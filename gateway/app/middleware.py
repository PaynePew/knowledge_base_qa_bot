"""Shallow module per Ousterhout. Public surface: ``ProdMiddleware``,
``read_sem``, ``admin_sem``, ``read_saturated``, ``is_provider_quota_error``,
``READ_PATHS``, ``ADMIN_PATHS``.

Production overload + cost-protection middleware for the demo VPS deploy
(issue #269).  One ASGI middleware on the Gateway parent app enforces five
guards, in this order, for every request:

  1. **Admin kill-switch** — if ``KB_ADMIN_TOKEN`` is *set*, admin/mutating
     heavy paths require ``Authorization: Bearer <token>`` (401 otherwise).
     Unset (the demo default) → admin paths are open so the Console works
     unauthenticated.  Read paths are never token-gated.
  2. **Daily USD budget** — heavy paths are charged a conservative per-endpoint
     estimate (see ``budget``); once the UTC-day total reaches
     ``KB_DAILY_USD_CAP`` they get 503 ``{"detail":"daily demo budget reached"}``.
  3. **Concurrency caps** — a read semaphore (``KB_MAX_INFLIGHT``, default 6)
     over the answer paths and an admin semaphore (``KB_MAX_ADMIN``, default 2)
     over the index/mutate paths.  Over-limit → 503.  The read semaphore's
     saturation is what ``GET /healthz/shed`` reports (``read_saturated()``).
  4. **Graceful provider failure** — an OpenAI ``insufficient_quota`` / 429
     bubbling out of a *non-streaming* heavy handler is mapped to a friendly
     503 instead of an unhandled 500 / worker crash (the streaming path already
     renders provider errors as a terminal SSE ``error`` event in
     ``gateway/app/routes.py`` — AC5).

All conditional logic that *decides an answer* lives in the deep stacks; this
module only wires guards around them (CODING_STANDARD §2.3).  Single-worker
model: the semaphores and budget are plain in-process objects (§2.6/§2.7).

Path matching uses the **full mounted path** with any trailing slash stripped
(``/wiki/chat/`` == ``/wiki/chat``).  Three parameterized heavy paths are
canonicalised before classification and charging — an exact-match set
cannot see a per-slug concrete path, and an unclassified path bypasses ALL
guards (the #376 bug class): ``POST /wiki/qa/{slug}/refile`` (ADR-0026
decision 1, issue #380) to ``QA_REFILE_TEMPLATE``, ``DELETE
/wiki/pages/{slug}`` (ADR-0025, issue #381) to ``PAGES_DELETE_TEMPLATE``,
and ``POST /wiki/pages/{slug}/aliases`` (ADR-0030 decision 3, issue #409) to
``ALIAS_ASSIGN_TEMPLATE`` — the latter two call no LLM (Confirmed
Remediation / Direct-class assign-alias, ADR-0024/ADR-0030 Invariants) but
still mutate the live corpus, so they stay admin-semaphore + kill-switch
gated even at a $0.00 budget estimate. ``POST /wiki/qa/promote-batch``
(ADR-0023, issue #382) needs no such canonicalisation — the path carries no
slug — but is classified as ``ADMIN_PATHS`` for the same reason as the
delete path: no LLM call, but a batch of live-corpus mutations plus a BM25
reindex is exactly the kind of write the admin gate exists for.
"""

from __future__ import annotations

import json
import os
import re
import threading

import openai
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import budget as _budget
from .logger import log_event as _log_event

# ---------------------------------------------------------------------------
# Heavy-path classification (full mounted paths, trailing slash stripped).
# ---------------------------------------------------------------------------

# Read / answer paths — gated by the read semaphore and the budget guard.
READ_PATHS: frozenset[str] = frozenset(
    {
        "/chat/stream",
        "/wiki/chat",
        "/rag/chat",
    }
)

# Canonical billing/classification key for the one parameterized heavy path.
# ``qa.refile`` runs a full grounded answer round through the chat pipeline
# (LLM draft + verify) and demotes a live page in place, so every concrete
# ``/wiki/qa/<slug>/refile`` must hit the admin guards; ``_canonical_path``
# collapses them to this template (slug is a single path segment — FastAPI's
# ``{slug}`` cannot contain ``/``, so the regex cannot under-match).
QA_REFILE_TEMPLATE = "/wiki/qa/{slug}/refile"
_QA_REFILE_RE = re.compile(r"^/wiki/qa/[^/]+/refile$")

# Canonical billing/classification key for the C11 orphan-delete path (ADR-0025,
# issue #381). Excludes the "reconcile" literal segment so it never collapses
# ``/wiki/pages/reconcile`` (a distinct, already-classified path) onto this
# template; ``/wiki/pages/collision/...`` has an extra path segment and never
# matches the single-segment ``[^/]+$`` shape in the first place.
PAGES_DELETE_TEMPLATE = "/wiki/pages/{slug}"
_PAGES_DELETE_RE = re.compile(r"^/wiki/pages/(?!reconcile$)[^/]+$")

# Canonical billing/classification key for the assign-alias path (ADR-0030
# decision 3, issue #409). Excludes "reconcile"/"collision" as the slug
# segment, mirroring PAGES_DELETE_TEMPLATE's own exclusion, so a hypothetical
# page slug of exactly that name can never be confused with the (distinctly
# classified) reconcile/collision sub-routes — those never carry a further
# "/aliases" segment today, but the same defensive shape keeps this pattern
# honest if that ever changes.
ALIAS_ASSIGN_TEMPLATE = "/wiki/pages/{slug}/aliases"
_ALIAS_ASSIGN_RE = re.compile(r"^/wiki/pages/(?!reconcile$|collision$)[^/]+/aliases$")

# Admin / index / mutating paths — gated by the admin semaphore, the budget
# guard, AND the optional admin-token kill-switch.
ADMIN_PATHS: frozenset[str] = frozenset(
    {
        QA_REFILE_TEMPLATE,  # ADR-0026 decision 1: chained re-file — LLM re-synthesis + grounding + demote (issue #380)
        "/rag/index",
        "/wiki/index",
        "/wiki/ingest",
        "/wiki/lint",
        "/wiki/import",
        "/upload",
        "/hybrid/index",  # ADR-0022: operator-triggered dense re-embed (issue #348)
        "/wiki/pages/reconcile",  # ADR-0028: C5 reconcile draft — LLM draft + grounding (issue #376)
        "/wiki/pages/reconcile/apply",  # ADR-0028: grounding re-check + two-page rewrite + reindex (issue #376)
        "/wiki/pages/collision/merge",  # ADR-0028: C4 merge draft — LLM draft + grounding (issue #378)
        "/wiki/pages/collision/merge/apply",  # ADR-0028: grounding re-check + base rewrite + variant deletes + reindex (issue #378)
        "/wiki/pages/collision/differentiate",  # ADR-0028: C4 differentiate draft — LLM draft + grounding (issue #378)
        "/wiki/pages/collision/differentiate/apply",  # ADR-0028: grounding re-check + N-page rewrite + reindex (issue #378)
        PAGES_DELETE_TEMPLATE,  # ADR-0025: C11 Confirmed orphan-delete — no LLM, still a live-corpus mutation (issue #381)
        "/wiki/qa/promote-batch",  # ADR-0023: Direct-tier batch promote — no LLM, still a live-corpus mutation (issue #382)
        ALIAS_ASSIGN_TEMPLATE,  # ADR-0030 decision 3: Direct-tier assign-alias — no LLM, still a live-corpus mutation (issue #409)
    }
)


def _read_int(name: str, default: int) -> int:
    """Read a positive int env var at construction time, falling back safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Module-level semaphores — read once at import so tests can drain/restore them.
# BoundedSemaphore catches an accidental over-release (a programming error).
read_sem = threading.BoundedSemaphore(_read_int("KB_MAX_INFLIGHT", 6))
admin_sem = threading.BoundedSemaphore(_read_int("KB_MAX_ADMIN", 2))


def read_saturated() -> bool:
    """True when the read semaphore has no free slots (drives /healthz/shed).

    Probes by a non-blocking acquire: if it succeeds there was a free slot, so
    we immediately release and report *not* saturated; if it fails, every slot
    is held and we report saturated.  Admin-semaphore state is intentionally
    ignored — only read saturation sheds (AC2).
    """
    if read_sem.acquire(blocking=False):
        read_sem.release()
        return False
    return True


def is_provider_quota_error(exc: BaseException) -> bool:
    """True for an OpenAI ``insufficient_quota`` / 429 rate-limit failure.

    ``insufficient_quota`` arrives as a ``RateLimitError`` (HTTP 429) from the
    OpenAI SDK; ``APIStatusError`` with ``status_code == 429`` covers the
    generic 429.  We treat both as a transient provider exhaustion to map to a
    friendly 503 (AC5).
    """
    if isinstance(exc, openai.RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(exc, openai.APIError) and status == 429


def _normalise_path(raw_path: str) -> str:
    """Return the path with a single trailing slash stripped (root preserved)."""
    if len(raw_path) > 1 and raw_path.endswith("/"):
        return raw_path.rstrip("/")
    return raw_path


def _canonical_path(raw_path: str) -> str:
    """Normalise, then collapse a parameterized heavy path to its template key.

    ``/wiki/qa/<slug>/refile`` → ``QA_REFILE_TEMPLATE``,
    ``/wiki/pages/<slug>`` → ``PAGES_DELETE_TEMPLATE``, and
    ``/wiki/pages/<slug>/aliases`` → ``ALIAS_ASSIGN_TEMPLATE`` so the
    exact-match classification sets and the budget table see one stable key
    per endpoint (also keeps per-slug cardinality out of the budget/log
    keys). Every other path passes through unchanged.
    """
    path = _normalise_path(raw_path)
    if _QA_REFILE_RE.fullmatch(path):
        return QA_REFILE_TEMPLATE
    if _ALIAS_ASSIGN_RE.fullmatch(path):
        return ALIAS_ASSIGN_TEMPLATE
    if _PAGES_DELETE_RE.fullmatch(path):
        return PAGES_DELETE_TEMPLATE
    return path


def _bearer_token(scope: Scope) -> str | None:
    """Extract the Bearer token from the ASGI ``Authorization`` header, if any."""
    for key, value in scope.get("headers", []):
        if key == b"authorization":
            text = value.decode("latin-1")
            if text.lower().startswith("bearer "):
                return text[len("bearer ") :].strip()
    return None


async def _send_json(send: Send, status: int, payload: dict) -> None:
    """Emit a minimal JSON ASGI response (used for all reject short-circuits)."""
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class ProdMiddleware:
    """Pure-ASGI overload + cost-protection guard (issue #269).

    Pure ASGI (not ``BaseHTTPMiddleware``) so a held semaphore is released only
    when the *whole* response — including a streaming SSE body — has finished
    sending.  ``BaseHTTPMiddleware`` would release on the response object before
    the stream drains, undercounting in-flight reads and breaking shed.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = _canonical_path(scope.get("path", ""))
        is_read = path in READ_PATHS
        is_admin = path in ADMIN_PATHS

        # Non-heavy paths (/, /healthz, /static, /read/*, favicons, ...) bypass
        # every guard — liveness and the UI must always be reachable (AC1).
        if not (is_read or is_admin):
            await self.app(scope, receive, send)
            return

        # 1. Admin kill-switch (admin paths only; read paths never token-gated).
        if is_admin:
            token = os.getenv("KB_ADMIN_TOKEN")
            if token and _bearer_token(scope) != token:
                await _send_json(send, 401, {"detail": "admin token required"})
                return

        # 2. Daily USD budget GATE — reject once the UTC-day total has hit the
        #    cap. The actual charge is deferred until AFTER the concurrency gate
        #    admits the request (below), so a request that is shed by the
        #    semaphore — and therefore never reaches the handler / OpenAI — does
        #    NOT consume budget. Charging here instead would let a burst of
        #    shed traffic drain the $/day ceiling on zero real spend and 503 all
        #    heavy paths for the rest of the UTC day.
        if _budget.budget.over_cap():
            _log_event("budget_block", f"path={path} cap={_budget.budget.cap_usd}")
            await _send_json(send, 503, {"detail": "daily demo budget reached"})
            return

        # 3. Concurrency cap — non-blocking acquire; over-limit sheds immediately.
        sem = read_sem if is_read else admin_sem
        if not sem.acquire(blocking=False):
            _log_event("overload_shed", f"path={path} kind={'read' if is_read else 'admin'}")
            await _send_json(send, 503, {"detail": "server busy, please retry"})
            return

        # Admitted past both gates → only now charge the conservative estimate.
        # Only requests that actually proceed to the handler (and may call
        # OpenAI) consume budget; over-cap and shed rejections never do.
        _budget.budget.charge(path)

        try:
            # 4. Graceful provider failure for NON-streaming heavy handlers.
            #    The streaming /chat/stream path commits HTTP 200 before any LLM
            #    call and renders provider errors as a terminal SSE error event
            #    (gateway/app/routes.py), so we never wrap it here — an exception
            #    after the response has started cannot be turned into a 503.
            await self._call_with_quota_guard(scope, receive, send, path, is_read)
        finally:
            sem.release()

    async def _call_with_quota_guard(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        path: str,
        is_read: bool,
    ) -> None:
        """Run the downstream app, mapping a provider quota/429 to a 503.

        Tracks whether the response has started: once the first
        ``http.response.start`` is sent we can no longer change the status, so a
        late provider error (the streaming case) must fall through to the app's
        own terminal-error handling rather than this 503 mapping.
        """
        started = False

        async def _tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, _tracking_send)
        except Exception as exc:  # noqa: BLE001 — last-resort transport guard
            if is_provider_quota_error(exc) and not started:
                _log_event("provider_quota_503", f"path={path} exc={type(exc).__name__}")
                await _send_json(
                    send,
                    503,
                    {"detail": "LLM provider quota exhausted, please retry later."},
                )
                return
            # Not a quota error, or the response already started — re-raise so
            # the app's own error handling (or the ASGI server) deals with it.
            raise
