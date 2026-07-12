"""Shallow module per Ousterhout. Public surface: ``ProdMiddleware``,
``read_sem``, ``admin_sem``, ``rate_limiter``, ``read_saturated``,
``is_provider_quota_error``, ``READ_PATHS``, ``ADMIN_PATHS``.

Production overload + cost-protection middleware for the demo VPS deploy
(issue #269).  One ASGI middleware on the Gateway parent app enforces six
guards, in this order, for every request:

  1. **Admin kill-switch** — if ``KB_ADMIN_TOKEN`` is *set*, admin/mutating
     heavy paths require ``Authorization: Bearer <token>`` (401 otherwise).
     Unset (the demo default) → admin paths are open so the Console works
     unauthenticated.  Read paths are never token-gated.
  2. **Daily USD budget** — heavy paths are charged a conservative per-endpoint
     estimate (see ``budget``); once the UTC-day total reaches the relevant
     ceiling they get 503 ``{"detail":"daily demo budget reached"}``. READ_PATHS
     gate on the full ``KB_DAILY_USD_CAP``; ADMIN_PATHS gate on the reduced
     ``KB_DAILY_USD_CAP - KB_READ_RESERVED_USD`` ceiling (issue #598 Slice A),
     so admin/mutating traffic can never spend the read-reserved floor.
  3. **Per-IP rate limit** — a fixed-window counter (``KB_RATE_LIMIT_PER_IP``,
     default 30 requests / 5 min per IP; ``0`` disables) over every heavy path,
     read or admin (see ``ratelimit``). Over-limit → 429
     ``{"detail":"rate limited, please retry later"}`` (issue #598 Slice A).
     Runs BEFORE the concurrency gate, mirroring the budget guard's
     charge-after-admission rationale: a rate-limited request never occupies
     a semaphore slot or charges the daily budget.
  4. **Concurrency caps** — a read semaphore (``KB_MAX_INFLIGHT``, default 6)
     over the answer paths and an admin semaphore (``KB_MAX_ADMIN``, default 2)
     over the index/mutate paths.  Over-limit → 503.  The read semaphore's
     saturation is what ``GET /healthz/shed`` reports (``read_saturated()``).
     ``/chat/stream`` additionally passes through a dedicated, tighter SSE cap
     (``KB_SSE_MAX_CONCURRENT`` — see ``gateway/app/sse_capacity.py``, issue
     #599): an SSE connection is held for the FULL stream duration (seconds),
     unlike a normal request/response cycle, so a slow/stalled reader can
     starve capacity even while the general read semaphore still has room.
  5. **Graceful provider failure** — an OpenAI ``insufficient_quota`` / 429
     bubbling out of a *non-streaming* heavy handler is mapped to a friendly
     503 instead of an unhandled 500 / worker crash (the streaming path already
     renders provider errors as a terminal SSE ``error`` event in
     ``gateway/app/routes.py`` — AC5).
  6. **SSE heartbeat + idle-read timeout** — ``/chat/stream`` only (issue
     #599): a ``: ping`` comment frame is injected every
     ``KB_SSE_HEARTBEAT_INTERVAL_SEC`` while the downstream generator is
     idle, and any ``send()`` to the client that blocks past
     ``KB_SSE_IDLE_TIMEOUT_SEC`` (no read progress) closes the stream
     server-side. See ``gateway/app/sse_capacity.py`` for the mechanism.

All conditional logic that *decides an answer* lives in the deep stacks; this
module only wires guards around them (CODING_STANDARD §2.3).  Single-worker
model: the semaphores and budget are plain in-process objects (§2.6/§2.7).

Path matching uses the **full mounted path** with any trailing slash stripped
(``/wiki/chat/`` == ``/wiki/chat``).  Seven parameterized heavy paths are
canonicalised before classification and charging — an exact-match set
cannot see a per-slug concrete path, and an unclassified path bypasses ALL
guards (the #376 bug class): ``POST /wiki/qa/{slug}/refile`` (ADR-0026
decision 1, issue #380) to ``QA_REFILE_TEMPLATE``, ``DELETE
/wiki/pages/{slug}`` (ADR-0025, issue #381) to ``PAGES_DELETE_TEMPLATE``,
``POST /wiki/pages/{slug}/aliases`` (ADR-0030 decision 3, issue #409) to
``ALIAS_ASSIGN_TEMPLATE``, and ``DELETE /wiki/pages/{slug}/aliases/{alias}``
(ADR-0030 extension, issue #491) to ``ALIAS_REMOVE_TEMPLATE`` — none of the
four call any LLM (Confirmed Remediation / Direct-class assign-alias /
remove-alias, ADR-0024/ADR-0030 Invariants) but each still mutates the live
corpus, so they stay admin-semaphore + kill-switch gated even at a $0.00
budget estimate. ``POST /wiki/qa/promote-batch``
(ADR-0023, issue #382) needs no such canonicalisation — the path carries no
slug — but is classified as ``ADMIN_PATHS`` for the same reason as the
delete path: no LLM call, but a batch of live-corpus mutations plus a BM25
reindex is exactly the kind of write the admin gate exists for.

Issue #583's coverage audit (a dynamic test asserting every mounted mutating
route is classified — see ``gateway/tests/test_admin_path_coverage.py``)
caught three more mounted-but-unclassified single-item QA routes, the same
"#376 bug class" repeated: ``POST /wiki/qa/{slug}/promote`` (Phase 6 Slice
6-4) and ``POST /wiki/qa/{slug}/demote`` (ADR-0037, issue #535) canonicalise
via ``QA_PROMOTE_TEMPLATE`` / ``QA_DEMOTE_TEMPLATE`` — no LLM call, but each
flips a Filed Answer's lifecycle status and triggers a full BM25 reindex.
``DELETE /wiki/qa/{slug}`` (ADR-0012) and ``PUT /wiki/qa/{slug}`` (ADR-0026
decision 2) share one path with two mutating verbs and canonicalise via the
single ``QA_ITEM_TEMPLATE`` (excludes the literal ``promote-batch`` segment
so it can never be swallowed by this template, mirroring
``PAGES_DELETE_TEMPLATE``'s own ``reconcile`` exclusion below).

``GET /wiki/transcribe/jobs/{job_id}`` and ``GET /wiki/transcribe/page-count``
(issue #447) are deliberately left OUT of both sets — each is a read-only,
in-memory or mechanical lookup (job-status poll; PDF page count) with no LLM
call and no corpus mutation, the same rationale that already leaves
``GET /read/*`` and ``GET /healthz*`` unclassified. ``POST
/wiki/transcribe/batch`` is the one that actually starts real (billed) work,
so it — not its poll/preflight siblings — is the path that needs the admin
gate. ``POST /wiki/import/jobs`` / ``GET /wiki/import/jobs/{job_id}``
(issue #497) split identically: the submit starts the same billed work as
``/wiki/import`` and is classified; its poll sibling stays unclassified.

``POST /sources/retire`` / ``POST /sources/restore`` (issue #604, ADR-0041)
mutate the live corpus (a Source file move) with no LLM call, the same
rationale as ``PAGES_DELETE_TEMPLATE`` / the alias endpoints — both are
classified below. Neither is parameterized (the relpath/timestamp travel in
the request body, not the path) so, unlike those two, no canonicalisation
template is needed. ``POST /sources/rename`` (issue #605, ADR-0041 decision
5) is classified for the same reason plus more: a Source-file move AND a
mechanical re-point of every derived page's frontmatter AND a BM25 reindex
— still no LLM call, but the heaviest of the three lifecycle acts, so it
stays admin-semaphore + kill-switch gated exactly like its S1 siblings.
``GET /sources/{relpath}/impact`` and ``GET /sources/trash`` are
deliberately left OUT of both sets: read-only, no LLM, no mutation — the
same rationale that already leaves ``GET /read/*`` and ``GET
/pages/resolution-map`` unclassified.
"""

from __future__ import annotations

import json
import os
import re
import threading

import openai
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import budget as _budget
from . import ratelimit as _ratelimit
from . import sse_capacity as _sse_capacity
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

# Canonical billing/classification key for the single-item Promote path
# (Phase 6 Slice 6-4). Found unclassified by the issue #583 coverage audit —
# no LLM call, but flips a Filed Answer draft -> live and triggers a full
# BM25 reindex (the #376 bug class again).
QA_PROMOTE_TEMPLATE = "/wiki/qa/{slug}/promote"
_QA_PROMOTE_RE = re.compile(r"^/wiki/qa/[^/]+/promote$")

# Canonical billing/classification key for the single-item Demote path
# (ADR-0037, issue #535) — the reversible inverse of Promote. Same #583
# audit gap: no LLM call, but a live -> draft lifecycle flip plus reindex.
QA_DEMOTE_TEMPLATE = "/wiki/qa/{slug}/demote"
_QA_DEMOTE_RE = re.compile(r"^/wiki/qa/[^/]+/demote$")

# Canonical billing/classification key for the single-QA-item path — covers
# BOTH ``DELETE /wiki/qa/{slug}`` (discard an inert draft, ADR-0012) and
# ``PUT /wiki/qa/{slug}`` (edit a draft in place, ADR-0026 decision 2): one
# path, two mutating verbs, same #583 audit gap. Excludes "promote-batch" so
# the already-classified literal ``/wiki/qa/promote-batch`` route (no slug
# segment) is never swallowed by this template, mirroring
# PAGES_DELETE_TEMPLATE's own "reconcile" exclusion below.
QA_ITEM_TEMPLATE = "/wiki/qa/{slug}"
_QA_ITEM_RE = re.compile(r"^/wiki/qa/(?!promote-batch$)[^/]+$")

# Canonical billing/classification key for the C11 orphan-delete path (ADR-0025,
# issue #381). Excludes the "reconcile" literal segment so it never collapses
# ``/wiki/pages/reconcile`` (a distinct, already-classified path) onto this
# template; ``/wiki/pages/collision/...`` has an extra path segment and never
# matches the single-segment ``[^/]+$`` shape in the first place. Also
# excludes "resolution-map": ``GET /wiki/pages/resolution-map`` (ADR-0030
# Invariant — the ONE shared slug map every wikilink surface reads) is a
# read-only, LLM-free lookup that was silently swallowed by this template
# and classified ADMIN, so enabling the KB_ADMIN_TOKEN kill-switch would
# have 401'd every wikilink render (found by issue #583's bare-fetch audit;
# it also burned an admin-semaphore slot per call). Accepted trade-off,
# identical to the "reconcile" exclusion above: a page whose slug is
# literally "resolution-map" (or "reconcile") cannot be admin-gated for
# delete — prompt-derived slugs of that exact shape do not occur in this
# corpus, and the alias templates below keep their own exclusion lists
# unchanged (a slug named "resolution-map" WOULD still be gated there).
PAGES_DELETE_TEMPLATE = "/wiki/pages/{slug}"
_PAGES_DELETE_RE = re.compile(r"^/wiki/pages/(?!reconcile$|resolution-map$)[^/]+$")

# Canonical billing/classification key for the assign-alias path (ADR-0030
# decision 3, issue #409). Excludes "reconcile"/"collision" as the slug
# segment, mirroring PAGES_DELETE_TEMPLATE's own exclusion, so a hypothetical
# page slug of exactly that name can never be confused with the (distinctly
# classified) reconcile/collision sub-routes — those never carry a further
# "/aliases" segment today, but the same defensive shape keeps this pattern
# honest if that ever changes.
ALIAS_ASSIGN_TEMPLATE = "/wiki/pages/{slug}/aliases"
_ALIAS_ASSIGN_RE = re.compile(r"^/wiki/pages/(?!reconcile$|collision$)[^/]+/aliases$")

# Canonical billing/classification key for the remove-alias path (ADR-0030
# extension, issue #491) — the mirror-image of ALIAS_ASSIGN_TEMPLATE, one path
# segment deeper (the alias itself). Same "reconcile"/"collision" exclusion on
# the slug segment, same defensive rationale: neither sub-route carries a
# further "/aliases/<alias>" shape today, but the exclusion keeps this
# canonicalisation honest if that ever changes. Anchored with its own ``$``
# so it can never be confused with ALIAS_ASSIGN_TEMPLATE's 3-segment shape
# (that regex's own ``$`` immediately follows "aliases", so a 4-segment path
# never matches it either — the two templates are mutually exclusive).
ALIAS_REMOVE_TEMPLATE = "/wiki/pages/{slug}/aliases/{alias}"
_ALIAS_REMOVE_RE = re.compile(r"^/wiki/pages/(?!reconcile$|collision$)[^/]+/aliases/[^/]+$")

# Admin / index / mutating paths — gated by the admin semaphore, the budget
# guard, AND the optional admin-token kill-switch.
ADMIN_PATHS: frozenset[str] = frozenset(
    {
        QA_REFILE_TEMPLATE,  # ADR-0026 decision 1: chained re-file — LLM re-synthesis + grounding + demote (issue #380)
        QA_PROMOTE_TEMPLATE,  # Phase 6 Slice 6-4: draft -> live status flip + reindex, no LLM (issue #583 audit)
        QA_DEMOTE_TEMPLATE,  # ADR-0037: live -> draft status flip + reindex, no LLM (issue #535; issue #583 audit)
        QA_ITEM_TEMPLATE,  # ADR-0012 delete / ADR-0026 decision 2 edit: draft-only mutation, no LLM (issue #583 audit)
        "/rag/index",
        "/wiki/index",
        "/wiki/ingest",
        "/wiki/lint",
        "/wiki/import",
        "/wiki/import/jobs",  # issue #497: async submit for the SAME import surface as /wiki/import — classified for the same reason /wiki/transcribe/batch is (the submit, not its poll sibling, starts the billed work)
        "/wiki/transcribe",  # issue #460: forced Transcribe — same cost-exposure surface as /wiki/import's auto-route
        "/wiki/transcribe/batch",  # issue #447: async submit for the SAME force-transcribe surface as /wiki/transcribe — was absent from this set (issue #459 added the route after #460 classified only the sync one), so it bypassed the admin token/semaphore entirely until this line
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
        ALIAS_REMOVE_TEMPLATE,  # ADR-0030 extension: Direct-tier remove-alias — no LLM, still a live-corpus mutation (issue #491)
        "/sources/retire",  # ADR-0041: Confirmed retire — no LLM, a Source-file move (issue #604)
        "/sources/restore",  # ADR-0041: Direct restore — no LLM, a Source-file move (issue #604)
        "/sources/rename",  # ADR-0041 decision 5: Direct rename — no LLM, a Source-file move + citation re-point + reindex (issue #605)
    }
)

# POST /feedback and GET /feedback (issue #558) are deliberately absent from
# BOTH sets above: the ADMIN_PATHS precedent keys on *live-corpus* mutation,
# and Reader Feedback is opinion data ABOUT the corpus that never enters it
# (CONTEXT.md "Reader Feedback") — so it stays a public, ungated surface with
# its own payload/store-size caps (gateway/app/feedback.py) instead.


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

# Module-level per-IP rate limiter (issue #598 Slice A) — constructed here
# (not in ``ratelimit.py``) so it is recreated fresh whenever THIS module is
# reloaded, exactly like ``read_sem`` / ``admin_sem`` above; see
# ``ratelimit.py``'s trailing comment for why that matters for test isolation.
rate_limiter = _ratelimit.RateLimiter(
    limit=_ratelimit.RATE_LIMIT_PER_IP, window_sec=_ratelimit.RATE_LIMIT_WINDOW_SEC
)


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
    ``/wiki/pages/<slug>`` → ``PAGES_DELETE_TEMPLATE``,
    ``/wiki/pages/<slug>/aliases`` → ``ALIAS_ASSIGN_TEMPLATE``, and
    ``/wiki/pages/<slug>/aliases/<alias>`` → ``ALIAS_REMOVE_TEMPLATE`` so the
    exact-match classification sets and the budget table see one stable key
    per endpoint (also keeps per-slug/alias cardinality out of the
    budget/log keys). Every other path passes through unchanged.
    """
    path = _normalise_path(raw_path)
    if _QA_REFILE_RE.fullmatch(path):
        return QA_REFILE_TEMPLATE
    if _QA_PROMOTE_RE.fullmatch(path):
        return QA_PROMOTE_TEMPLATE
    if _QA_DEMOTE_RE.fullmatch(path):
        return QA_DEMOTE_TEMPLATE
    if _ALIAS_REMOVE_RE.fullmatch(path):
        return ALIAS_REMOVE_TEMPLATE
    if _ALIAS_ASSIGN_RE.fullmatch(path):
        return ALIAS_ASSIGN_TEMPLATE
    if _PAGES_DELETE_RE.fullmatch(path):
        return PAGES_DELETE_TEMPLATE
    if _QA_ITEM_RE.fullmatch(path):
        return QA_ITEM_TEMPLATE
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
        # The one SSE surface (issue #599) — drives the extra concurrency cap
        # and the heartbeat/idle-timeout wrap below, on top of the read guards.
        is_sse = path == "/chat/stream"

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

        # 2. Daily USD budget GATE — reject once the relevant ceiling has been
        #    hit. ADMIN_PATHS gate on the reduced over_admin_cap() ceiling
        #    (cap - KB_READ_RESERVED_USD, issue #598); READ_PATHS keep the
        #    unchanged over_cap() (full cap). The actual charge is deferred
        #    until AFTER the concurrency gate admits the request (below), so a
        #    request that is shed by the semaphore — and therefore never
        #    reaches the handler / OpenAI — does NOT consume budget. Charging
        #    here instead would let a burst of shed traffic drain the $/day
        #    ceiling on zero real spend and 503 all heavy paths for the rest
        #    of the UTC day.
        budget_exhausted = (
            _budget.budget.over_admin_cap() if is_admin else _budget.budget.over_cap()
        )
        if budget_exhausted:
            _log_event(
                "budget_block",
                f"path={path} cap={_budget.budget.cap_usd} read_reserved={_budget.budget.read_reserved_usd}",
            )
            await _send_json(send, 503, {"detail": "daily demo budget reached"})
            return

        # 3. Per-IP rate limit — heavy paths only (read AND admin, issue #598
        #    Slice A). Runs BEFORE the semaphore/charge so a rate-limited
        #    request never consumes budget or an inflight slot (same
        #    charge-after-admission rationale as the budget gate above).
        ip = _ratelimit.client_ip(scope)
        if not rate_limiter.allow(ip):
            _log_event("rate_limited", f"path={path} ip={ip}")
            await _send_json(send, 429, {"detail": "rate limited, please retry later"})
            return

        # 4. Concurrency cap — non-blocking acquire; over-limit sheds immediately.
        sem = read_sem if is_read else admin_sem
        if not sem.acquire(blocking=False):
            _log_event("overload_shed", f"path={path} kind={'read' if is_read else 'admin'}")
            await _send_json(send, 503, {"detail": "server busy, please retry"})
            return

        # 4b. SSE-specific concurrency cap (issue #599) — a SEPARATE, tighter
        # pool just for /chat/stream: an SSE connection is held for the FULL
        # stream duration (seconds), unlike a normal request/response cycle,
        # so a slow/stalled reader can starve capacity even while the general
        # read semaphore above still has room. Checked in ADDITION to (not
        # instead of) the read semaphore; over-limit releases the read-sem
        # slot just taken and sheds with the same busy-retry shape.
        if is_sse and not _sse_capacity.sse_sem.acquire(blocking=False):
            sem.release()
            _log_event("overload_shed", f"path={path} kind=sse")
            await _send_json(send, 503, {"detail": "server busy, please retry"})
            return

        # Admitted past every gate → only now charge the conservative estimate.
        # Only requests that actually proceed to the handler (and may call
        # OpenAI) consume budget; over-cap, rate-limited, and shed rejections
        # never do.
        _budget.budget.charge(path)

        try:
            # 5. Graceful provider failure for NON-streaming heavy handlers.
            #    The streaming /chat/stream path commits HTTP 200 before any LLM
            #    call and renders provider errors as a terminal SSE error event
            #    (gateway/app/routes.py), so we never wrap it here — an exception
            #    after the response has started cannot be turned into a 503.
            if is_sse:
                # 6. SSE heartbeat + idle-read timeout (issue #599) — wraps the
                #    same quota-guarded call, so the provider-failure mapping
                #    above still applies unchanged; this layer only injects
                #    heartbeat comment frames and enforces the idle-read cap.
                await _sse_capacity.run_with_heartbeat(
                    lambda queued_send: self._call_with_quota_guard(
                        scope, receive, queued_send, path, is_read
                    ),
                    send,
                    log_summary=f"path={path}",
                )
            else:
                await self._call_with_quota_guard(scope, receive, send, path, is_read)
        finally:
            sem.release()
            if is_sse:
                _sse_capacity.sse_sem.release()

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
