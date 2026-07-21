"""Shallow module per Ousterhout. Public surface: ``ErrorLoggingMiddleware``.

Gateway-wide unhandled-exception logging (issue #648).

Today an exception that escapes a route handler surfaces ONLY as a
traceback on the ASGI server's stderr — ``gateway/log.md`` (the Wiki Log's
Gateway channel, ``project-docs/log-kinds.md``) carries no record of it, so
an operator grepping the log sees a silent gap where the 500 happened. Every
*mapped* failure already has a kind (``budget_block``, ``provider_quota_503``,
...); this module closes the blind spot for the *unmapped* crash.

Design (per the issue's design notes): a dedicated ASGI wrapper, not new
logic threaded through ``ProdMiddleware``'s guard chain. ``main.py`` installs
this middleware AFTER ``ProdMiddleware`` (``app.add_middleware`` makes the
most-recently-added middleware the OUTERMOST one Starlette wraps), so it
observes any exception ``ProdMiddleware`` re-raises — including one from the
mounted ``/wiki`` and ``/rag`` sub-apps, which pass through the same
middleware stack as the parent app's own routes.

No double-logging: ``ProdMiddleware``'s OpenAI-quota mapping
(``provider_quota_503``) already handles its exception by sending a 503 and
RETURNING (not re-raising), so that exception never reaches here.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

from .logger import log_event as _log_event

# Streaming is out of scope (issue #648): mid-stream errors on this one path
# are already rendered as a terminal SSE ``error`` event by the existing
# plumbing (``gateway/app/routes.py``), so an exception logged here would
# either never occur for this path or would duplicate an already-reported
# failure.
_STREAM_PATH = "/chat/stream"

# Truncation bound for the logged exception message (issue #648 AC).
_MAX_MESSAGE_CHARS = 200


def _format_exc(exc: BaseException) -> str:
    """Render ``<ExcClass>: <message repr, truncated to <= 200 chars>``."""
    message = repr(str(exc))
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[:_MAX_MESSAGE_CHARS]
    return f"{type(exc).__name__}: {message}"


class ErrorLoggingMiddleware:
    """Pure-ASGI log-and-re-raise net for exceptions escaping a non-streaming
    request path (issue #648).

    Logs exactly one ``unhandled_error`` line to ``gateway/log.md`` via
    ``log_event``, then re-raises unchanged — the response stays whatever it
    is today (Starlette's own ``ServerErrorMiddleware`` 500) and the full
    traceback still reaches stderr. This middleware never swallows an
    exception and never touches ``send``, so it cannot change a response
    that has already started.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path", "") == _STREAM_PATH:
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        except Exception as exc:  # noqa: BLE001 — last-resort observability net
            path = scope.get("path", "")
            _log_event("unhandled_error", f"path={path} exc={_format_exc(exc)}")
            raise
