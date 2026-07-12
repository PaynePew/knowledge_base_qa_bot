"""Deep module per Ousterhout. Public surface: ``sse_sem``, ``run_with_heartbeat``,
``encode_comment``, ``IDLE_TIMEOUT_SEC``, ``HEARTBEAT_INTERVAL_SEC``, ``MAX_CONCURRENT``.

SSE slow-client capacity guard for ``POST /chat/stream`` (issue #599, split
from #580's ops availability hardening). An open SSE connection occupies a
serving slot for the FULL duration of the stream (sources -> LLM draft+verify
-> tokens -> done — several seconds), unlike a normal request/response cycle.
A handful of slow or abandoned readers can therefore exhaust capacity even
while ``KB_MAX_INFLIGHT`` (the general read cap ``/chat/stream`` already
shares with ``/wiki/chat``/``/rag/chat`` via ``gateway.app.middleware``)
still has room. This module adds three independently-tunable guards, wired
into ``ProdMiddleware`` for the ``/chat/stream`` path only:

  1. **Per-server concurrent SSE cap** (``KB_SSE_MAX_CONCURRENT``) — a
     dedicated semaphore (``sse_sem``), checked in ADDITION to (not instead
     of) the general read semaphore. Over-limit -> the same clean
     ``{"detail": "server busy, please retry"}`` 503 shape ``read_sem``/
     ``admin_sem`` already use — the reader UI already handles this class of
     rejection on this exact path, so no client-side change is needed.
  2. **Heartbeat** — a ``: ping`` SSE comment frame emitted every
     ``KB_SSE_HEARTBEAT_INTERVAL_SEC`` while the downstream generator sits
     idle between real frames (typically during the several-second draft+
     verify LLM call). ADR-0009: a comment frame is invisible to the event
     contract — never a new ``event:`` type — so ``token``/``status``/``done``
     framing is unchanged; every conforming SSE parser (including the
     hand-rolled one in ``gateway/static/index.html``) ignores a line that
     does not start with ``event:``/``data:``.
  3. **Idle-read timeout** (``KB_SSE_IDLE_TIMEOUT_SEC``) — if a single
     ``send()`` call to the ASGI transport does not complete within this many
     seconds, the client has made no read progress (its OS socket receive
     buffer is full and draining nothing), so the stream is closed
     server-side instead of held open indefinitely.

``run_with_heartbeat`` is the mechanism behind (2) and (3): it runs the
downstream call as a background task, funnels its ASGI ``send`` messages
through a queue, and pumps that queue on the event loop — injecting a
heartbeat frame whenever the queue sits idle for a full interval, and
wrapping every real ``send()`` in the idle-read timeout. This works because
Starlette's ``StreamingResponse`` drives a *sync* generator (the per-stack
``stream_query()`` chain, via ``anyio``'s ``iterate_in_threadpool``) — the
event loop stays free to run this pump concurrently with the blocking LLM
call underneath it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import Awaitable, Callable

from starlette.types import Message, Send

from .logger import log_event as _log_event

# ---------------------------------------------------------------------------
# Config — read once at import (restart to apply), mirroring
# gateway.app.middleware._read_int / gateway.app.budget._read_cap
# (CODING_STANDARD §2.6/§2.7: module-level globals read once at startup).
# ---------------------------------------------------------------------------


def _read_positive_float(name: str, default: float) -> float:
    """Read a positive float env var at import time, falling back safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_positive_int(name: str, default: int) -> int:
    """Read a positive int env var at import time, falling back safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Idle-read timeout: close a stream making zero read progress for this long
# (default 45s — issue #599 scope decision).
IDLE_TIMEOUT_SEC = _read_positive_float("KB_SSE_IDLE_TIMEOUT_SEC", 45.0)
# Heartbeat cadence: default 15s, a third of the default idle timeout, so at
# least two heartbeats land — keeping proxies/clients alive and surfacing a
# dead peer — before the idle-read timeout would otherwise fire.
HEARTBEAT_INTERVAL_SEC = _read_positive_float("KB_SSE_HEARTBEAT_INTERVAL_SEC", 15.0)
# Per-server concurrent SSE cap: default 6, the same order of magnitude as
# KB_MAX_INFLIGHT's default (the general read cap /chat/stream already
# shares with /wiki/chat, /rag/chat). A SEPARATE, dedicated knob: an SSE
# connection is held for the full stream duration (seconds), far longer than
# a typical request/response cycle, so operators may want to tune it
# independently of the general read cap.
MAX_CONCURRENT = _read_positive_int("KB_SSE_MAX_CONCURRENT", 6)

# Module-level semaphore — read once at import so tests can drain/restore it
# (mirrors gateway.app.middleware.read_sem / admin_sem).
sse_sem = threading.BoundedSemaphore(MAX_CONCURRENT)


def encode_comment(text: str) -> bytes:
    """Encode one SSE comment frame (RFC 8895 §3.1) as heartbeat bytes.

    A line starting with ``:`` is a comment — every conforming SSE parser
    ignores it (including the hand-rolled one in
    ``gateway/static/index.html`` and its Python mirror in
    ``gateway/tests/test_sse_parser.py``), so a heartbeat never collides with
    the ``event:``/``data:`` event contract (ADR-0009).
    """
    return f": {text}\n\n".encode()


_PING_FRAME = encode_comment("ping")


async def run_with_heartbeat(
    call: Callable[[Send], Awaitable[None]],
    send: Send,
    *,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SEC,
    idle_timeout: float = IDLE_TIMEOUT_SEC,
    log_summary: str = "",
) -> None:
    """Run ``call(queued_send)`` while pumping its ASGI messages through ``send``.

    Injects a ``: ping`` heartbeat comment frame into any gap between real
    messages longer than ``heartbeat_interval``, and enforces ``idle_timeout``
    on every ``send()`` to the real ASGI transport — a client that has
    stopped draining its socket makes ``send()`` block, so a timed-out
    ``send()`` IS "no read progress for idle_timeout seconds" (issue #599).

    ``call`` receives a queue-backed ``send`` (not the real one) so its
    messages can be interleaved with heartbeat frames on the event loop
    while ``call`` itself runs as a background task — this is what lets a
    heartbeat fire DURING the several-second gap while ``call`` is blocked
    inside a downstream LLM call (that blocking happens on a worker thread
    via ``anyio.to_thread``, per the module docstring, so the event loop
    driving this pump stays free).

    On an idle-read timeout, the background task is cancelled and this
    coroutine returns (rather than raising) — ``StreamingResponse`` has
    already committed HTTP 200, so there is nothing left to map to an HTTP
    error; the connection simply closes. Any other exception surfacing from
    ``call`` (the downstream generator already turns every known failure
    into a terminal SSE ``error`` event — see ``gateway/app/routes.py`` —
    so this is a last-resort path) is likewise swallowed here: the ``finally``
    below always drains ``runner`` so the pump never hangs waiting on it.
    """
    queue: asyncio.Queue[Message | None] = asyncio.Queue()

    async def _queued_send(message: Message) -> None:
        await queue.put(message)

    async def _run() -> None:
        try:
            await call(_queued_send)
        finally:
            # Always unblocks the pump loop below, whether `call` returned
            # normally or raised — guarantees no hang regardless of outcome.
            await queue.put(None)

    runner = asyncio.ensure_future(_run())
    started = False
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
            except TimeoutError:
                if not started:
                    continue  # headers not sent yet — nothing to comment onto
                await asyncio.wait_for(
                    send({"type": "http.response.body", "body": _PING_FRAME, "more_body": True}),
                    timeout=idle_timeout,
                )
                continue

            if message is None:
                return

            await asyncio.wait_for(send(message), timeout=idle_timeout)
            if message["type"] == "http.response.start":
                started = True
            elif message["type"] == "http.response.body" and not message.get("more_body", True):
                return
    except TimeoutError:
        # idle-read timeout (AC2): the client made no read progress within
        # idle_timeout seconds on either a real frame or a heartbeat.
        _log_event(
            "sse_idle_timeout",
            f"idle_timeout_sec={idle_timeout} {log_summary}".strip(),
        )
    finally:
        if not runner.done():
            runner.cancel()
        # Last-resort cleanup: drain `runner` so a downstream exception or a
        # cancellation is never left un-retrieved, and never propagates out
        # of this cleanup path (the response may already be mid-stream).
        with contextlib.suppress(BaseException):  # noqa: BLE001
            await runner
