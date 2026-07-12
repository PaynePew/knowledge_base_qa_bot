"""Shallow module per Ousterhout. Public surface: ``RateLimiter``,
``rate_limiter``, ``client_ip``, ``RATE_LIMIT_PER_IP``, ``RATE_LIMIT_WINDOW_SEC``.

Per-IP fixed-window rate limiter for the demo VPS deploy's heavy paths (issue
#598 Slice A / Q3). The public demo box is key-free, so a single client
hammering a heavy path (read or admin) could otherwise burn the daily USD
budget or saturate the concurrency semaphores on its own. ``ProdMiddleware``
checks this gate for every heavy request, BEFORE the concurrency semaphore
(see that module's guard-order docstring): a rate-limited request is rejected
with a 429 before it ever occupies a semaphore slot or charges the daily
budget.

Fixed window (not a sliding log): each IP gets a ``window_sec``-wide counting
window; the first request from a fresh (or fully expired) IP starts a new
window, and the window resets wholesale once ``window_sec`` has elapsed since
it started — this is the same "reserve-before-spend accumulator" shape
``gateway.app.budget.DailyBudget`` uses, just keyed by IP instead of UTC day.

IP-spoofing caveat: ``client_ip`` trusts the FIRST ``X-Forwarded-For`` hop
verbatim. On a real multi-hop edge a client can forge this header to evade
per-IP limiting (the trusted-edge case is the reverse — trust the RIGHTMOST
hop the edge itself appended). This project's demo deploy sits directly
behind one shared, first-party edge with no untrusted intermediary, so the
naive first-hop read is an accepted single-tenant posture, not a general
defense — do not reuse this helper for a multi-tenant or public-internet-facing
proxy chain without revisiting this trust model.

Single-worker model (CODING_STANDARD §2.6/§2.7): one in-process ``dict`` is
the whole store, mirroring ``DailyBudget``'s. ``allow()`` holds ``self._lock``
for its full duration (read, prune, and the admit-or-block decision) so
concurrent callers on the event loop never race past the limit.
"""

from __future__ import annotations

import os
import threading
import time

from starlette.types import Scope

# Fixed window width — not configurable (only the per-window count is), per
# issue #598's Q3 scope decision.
RATE_LIMIT_WINDOW_SEC = 300.0  # 5 minutes


def _read_rate_limit_per_ip() -> int:
    """Read ``KB_RATE_LIMIT_PER_IP`` at construction time (default 30; ``0`` disables).

    Unlike the concurrency semaphores' ``_read_int`` in ``middleware.py``, ``0``
    is a valid, meaningful value here (rate limiting off) and must NOT fall
    back to the default — only a missing or unparsable value does.
    """
    raw = os.getenv("KB_RATE_LIMIT_PER_IP")
    if raw is None:
        return 30
    try:
        value = int(raw)
    except ValueError:
        return 30
    return value if value >= 0 else 30


# Module-level default, read once at import (restart to apply a new value).
RATE_LIMIT_PER_IP = _read_rate_limit_per_ip()


class RateLimiter:
    """Fixed-window per-IP request counter.

    Public surface:
      - ``allow(ip, now=None)`` — True (and records the hit) if ``ip`` is
        under ``limit`` for its current window; False (no record) once ``ip``
        has hit ``limit`` for the window. ``limit <= 0`` disables the gate:
        always True, and — bounded-store guarantee — never records anything.
        ``now`` is an injectable ``time.monotonic()``-style float so window
        expiry is deterministic in tests; defaults to the real clock.

    The store is a ``{ip: (window_start, count)}`` dict. Every ``allow()``
    call prunes any OTHER ip whose window has fully expired before deciding
    the current one (bounded store — issue #598), so the dict's size tracks
    recently-active IPs rather than growing for the life of the process.
    """

    def __init__(self, *, limit: int, window_sec: float) -> None:
        self.limit = limit
        self.window_sec = window_sec
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allow(self, ip: str, *, now: float | None = None) -> bool:
        """Return True and record one hit for ``ip``, or False if already at ``limit``."""
        if self.limit <= 0:
            return True
        moment = now if now is not None else time.monotonic()
        with self._lock:
            self._prune_expired(moment)
            window_start, count = self._windows.get(ip, (moment, 0))
            if count >= self.limit:
                return False
            self._windows[ip] = (window_start, count + 1)
            return True

    def _prune_expired(self, moment: float) -> None:
        """Drop any window that fully elapsed (``>= window_sec`` old).

        Must be called with ``self._lock`` already held.
        """
        expired = [
            ip for ip, (start, _) in self._windows.items() if moment - start >= self.window_sec
        ]
        for ip in expired:
            del self._windows[ip]


def client_ip(scope: Scope) -> str:
    """Return the request's client IP per issue #598's trust model.

    First ``X-Forwarded-For`` hop when present (the shared edge always sets
    it — see the module docstring's spoofing caveat), else the ASGI
    ``scope["client"]`` peer address, else ``"unknown"``.
    """
    for key, value in scope.get("headers", []):
        if key == b"x-forwarded-for":
            text = value.decode("latin-1")
            first = text.split(",")[0].strip()
            if first:
                return first
    client = scope.get("client")
    if client:
        return client[0]
    return "unknown"


# Module-level singleton — the one counter the middleware shares across
# requests (single-worker model; CODING_STANDARD §2.7).
rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_IP, window_sec=RATE_LIMIT_WINDOW_SEC)
