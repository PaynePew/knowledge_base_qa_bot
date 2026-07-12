"""Shallow module per Ousterhout. Public surface: ``RateLimiter``,
``client_ip``, ``RATE_LIMIT_PER_IP``, ``RATE_LIMIT_WINDOW_SEC``.

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

IP-spoofing hardening (issue #598 scope addendum point 4 — adversarial-verify
follow-up grill): ``client_ip`` trusts the RIGHTMOST ``X-Forwarded-For`` hop.
A trusted edge appends the real client IP as the LAST entry in the chain (any
hops a client itself supplies are forwarded AHEAD of it, never after) — the
previous first-hop read was directly client-spoofable: a client could set an
arbitrary ``X-Forwarded-For: <anything>, ...`` header and rotate that forged
first hop to evade per-IP rate limiting entirely (bypass, not merely a
theoretical caveat). This project's demo deploy sits directly behind one
shared, first-party edge that appends (not merely forwards) the peer address
it itself observed, so the rightmost-hop read is trustworthy for THIS
deployment's proxy topology — do not reuse this helper for a multi-hop chain
where an untrusted intermediary sits between the edge and this app without
revisiting which hop is actually trustworthy there.

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
      - ``retry_after_seconds(ip, now=None)`` — seconds until ``ip``'s current
        window resets (issue #598 scope addendum point 5 — backs the 429
        response's ``Retry-After`` header). Read-only; never mutates state.

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

    def retry_after_seconds(self, ip: str, *, now: float | None = None) -> float:
        """Seconds remaining until ``ip``'s current fixed window resets (``>= 0``).

        Called by ``ProdMiddleware`` right after ``allow()`` returns False, to
        populate the 429 response's ``Retry-After`` header (issue #598 scope
        addendum point 5) — so a client knows how long to back off instead of
        immediately retrying and tripping the same limit again. Read-only: it
        does not prune or otherwise mutate ``self._windows``. If ``ip`` has no
        recorded window (a caller querying without first calling ``allow()``,
        or a window that was already pruned by another IP's call), the full
        ``window_sec`` is returned — the safe, conservative answer when the
        actual reset time is unknown.
        """
        moment = now if now is not None else time.monotonic()
        with self._lock:
            window = self._windows.get(ip)
        if window is None:
            return self.window_sec
        window_start, _count = window
        return max(self.window_sec - (moment - window_start), 0.0)

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

    Rightmost ``X-Forwarded-For`` hop when present (the shared edge always
    appends it last — see the module docstring's "IP-spoofing hardening"
    paragraph, issue #598 scope addendum point 4), else the ASGI
    ``scope["client"]`` peer address, else ``"unknown"``.
    """
    for key, value in scope.get("headers", []):
        if key == b"x-forwarded-for":
            text = value.decode("latin-1")
            hops = [hop.strip() for hop in text.split(",")]
            last = hops[-1] if hops else ""
            if last:
                return last
    client = scope.get("client")
    if client:
        return client[0]
    return "unknown"


# No module-level ``RateLimiter`` singleton here (unlike ``budget.py``'s
# ``budget``): the live instance is constructed in ``middleware.py`` next to
# ``read_sem`` / ``admin_sem`` instead, so reloading THAT module — the
# pattern every existing ``ProdMiddleware`` test's ``_fresh_app()`` helper
# already uses for the semaphores — also resets the rate-limit window,
# without requiring an edit to any pre-existing test file (issue #598).
