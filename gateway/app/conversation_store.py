"""Deep module per Ousterhout. Public surface: ``ConversationStore``, ``store``.

In-memory Conversation Store for the Gateway (Phase 11 — issues #159, #160).

Keyed by ``session_id`` (a UUID string minted by the route layer). Each session
holds an ordered list of turn records; the store is the single source of truth
for conversation history within a process lifetime.

Turn record shape (all fields required):
    {
        "question":         str   — the rewritten self-contained query (NOT raw),
        "answer":           str   — grounded answer text or CANNOT_CONFIRM_PHRASE,
        "stack":            str   — "wiki" | "rag",
        "grounding_reason": str   — e.g. "claim_supported", "below_threshold", …,
        "ts":               str   — ISO-8601 UTC timestamp string,
    }

The turn is appended **only** by the route layer, and **only** on a normal
``done`` event (grounding-pass, grounding-fail, or Cannot Confirm all write;
``error``-before-``done`` writes nothing — that invariant is enforced by the
route, not here).

Sliding-window cap (S2):
    Each session is capped at ``WINDOW_SIZE`` turns (default 10). When the
    (N+1)-th turn is appended the oldest turn is silently dropped so that
    the session holds exactly N turns.

TTL eviction (S2):
    The store tracks the monotonic time of the last append per session. When
    ``evict_expired()`` is called, any session whose last-access time is more
    than ``TTL_SECONDS`` seconds ago is deleted in its entirety. The sweep
    iterates over a **snapshot** of keys to avoid ``RuntimeError: dictionary
    changed size during iteration`` (CODING_STANDARD §2.6).

Thread-safety note: CPython's GIL makes individual dict / list mutations
atomic for single-statement ops (CODING_STANDARD §2.6). This is sufficient
for single-process use (the current deployment model). A multi-worker
upgrade would swap this class for a Redis-backed implementation (same public
interface).

Redis is the documented multi-worker upgrade and stays out of scope
(CODING_STANDARD §2.6/§2.7).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Module-level configurable defaults
# ---------------------------------------------------------------------------

#: Default maximum number of turns kept per session (oldest evicted on overflow).
WINDOW_SIZE: int = 10

#: Default idle TTL in seconds before a whole session is evicted (30 minutes).
TTL_SECONDS: int = 30 * 60  # 1800


class ConversationStore:
    """In-memory session store with sliding-window cap and TTL eviction.

    Args:
        window_size: Maximum turns per session.  Defaults to ``WINDOW_SIZE``.
        ttl_seconds: Idle TTL in seconds before a session is evicted.
                     Defaults to ``TTL_SECONDS``.
        clock: A zero-argument callable returning the current time as a float
               (monotonic seconds).  Defaults to ``time.monotonic``.  Inject a
               fake clock in tests so they do not sleep.

    Thread-safety note: CPython's GIL makes individual dict / list
    mutations atomic for single-statement ops (CODING_STANDARD §2.6).
    This is sufficient for single-process use (the current deployment
    model). A multi-worker upgrade would swap this class for a
    Redis-backed implementation (same public interface).
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        ttl_seconds: int = TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._window_size = window_size
        self._ttl_seconds = ttl_seconds
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        # _sessions: session_id → deque[turn_dict] (bounded to window_size)
        self._sessions: dict[str, deque[dict]] = {}
        # _last_access: session_id → float (clock value of last append)
        self._last_access: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append_turn(self, session_id: str, turn: dict) -> None:
        """Append a completed turn to the session's history.

        Creates the session entry on first use (no explicit session-init
        call needed — the route mints the UUID and the first append
        implicitly initialises the session).

        When the session already holds ``window_size`` turns, the oldest
        turn is silently discarded before the new one is inserted so that
        the session always holds at most ``window_size`` entries.

        Updates the last-access timestamp for TTL purposes.

        Args:
            session_id: UUID string identifying the conversation session.
            turn: turn record dict (see module docstring for required keys).
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self._window_size)
        self._sessions[session_id].append(turn)
        self._last_access[session_id] = self._clock()

    # ------------------------------------------------------------------
    # TTL eviction
    # ------------------------------------------------------------------

    def evict_expired(self) -> None:
        """Delete all sessions that have been idle longer than ``ttl_seconds``.

        The sweep iterates over a **snapshot** of keys so that deleting entries
        from ``self._sessions`` inside the loop never triggers
        ``RuntimeError: dictionary changed size during iteration``.

        Call this periodically (e.g. via a background task or before each
        request) to reclaim memory from abandoned sessions.
        """
        now = self._clock()
        # Snapshot keys first to avoid mutating the dict during iteration.
        expired = [
            sid
            for sid in list(self._sessions)
            if now - self._last_access.get(sid, 0.0) > self._ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
            self._last_access.pop(sid, None)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        """Return the ordered turn history for ``session_id``.

        Returns an empty list for an unknown / evicted session (no side
        effect — does **not** update the last-access timestamp).

        Args:
            session_id: UUID string.

        Returns:
            List of turn dicts in insertion order (oldest first), capped at
            ``window_size``.
        """
        return list(self._sessions.get(session_id, []))

    def dump(self, session_id: str) -> list[dict]:
        """Return a full copy of the session's surviving turn window.

        This is the **Phase 10 Hot Cache seam**: the Hot Cache writer calls
        ``dump(session_id)`` to summarise a completed session without
        re-deriving session state.

        After a TTL eviction or window-cap eviction, ``dump`` returns only
        the turns that survived (i.e. the current window), not the discarded
        ones — evicted turns are gone.

        Returns a shallow copy; the caller may freely modify the returned
        list without affecting the store.
        """
        return list(self._sessions.get(session_id, []))


# ---------------------------------------------------------------------------
# Production singleton
# ---------------------------------------------------------------------------

#: Module-level singleton used by ``gateway.app.routes``.
#: Tests construct their own ``ConversationStore()`` instances.
store: ConversationStore = ConversationStore()
