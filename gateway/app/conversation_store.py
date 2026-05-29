"""Deep module per Ousterhout. Public surface: ``ConversationStore``, ``store``.

In-memory Conversation Store for the Gateway (Phase 11 Slice 1 — issue #159).

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

Intentional omissions (later slices):
- Sliding-window cap (S2)
- TTL eviction (S2)
- Redis-backed multi-worker store (§2.6 upgrade path — not built)

The ``store`` singleton at module level is the production instance; tests
construct their own ``ConversationStore()`` instances so there is no shared
state between test runs.
"""

from __future__ import annotations


class ConversationStore:
    """In-memory session store.

    Thread-safety note: CPython's GIL makes individual dict / list
    mutations atomic for single-statement ops (CODING_STANDARD §2.6).
    This is sufficient for single-process use (the current deployment
    model). A multi-worker upgrade would swap this class for a
    Redis-backed implementation (same public interface).
    """

    def __init__(self) -> None:
        # _sessions: session_id → list[turn_dict]
        self._sessions: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append_turn(self, session_id: str, turn: dict) -> None:
        """Append a completed turn to the session's history.

        Creates the session entry on first use (no explicit session-init
        call needed — the route mints the UUID and the first append
        implicitly initialises the session).

        Args:
            session_id: UUID string identifying the conversation session.
            turn: turn record dict (see module docstring for required keys).
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(turn)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        """Return the ordered turn history for ``session_id``.

        Returns an empty list for an unknown / new session (no side effect).

        Args:
            session_id: UUID string.

        Returns:
            List of turn dicts in insertion order (oldest first).
        """
        return list(self._sessions.get(session_id, []))

    def dump(self, session_id: str) -> list[dict]:
        """Return a full copy of the session's turn history.

        This is the **Phase 10 Hot Cache seam**: the Hot Cache writer calls
        ``dump(session_id)`` to summarise a completed session without
        re-deriving session state.

        Behaviour is identical to ``get_history`` — a separate method
        is provided so callers can express intent ("I want the full record
        for archival / summarisation") rather than just ("I need the last N
        turns for query rewriting").

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
