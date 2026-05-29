"""Unit tests for the gateway Conversation Store (Phase 11 Slice 1 — issue #159).

All tests are hermetic: no LLM, no OPENAI_API_KEY.
The store is an in-memory deep module (§2.7 monkeypatch-swappable).

Coverage:
- New session mint: get_history on unknown id returns []
- Append turn: appended turn appears in get_history
- Multiple turns: ordering preserved
- Dump: returns full turn history (same as get_history)
- No append on error: error-path callers must not call append_turn
  (the AC is that error-before-done appends nothing — that constraint
  lives in the route layer, not the store itself; the store is dumb)
- Store exposes dump() as the seam for Phase 10 Hot Cache

Phase 11 Slice 2 (issue #160) additions:
- Sliding-window cap at 10 turns (oldest evicted on 11th append)
- TTL eviction: whole session evicted after 30 min idle
- TTL sweep safe against mid-iteration mutation (no RuntimeError)
- dump() returns surviving window after TTL or window eviction
- Window size and TTL are configurable constants on ConversationStore
- Time is injectable/monkeypatchable so tests do not sleep
"""

from __future__ import annotations

import datetime

import pytest

from gateway.app.conversation_store import ConversationStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turn(question: str = "q", answer: str = "a", stack: str = "wiki") -> dict:
    return {
        "question": question,
        "answer": answer,
        "stack": stack,
        "grounding_reason": "claim_supported",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_get_history_unknown_session_returns_empty():
    """get_history on a never-seen session_id returns an empty list."""
    store = ConversationStore()
    assert store.get_history("nonexistent-session") == []


def test_append_then_get_history():
    """A turn appended to a session appears in get_history."""
    store = ConversationStore()
    turn = _turn("How long do refunds take?", "5-7 business days.")
    store.append_turn("session-1", turn)
    history = store.get_history("session-1")
    assert len(history) == 1
    assert history[0]["question"] == "How long do refunds take?"
    assert history[0]["answer"] == "5-7 business days."


def test_append_multiple_turns_preserves_order():
    """Multiple appended turns are returned in insertion order."""
    store = ConversationStore()
    store.append_turn("sess", _turn("q1", "a1"))
    store.append_turn("sess", _turn("q2", "a2"))
    store.append_turn("sess", _turn("q3", "a3"))
    history = store.get_history("sess")
    assert [t["question"] for t in history] == ["q1", "q2", "q3"]


def test_different_sessions_are_isolated():
    """Turns from different sessions do not bleed into each other."""
    store = ConversationStore()
    store.append_turn("sess-A", _turn("question-A"))
    store.append_turn("sess-B", _turn("question-B"))
    assert len(store.get_history("sess-A")) == 1
    assert store.get_history("sess-A")[0]["question"] == "question-A"
    assert len(store.get_history("sess-B")) == 1


# ---------------------------------------------------------------------------
# Dump (full session-window export)
# ---------------------------------------------------------------------------


def test_dump_returns_full_history():
    """dump(session_id) returns the complete turn history for the session."""
    store = ConversationStore()
    store.append_turn("s1", _turn("q1", "a1"))
    store.append_turn("s1", _turn("q2", "a2"))
    dumped = store.dump("s1")
    assert len(dumped) == 2
    assert dumped[0]["question"] == "q1"
    assert dumped[1]["question"] == "q2"


def test_dump_unknown_session_returns_empty():
    """dump on a never-seen session returns []."""
    store = ConversationStore()
    assert store.dump("no-such-session") == []


def test_dump_returns_copy_not_live_reference():
    """dump() returns a copy — mutating the returned list does not affect the store."""
    store = ConversationStore()
    store.append_turn("s1", _turn("q1"))
    dumped = store.dump("s1")
    dumped.clear()
    # The store's internal list must be unaffected.
    assert len(store.get_history("s1")) == 1


# ---------------------------------------------------------------------------
# Turn record shape: required fields
# ---------------------------------------------------------------------------


def test_turn_fields_preserved():
    """All expected turn fields survive the round-trip through the store."""
    store = ConversationStore()
    ts_str = "2026-05-29T10:00:00.000000Z"
    turn = {
        "question": "how long do exchanges take?",
        "answer": "3-5 business days.",
        "stack": "wiki",
        "grounding_reason": "claim_supported",
        "ts": ts_str,
    }
    store.append_turn("sess", turn)
    stored = store.get_history("sess")[0]
    assert stored["question"] == "how long do exchanges take?"
    assert stored["answer"] == "3-5 business days."
    assert stored["stack"] == "wiki"
    assert stored["grounding_reason"] == "claim_supported"
    assert stored["ts"] == ts_str


# ---------------------------------------------------------------------------
# Phase 11 Slice 2: Sliding-window cap (AC1, AC5)
# ---------------------------------------------------------------------------


def test_window_cap_default_is_10():
    """Default WINDOW_SIZE constant must be 10."""
    from gateway.app.conversation_store import WINDOW_SIZE

    assert WINDOW_SIZE == 10


def test_session_capped_at_10_turns():
    """A session never holds more than 10 turns; the 11th append drops the oldest."""
    store = ConversationStore()
    for i in range(11):
        store.append_turn("sess", _turn(f"q{i}", f"a{i}"))
    history = store.get_history("sess")
    # Must have exactly 10 turns.
    assert len(history) == 10
    # The oldest turn (q0) must be gone; q1 is now the oldest.
    assert history[0]["question"] == "q1"
    # The newest turn must be q10.
    assert history[-1]["question"] == "q10"


def test_window_cap_keeps_newest_after_many_appends():
    """After N > 10 appends only the last 10 are kept."""
    store = ConversationStore()
    for i in range(25):
        store.append_turn("sess", _turn(f"q{i}"))
    history = store.get_history("sess")
    assert len(history) == 10
    assert history[0]["question"] == "q15"
    assert history[-1]["question"] == "q24"


def test_window_cap_does_not_affect_other_sessions():
    """Window eviction for one session must not touch other sessions."""
    store = ConversationStore()
    for i in range(12):
        store.append_turn("sess-A", _turn(f"qA{i}"))
    store.append_turn("sess-B", _turn("qB0"))
    assert len(store.get_history("sess-A")) == 10
    assert len(store.get_history("sess-B")) == 1


def test_configurable_window_size():
    """ConversationStore accepts a custom window_size at construction time."""
    store = ConversationStore(window_size=3)
    for i in range(5):
        store.append_turn("sess", _turn(f"q{i}"))
    history = store.get_history("sess")
    assert len(history) == 3
    assert history[0]["question"] == "q2"
    assert history[-1]["question"] == "q4"


# ---------------------------------------------------------------------------
# Phase 11 Slice 2: TTL eviction (AC2, AC3, AC5)
# ---------------------------------------------------------------------------


def test_ttl_default_is_30_minutes():
    """Default TTL_SECONDS constant must be 1800 (30 min)."""
    from gateway.app.conversation_store import TTL_SECONDS

    assert TTL_SECONDS == 1800


def test_ttl_evicts_idle_session():
    """A session idle for longer than TTL is evicted; later get_history returns []."""
    # Inject a clock so we do not sleep 30 real minutes.
    fake_time = [0.0]

    def _clock() -> float:
        return fake_time[0]

    store = ConversationStore(clock=_clock)
    store.append_turn("sess", _turn("q1"))

    # Advance time past TTL (1801 seconds beyond the last append).
    fake_time[0] = 1801.0

    # Trigger eviction sweep.
    store.evict_expired()

    # The session must be gone.
    assert store.get_history("sess") == []


def test_ttl_does_not_evict_active_session():
    """A session touched within TTL survives the eviction sweep."""
    fake_time = [0.0]

    def _clock() -> float:
        return fake_time[0]

    store = ConversationStore(clock=_clock)
    store.append_turn("sess", _turn("q1"))

    # Advance time to just under TTL.
    fake_time[0] = 1799.0
    store.evict_expired()

    # Session must still be present.
    assert len(store.get_history("sess")) == 1


def test_ttl_resets_on_append():
    """A new append refreshes the idle clock; the session survives a sweep that
    would have evicted it based on the old timestamp."""
    fake_time = [0.0]

    def _clock() -> float:
        return fake_time[0]

    store = ConversationStore(clock=_clock)
    store.append_turn("sess", _turn("q1"))

    # Advance to just before TTL, then append a second turn.
    fake_time[0] = 1799.0
    store.append_turn("sess", _turn("q2"))

    # Now advance past the *original* TTL boundary (1801 s from t=0),
    # but still under 30 min from the refreshed ts (fake_time[0] == 1800+).
    fake_time[0] = 1801.0
    store.evict_expired()

    # Session must survive because last append was at t=1799, and t=1801 < 1799+1800.
    assert len(store.get_history("sess")) == 2


def test_ttl_evicts_multiple_sessions_without_runtime_error():
    """AC3: The TTL sweep must NOT raise RuntimeError when multiple sessions
    are evicted in a single call (no mutation of the dict during iteration)."""
    fake_time = [0.0]

    def _clock() -> float:
        return fake_time[0]

    store = ConversationStore(clock=_clock)
    for sid in ["s1", "s2", "s3"]:
        store.append_turn(sid, _turn("q"))

    fake_time[0] = 9999.0
    # Must not raise RuntimeError.
    store.evict_expired()

    assert store.get_history("s1") == []
    assert store.get_history("s2") == []
    assert store.get_history("s3") == []


def test_configurable_ttl():
    """ConversationStore accepts a custom ttl_seconds at construction time."""
    fake_time = [0.0]

    def _clock() -> float:
        return fake_time[0]

    store = ConversationStore(ttl_seconds=60, clock=_clock)
    store.append_turn("sess", _turn("q1"))

    fake_time[0] = 61.0
    store.evict_expired()

    assert store.get_history("sess") == []


# ---------------------------------------------------------------------------
# Phase 11 Slice 2: dump after eviction (AC4)
# ---------------------------------------------------------------------------


def test_dump_after_window_eviction():
    """dump() returns the surviving window (not the evicted turns)."""
    store = ConversationStore(window_size=3)
    for i in range(5):
        store.append_turn("sess", _turn(f"q{i}"))
    dumped = store.dump("sess")
    assert len(dumped) == 3
    assert dumped[0]["question"] == "q2"
    assert dumped[-1]["question"] == "q4"


def test_dump_after_ttl_eviction():
    """dump() returns [] for a TTL-evicted session."""
    fake_time = [0.0]

    def _clock() -> float:
        return fake_time[0]

    store = ConversationStore(clock=_clock)
    store.append_turn("sess", _turn("q1"))

    fake_time[0] = 9999.0
    store.evict_expired()

    assert store.dump("sess") == []
