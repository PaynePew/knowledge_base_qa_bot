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
# Dump (Phase 10 Hot Cache seam)
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
