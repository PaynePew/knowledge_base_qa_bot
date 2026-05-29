"""Hermetic endpoint tests for Phase 11 Slice 1 multi-turn wiring (issue #159).

Tests cover the full session lifecycle on POST /chat/stream?stack=wiki:
- No session param → mints UUID → done.session present
- Echo ?session=<id> → continues conversation → history used for turn 2
- Turn 1 (no history) → passthrough (no rewrite LLM call)
- Turn 2+ → rewrite_query called with history
- done event carries session id
- turn is appended to store only on done (not on error)
- stored turn's question is the rewritten query
- Conversation Store dump() returns full history

All hermetic: LLM and rewriter mocked; no OPENAI_API_KEY.
Prior art: gateway/tests/test_chat_stream.py
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _logger
import markdown_kb.app.retrieval as _retrieval
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

import gateway.app.conversation_store as _store_module
import gateway.app.query_rewriting as _rewrite_module

_FIXTURE_DOCS = Path(__file__).resolve().parents[2] / "markdown_kb" / "tests" / "fixtures" / "docs"


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    CANNED_ANSWER = (
        "Approved refunds are processed within 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


class _FakeRewriteLLM:
    """Structured-output rewrite LLM stub that returns a fixed rewritten query."""

    def __init__(self, rewritten: str = "how long do exchanges take?") -> None:
        self._rewritten = rewritten

    def with_structured_output(self, schema):
        chain = MagicMock()
        result = MagicMock()
        result.rewritten_query = self._rewritten
        chain.invoke.return_value = result
        return chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approved_outcome() -> GroundingOutcome:
    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims trace to the cited section.",
            claims=[
                GroundingClaim(
                    text="Approved refunds are processed within 5-7 business days.",
                    supported=True,
                    citing_section_ids=["refund_policy.md#refund-timeline"],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


def _parse_sse_response(content: str) -> list[dict]:
    """Parse a multi-frame SSE response into a list of {type, data} dicts."""
    events = []
    for frame in content.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        lines = frame.split("\n")
        event_type = "message"
        data_str = ""
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:]
        if data_str:
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"raw": data_str}
            events.append({"type": event_type, "data": data})
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    """Redirect INDEX_PATH, LOG_PATH, WIKI_DIR to tmp for all multiturn tests."""
    monkeypatch.setattr(_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    """Use a fresh ConversationStore per test to avoid cross-test bleed."""
    from gateway.app.conversation_store import ConversationStore

    fresh = ConversationStore()
    monkeypatch.setattr(_store_module, "store", fresh)
    return fresh


@pytest.fixture(autouse=True)
def _reset_rewrite_llm(monkeypatch):
    """Reset the rewrite LLM singleton between tests."""
    monkeypatch.setattr(_rewrite_module, "_rewrite_llm", None)


@pytest.fixture()
def indexed_wiki_corpus(tmp_path):
    """Build the Section Index from the 3-Source hermetic fixture docs."""
    _indexer.build_index(_FIXTURE_DOCS)
    yield
    _indexer.sections.clear()


@pytest.fixture()
def multiturn_client(indexed_wiki_corpus, monkeypatch):
    """TestClient for the Gateway with mocked LLM, grounding, and rewrite LLM."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    # Default rewrite LLM: returns a fixed self-contained query for turn 2+.
    fake_rewrite = _FakeRewriteLLM("how long do exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


# ---------------------------------------------------------------------------
# AC: done payload carries session
# ---------------------------------------------------------------------------


def test_no_session_param_mints_uuid_in_done(multiturn_client):
    """A request with no ?session param gets a session UUID in done.session."""
    resp = multiturn_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    done = next(e for e in events if e["type"] == "done")
    assert "session" in done["data"], f"done must carry 'session': {done['data']}"
    session_id = done["data"]["session"]
    assert session_id, "session must be a non-empty UUID string"
    # Validate it's a valid UUID
    uuid.UUID(session_id)  # raises ValueError if invalid


def test_session_param_echoed_in_done(multiturn_client):
    """Echoing ?session=<existing_id> returns the same id in done.session."""
    # Seed a session in the store first.
    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00.000000Z",
        },
    )

    resp = multiturn_client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    done = next(e for e in events if e["type"] == "done")
    assert done["data"]["session"] == existing_id


# ---------------------------------------------------------------------------
# AC: turn 1 passthrough — no rewrite LLM call
# ---------------------------------------------------------------------------


def test_turn1_no_history_passthrough_no_rewrite_call(indexed_wiki_corpus, monkeypatch):
    """Turn 1 (empty history / no ?session) does NOT call the rewrite LLM."""
    get_llm_calls = []

    def _mock_get_llm():
        get_llm_calls.append(True)
        return _FakeRewriteLLM()

    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", _mock_get_llm)

    # Wire the retrieval LLM.
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    assert get_llm_calls == [], "rewrite LLM must not be called on turn 1"


# ---------------------------------------------------------------------------
# AC: turn 2+ uses rewrite
# ---------------------------------------------------------------------------


def test_turn2_triggers_rewrite_llm_call(indexed_wiki_corpus, monkeypatch):
    """Turn 2 (prior history exists for session) calls the rewrite LLM."""
    rewrite_invocations = []

    class _TrackingRewriteLLM:
        def with_structured_output(self, schema):
            chain = MagicMock()
            result = MagicMock()
            result.rewritten_query = "how long do exchanges take?"
            chain.invoke.side_effect = lambda msgs: rewrite_invocations.append(msgs) or result
            return chain

    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: _TrackingRewriteLLM())

    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )

    # Seed session with 1 prior turn.
    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200
    assert len(rewrite_invocations) == 1, (
        f"rewrite LLM must be called exactly once on turn 2, got {len(rewrite_invocations)}"
    )


# ---------------------------------------------------------------------------
# AC: turn appended to store on done; stored question = rewritten query
# ---------------------------------------------------------------------------


def test_turn_appended_to_store_on_done(multiturn_client):
    """A completed turn is appended to the Conversation Store on done."""
    resp = multiturn_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    done = next(e for e in events if e["type"] == "done")
    session_id = done["data"]["session"]

    history = _store_module.store.get_history(session_id)
    assert len(history) == 1, f"Expected 1 turn, got {len(history)}"
    assert history[0]["question"] == "How long do refunds take?"


def test_stored_question_is_rewritten_query_on_turn2(indexed_wiki_corpus, monkeypatch):
    """The stored turn's question is the rewritten query, not the raw follow-up."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    rewritten = "how long do exchanges take?"
    fake_rewrite = _FakeRewriteLLM(rewritten)
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "and exchanges?"},  # raw elliptical follow-up
    )
    assert resp.status_code == 200
    # The store should now have 2 turns.
    history = _store_module.store.get_history(existing_id)
    assert len(history) == 2
    # The second turn's question must be the rewritten query, not the raw follow-up.
    assert history[1]["question"] == rewritten, (
        f"Stored question must be rewritten query '{rewritten}', got '{history[1]['question']}'"
    )


# ---------------------------------------------------------------------------
# AC: error before done does NOT append to store
# ---------------------------------------------------------------------------


def test_error_before_done_does_not_append_to_store(indexed_wiki_corpus, monkeypatch):
    """An LLM error (no done event) does NOT write to the Conversation Store."""

    def _raise_503(*_args, **_kwargs):
        raise HTTPException(status_code=503, detail="LLM service temporarily unavailable.")

    fake_llm_error = type("_ErrLLM", (), {"invoke": staticmethod(_raise_503)})()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm_error)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm_error)

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]
    # Verify stream terminated with error (no done)
    assert "error" in types
    assert "done" not in types

    # The store must have no turns for any session.
    # (The session_id was minted but never committed.)
    # We check all sessions are empty by checking the internal store state.
    # Since we patched to a fresh store, any turns would have been on the
    # fresh store — verify it has no sessions.
    assert _store_module.store._sessions == {}, (
        f"Store must be empty after error-before-done; got: {_store_module.store._sessions}"
    )


# ---------------------------------------------------------------------------
# AC: Conversation Store dump returns full history
# ---------------------------------------------------------------------------


def test_dump_returns_full_history_after_two_turns(multiturn_client):
    """dump(session_id) returns all turns after two sequential requests."""
    # Turn 1.
    resp1 = multiturn_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp1.status_code == 200
    events1 = _parse_sse_response(resp1.text)
    session_id = next(e for e in events1 if e["type"] == "done")["data"]["session"]

    # Turn 2 (echo session).
    resp2 = multiturn_client.post(
        f"/chat/stream?stack=wiki&session={session_id}",
        json={"query": "and exchanges?"},
    )
    assert resp2.status_code == 200

    dumped = _store_module.store.dump(session_id)
    assert len(dumped) == 2, f"Expected 2 turns in dump, got {len(dumped)}"


# ---------------------------------------------------------------------------
# AC: existing SSE event order is unchanged (regression guard)
# ---------------------------------------------------------------------------


def test_multiturn_event_order_unchanged(multiturn_client):
    """Multi-turn requests preserve the Phase 9 SSE event order: sources→status→token(s)→done."""
    resp = multiturn_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    assert types[0] == "sources"
    assert types[-1] == "done"
    # done carries session (Phase 11 addition)
    done_data = next(e for e in events if e["type"] == "done")["data"]
    assert "session" in done_data


# ---------------------------------------------------------------------------
# AC (Slice 4): status:rewriting ordering — turn 1 absent, turn 2+ before sources
# ---------------------------------------------------------------------------


def test_turn1_no_status_rewriting_event(multiturn_client):
    """Turn 1 (no history) MUST NOT emit a status:rewriting event (turn 1 is passthrough)."""
    resp = multiturn_client.post(
        "/chat/stream?stack=wiki",
        json={"query": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    rewriting_events = [
        e for e in events if e["type"] == "status" and e.get("data", {}).get("phase") == "rewriting"
    ]
    assert rewriting_events == [], f"Turn 1 must NOT emit status:rewriting; got: {rewriting_events}"


def test_turn2_emits_status_rewriting_before_sources(indexed_wiki_corpus, monkeypatch):
    """Turn 2+ MUST emit status:rewriting BEFORE the sources event.

    The ordering is: status:rewriting → sources → status:verifying → token(s) → done.
    This is the Phase 11 Slice 4 AC.
    """
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    fake_rewrite = _FakeRewriteLLM("how long do exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    # Seed a session with 1 prior turn so this is turn 2.
    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    # Must have a status:rewriting event
    rewriting_indices = [
        i
        for i, e in enumerate(events)
        if e["type"] == "status" and e.get("data", {}).get("phase") == "rewriting"
    ]
    assert rewriting_indices, f"Turn 2 must emit status:rewriting; got event types: {types}"

    # status:rewriting must appear BEFORE sources
    rewriting_idx = rewriting_indices[0]
    sources_indices = [i for i, t in enumerate(types) if t == "sources"]
    assert sources_indices, f"Turn 2 must emit sources; got: {types}"
    sources_idx = sources_indices[0]

    assert rewriting_idx < sources_idx, (
        f"status:rewriting (index {rewriting_idx}) must precede sources (index {sources_idx}); "
        f"event order: {types}"
    )


def test_turn2_full_event_order_with_rewriting(indexed_wiki_corpus, monkeypatch):
    """Turn 2 full SSE event order: status:rewriting → sources → status:verifying → token(s) → done."""
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )
    fake_rewrite = _FakeRewriteLLM("how long do exchanges take?")
    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: fake_rewrite)

    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    # First event: status:rewriting
    assert types[0] == "status", f"First event must be status:rewriting, got: {types}"
    assert events[0]["data"]["phase"] == "rewriting", (
        f"First status event must have phase=rewriting, got: {events[0]['data']}"
    )
    # Second event: sources
    assert types[1] == "sources", f"Second event must be sources, got: {types}"
    # Last event: done with session
    assert types[-1] == "done"
    assert "session" in events[-1]["data"]


def test_rewrite_error_after_status_rewriting_yields_terminal_error(
    indexed_wiki_corpus, monkeypatch
):
    """If rewrite_query raises after status:rewriting is committed, a terminal error event is emitted.

    The store must NOT be written (consistent with error-before-done invariant).
    """
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)

    class _ErrorRewriteLLM:
        def with_structured_output(self, schema):
            chain = MagicMock()
            chain.invoke.side_effect = RuntimeError("rewrite LLM exploded")
            return chain

    monkeypatch.setattr(_rewrite_module, "get_rewrite_llm", lambda: _ErrorRewriteLLM())

    existing_id = str(uuid.uuid4())
    _store_module.store.append_turn(
        existing_id,
        {
            "question": "How long do refunds take?",
            "answer": "5-7 business days.",
            "stack": "wiki",
            "grounding_reason": "claim_supported",
            "ts": "2026-05-29T10:00:00Z",
        },
    )

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "and exchanges?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    types = [e["type"] for e in events]

    # Must have a terminal error event
    assert "error" in types, f"Expected error event; got: {types}"
    # Must NOT have a done event (error-before-done invariant)
    assert "done" not in types, f"Must NOT have done after rewrite error; got: {types}"
    # Store must still have only 1 turn (the seeded one, not a new one)
    history = _store_module.store.get_history(existing_id)
    assert len(history) == 1, f"Rewrite error must not write a turn; store has {len(history)} turns"
