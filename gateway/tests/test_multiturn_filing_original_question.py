"""Gateway endpoint tests for issue #579 — filing under the original question.

Turn 2+ retrieval uses the Gateway's rewritten, self-contained query (Phase 11
Slice 4), but the filed ``wiki/qa/<slug>.md`` page must record the user's
LITERAL follow-up as ``question`` (never the rewrite) — the rewrite instead
lands in the page's ``retrieval_query`` audit field. The Conversation Store's
stored turn is UNCHANGED (it still needs the rewrite as history input for the
NEXT turn's rewrite call) — ``test_multiturn_routes.py::
test_stored_question_is_rewritten_query_on_turn2`` already pins that; this
file only adds the filed-page-level assertions.

Prior art / fixture reuse: ``gateway/tests/test_multiturn_routes.py`` (rewrite
LLM stub + multiturn fixtures) and ``gateway/tests/test_chat_stream_filing.py``
(reading the written qa page off disk).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from unittest.mock import MagicMock

import markdown_kb.app.indexer as _indexer
import markdown_kb.app.logger as _logger
import markdown_kb.app.retrieval as _retrieval
import pytest
from fastapi.testclient import TestClient
from markdown_kb.app.grounding import GroundingClaim, GroundingOutcome, GroundingResult

import gateway.app.conversation_store as _store_module
import gateway.app.logger as _gateway_logger
import gateway.app.query_rewriting as _rewrite_module

from .test_multiturn_routes import _FIXTURE_DOCS, _parse_sse_response

REFUND_SECTION_ID = "refund_policy.md#refund-timeline"


@dataclass(frozen=True)
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    CANNED_ANSWER = (
        f"Approved refunds are processed within 5-7 business days. [Source: {REFUND_SECTION_ID}]"
    )

    def invoke(self, messages):
        return _FakeLLMResponse(content=self.CANNED_ANSWER)


class _FakeRewriteLLM:
    """Structured-output rewrite LLM stub that returns a fixed rewritten query."""

    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten

    def with_structured_output(self, schema):
        chain = MagicMock()
        result = MagicMock()
        result.rewritten_query = self._rewritten
        chain.invoke.return_value = result
        return chain


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
                    citing_section_ids=[REFUND_SECTION_ID],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


@pytest.fixture(autouse=True)
def _redirect_paths_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(_gateway_logger, "LOG_PATH", tmp_path / "gateway" / "log.md")
    monkeypatch.setattr(_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(_indexer, "WIKI_DIR", tmp_path / "wiki")


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    from gateway.app.conversation_store import ConversationStore

    fresh = ConversationStore()
    monkeypatch.setattr(_store_module, "store", fresh)
    return fresh


@pytest.fixture(autouse=True)
def _reset_rewrite_llm(monkeypatch):
    monkeypatch.setattr(_rewrite_module, "_rewrite_llm", None)


@pytest.fixture()
def indexed_wiki_corpus():
    _indexer.build_index(_FIXTURE_DOCS)
    yield
    _indexer.sections.clear()


@pytest.fixture()
def grounded_client(indexed_wiki_corpus, monkeypatch):
    fake_llm = _FakeLLM()
    monkeypatch.setattr(_retrieval, "_llm", fake_llm)
    monkeypatch.setattr(_retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        _retrieval.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(),
    )

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


def test_turn2_files_under_original_question_with_rewrite_as_retrieval_query(
    grounded_client, tmp_path, monkeypatch
):
    """Turn 2+: the filed page's question is the raw follow-up; retrieval_query
    is the rewrite. The stored Conversation Store turn is unaffected."""
    from markdown_kb.app.qa import compute_slug

    rewritten = "how long does it take for the refund to arrive?"
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

    raw_followup = "and when will it arrive?"
    resp = grounded_client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": raw_followup},
    )
    assert resp.status_code == 200

    events = _parse_sse_response(resp.text)
    done_events = [e for e in events if e["type"] == "done"]
    assert done_events, "Expected a done event"
    filed = done_events[-1]["data"].get("filed")
    assert filed is not None, "Turn 2 grounded answer must file"

    # The filed slug must be computed from the ORIGINAL follow-up, not the rewrite.
    assert filed["slug"] == compute_slug(raw_followup)

    qa_path = tmp_path / "wiki" / "qa" / f"{filed['slug']}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert f"question: {raw_followup}" in content, (
        f"question must be the literal follow-up, got:\n{content}"
    )
    assert f"retrieval_query: {rewritten}" in content, (
        f"retrieval_query must carry the rewrite, got:\n{content}"
    )

    # Regression trip-wire: the Conversation Store must still hold the
    # REWRITTEN question (needed as history input for the next rewrite) —
    # if this goes red, the store was touched, which is out of scope (#579).
    history = _store_module.store.get_history(existing_id)
    assert history[1]["question"] == rewritten


def test_turn1_retrieval_query_equals_question_on_passthrough(grounded_client, tmp_path):
    """Turn 1 (no session): no rewrite happened, so both fields are identical."""
    query = "How long do refunds take?"
    resp = grounded_client.post("/chat/stream?stack=wiki", json={"query": query})
    assert resp.status_code == 200

    events = _parse_sse_response(resp.text)
    done_events = [e for e in events if e["type"] == "done"]
    filed = done_events[-1]["data"].get("filed")
    assert filed is not None

    qa_path = tmp_path / "wiki" / "qa" / f"{filed['slug']}.md"
    content = qa_path.read_text(encoding="utf-8")
    assert f"question: {query}" in content
    assert f"retrieval_query: {query}" in content

    # No rewritten_query on the sources event for turn 1 (nothing to surface).
    sources_data = next(e["data"] for e in events if e["type"] == "sources")
    assert "rewritten_query" not in sources_data


def test_turn2_sources_event_carries_rewritten_query(grounded_client, monkeypatch):
    """Issue #579 UI surface: the sources event carries rewritten_query on turn 2+."""
    rewritten = "how long does it take for the refund to arrive?"
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

    resp = grounded_client.post(
        f"/chat/stream?stack=wiki&session={existing_id}",
        json={"query": "and when will it arrive?"},
    )
    assert resp.status_code == 200
    events = _parse_sse_response(resp.text)
    sources_data = next(e["data"] for e in events if e["type"] == "sources")
    assert sources_data.get("rewritten_query") == rewritten
