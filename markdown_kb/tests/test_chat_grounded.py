"""Integration tests for the grounded /chat happy path — Slice 2.

Tests translate every acceptance criterion from issue #3 directly into
executable assertions. All tests use a FakeLLM stub (no live OpenAI calls).

Run with:
    pytest -m "not live"   (from markdown_kb/)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer
import app.retrieval as retrieval_module
from app.retrieval import CANNOT_CONFIRM_PHRASE

from .conftest import FakeLLMResponse

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

REFUND_SECTION_ID = "refund_policy.md#refund-timeline"
EMAIL_SECTION_ID = "account_help.md#change-email-address"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeLLM:
    """Minimal LLM stub that returns a canned grounded answer.

    The answer contains a [Source: ...] citation token so the integration
    tests can verify the real prompt was constructed and passed in.
    The stub records the last prompt it received for inspection.
    """

    def __init__(self, source_id: str = REFUND_SECTION_ID):
        self.source_id = source_id
        self.last_messages: list = []

    def invoke(self, messages: list):
        self.last_messages = messages
        return FakeLLMResponse(
            content=(
                f"Approved refunds are processed within 5-7 business days. "
                f"[Source: {self.source_id}]"
            )
        )


@pytest.fixture()
def fake_llm_refund():
    return FakeLLM(source_id=REFUND_SECTION_ID)


@pytest.fixture()
def fake_llm_email():
    return FakeLLM(source_id=EMAIL_SECTION_ID)


@pytest.fixture()
def client_with_fake_llm(indexed_corpus, fake_llm_refund, monkeypatch):
    """TestClient with a FakeLLM injected for refund-related queries."""
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm_refund)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm_refund)

    from app.main import app

    return TestClient(app), fake_llm_refund, indexed_corpus


@pytest.fixture()
def client_with_email_llm(indexed_corpus, fake_llm_email, monkeypatch):
    """TestClient with a FakeLLM injected for email-related queries."""
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm_email)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm_email)

    from app.main import app

    return TestClient(app), fake_llm_email, indexed_corpus


# ---------------------------------------------------------------------------
# AC 1: POST /chat with refund query cites refund_policy.md#refund-timeline
# ---------------------------------------------------------------------------


def test_chat_refund_query_returns_200_with_citation(client_with_fake_llm):
    client, fake_llm, _ = client_with_fake_llm
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()

    # Response shape
    assert "answer" in body, "Response must have 'answer' field"
    assert "sources" in body, "Response must have 'sources' field"

    # Citation in answer (the fake LLM includes it)
    assert REFUND_SECTION_ID in body["answer"], (
        f"answer must contain citation '{REFUND_SECTION_ID}', got: {body['answer']}"
    )

    # Sources array contains matching entry
    source_ids = [s["source"] for s in body["sources"]]
    assert REFUND_SECTION_ID in source_ids, (
        f"sources must contain '{REFUND_SECTION_ID}', got: {source_ids}"
    )


# ---------------------------------------------------------------------------
# AC 2: POST /chat with email query cites account_help.md#change-email-address
# ---------------------------------------------------------------------------


def test_chat_email_query_returns_200_with_citation(client_with_email_llm):
    client, fake_llm, _ = client_with_email_llm
    resp = client.post("/chat", json={"query": "Can I change my email address?"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()

    assert "answer" in body
    assert "sources" in body

    # Citation in sources
    source_ids = [s["source"] for s in body["sources"]]
    assert EMAIL_SECTION_ID in source_ids, (
        f"sources must contain '{EMAIL_SECTION_ID}', got: {source_ids}"
    )


# ---------------------------------------------------------------------------
# AC 3: wiki/log.md has a chat | ... entry per request
# ---------------------------------------------------------------------------


def test_chat_writes_log_entry(client_with_fake_llm):
    client, _, corpus = client_with_fake_llm
    log_path: Path = corpus["log_path"]

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert log_path.exists(), "wiki/log.md must exist after /chat request"
    content = log_path.read_text(encoding="utf-8")

    # Must contain at least one chat log entry
    assert "chat |" in content, (
        f"Expected 'chat |' entry in log, got:\n{content}"
    )

    # Must contain query (truncated or full)
    assert "refund" in content.lower(), (
        f"Log must mention the query keyword, got:\n{content}"
    )

    # Must contain top section id
    assert REFUND_SECTION_ID in content, (
        f"Log must contain top section id '{REFUND_SECTION_ID}', got:\n{content}"
    )


# ---------------------------------------------------------------------------
# AC 4: test_build_prompt_structure — CONTEXT before QUESTION, [Source:...],
#        Heading: parent > leaf breadcrumb
# ---------------------------------------------------------------------------


def test_build_prompt_structure(indexed_corpus):
    """Prompt has CONTEXT: before QUESTION:, [Source: ...] header, Heading: breadcrumb."""
    from app.prompt_builder import build_prompt

    sections_with_scores = indexer.search("How long do refunds take?", k=3)
    assert sections_with_scores, "Expected at least one ranked section"

    ranked_sections = [sec for sec, _score in sections_with_scores]
    prompt = build_prompt("How long do refunds take?", ranked_sections)

    # CONTEXT: must appear before QUESTION:
    ctx_pos = prompt.find("CONTEXT:")
    q_pos = prompt.find("QUESTION:")
    assert ctx_pos != -1, "Prompt must contain 'CONTEXT:'"
    assert q_pos != -1, "Prompt must contain 'QUESTION:'"
    assert ctx_pos < q_pos, (
        f"CONTEXT: (pos {ctx_pos}) must appear before QUESTION: (pos {q_pos})"
    )

    # Each cited Section appears under a [Source: filename#heading] header line
    for sec in ranked_sections:
        source_header = f"[Source: {sec.id}]"
        assert source_header in prompt, (
            f"Prompt must contain '{source_header}', prompt was:\n{prompt}"
        )

    # Each Section shows a Heading: parent > leaf breadcrumb line
    for sec in ranked_sections:
        breadcrumb = " > ".join(sec.heading_path)
        heading_line = f"Heading: {breadcrumb}"
        assert heading_line in prompt, (
            f"Prompt must contain 'Heading: {breadcrumb}', prompt was:\n{prompt}"
        )


# ---------------------------------------------------------------------------
# AC 5: test_build_prompt_citation_ids_match_ranked_sections
# ---------------------------------------------------------------------------


def test_build_prompt_citation_ids_match_ranked_sections(indexed_corpus):
    """Every ranked section's id appears as a [Source: ...] in the prompt."""
    from app.prompt_builder import build_prompt

    sections_with_scores = indexer.search("How long do refunds take?", k=3)
    ranked_sections = [sec for sec, _score in sections_with_scores]
    prompt = build_prompt("How long do refunds take?", ranked_sections)

    for sec in ranked_sections:
        assert f"[Source: {sec.id}]" in prompt, (
            f"[Source: {sec.id}] missing from prompt.\nPrompt:\n{prompt}"
        )


# ---------------------------------------------------------------------------
# AC 6: test_search_ranks_refund_timeline_first
# ---------------------------------------------------------------------------


def test_search_ranks_refund_timeline_first(indexed_corpus):
    """BM25 search ranks refund_policy.md#refund-timeline first for 'how long do refunds take'."""
    results = indexer.search("how long do refunds take", k=3)
    assert results, "search must return at least one result"
    top_section, top_score = results[0]
    assert top_section.id == REFUND_SECTION_ID, (
        f"Expected top result to be '{REFUND_SECTION_ID}', got '{top_section.id}' "
        f"(score={top_score:.3f}). Full results: {[(s.id, round(sc, 3)) for s, sc in results]}"
    )


# ---------------------------------------------------------------------------
# AC 7: Integration tests use fake LLM by default; live tests are opt-in
# ---------------------------------------------------------------------------


def test_no_live_llm_calls_in_default_tests(client_with_fake_llm):
    """Verify the FakeLLM stub was invoked (not a real LLM call)."""
    client, fake_llm, _ = client_with_fake_llm
    assert fake_llm.last_messages == [], "FakeLLM should not have been called yet"

    client.post("/chat", json={"query": "How long do refunds take?"})

    assert fake_llm.last_messages != [], "FakeLLM should have been invoked after /chat"
    # The messages passed must not be empty — the real prompt was built and sent
    assert len(fake_llm.last_messages) >= 2, (
        "Expected at least [SystemMessage, HumanMessage] in FakeLLM.last_messages"
    )


# ---------------------------------------------------------------------------
# Cannot Confirm gate — pre-LLM, no LLM call when no candidates
# (ADR-0001: block before LLM when retrieval yields nothing)
# ---------------------------------------------------------------------------


def test_cannot_confirm_no_llm_call(indexed_corpus, monkeypatch):
    """When retrieval yields no results, return Cannot Confirm WITHOUT invoking LLM."""
    call_count = {"n": 0}

    class _SentinelLLM:
        def invoke(self, messages):
            call_count["n"] += 1
            raise AssertionError("LLM should NOT be called when no candidates")

    monkeypatch.setattr(retrieval_module, "_llm", _SentinelLLM())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: _SentinelLLM())

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == CANNOT_CONFIRM_PHRASE, (
        f"Expected CANNOT_CONFIRM_PHRASE, got: {body['answer']!r}"
    )
    assert call_count["n"] == 0, "LLM must NOT be called when retrieval yields no candidates"
