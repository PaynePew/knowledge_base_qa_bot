"""Integration tests for the grounded /chat happy path — Slice 2.

Tests translate every acceptance criterion from issue #3 directly into
executable assertions. All tests use a FakeLLM stub (no live OpenAI calls).

Run with:
    pytest -m "not live"   (from markdown_kb/)
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer
import app.retrieval as retrieval_module
from app.grounding import GroundingClaim, GroundingOutcome, GroundingResult
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


def _make_approved_outcome(source_id: str) -> GroundingOutcome:
    """Build a GroundingOutcome(passed=True) stub for fixture use.

    The claim text and citing_section_ids match the FakeLLM canned answer.
    """
    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims trace to the cited section.",
            claims=[
                GroundingClaim(
                    text="Approved refunds are processed within 5-7 business days.",
                    supported=True,
                    citing_section_ids=[source_id],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


@pytest.fixture()
def client_with_fake_llm(indexed_corpus, fake_llm_refund, monkeypatch):
    """TestClient with a FakeLLM injected for refund-related queries.

    grounding.verify() is patched to return claim_supported so the test
    exercises the full route → retrieval → grounding path without real API calls.
    """
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm_refund)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm_refund)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: _make_approved_outcome(REFUND_SECTION_ID),
    )

    from app.main import app

    return TestClient(app), fake_llm_refund, indexed_corpus


@pytest.fixture()
def client_with_email_llm(indexed_corpus, fake_llm_email, monkeypatch):
    """TestClient with a FakeLLM injected for email-related queries.

    grounding.verify() is patched to return claim_supported so the test
    exercises the full route → retrieval → grounding path without real API calls.
    """
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm_email)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm_email)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: _make_approved_outcome(EMAIL_SECTION_ID),
    )

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
    assert "grounding" in body, "Response must have 'grounding' field"

    # Citation in answer (the fake LLM includes it)
    assert REFUND_SECTION_ID in body["answer"], (
        f"answer must contain citation '{REFUND_SECTION_ID}', got: {body['answer']}"
    )

    # Sources array contains matching entry
    source_ids = [s["source"] for s in body["sources"]]
    assert REFUND_SECTION_ID in source_ids, (
        f"sources must contain '{REFUND_SECTION_ID}', got: {source_ids}"
    )

    # Grounding field reflects verifier approval (AC #6 from issue #12)
    assert body["grounding"]["passed"] is True, (
        f"Expected grounding.passed=True, got: {body['grounding']}"
    )
    assert body["grounding"]["reason"] == "claim_supported", (
        f"Expected grounding.reason=claim_supported, got: {body['grounding']['reason']!r}"
    )
    assert body["grounding"]["claims"] is not None and len(body["grounding"]["claims"]) > 0, (
        f"Expected grounding.claims to be non-empty, got: {body['grounding']['claims']}"
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
    assert "grounding" in body

    # Citation in sources
    source_ids = [s["source"] for s in body["sources"]]
    assert EMAIL_SECTION_ID in source_ids, (
        f"sources must contain '{EMAIL_SECTION_ID}', got: {source_ids}"
    )

    # Grounding field reflects verifier approval (AC #6 from issue #12)
    assert body["grounding"]["passed"] is True, (
        f"Expected grounding.passed=True, got: {body['grounding']}"
    )
    assert body["grounding"]["reason"] == "claim_supported", (
        f"Expected grounding.reason=claim_supported, got: {body['grounding']['reason']!r}"
    )
    assert body["grounding"]["claims"] is not None and len(body["grounding"]["claims"]) > 0, (
        f"Expected grounding.claims to be non-empty, got: {body['grounding']['claims']}"
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
    assert "chat |" in content, f"Expected 'chat |' entry in log, got:\n{content}"

    # Must contain query (truncated or full)
    assert "refund" in content.lower(), f"Log must mention the query keyword, got:\n{content}"

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
    assert ctx_pos < q_pos, f"CONTEXT: (pos {ctx_pos}) must appear before QUESTION: (pos {q_pos})"

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
# B3 page expansion: prompt CONTEXT contains full parent page, not just BM25 hit
# (Slice 4-4 #49)
# ---------------------------------------------------------------------------


def test_prompt_contains_expanded_page_sections(client_with_fake_llm, indexed_corpus):
    """After B3: LLM prompt CONTEXT contains ALL sections of the hit page, not just the BM25 hit.

    For 'How long do refunds take?', the top BM25 hit is refund-timeline.
    After expand_to_pages(), the prompt must also contain cancellation-window
    and non-refundable-items (the other sections of refund_policy.md).
    """
    client, fake_llm, _ = client_with_fake_llm

    client.post("/chat", json={"query": "How long do refunds take?"})

    # Extract the prompt from FakeLLM's recorded messages
    assert fake_llm.last_messages, "FakeLLM must have been invoked"
    # The HumanMessage is index 1 (after SystemMessage)
    human_msg = fake_llm.last_messages[1]
    prompt_text = human_msg.content

    # The BM25 hit must be in the prompt
    assert "[Source: refund_policy.md#refund-timeline]" in prompt_text, (
        f"BM25 hit must be in prompt. Prompt:\n{prompt_text}"
    )
    # Sibling sections must also be present (B3 expansion)
    assert "[Source: refund_policy.md#cancellation-window]" in prompt_text, (
        f"Sibling section cancellation-window must be in prompt after B3 expansion. "
        f"Prompt:\n{prompt_text}"
    )
    assert "[Source: refund_policy.md#non-refundable-items]" in prompt_text, (
        f"Sibling section non-refundable-items must be in prompt after B3 expansion. "
        f"Prompt:\n{prompt_text}"
    )


def test_sources_are_bm25_hits_not_expanded(client_with_fake_llm, indexed_corpus):
    """After B3: ChatResponse.sources[] is still BM25 top-K, NOT the expanded set.

    The BM25 hit for 'How long do refunds take?' is only refund-timeline.
    sources must list only that section (the BM25 result), not all 3 page sections.
    """
    client, _, _ = client_with_fake_llm

    resp = client.post("/chat", json={"query": "How long do refunds take?"})
    body = resp.json()

    source_ids = [s["source"] for s in body["sources"]]

    # The BM25 hit must be in sources
    assert REFUND_SECTION_ID in source_ids, (
        f"BM25 top hit '{REFUND_SECTION_ID}' must be in sources, got: {source_ids}"
    )
    # Sibling sections should NOT be in sources (expansion is for LLM context, not client)
    assert "refund_policy.md#cancellation-window" not in source_ids, (
        f"Sibling section must NOT be in sources (B3 only expands LLM context). "
        f"Got sources: {source_ids}"
    )
    assert "refund_policy.md#non-refundable-items" not in source_ids, (
        f"Sibling section must NOT be in sources (B3 only expands LLM context). "
        f"Got sources: {source_ids}"
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

    # Grounding field reflects the pre-LLM gate reason (AC #2 from issue #12)
    assert "grounding" in body, "Response must have 'grounding' field"
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] in ("below_threshold", "retrieval_empty"), (
        f"Expected pre-LLM gate reason, got: {body['grounding']['reason']!r}"
    )
