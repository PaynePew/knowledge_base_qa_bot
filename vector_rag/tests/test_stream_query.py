"""Tests for vector_rag.app.retrieval.stream_query() (Phase 9 Slice 3 / issue #120).

Covers the two-step yield shape (sources_ready partial, then full result),
the pre-LLM gate path (early-exit on index missing / empty retrieval), and
the happy path (grounded answer with LLM invoked once).

All tests are hermetic: embeddings leaf and LLM are faked; no OPENAI_API_KEY.
"""

from __future__ import annotations

from unittest.mock import patch

import vector_rag.app.indexer as indexer
import vector_rag.app.retrieval as retrieval
from markdown_kb.app.grounding import GroundingOutcome

from .conftest import FakeLLMResponse

REFUND_SOURCE = "refund_policy.md#refund-timeline"


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------


class FakeLLM:
    """Returns a canned grounded answer with a [Source: ...] token."""

    CANNED_ANSWER = f"Refunds take 5-7 business days. [Source: {REFUND_SOURCE}]"

    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return FakeLLMResponse(content=self.CANNED_ANSWER)


def _approved() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported", result=None)


def _rejected() -> GroundingOutcome:
    return GroundingOutcome(passed=False, reason="claim_unsupported", result=None)


# ---------------------------------------------------------------------------
# stream_query() shape — always yields exactly two dicts
# ---------------------------------------------------------------------------


def test_stream_query_yields_exactly_two_dicts(indexed_corpus, monkeypatch):
    """stream_query() always yields two dicts: partial then full."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        results = list(retrieval.stream_query("How long do refunds take?"))

    assert len(results) == 2, f"Expected 2 yields, got {len(results)}"


def test_stream_query_first_yield_is_sources_ready(indexed_corpus, monkeypatch):
    """First yield has _phase='sources_ready' and a sources list."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        gen = retrieval.stream_query("How long do refunds take?")
        partial = next(gen)

    assert partial.get("_phase") == "sources_ready"
    assert "sources" in partial
    assert isinstance(partial["sources"], list)


def test_stream_query_sources_ready_emitted_before_llm(indexed_corpus, monkeypatch):
    """First yield (sources_ready) is emitted before the LLM is called."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        gen = retrieval.stream_query("How long do refunds take?")
        _ = next(gen)
        # LLM must NOT have been called yet
        assert fake_llm.call_count == 0, "LLM must not be called before first yield"


def test_stream_query_second_yield_is_full_result(indexed_corpus, monkeypatch):
    """Second yield has answer, sources, grounding_outcome — identical to query()."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        results = list(retrieval.stream_query("How long do refunds take?"))

    full = results[1]
    assert "answer" in full
    assert "sources" in full
    assert "grounding_outcome" in full
    assert "_phase" not in full, "Full result must not carry internal _phase key"


def test_stream_query_llm_called_once_for_happy_path(indexed_corpus, monkeypatch):
    """LLM is invoked exactly once during a happy-path stream_query."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        list(retrieval.stream_query("How long do refunds take?"))

    assert fake_llm.call_count == 1


def test_stream_query_full_result_matches_query(indexed_corpus, monkeypatch):
    """Second yield from stream_query() matches the result from query()."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        stream_results = list(retrieval.stream_query("How long do refunds take?"))

    # Reset LLM call count and run query() separately
    fake_llm2 = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm2)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm2)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        query_result = retrieval.query("How long do refunds take?")

    full = stream_results[1]
    assert full["answer"] == query_result["answer"]
    assert full["grounding_outcome"].passed == query_result["grounding_outcome"].passed
    assert full["grounding_outcome"].reason == query_result["grounding_outcome"].reason


# ---------------------------------------------------------------------------
# Pre-LLM gate — index missing: early-exit path yields 2 dicts, no LLM call
# ---------------------------------------------------------------------------


def test_stream_query_index_missing_yields_two_dicts(fake_embeddings, monkeypatch):
    """stream_query() with no index built yields exactly two dicts, no LLM call."""
    indexer.vectorstore = None
    sentinel_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", sentinel_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: sentinel_llm)

    results = list(retrieval.stream_query("anything"))

    assert len(results) == 2
    assert sentinel_llm.call_count == 0, "LLM must not be called on early-exit path"


def test_stream_query_index_missing_first_yield_sources_ready(
    fake_embeddings, monkeypatch
):
    """On index-missing path, first yield has _phase='sources_ready'."""
    indexer.vectorstore = None
    monkeypatch.setattr(retrieval, "get_llm", lambda: FakeLLM())

    results = list(retrieval.stream_query("anything"))
    assert results[0].get("_phase") == "sources_ready"


def test_stream_query_index_missing_full_result_not_indexed(
    fake_embeddings, monkeypatch
):
    """On index-missing path, second yield has passed=False reason=index_missing."""
    indexer.vectorstore = None
    monkeypatch.setattr(retrieval, "get_llm", lambda: FakeLLM())

    results = list(retrieval.stream_query("anything"))
    full = results[1]
    assert full["grounding_outcome"].passed is False
    assert full["grounding_outcome"].reason == "index_missing"


# ---------------------------------------------------------------------------
# Pre-LLM gate — retrieval empty: early-exit, no LLM call
# ---------------------------------------------------------------------------


def test_stream_query_empty_retrieval_no_llm_call(indexed_corpus, monkeypatch):
    """stream_query() with empty retrieval yields two dicts; LLM is NOT called."""
    monkeypatch.setattr(indexer, "search", lambda q, k=3: [])
    sentinel_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", sentinel_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: sentinel_llm)

    results = list(retrieval.stream_query("anything"))

    assert len(results) == 2
    assert sentinel_llm.call_count == 0


def test_stream_query_empty_retrieval_full_result_cannot_confirm(
    indexed_corpus, monkeypatch
):
    """On empty-retrieval path, second yield is Cannot Confirm."""
    monkeypatch.setattr(indexer, "search", lambda q, k=3: [])
    monkeypatch.setattr(retrieval, "get_llm", lambda: FakeLLM())

    results = list(retrieval.stream_query("anything"))
    full = results[1]
    assert full["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert full["grounding_outcome"].passed is False


# ---------------------------------------------------------------------------
# RAG source shape: NO score, NO derived_from (issue #120 spec)
# ---------------------------------------------------------------------------


def test_stream_query_sources_have_no_score_field(indexed_corpus, monkeypatch):
    """RAG sources in stream_query output carry NO 'score' field."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        results = list(retrieval.stream_query("How long do refunds take?"))

    # Check both yields
    for result in results:
        for src in result.get("sources", []):
            assert "score" not in src, f"RAG source must not carry 'score': {src}"


def test_stream_query_sources_have_no_derived_from_field(indexed_corpus, monkeypatch):
    """RAG sources in stream_query output carry NO 'derived_from' field."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        results = list(retrieval.stream_query("How long do refunds take?"))

    for result in results:
        for src in result.get("sources", []):
            assert "derived_from" not in src, (
                f"RAG source must not carry 'derived_from': {src}"
            )


def test_stream_query_sources_have_required_fields(indexed_corpus, monkeypatch):
    """RAG sources carry exactly: source, heading, content."""
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        results = list(retrieval.stream_query("How long do refunds take?"))

    partial = results[0]
    assert len(partial["sources"]) >= 1
    for src in partial["sources"]:
        assert "source" in src
        assert "heading" in src
        assert "content" in src
