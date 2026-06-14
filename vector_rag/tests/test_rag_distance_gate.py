"""Pre-LLM distance-relevance gate for RAG (issue #257).

FAISS k-NN always returns k neighbours, so the wiki path's pre-LLM relevance gate
had no RAG counterpart — the FAISS distance was computed and discarded. These
tests cover the new gate: env parsing, OFF-by-default (non-breaking), fires when
the closest distance exceeds the ceiling, passes within it, the RAG-no-score
invariant on the gate path, and parity (the public ``query`` inherits the gate).

Hermetic: ``indexed_corpus`` builds a real FAISS index with deterministic offline
embeddings — no deep-module mock (CODING_STANDARD §6.3), no LLM (the gate is
pre-LLM, so ``query`` returns before any synthesis call).
"""

from __future__ import annotations

import vector_rag.app.retrieval as retrieval


def test_max_rag_distance_unset_is_none(monkeypatch):
    monkeypatch.delenv("KB_RAG_DISTANCE_THRESHOLD", raising=False)
    assert retrieval._max_rag_distance() is None


def test_max_rag_distance_parses_env(monkeypatch):
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "1.5")
    assert retrieval._max_rag_distance() == 1.5


def test_gate_off_by_default_lets_retrieval_through(indexed_corpus, monkeypatch):
    """Unset threshold → gate disabled → a query that retrieves does NOT early-exit."""
    monkeypatch.delenv("KB_RAG_DISTANCE_THRESHOLD", raising=False)
    gate = retrieval._retrieve_and_gate("refund policy")
    assert gate["early_exit"] is False


def test_gate_fires_when_closest_distance_exceeds_ceiling(indexed_corpus, monkeypatch):
    """Ceiling 0.0 → every positive FAISS distance exceeds it → Cannot Confirm, no LLM."""
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    gate = retrieval._retrieve_and_gate("refund policy")
    assert gate["early_exit"] is True
    assert gate["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert gate["grounding_outcome"].reason == "below_threshold"
    assert gate["chunks"] == []


def test_gate_passes_when_within_ceiling(indexed_corpus, monkeypatch):
    """A large ceiling admits the retrieval (closest distance < ceiling)."""
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "1000.0")
    gate = retrieval._retrieve_and_gate("refund policy")
    assert gate["early_exit"] is False


def test_gate_path_sources_carry_no_score(indexed_corpus, monkeypatch):
    """RAG-no-score invariant holds on the gate path — distance is gate-only."""
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    gate = retrieval._retrieve_and_gate("refund policy")
    assert gate["sources"]  # populated even on the early-exit path
    for src in gate["sources"]:
        assert "score" not in src


def test_public_query_inherits_the_gate_without_llm(indexed_corpus, monkeypatch):
    """Parity: the public query() (every interface's entry) refuses pre-LLM.

    No get_llm/verify fake is needed — the gate early-exits before any LLM call,
    which is exactly why CLI / MCP / Browser all inherit the same behaviour.
    """
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    result = retrieval.query("anything at all")
    assert result["answer"] == retrieval.CANNOT_CONFIRM_PHRASE
    assert result["grounding_outcome"].reason == "below_threshold"
