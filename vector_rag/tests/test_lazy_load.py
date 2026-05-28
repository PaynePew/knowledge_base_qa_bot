"""Hermetic tests for lazy-load of a persisted FAISS index in _retrieve_and_gate.

Issue #133: a fresh Gateway process (vectorstore=None) must auto-load a
persisted FAISS index from FAISS_INDEX_DIR before declaring index_missing.

All tests are offline: fake embeddings, no OPENAI_API_KEY.

Three AC groups covered here:
  1. fresh-process lazy-load: vectorstore=None + persisted index → real chunks.
  2. index_missing only when no persisted index on disk.
  3. no double-load when vectorstore is already populated.

The Gateway grounded-stream AC is covered in
``gateway/tests/test_chat_stream_rag_lazy.py`` (Gateway-layer hermetic test).
"""

from __future__ import annotations

from unittest.mock import patch

import vector_rag.app.indexer as indexer
import vector_rag.app.retrieval as retrieval
from markdown_kb.app.grounding import GroundingOutcome

from .conftest import FakeLLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLLM:
    CANNED_ANSWER = (
        "Refunds take 5-7 business days. [Source: refund_policy.md#refund-timeline]"
    )

    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return FakeLLMResponse(content=self.CANNED_ANSWER)


def _approved() -> GroundingOutcome:
    return GroundingOutcome(passed=True, reason="claim_supported", result=None)


# ---------------------------------------------------------------------------
# AC 1 — Fresh-process lazy load: vectorstore=None + persisted index → real chunks
# ---------------------------------------------------------------------------


def test_retrieve_and_gate_lazy_loads_from_disk_when_vectorstore_is_none(
    indexed_corpus,
):
    """_retrieve_and_gate lazy-loads a persisted index when vectorstore is None.

    Simulates a fresh Gateway process by clearing the in-memory singleton after
    build_index() has persisted the index to disk. The gate must load from disk
    and return real chunks (not index_missing).
    """
    # Verify that the persisted index is on disk.
    assert indexer.FAISS_INDEX_DIR.exists(), (
        "indexed_corpus fixture must persist the FAISS index to disk"
    )

    # Simulate fresh process: drop in-memory singleton.
    indexer.vectorstore = None
    assert indexer.vectorstore is None

    gate = retrieval._retrieve_and_gate("How long do refunds take?")

    assert gate["early_exit"] is False, (
        f"Expected no early_exit after lazy-load, got reason="
        f"{gate['grounding_outcome'].reason}"
    )
    assert len(gate["chunks"]) > 0, (
        "Lazy-loaded index must return Chunks, not index_missing"
    )
    assert gate["grounding_outcome"].reason != "index_missing", (
        "Reason must NOT be index_missing when a persisted index is on disk"
    )


def test_retrieve_and_gate_populates_vectorstore_after_lazy_load(indexed_corpus):
    """After lazy-load, indexer.vectorstore is no longer None."""
    indexer.vectorstore = None

    retrieval._retrieve_and_gate("any question")

    assert indexer.vectorstore is not None, (
        "lazy-load must repopulate indexer.vectorstore"
    )


def test_query_lazy_loads_and_returns_answer(indexed_corpus, monkeypatch):
    """query() via a fresh process (vectorstore=None) returns a real answer.

    End-to-end path: lazy-load → search → LLM draft → grounded answer.
    """
    indexer.vectorstore = None

    fake_llm = _FakeLLM()
    monkeypatch.setattr(retrieval, "_llm", fake_llm)
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    with patch.object(retrieval.grounding_module, "verify", return_value=_approved()):
        result = retrieval.query("How long do refunds take?")

    assert result["grounding_outcome"].reason != "index_missing", (
        "query() must NOT return index_missing when a persisted index is on disk"
    )
    assert result["grounding_outcome"].passed is True
    assert len(result["sources"]) > 0


# ---------------------------------------------------------------------------
# AC 2 — index_missing ONLY when no persisted index on disk
# ---------------------------------------------------------------------------


def test_retrieve_and_gate_returns_index_missing_when_no_disk_index(
    fake_embeddings,
):
    """index_missing is returned ONLY when no persisted index exists on disk.

    The autouse fixture redirects FAISS_INDEX_DIR to a fresh tmp dir (no files),
    so vectorstore=None + no disk index → index_missing.
    """
    assert not indexer.FAISS_INDEX_DIR.exists(), (
        "tmp FAISS_INDEX_DIR must not exist before any build"
    )
    indexer.vectorstore = None

    gate = retrieval._retrieve_and_gate("any question")

    assert gate["early_exit"] is True
    assert gate["grounding_outcome"].reason == "index_missing"
    assert gate["answer"] == retrieval.NOT_INDEXED_MESSAGE


# ---------------------------------------------------------------------------
# AC 3 — No double-load when vectorstore is already populated
# ---------------------------------------------------------------------------


def test_retrieve_and_gate_no_double_load_when_vectorstore_populated(
    indexed_corpus, monkeypatch
):
    """_retrieve_and_gate does NOT call load_vector_index when vectorstore is set.

    When the index is already in memory (e.g. after a POST /index in the same
    process), a second call must NOT re-load from disk.
    """
    assert indexer.vectorstore is not None, (
        "indexed_corpus fixture must populate vectorstore"
    )

    load_calls = {"count": 0}
    original_load = indexer.load_vector_index

    def _counting_load(*args, **kwargs):
        load_calls["count"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(indexer, "load_vector_index", _counting_load)

    retrieval._retrieve_and_gate("How long do refunds take?")

    assert load_calls["count"] == 0, (
        f"load_vector_index must NOT be called when vectorstore is already populated "
        f"(called {load_calls['count']} times)"
    )
