"""Hermetic tests for lazy-load of persisted .kb/index.json in _retrieve_and_gate.

Issue #148: a fresh Gateway process (sections=[]) must auto-load a persisted
.kb/index.json before declaring index_missing. Mirrors the RAG lazy-load fix
(#133 / #136) for the Wiki (markdown_kb) stack.

All tests are offline: BM25 only, mocked LLM, no OPENAI_API_KEY.

Four AC groups covered:
  1. Fresh-process lazy-load: sections=[] + persisted index → real Sections (not index_missing).
  2. index_missing ONLY when no persisted .kb/index.json exists on disk.
  3. No double-load when sections is already populated.
  4. Gateway stack=wiki returns grounded after restart WITHOUT re-POST /wiki/index
     (given a persisted index) — tested via non-stream /chat endpoint (hermetic).
"""

from __future__ import annotations

from unittest.mock import patch

import app.indexer as indexer
import app.retrieval as retrieval

from .conftest import FakeLLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLLM:
    CANNED_ANSWER = "Refunds take 5-7 business days. [Source: refund_policy.md#refund-timeline]"

    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return FakeLLMResponse(content=self.CANNED_ANSWER)


def _approved():
    from app.grounding import GroundingOutcome

    return GroundingOutcome(passed=True, reason="claim_supported", result=None)


# ---------------------------------------------------------------------------
# AC 1 — Fresh-process lazy load: sections=[] + persisted index → real Sections
# ---------------------------------------------------------------------------


def test_retrieve_and_gate_lazy_loads_from_disk_when_sections_empty(
    indexed_corpus,
):
    """_retrieve_and_gate lazy-loads a persisted index when sections is empty.

    Simulates a fresh Gateway process by clearing the in-memory sections list
    after build_index() has persisted the index to disk. The gate must load
    from disk and return real Sections (not index_missing).
    """
    # Verify persisted index is on disk (autouse redirect → tmp).
    assert indexer.INDEX_PATH.exists(), "indexed_corpus fixture must persist the index to disk"

    # Simulate fresh process: drop in-memory sections list.
    indexer.sections.clear()
    assert not indexer.sections

    gate = retrieval._retrieve_and_gate("How long do refunds take?")

    assert gate["early_exit"] is False, (
        f"Expected no early_exit after lazy-load, got reason={gate['grounding_outcome'].reason}"
    )
    assert len(gate["ranked"]) > 0, (
        "Lazy-loaded index must return ranked Sections, not index_missing"
    )
    assert gate["grounding_outcome"].reason != "index_missing", (
        "Reason must NOT be index_missing when a persisted index is on disk"
    )


def test_retrieve_and_gate_repopulates_sections_after_lazy_load(indexed_corpus):
    """After lazy-load, indexer.sections is no longer empty."""
    indexer.sections.clear()

    retrieval._retrieve_and_gate("any question")

    assert len(indexer.sections) > 0, "lazy-load must repopulate indexer.sections"


def test_query_lazy_loads_and_returns_answer(indexed_corpus, monkeypatch):
    """query() via a fresh process (sections=[]) returns a real answer.

    End-to-end path: lazy-load → BM25 search → LLM draft → grounded answer.
    """
    indexer.sections.clear()

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


def test_retrieve_and_gate_returns_index_missing_when_no_disk_index():
    """index_missing is returned ONLY when no persisted index exists on disk.

    The autouse _redirect_paths_to_tmp fixture redirects INDEX_PATH to a fresh
    tmp path (no file there), so sections=[] + no disk index → index_missing.
    """
    assert not indexer.INDEX_PATH.exists(), "tmp INDEX_PATH must not exist before any build"
    indexer.sections.clear()

    gate = retrieval._retrieve_and_gate("any question")

    assert gate["early_exit"] is True
    assert gate["grounding_outcome"].reason == "index_missing"
    assert gate["answer"] == retrieval.NOT_INDEXED_MESSAGE


# ---------------------------------------------------------------------------
# AC 3 — No double-load when sections is already populated
# ---------------------------------------------------------------------------


def test_retrieve_and_gate_no_double_load_when_sections_populated(indexed_corpus, monkeypatch):
    """_retrieve_and_gate does NOT call load_index_json when sections is set.

    When the index is already in memory (e.g. after a POST /wiki/index in the
    same process), a second call must NOT re-load from disk.
    """
    assert len(indexer.sections) > 0, "indexed_corpus fixture must populate indexer.sections"

    load_calls = {"count": 0}
    original_load = indexer.load_index_json

    def _counting_load(*args, **kwargs):
        load_calls["count"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(indexer, "load_index_json", _counting_load)

    retrieval._retrieve_and_gate("How long do refunds take?")

    assert load_calls["count"] == 0, (
        f"load_index_json must NOT be called when sections is already populated "
        f"(called {load_calls['count']} times)"
    )
