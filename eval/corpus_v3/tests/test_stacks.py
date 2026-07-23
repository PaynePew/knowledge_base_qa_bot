"""Dense-over-wiki arm tests — external behaviour only (CODING_STANDARD §0.2).

The dense-over-wiki arm (ADR-0045 Prerequisite 1, the missing 2x2 cell) is
driven as a plain in-process callable
``dense_over_wiki_retrieval(query, k) -> list[RetrievedItem]`` — no HTTP, no
LLM calls. These tests build the real FAISS path over the committed corpus v3
wiki fixtures with ``fake_dense_embeddings`` (CODING_STANDARD §6.3: mock the
embeddings leaf, not the deep retrieval module) and assert the normalised
output shape, cutoff behaviour, and the registry registration point.
"""

from __future__ import annotations

import hybrid_kb.app.dense_index as hk_dense
import markdown_kb.app.indexer as mk_indexer
from eval.corpus_v3 import stacks
from eval.corpus_v3.models import RetrievedItem


def _index() -> None:
    """Build the wiki Section list then the dense-over-wiki index from it."""
    stacks.index_wiki_corpus()  # populates mk_indexer.sections over the fixtures
    stacks.index_dense_over_wiki()  # dense arm built from that exact Section list


def test_dense_over_wiki_returns_normalised_retrieved_items(fake_dense_embeddings):
    """The arm returns ``RetrievedItem``s in the common normalised shape."""
    _index()
    items = stacks.dense_over_wiki_retrieval("forgot my password", k=3)

    assert items, "an in-scope query must retrieve at least one Section"
    for item in items:
        assert isinstance(item, RetrievedItem)
        assert isinstance(item.source_section_id, str) and item.source_section_id
        assert isinstance(item.content, str) and item.content
        assert isinstance(item.heading_path, list)


def test_dense_over_wiki_recovers_the_relevant_wiki_page(fake_dense_embeddings):
    """A query about a topic surfaces that topic's wiki Section (real FAISS path)."""
    _index()
    items = stacks.dense_over_wiki_retrieval("how do I reset my password", k=3)
    assert any(it.source_section_id.startswith("password-reset") for it in items)


def test_dense_over_wiki_truncates_to_cutoff_k(fake_dense_embeddings):
    """The returned list honours the cutoff ``k`` (no RRF pool to overfetch)."""
    _index()
    assert len(stacks.dense_over_wiki_retrieval("return refund warranty", k=1)) <= 1
    assert len(stacks.dense_over_wiki_retrieval("return refund warranty", k=3)) <= 3


def test_index_dense_over_wiki_embeds_the_bm25_section_list(fake_dense_embeddings):
    """The dense index is built from the SAME Section list BM25 indexes (1:1 ids)."""
    stacks.index_wiki_corpus()
    n = stacks.index_dense_over_wiki()
    assert n == len(mk_indexer.sections)
    assert hk_dense.sections_indexed == n


def test_arm_registry_registers_dense_over_wiki_by_name():
    """The scaffold's adapter registry is the registration point for later arms."""
    assert stacks.ARM_REGISTRY["dense_over_wiki"] is stacks.dense_over_wiki_retrieval


def test_arm_registry_entries_conform_to_the_common_callable_shape(
    fake_dense_embeddings,
):
    """Every registered arm returns ``list[RetrievedItem]`` given (query, k)."""
    _index()
    for name, arm in stacks.ARM_REGISTRY.items():
        items = arm("password reset", k=2)
        assert isinstance(items, list), f"{name} arm must return a list"
        assert all(isinstance(it, RetrievedItem) for it in items), (
            f"{name} arm must return RetrievedItems"
        )
