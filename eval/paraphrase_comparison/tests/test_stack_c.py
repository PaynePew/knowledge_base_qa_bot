"""Stack C (Hybrid) eval arm tests — external behaviour only (CODING_STANDARD §0.2).

Stack C is the third comparison arm: BM25 + dense-over-wiki, fused by Reciprocal
Rank Fusion (reused from ``hybrid_kb`` — NOT reimplemented here), driven as a
plain in-process callable ``stack_c_retrieval(query, k) -> list[RetrievedItem]``
with the SAME shape as the existing two arms (no HTTP). These tests build both of
Stack C's arms over the eval wiki fixtures offline (BM25 via the real markdown_kb
indexer; dense via the real FAISS path with ``fake_dense_embeddings``) and assert
the fused output, the docs-id normalisation, and overfetch/cutoff decoupling.
"""

from __future__ import annotations

import hybrid_kb.app.dense_index as hk_dense
from eval.paraphrase_comparison import stacks
from eval.paraphrase_comparison.loader import load_paraphrases
from eval.paraphrase_comparison.models import RetrievedItem


def _index_both_arms() -> None:
    """Build Stack A's BM25 index then Stack C's dense arm from the SAME Sections."""
    stacks.index_stack_a()  # populates mk_indexer.sections over the eval wiki
    stacks.index_stack_c()  # dense arm built from that exact Section list


def test_stack_c_returns_retrieved_items_normalised_to_docs_ids(fake_dense_embeddings):
    """The fused arm returns RetrievedItems resolved to docs Gold Section ids."""
    _index_both_arms()
    para = next(p for p in load_paraphrases() if p.paraphrase_type == "synonym_swap")
    items = stacks.stack_c_retrieval(para.text, k=3)

    assert items, "an in-scope query must fuse to at least one Section"
    for item in items:
        assert isinstance(item, RetrievedItem)
        # Both arms normalise hits to docs Gold Section ids ('<file>.md#<slug>').
        assert item.source_section_id.split("#")[0].endswith(".md")


def test_stack_c_truncates_to_cutoff_k(fake_dense_embeddings):
    """The fused output is truncated to the final cutoff ``k`` (not the deep pool)."""
    _index_both_arms()
    para = next(p for p in load_paraphrases() if p.paraphrase_type == "synonym_swap")
    assert len(stacks.stack_c_retrieval(para.text, k=3)) <= 3
    assert len(stacks.stack_c_retrieval(para.text, k=1)) <= 1


def test_stack_c_overfetch_depth_decoupled_from_cutoff(fake_dense_embeddings):
    """A deep candidate pool feeds fusion independently of the final cutoff.

    'password' matches the password-reset wiki Section; with a deep pool the
    fused top-k is filled from a wide candidate set, but the returned list still
    honours the small cutoff ``k`` — overfetch (candidate_depth) and cutoff are
    separate knobs (AC2).
    """
    _index_both_arms()
    deep = stacks.stack_c_retrieval("password reset login", k=10, candidate_depth=50)
    cut = stacks.stack_c_retrieval("password reset login", k=3, candidate_depth=50)
    assert len(cut) <= 3
    assert len(deep) >= len(cut), "a deeper cutoff over the same pool returns >= items"


def test_stack_c_recovers_a_stack_a_keyword_hit(fake_dense_embeddings):
    """Fusion preserves BM25's keyword hit (recall-union includes the BM25 arm)."""
    _index_both_arms()
    # Stack A resolves this query to 'account_management.md#password-reset'; Stack
    # C fuses BM25 + dense, so that same docs id must still surface in the union.
    a_items = stacks.stack_a_retrieval("forgot my login passphrase reset", k=5)
    c_items = stacks.stack_c_retrieval("forgot my login passphrase reset", k=5)
    a_ids = {it.source_section_id for it in a_items}
    c_ids = {it.source_section_id for it in c_items}
    assert a_ids & c_ids, "Hybrid fusion must retain at least one BM25 keyword hit"


def test_stack_c_dense_arm_built_from_bm25_section_list(fake_dense_embeddings):
    """index_stack_c builds the dense index from BM25's Section list (1:1 ids)."""
    import markdown_kb.app.indexer as mk_indexer

    stacks.index_stack_a()
    n = stacks.index_stack_c()
    # One dense embedding per BM25 wiki Section — same-corpus invariant (ADR-0018).
    assert n == len(mk_indexer.sections)
    assert hk_dense.sections_indexed == n
