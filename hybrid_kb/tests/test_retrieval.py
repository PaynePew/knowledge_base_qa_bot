"""Hybrid Retrieval (Stack C) retrieval core — RRF + OR-gate + overfetch (S2, #312).

External behaviour only (CODING_STANDARD §0.2 / §6.2). Three deterministic seams,
all offline:

  * RRF fusion (pure) — synthetic ranked Section lists; assert summed contribution
    for a shared id, dedup, fused order, and overfetch→top_k truncation.
  * Pre-LLM OR-gate (pure) — synthetic native scores + language; assert proceed vs
    Cannot Confirm(``below_threshold``), per-language threshold selection, and that
    the gate is OR (not AND).
  * ``retrieve_and_gate`` (deep module, no mocking) — both arms built over a small
    synthetic wiki corpus (BM25 via the real ``markdown_kb`` indexer, dense via the
    real FAISS path with offline ``fake_embeddings``). Asserts the gate is enforced
    inside the module, fused Sections satisfy ``CitableContent``, and overfetch
    depth is decoupled from the final top-k.

No ``@pytest.mark.live`` here — the dense arm runs on the deterministic offline
fake (#311 hermetic pattern); the one live smoke belongs to S1's dense build.
"""

from __future__ import annotations

import pytest

import hybrid_kb.app.dense_index as dense_index
import hybrid_kb.app.retrieval as retrieval
import markdown_kb.app.indexer as bm25_indexer
import markdown_kb.app.retrieval as bm25_gate
import vector_rag.app.retrieval as dense_gate_module
from markdown_kb.app.grounding import CitableContent, GroundingOutcome
from markdown_kb.app.indexer import Section

from .conftest import make_section


# ===========================================================================
# AC2 — RRF is a pure function over two ranked lists
# ===========================================================================
def _ranked(*ids: str) -> list[Section]:
    """A rank-ordered Section list keyed only by id (RRF reads id + rank)."""
    return [make_section(i, content=f"body of {i}") for i in ids]


def test_rrf_sums_reciprocal_rank_contributions_for_shared_id():
    """A Section id present in BOTH arms sums its two 1/(K+rank) contributions."""
    k = retrieval.RRF_K
    # "shared" is rank 1 in arm A and rank 2 in arm B.
    arm_a = _ranked("shared", "a-only")
    arm_b = _ranked("b-only", "shared")

    fused = retrieval.reciprocal_rank_fusion(arm_a, arm_b, k=k, top_k=10)
    scores = {section.id: score for section, score in fused}

    assert scores["shared"] == pytest.approx(1.0 / (k + 1) + 1.0 / (k + 2))
    assert scores["a-only"] == pytest.approx(1.0 / (k + 2))
    assert scores["b-only"] == pytest.approx(1.0 / (k + 1))


def test_rrf_dedups_shared_id_to_one_entry():
    """An id in both arms appears exactly once in the fused output (dedup by id)."""
    fused = retrieval.reciprocal_rank_fusion(
        _ranked("shared", "x"), _ranked("shared", "y"), top_k=10
    )
    ids = [section.id for section, _ in fused]
    assert ids.count("shared") == 1
    assert set(ids) == {"shared", "x", "y"}


def test_rrf_orders_by_summed_contribution():
    """A Section both arms rank highly outranks one only a single arm ranks high."""
    # "both" is rank 1 in A and rank 1 in B → highest summed score.
    # "a-top" is rank 1 in A only; "b-top" is rank 1 in B only.
    arm_a = _ranked("both", "a-top")
    arm_b = _ranked("both", "b-top")
    fused = retrieval.reciprocal_rank_fusion(arm_a, arm_b, top_k=10)
    assert fused[0][0].id == "both", "a Section ranked #1 in both arms must fuse to #1"


def test_rrf_truncates_to_top_k_independent_of_input_depth():
    """A deep candidate pool per arm collapses to the final top_k cutoff."""
    arm_a = _ranked(*[f"a{i}" for i in range(50)])
    arm_b = _ranked(*[f"b{i}" for i in range(50)])
    fused = retrieval.reciprocal_rank_fusion(arm_a, arm_b, top_k=3)
    assert len(fused) == 3, "overfetched pools must truncate to top_k"


def test_rrf_keeps_disjoint_ids_from_both_arms():
    """Ids unique to one arm still surface (recall-union)."""
    fused = retrieval.reciprocal_rank_fusion(_ranked("a1"), _ranked("b1"), top_k=10)
    assert {section.id for section, _ in fused} == {"a1", "b1"}


def test_rrf_empty_inputs_return_empty():
    assert retrieval.reciprocal_rank_fusion([], [], top_k=3) == []


# ===========================================================================
# AC3 / AC4 — pre-LLM OR-gate is a pure function; reuses the two thresholds
# ===========================================================================
# Native-score fixtures relative to the reused calibrated thresholds:
#   BM25  en clears at >= _SCORE_THRESHOLD (0.5), zh at >= _SCORE_THRESHOLD_ZH (4.0)
#   dense clears at min-distance <= _max_rag_distance() (1.1)
_DENSE_CLEARS = 0.5  # <= 1.1
_DENSE_FAILS = 5.0  # > 1.1


def test_gate_proceeds_when_only_bm25_clears():
    """BM25 clears, dense fails → proceed. (AND would refuse — this asserts OR.)"""
    outcome = retrieval.evaluate_or_gate(
        bm25_top_score=2.0, dense_best_distance=_DENSE_FAILS, lang="en"
    )
    assert outcome.passed is True
    assert outcome.reason == "claim_supported"


def test_gate_proceeds_when_only_dense_clears():
    """Dense clears, BM25 fails → proceed. (AND would refuse — this asserts OR.)"""
    outcome = retrieval.evaluate_or_gate(
        bm25_top_score=0.1, dense_best_distance=_DENSE_CLEARS, lang="en"
    )
    assert outcome.passed is True
    assert outcome.reason == "claim_supported"


def test_gate_cannot_confirm_when_both_arms_below_threshold():
    """Neither arm clears → Cannot Confirm with the shared below_threshold reason."""
    outcome = retrieval.evaluate_or_gate(
        bm25_top_score=0.1, dense_best_distance=_DENSE_FAILS, lang="en"
    )
    assert outcome.passed is False
    # The reason is validated against GroundingOutcome's Literal, not a bare
    # string — a typo would fail Pydantic validation (the Cannot Confirm contract).
    assert outcome == GroundingOutcome(passed=False, reason="below_threshold")


def test_gate_proceeds_when_both_clear():
    outcome = retrieval.evaluate_or_gate(
        bm25_top_score=2.0, dense_best_distance=_DENSE_CLEARS, lang="en"
    )
    assert outcome.passed is True


def test_gate_and_semantics_explicitly_rejected():
    """OR, not AND: every single-arm-clears case proceeds; only all-weak refuses.

    AND-semantics would defeat the recall-union purpose (ADR-0018) — a Section one
    arm ranks low but the other ranks high must still reach fusion. Encoded as a
    test so the OR contract cannot silently regress to AND.
    """
    only_bm25 = retrieval.evaluate_or_gate(2.0, _DENSE_FAILS, "en")
    only_dense = retrieval.evaluate_or_gate(0.1, _DENSE_CLEARS, "en")
    both_weak = retrieval.evaluate_or_gate(0.1, _DENSE_FAILS, "en")

    assert only_bm25.passed and only_dense.passed, "OR proceeds when EITHER arm clears"
    assert both_weak.passed is False, "only all-weak refuses"
    # If the gate were AND, only_bm25 and only_dense would BOTH be False.
    assert (only_bm25.passed and only_dense.passed) != both_weak.passed


def test_gate_selects_per_language_bm25_threshold():
    """The zh band (4.0) and en band (0.5) gate the same BM25 score differently.

    A score of 3.0 clears the English 0.5 gate but not the Chinese 4.0 gate (#261):
    proves the per-language threshold is selected by the query language, reusing
    markdown_kb's two calibrated constants.
    """
    dense_fails = _DENSE_FAILS
    assert retrieval.evaluate_or_gate(3.0, dense_fails, "en").passed is True
    assert retrieval.evaluate_or_gate(3.0, dense_fails, "zh").passed is False


def test_gate_reuses_thresholds_by_reference_not_a_redefined_constant(monkeypatch):
    """Monkeypatching the OWNING modules' thresholds changes the gate verdict.

    Proves S2 reuses the existing eval-calibrated thresholds by reference (no new
    threshold introduced, ADR-0018): if hybrid had its own copy, these patches
    would have no effect.
    """
    # Raise the en BM25 bar above the score → BM25 no longer clears.
    monkeypatch.setattr(bm25_gate, "_SCORE_THRESHOLD", 100.0)
    # Shrink the dense ceiling below the distance → dense no longer clears.
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    outcome = retrieval.evaluate_or_gate(2.0, 0.5, "en")
    assert outcome.passed is False, "patched thresholds must flip the verdict"


def test_gate_handles_empty_arms_as_no_clear():
    """No candidates from either arm (None scores) → Cannot Confirm."""
    outcome = retrieval.evaluate_or_gate(None, None, "en")
    assert outcome.passed is False
    assert outcome.reason == "below_threshold"


# ===========================================================================
# AC1 / AC4 / AC6 — retrieve_and_gate over real (test-built) arms, no mocking
# ===========================================================================
def _wiki_section(section_id: str, content: str, heading_path: list[str]) -> Section:
    """A BM25-ready English wiki Section (real tokens via the production tokenizer)."""
    return Section(
        id=section_id,
        file=section_id.split("#")[0],
        heading=heading_path[-1],
        heading_path=heading_path,
        content=content,
        tokens=bm25_indexer.tokenize(content),
        metadata={"lang": "en"},
    )


_CORPUS = [
    _wiki_section(
        "refund-policy#refund-policy",
        "Refund policy: refunds are processed within seven business days after "
        "approval. How long a refund takes is usually about one week.",
        ["Refund Policy"],
    ),
    _wiki_section(
        "shipping-policy#shipping-policy",
        "Shipping policy: standard shipping policy delivery takes three to five "
        "business days within the country.",
        ["Shipping Policy"],
    ),
    _wiki_section(
        "return-policy#return-policy",
        "Return policy: our policy lets a customer return purchased items within "
        "thirty days to receive a refund.",
        ["Return Policy"],
    ),
    _wiki_section(
        "privacy-policy#privacy-policy",
        "Privacy policy: this policy explains how the company handles your personal "
        "account data and contact details.",
        ["Privacy Policy"],
    ),
]


@pytest.fixture()
def wired_corpus(fake_embeddings):
    """Build BOTH arms over the synthetic corpus with no deep-module mocking.

    BM25: populate ``markdown_kb`` indexer state directly (no build_index, so the
    committed ``.kb/index.json`` seed is never written). Dense: real FAISS build
    over the same Section list via the offline fake embeddings. The shared Section
    objects guarantee 1:1 id alignment (the ADR-0018 same-corpus invariant).
    """
    bm25_indexer.sections = list(_CORPUS)
    bm25_indexer.rebuild_stats()
    dense_index.build_index(sections=list(_CORPUS))
    yield _CORPUS
    bm25_indexer.sections = []
    bm25_indexer.rebuild_stats()
    dense_index.vectorstore = None
    dense_index.sections_indexed = 0


def test_retrieve_proceeds_and_returns_citable_sections(wired_corpus):
    """An in-scope query clears the gate and returns fused CitableContent Sections."""
    result = retrieval.retrieve_and_gate("how long do refunds take")

    assert result["early_exit"] is False
    assert isinstance(result["grounding_outcome"], GroundingOutcome)
    assert result["grounding_outcome"].passed is True

    sections = result["sections"]
    assert sections, "an in-scope query must return fused Sections"
    assert len(sections) <= retrieval.DEFAULT_TOP_K
    for section in sections:
        assert isinstance(section, Section)
        # Satisfies the downstream-synthesis input contract (ADR-0004 Q9).
        assert isinstance(section, CitableContent)
        assert section.id and section.heading_path and section.content


def test_retrieve_top_hit_is_the_refund_section(wired_corpus):
    """Fusion surfaces the on-topic Section first for a clear keyword query."""
    result = retrieval.retrieve_and_gate("refund refunds processed approval")
    top_ids = [s.id for s in result["sections"]]
    assert "refund-policy#refund-policy" in top_ids


def test_retrieve_cannot_confirm_when_both_arms_weak(wired_corpus, monkeypatch):
    """Out-of-scope query with the dense gate disabled → Cannot Confirm(below_threshold).

    Forcing the dense ceiling to 0 (so the dense arm can never clear) isolates the
    BM25 arm: a query with no keyword overlap leaves BOTH arms below threshold, and
    the gate — enforced INSIDE the deep module — returns Cannot Confirm.
    """
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "0.0")
    result = retrieval.retrieve_and_gate("xylophone quokka zugzwang nonsense")

    assert result["early_exit"] is True
    assert result["grounding_outcome"].passed is False
    assert result["grounding_outcome"].reason == "below_threshold"


def test_overfetch_candidate_depth_is_decoupled_from_top_k(wired_corpus):
    """A deeper candidate_depth widens each arm's pool independently of top_k.

    The query 'policy' matches all four corpus Sections. At candidate_depth=1 each
    arm yields a single hit, so fusion has at most two distinct Sections; at
    candidate_depth=5 each arm overfetches the full corpus and fusion fills the
    top_k=3 cutoff. Same top_k, different pool depth → AC1.
    """
    shallow = retrieval.retrieve_and_gate("policy", candidate_depth=1, top_k=3)
    deep = retrieval.retrieve_and_gate("policy", candidate_depth=5, top_k=3)

    assert len(shallow["sections"]) <= 2, "depth=1 limits each arm to one hit"
    assert len(deep["sections"]) == 3, "a deep pool fills the top_k cutoff"
    assert len(deep["sections"]) > len(shallow["sections"]), (
        "candidate_depth must control arm overfetch independently of top_k"
    )


# ===========================================================================
# #327 — _dense_arm_clears None-ceiling gate parity with vector_rag
#
# vector_rag treats None ceiling as "gate disabled → proceed" (its refuse
# condition is  min_dist > _max_rag_distance(), so None disables the gate).
# hybrid_kb must match that semantics exactly (ADR-0018 §4.3 gate-parity).
# Monkeypatching dense_gate_module._max_rag_distance patches the SAME object
# that hybrid_kb.app.retrieval binds to its module-level _dense_gate alias —
# same Python object, so the patch is seen at call time.
# ===========================================================================


def test_dense_arm_clears_none_ceiling_gate_disabled(monkeypatch):
    """None ceiling → gate disabled → a present dense hit clears.

    Parity with vector_rag: when _max_rag_distance() returns None the dense
    gate is disabled, meaning any hit that is present (best_distance is not
    None) should be treated as clearing. The pre-#327 code returned False here,
    the opposite of vector_rag's semantics (#327).
    """
    monkeypatch.setattr(dense_gate_module, "_max_rag_distance", lambda: None)
    assert retrieval._dense_arm_clears(0.5) is True


def test_dense_arm_no_hit_never_clears_regardless_of_ceiling(monkeypatch):
    """No dense hit (best_distance=None) never clears, even with gate disabled.

    This is orthogonal to the ceiling: a missing dense hit means the arm
    produced nothing, which cannot constitute a clearance regardless of whether
    the distance gate is disabled or not.
    """
    monkeypatch.setattr(dense_gate_module, "_max_rag_distance", lambda: None)
    assert retrieval._dense_arm_clears(None) is False


def test_dense_arm_clears_hit_within_ceiling(monkeypatch):
    """A hit within the non-None ceiling clears (non-None ceiling, distance ≤ ceiling)."""
    monkeypatch.setattr(dense_gate_module, "_max_rag_distance", lambda: 1.1)
    assert retrieval._dense_arm_clears(0.5) is True


def test_dense_arm_clears_hit_exceeds_ceiling(monkeypatch):
    """A hit beyond the non-None ceiling does not clear (distance > ceiling)."""
    monkeypatch.setattr(dense_gate_module, "_max_rag_distance", lambda: 1.1)
    assert retrieval._dense_arm_clears(5.0) is False
