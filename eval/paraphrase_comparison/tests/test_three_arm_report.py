"""Three-arm report + cutoff-sweep scoring tests (#316).

Fast and synthetic: ``render_report`` is exercised with hand-built ``StackScores``
/ ``SweepScores`` / ``ThreeArmStats`` (no indexing, no FAISS), and
``score_three_arms`` is driven with in-process fake retrieval callables so the
overfetch-decoupled-from-cutoff sweep (AC2) and the Cochran's Q omnibus wiring
(AC3) are asserted without building any index. These cover the new rendering and
scoring paths the heavier ``run_comparison`` integration tests also exercise.
"""

from __future__ import annotations

from eval.paraphrase_comparison.models import (
    CORE_PARAPHRASE_TYPES,
    PARAPHRASE_TYPES,
    Paraphrase,
    RetrievedItem,
)
from eval.paraphrase_comparison.runner import (
    StackScores,
    SweepScores,
    ThreeArmStats,
    render_report,
    score_three_arms,
)
from eval.paraphrase_comparison.statistics import cochran_q

SWEEP_CUTOFFS = (1, 3, 5, 10)


def _stack_scores(name: str, hit: float, mrr: float) -> StackScores:
    """A StackScores with the same hit/MRR for every Paraphrase Type (PRIMARY k=3)."""
    return StackScores(
        stack=name,
        k=3,
        by_type={t: hit for t in PARAPHRASE_TYPES},
        mrr_by_type={t: mrr for t in PARAPHRASE_TYPES},
        n_by_type={t: 10 for t in PARAPHRASE_TYPES},
    )


def _sweep_scores(name: str, base: float) -> SweepScores:
    """A SweepScores whose hit/MRR climb with the cutoff (monotone, like real data)."""
    hit_by_cutoff = {
        c: {t: min(1.0, base + 0.05 * c) for t in PARAPHRASE_TYPES}
        for c in SWEEP_CUTOFFS
    }
    mrr_by_cutoff = {c: {t: base for t in PARAPHRASE_TYPES} for c in SWEEP_CUTOFFS}
    return SweepScores(
        stack=name,
        cutoffs=SWEEP_CUTOFFS,
        hit_by_cutoff=hit_by_cutoff,
        mrr_by_cutoff=mrr_by_cutoff,
        n_by_type={t: 10 for t in PARAPHRASE_TYPES},
    )


def _three_arm_stats() -> ThreeArmStats:
    # A separates from B and C → significant omnibus, post-hoc gate open.
    hits_a = [1, 1, 1, 1]
    hits_b = [0, 0, 0, 0]
    hits_c = [0, 0, 1, 0]
    cochran = cochran_q(hits_a, hits_b, hits_c)
    return ThreeArmStats(
        cutoff=3,
        cochran=cochran,
        posthoc_significant=cochran.p_value < 0.05,
        pair_labels=("Wiki ↔ RAG", "Hybrid ↔ Wiki", "Hybrid ↔ RAG"),
        pair_bc=((4, 0), (0, 3), (1, 0)),
        pair_raw_p=(0.0625, 0.25, 1.0),
        pair_holm_p=(0.1875, 0.5, 1.0),
    )


def _render_three_arm_report() -> str:
    stack_a = _stack_scores("Stack A", 0.6, 0.5)
    stack_b = _stack_scores("Stack B", 0.5, 0.4)
    stack_c = _stack_scores("Stack C", 0.7, 0.6)
    sweep = (
        _sweep_scores("Stack A", 0.5),
        _sweep_scores("Stack B", 0.4),
        _sweep_scores("Stack C", 0.6),
    )
    return render_report(
        stack_a,
        stack_b,
        embedding_mode="fake",
        stack_c=stack_c,
        sweep=sweep,
        three_arm=_three_arm_stats(),
    )


def test_report_renders_stack_c_columns():
    report = _render_three_arm_report()
    assert "hit_rate@3 (C)" in report
    assert "MRR (C)" in report
    assert "Δ (C−A)" in report
    # The legacy two-arm columns are still present (backward compatible).
    assert "hit_rate@3 (A)" in report
    assert "hit_rate@3 (B)" in report


def test_report_renders_cutoff_sweep_for_all_arms():
    report = _render_three_arm_report()
    assert "## Cutoff Sweep" in report
    # Every cutoff in the sweep has a row.
    for cutoff in SWEEP_CUTOFFS:
        assert f"hit@{cutoff}" in report
    # The sweep table carries all three arms' columns.
    assert "hit_rate (C)" in report
    assert "MRR (C)" in report


def test_report_renders_cochran_q_and_posthoc_pairs():
    report = _render_three_arm_report()
    assert "Cochran's Q" in report
    assert "Three-Arm Statistical Tests" in report
    # The three post-hoc pairwise McNemar comparisons appear with Holm correction.
    assert "Wiki ↔ RAG" in report
    assert "Hybrid ↔ Wiki" in report
    assert "Hybrid ↔ RAG" in report
    assert "Holm p" in report


def test_report_documents_supersedes_phase_8():
    report = _render_three_arm_report()
    assert "supersede" in report.lower()
    assert "Phase 8" in report
    # The offline banner must still warn these are not real-embedding numbers.
    assert "OFFLINE TRACER NUMBERS" in report


def test_report_two_arm_fallback_omits_three_arm_sections():
    """A legacy two-arm render_report call must not emit the Hybrid/sweep sections."""
    report = render_report(
        _stack_scores("Stack A", 0.6, 0.5),
        _stack_scores("Stack B", 0.5, 0.4),
        embedding_mode="fake",
    )
    assert "hit_rate@3 (C)" not in report
    assert "## Cutoff Sweep" not in report
    assert "Cochran's Q" not in report


# ---------------------------------------------------------------------------
# score_three_arms — overfetch decoupled from cutoff (AC2), no indexing
# ---------------------------------------------------------------------------
def _para(pid: str, ptype: str = "synonym_swap") -> Paraphrase:
    return Paraphrase(
        paraphrase_id=pid,
        paraphrase_type=ptype,
        text=f"query {pid}",
        gold_docs_section_id="gold.md#x",
        key_tokens_docs=["foo"],
        key_tokens_wiki=[],
    )


_MATCH = RetrievedItem(
    source_section_id="gold.md#x", content="foo bar", heading_path=[]
)
_MISS = RetrievedItem(
    source_section_id="other.md#y", content="baz qux", heading_path=[]
)


def test_score_three_arms_reads_multiple_cutoffs_from_one_pool():
    """A gold hit at rank 5 scores hit@5/@10 but not hit@1/@3 — one deep pass, sliced."""
    # rank-5 hit: four misses then the match (index 4).
    deep_list = [_MISS, _MISS, _MISS, _MISS, _MATCH]

    def retrieve(_text, _k):
        return list(deep_list)

    scoring = score_three_arms([_para("p1")], retrieve, retrieve, retrieve)
    hit_by_cutoff = scoring.sweep[0].hit_by_cutoff
    assert hit_by_cutoff[1]["synonym_swap"] == 0.0
    assert hit_by_cutoff[3]["synonym_swap"] == 0.0
    assert hit_by_cutoff[5]["synonym_swap"] == 1.0
    assert hit_by_cutoff[10]["synonym_swap"] == 1.0


def test_score_three_arms_omnibus_opens_gate_on_separation():
    """Arm A hits, B and C miss on every Core paraphrase → significant Cochran's Q."""
    paras = [_para(f"p{i}") for i in range(4)]

    def hit_all(_text, _k):
        return [_MATCH]

    def miss_all(_text, _k):
        return [_MISS]

    scoring = score_three_arms(paras, hit_all, miss_all, miss_all)
    assert scoring.three_arm.cochran.df == 2
    assert scoring.three_arm.posthoc_significant is True
    # Pooled Core hit vectors feed the omnibus at the primary cutoff (3).
    assert scoring.three_arm.cutoff == 3
    assert set(CORE_PARAPHRASE_TYPES)  # sanity: Core types exist to pool over
