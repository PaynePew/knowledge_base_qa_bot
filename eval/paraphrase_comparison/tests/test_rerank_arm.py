"""4th-arm (Stack C + rerank) scoring + report rendering — ADR-0019 / #310.

External behaviour only (CODING_STANDARD §0.2). Fast and synthetic where it can
be: ``score_rerank_comparison`` is driven with in-process fake retrieval
callables (no indexing, no model); ``_render_rerank_section`` is exercised via
``render_report`` with a hand-built ``RerankComparison``; and one offline
``run_comparison(with_rerank=True)`` integration test wires the real eval path
with the deterministic fake cross-encoder so the focused section renders without
torch or the 2.3 GB model.
"""

from __future__ import annotations

from eval.paraphrase_comparison.models import (
    CORE_PARAPHRASE_TYPES,
    PROBE_PARAPHRASE_TYPES,
    Paraphrase,
    RetrievedItem,
)
from eval.paraphrase_comparison.runner import (
    RerankComparison,
    StackScores,
    SweepScores,
    render_report,
    run_comparison,
    score_rerank_comparison,
)

SWEEP = (1, 3, 5, 10)
_ALL_TYPES = (*CORE_PARAPHRASE_TYPES, *PROBE_PARAPHRASE_TYPES)


def _para(pid: str, ptype: str = "synonym_swap") -> Paraphrase:
    return Paraphrase(
        paraphrase_id=pid,
        paraphrase_type=ptype,
        text=f"q {pid}",
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


# ---------------------------------------------------------------------------
# score_rerank_comparison — aligned paired pass over C and C+rerank
# ---------------------------------------------------------------------------
def test_score_rerank_comparison_pairs_c_and_d():
    """D hits hit@3 where C misses on every Core paraphrase → paired c=n, b=0."""
    paras = [_para(f"p{i}") for i in range(4)]

    def retrieve_c(_text, _k):
        return [_MISS, _MISS, _MISS, _MISS, _MATCH]  # gold at rank 5 → miss@3

    def retrieve_d(_text, _k):
        return [_MATCH, _MISS, _MISS, _MISS, _MISS]  # gold at rank 1 → hit@3

    comp = score_rerank_comparison(paras, retrieve_c, retrieve_d)

    assert comp.paired_c == 4, "C-miss/D-hit on all four Core paraphrases"
    assert comp.paired_b == 0
    assert comp.primary_d.by_type["synonym_swap"] == 1.0
    assert comp.primary_c.by_type["synonym_swap"] == 0.0
    assert comp.mean_added_latency_ms >= 0.0
    assert comp.n_queries == 4
    assert comp.paired_cutoff == 3


def test_score_rerank_comparison_reads_sweep_from_one_pool():
    """A gold hit at rank 5 (C) scores @5/@10 but not @1/@3 — one deep pass, sliced."""

    def retrieve_c(_text, _k):
        return [_MISS, _MISS, _MISS, _MISS, _MATCH]

    def retrieve_d(_text, _k):
        return [_MATCH]

    comp = score_rerank_comparison([_para("p1")], retrieve_c, retrieve_d)

    assert comp.sweep_c.hit_by_cutoff[3]["synonym_swap"] == 0.0
    assert comp.sweep_c.hit_by_cutoff[5]["synonym_swap"] == 1.0
    assert comp.sweep_d.hit_by_cutoff[1]["synonym_swap"] == 1.0


# ---------------------------------------------------------------------------
# _render_rerank_section (via render_report) — focused, additive section
# ---------------------------------------------------------------------------
def _stack(name: str, hit: float, mrr: float) -> StackScores:
    return StackScores(
        stack=name,
        k=3,
        by_type={t: hit for t in _ALL_TYPES},
        mrr_by_type={t: mrr for t in _ALL_TYPES},
        n_by_type={t: 10 for t in _ALL_TYPES},
    )


def _sweep(name: str, base: float) -> SweepScores:
    return SweepScores(
        stack=name,
        cutoffs=SWEEP,
        hit_by_cutoff={c: {t: base for t in CORE_PARAPHRASE_TYPES} for c in SWEEP},
        mrr_by_cutoff={c: {t: base for t in CORE_PARAPHRASE_TYPES} for c in SWEEP},
        n_by_type={t: 10 for t in CORE_PARAPHRASE_TYPES},
    )


def _comparison() -> RerankComparison:
    return RerankComparison(
        primary_c=_stack("Stack C", 0.6, 0.5),
        primary_d=_stack("Stack C + rerank", 0.7, 0.6),
        sweep_c=_sweep("Stack C", 0.6),
        sweep_d=_sweep("Stack C + rerank", 0.7),
        paired_cutoff=3,
        paired_b=2,
        paired_c=8,
        paired_mcnemar_p=0.1094,
        mean_added_latency_ms=42.5,
        n_queries=260,
    )


def test_render_report_includes_rerank_section_when_supplied():
    report = render_report(
        _stack("Stack A", 0.6, 0.5),
        _stack("Stack B", 0.5, 0.4),
        embedding_mode="real",
        rerank_comparison=_comparison(),
    )
    assert "## Reranker Evaluation (Stack C → Stack C + rerank)" in report
    assert "hit (C+rerank)" in report  # the side-by-side cutoff-sweep column
    assert "Mean added latency" in report
    assert "42.5 ms/query" in report
    assert "Gate check" in report
    # The reranker section never claims to feed the gate or surface a score.
    assert "never loaded" in report


def test_render_report_omits_rerank_section_by_default():
    report = render_report(
        _stack("Stack A", 0.6, 0.5),
        _stack("Stack B", 0.5, 0.4),
        embedding_mode="real",
    )
    assert "## Reranker Evaluation" not in report


def test_rerank_section_gate_check_reads_met_on_probe_lift_no_core_regression():
    """A probe lift with no Core regression renders a MET gate check."""
    comp = RerankComparison(
        primary_c=_stack("Stack C", 0.6, 0.5),
        primary_d=_stack("Stack C + rerank", 0.7, 0.6),  # +0.1 everywhere
        sweep_c=_sweep("Stack C", 0.6),
        sweep_d=_sweep("Stack C + rerank", 0.7),
        paired_cutoff=3,
        paired_b=0,
        paired_c=10,
        paired_mcnemar_p=0.002,
        mean_added_latency_ms=40.0,
        n_queries=260,
    )
    report = render_report(
        _stack("Stack A", 0.6, 0.5),
        _stack("Stack B", 0.5, 0.4),
        embedding_mode="real",
        rerank_comparison=comp,
    )
    assert "Gate check (ADR-0019): MET" in report


# ---------------------------------------------------------------------------
# run_comparison(with_rerank=True) — full offline path with the fake encoder
# ---------------------------------------------------------------------------
def test_run_comparison_with_rerank_renders_focused_section(
    tmp_path, fake_vector_index, fake_cross_encoder
):
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake", with_rerank=True)
    report = report_path.read_text(encoding="utf-8")

    # The focused 4th-arm section is present...
    assert "## Reranker Evaluation (Stack C → Stack C + rerank)" in report
    assert "Mean added latency" in report
    # ...flagged as offline tracer (fake cross-encoder)...
    assert "OFFLINE" in report
    # ...and the existing three-arm sections are untouched (additive).
    assert "## Cutoff Sweep" in report
    assert "Cochran's Q" in report


def test_run_comparison_without_rerank_has_no_focused_section(
    tmp_path, fake_vector_index
):
    """The default 3-arm run must not render the reranker section."""
    report_path = tmp_path / "report.md"
    run_comparison(report_path=report_path, embedding_mode="fake")
    report = report_path.read_text(encoding="utf-8")
    assert "## Reranker Evaluation" not in report
