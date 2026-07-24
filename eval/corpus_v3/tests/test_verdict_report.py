"""Verdict-report tests — external behaviour only (CODING_STANDARD §0.2).

Every axis result below is hand-built (CODING_STANDARD §6.5): this module
never calls an LLM or a stack adapter, so "canned axis results" is the ONLY
kind of input it ever sees, live or not (issue #662 AC 1).
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.verdict_report import (
    DEMOTE_CLAUSE_TEXT,
    KILL_CLAUSE_TEXT,
    SURVIVAL_CLAUSE_TEXT,
    DecisionMatrixCell,
    DecisionMatrixRow,
    PairwiseComparison,
    VerdictReportInput,
    demote_clause_verdict,
    kill_clause_verdict,
    render_axis_stratum_tables,
    render_clause_walkthrough,
    render_cost_chapter,
    render_decision_matrix,
    render_honest_limits,
    render_verdict_report,
    survival_entries,
)


def _cmp(
    axis: str,
    arm_a: str,
    arm_b: str,
    rate_a: float,
    rate_b: float,
    p_value: float,
    *,
    stratum: str = "macro",
    n: int = 100,
    test_name: str = "mcnemar",
) -> PairwiseComparison:
    return PairwiseComparison(
        axis=axis,
        stratum=stratum,
        arm_a=arm_a,
        arm_b=arm_b,
        rate_a=rate_a,
        rate_b=rate_b,
        n=n,
        p_value=p_value,
        test_name=test_name,
    )


# ---------------------------------------------------------------------------
# PairwiseComparison — polarity-aware advantage
# ---------------------------------------------------------------------------
def test_advantage_arm_higher_is_better_axis():
    c = _cmp("grounding_pass_rate", "wiki", "rag", 0.9, 0.7, p_value=0.01)
    assert c.advantage_arm() == "wiki"
    assert c.significant_advantage("wiki") is True
    assert c.significant_advantage("rag") is False


def test_advantage_arm_lower_is_better_axis():
    """contradiction_leak_rate: a LOWER rate is the advantage."""
    c = _cmp("contradiction_leak_rate", "wiki", "rag", 0.05, 0.30, p_value=0.01)
    assert c.advantage_arm() == "wiki"


def test_advantage_arm_none_on_a_tie():
    c = _cmp("grounding_pass_rate", "wiki", "rag", 0.8, 0.8, p_value=0.9)
    assert c.advantage_arm() is None
    assert c.significant_advantage("wiki") is False


def test_significant_advantage_false_when_not_significant_even_if_directionally_better():
    """ADR-0045: 'a non-significant advantage still kills' — direction alone
    is not enough."""
    c = _cmp("grounding_pass_rate", "wiki", "rag", 0.85, 0.80, p_value=0.40)
    assert c.advantage_arm() == "wiki"
    assert c.significant_advantage("wiki") is False


def test_advantage_arm_raises_on_unknown_axis():
    c = _cmp("made_up_axis", "wiki", "rag", 0.9, 0.7, p_value=0.01)
    with pytest.raises(ValueError, match="unknown axis"):
        c.advantage_arm()


# ---------------------------------------------------------------------------
# kill_clause_verdict
# ---------------------------------------------------------------------------
def _three_axis_comparisons(*, wiki_wins_all: bool) -> list[PairwiseComparison]:
    if wiki_wins_all:
        return [
            _cmp("contradiction_leak_rate", "wiki", "rag", 0.05, 0.30, p_value=0.001),
            _cmp("grounding_pass_rate", "wiki", "rag", 0.90, 0.60, p_value=0.001),
            _cmp("correct_refusal_rate", "wiki", "rag", 0.85, 0.50, p_value=0.001),
        ]
    # wiki loses (or ties) on correct_refusal_rate -> kill clause fires
    return [
        _cmp("contradiction_leak_rate", "wiki", "rag", 0.05, 0.30, p_value=0.001),
        _cmp("grounding_pass_rate", "wiki", "rag", 0.90, 0.60, p_value=0.001),
        _cmp("correct_refusal_rate", "wiki", "rag", 0.50, 0.50, p_value=1.0),
    ]


def test_kill_clause_fires_when_wiki_lacks_advantage_on_any_axis():
    verdict = kill_clause_verdict(
        _three_axis_comparisons(wiki_wins_all=False),
        wiki_arm="wiki",
        baseline_arm="rag",
    )
    assert verdict.outcome == "killed"
    assert verdict.clause_text == KILL_CLAUSE_TEXT


def test_kill_clause_does_not_fire_when_wiki_wins_all_three_axes():
    verdict = kill_clause_verdict(
        _three_axis_comparisons(wiki_wins_all=True), wiki_arm="wiki", baseline_arm="rag"
    )
    assert verdict.outcome == "survives_kill_clause"


def test_kill_clause_raises_when_an_axis_comparison_is_missing():
    comparisons = _three_axis_comparisons(wiki_wins_all=True)[:2]  # drop one axis
    with pytest.raises(ValueError, match="missing"):
        kill_clause_verdict(comparisons, wiki_arm="wiki", baseline_arm="rag")


def test_kill_clause_ignores_comparisons_for_other_arm_pairs():
    comparisons = _three_axis_comparisons(wiki_wins_all=True) + [
        _cmp(
            "grounding_pass_rate", "hybrid", "dense_over_wiki", 0.5, 0.9, p_value=0.001
        )
    ]
    verdict = kill_clause_verdict(comparisons, wiki_arm="wiki", baseline_arm="rag")
    assert verdict.outcome == "survives_kill_clause"
    assert len(verdict.comparisons) == 3


# ---------------------------------------------------------------------------
# demote_clause_verdict
# ---------------------------------------------------------------------------
def test_demote_clause_fires_on_a_non_significant_difference():
    """ADR-0045: 'ties and non-significant differences both demote'."""
    comparison = _cmp(
        "contradiction_leak_rate", "hybrid", "rag", 0.10, 0.12, p_value=0.60
    )
    verdict = demote_clause_verdict(comparison, hybrid_arm="hybrid", baseline_arm="rag")
    assert verdict.outcome == "demoted"
    assert verdict.clause_text == DEMOTE_CLAUSE_TEXT


def test_demote_clause_fires_on_an_exact_tie():
    comparison = _cmp(
        "contradiction_leak_rate", "hybrid", "rag", 0.10, 0.10, p_value=1.0
    )
    verdict = demote_clause_verdict(comparison, hybrid_arm="hybrid", baseline_arm="rag")
    assert verdict.outcome == "demoted"


def test_demote_clause_does_not_fire_on_a_significant_hybrid_advantage():
    comparison = _cmp(
        "contradiction_leak_rate", "hybrid", "rag", 0.05, 0.30, p_value=0.001
    )
    verdict = demote_clause_verdict(comparison, hybrid_arm="hybrid", baseline_arm="rag")
    assert verdict.outcome == "retains_narrative_claim"


def test_demote_clause_raises_on_the_wrong_axis():
    comparison = _cmp("grounding_pass_rate", "hybrid", "rag", 0.9, 0.6, p_value=0.001)
    with pytest.raises(ValueError, match="contradiction_leak_rate"):
        demote_clause_verdict(comparison, hybrid_arm="hybrid", baseline_arm="rag")


def test_demote_clause_raises_on_a_mismatched_arm_pair():
    comparison = _cmp(
        "contradiction_leak_rate", "wiki", "rag", 0.05, 0.30, p_value=0.001
    )
    with pytest.raises(ValueError, match="arms"):
        demote_clause_verdict(comparison, hybrid_arm="hybrid", baseline_arm="rag")


# ---------------------------------------------------------------------------
# survival_entries
# ---------------------------------------------------------------------------
def test_survival_entries_finds_significant_wins_over_baseline():
    comparisons = [
        _cmp("grounding_pass_rate", "wiki", "rag", 0.9, 0.6, p_value=0.001),
        _cmp("correct_refusal_rate", "hybrid", "rag", 0.9, 0.6, p_value=0.001),
        _cmp(
            "contradiction_leak_rate", "hybrid", "wiki", 0.05, 0.10, p_value=0.001
        ),  # not vs baseline
    ]
    entries = survival_entries(
        comparisons,
        wiki_backed_arms=["wiki", "hybrid", "dense_over_wiki"],
        baseline_arm="rag",
    )
    assert [(e.arm, e.axis) for e in entries] == [
        ("hybrid", "correct_refusal_rate"),
        ("wiki", "grounding_pass_rate"),
    ]


def test_survival_entries_excludes_non_significant_wins():
    comparisons = [_cmp("grounding_pass_rate", "wiki", "rag", 0.7, 0.6, p_value=0.5)]
    entries = survival_entries(
        comparisons, wiki_backed_arms=["wiki"], baseline_arm="rag"
    )
    assert entries == []


def test_survival_entries_excludes_non_wiki_backed_arms():
    comparisons = [
        _cmp(
            "grounding_pass_rate", "dense_docs_variant", "rag", 0.9, 0.6, p_value=0.001
        )
    ]
    entries = survival_entries(
        comparisons, wiki_backed_arms=["wiki", "hybrid"], baseline_arm="rag"
    )
    assert entries == []


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def test_render_axis_stratum_tables_groups_by_axis():
    comparisons = _three_axis_comparisons(wiki_wins_all=True)
    text = render_axis_stratum_tables(comparisons)
    assert "### contradiction_leak_rate" in text
    assert "### grounding_pass_rate" in text
    assert "### correct_refusal_rate" in text


def test_render_clause_walkthrough_quotes_all_three_clauses_verbatim():
    kill = kill_clause_verdict(
        _three_axis_comparisons(wiki_wins_all=False),
        wiki_arm="wiki",
        baseline_arm="rag",
    )
    demote = demote_clause_verdict(
        _cmp("contradiction_leak_rate", "hybrid", "rag", 0.10, 0.10, p_value=1.0),
        hybrid_arm="hybrid",
        baseline_arm="rag",
    )
    text = render_clause_walkthrough(kill, demote, survivals=[])
    assert KILL_CLAUSE_TEXT in text
    assert DEMOTE_CLAUSE_TEXT in text
    assert SURVIVAL_CLAUSE_TEXT in text
    assert "killed" in text
    assert "demoted" in text


def test_render_cost_chapter_handles_a_none_cost_per_grounded_correct():
    text = render_cost_chapter(
        cost_per_grounded_correct={"wiki": 0.05, "rag": None},
        amortization_curves={"wiki": {10: 0.44, 100: 0.044}},
    )
    assert "$0.0500" in text
    assert "n/a (0 grounded-correct answers)" in text
    assert "n=10" in text


def test_render_decision_matrix_tags_every_cell():
    rows = [
        DecisionMatrixRow(
            label="Contradiction control",
            cells={
                "wiki": DecisionMatrixCell("measured on corpus v3", "measured-local"),
                "rag": DecisionMatrixCell("baseline", "measured-local"),
            },
        )
    ]
    text = render_decision_matrix(["wiki", "rag"], rows)
    assert "[measured-local]" in text
    assert "Contradiction control" in text


def test_render_honest_limits_numbers_each_entry():
    text = render_honest_limits(
        ["Structural analogues only.", "MDD is calculated, not cited."]
    )
    assert "1. Structural analogues only." in text
    assert "2. MDD is calculated, not cited." in text


def test_render_verdict_report_assembles_every_section_with_trust_note():
    kill = kill_clause_verdict(
        _three_axis_comparisons(wiki_wins_all=True), wiki_arm="wiki", baseline_arm="rag"
    )
    demote = demote_clause_verdict(
        _cmp("contradiction_leak_rate", "hybrid", "rag", 0.05, 0.30, p_value=0.001),
        hybrid_arm="hybrid",
        baseline_arm="rag",
    )
    report_input = VerdictReportInput(
        title="Corpus v3 verdict",
        tldr="Placeholder TL;DR.",
        comparisons=_three_axis_comparisons(wiki_wins_all=True),
        kill=kill,
        demote=demote,
        survivals=survival_entries(
            _three_axis_comparisons(wiki_wins_all=True),
            wiki_backed_arms=["wiki"],
            baseline_arm="rag",
        ),
        cost_per_grounded_correct={"wiki": 0.05},
        amortization_curves={"wiki": {10: 0.44}},
        decision_matrix_columns=["wiki"],
        decision_matrix_rows=[
            DecisionMatrixRow(
                label="Row", cells={"wiki": DecisionMatrixCell("x", "argued")}
            )
        ],
        honest_limits=["Limit one."],
        trust_note="⚠️ PLACEHOLDER — NOT REAL DATA.",
    )

    text = render_verdict_report(report_input)

    assert text.startswith("⚠️ PLACEHOLDER — NOT REAL DATA.")
    assert "# Corpus v3 verdict" in text
    assert "## TL;DR" in text
    assert "## Per-axis, per-stratum results" in text
    assert "## ADR-0045 clause walkthrough" in text
    assert "## Cost chapter" in text
    assert "## Method-comparison decision matrix" in text
    assert "## Honest limits" in text
