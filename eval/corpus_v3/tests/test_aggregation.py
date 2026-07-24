"""Per-stratum-first, macro-second aggregation tests — external behaviour only
(CODING_STANDARD §0.2).

The centrepiece is the AC's required fixture: a macro winner that differs
from a stratum winner, proving "per stratum first, macro second" surfaces
information a single pooled number would hide (PRD #654 user stories 6-7, 9,
13).
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.aggregation import (
    BY_LANGUAGE,
    BY_OVERLAP,
    BY_SCENARIO,
    QueryOutcome,
    Stratum,
    arm_metric_means,
    evaluate_query,
    group_by_stratum,
    macro_metrics,
    stratified_metrics,
    winning_arm,
)
from eval.corpus_v3.metrics import HIT_AT_1, HIT_AT_3
from eval.corpus_v3.models import RetrievedItem
from eval.corpus_v3.query_schema import Query


def _outcome(
    query_id, scenario, overlap, language, arm, hit1, hit3, rr
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        stratum=Stratum(scenario, overlap, language),
        arm=arm,
        hit_at_1=hit1,
        hit_at_3=hit3,
        reciprocal_rank=rr,
    )


# ---------------------------------------------------------------------------
# evaluate_query — the Query + RetrievedItem -> QueryOutcome seam
# ---------------------------------------------------------------------------
def test_evaluate_query_scores_hit_at_1_hit_at_3_and_reciprocal_rank():
    query = Query(
        query_id="q-1",
        text="return window?",
        scenario_stratum="factoid",
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=["returns_policy.md#return-window"],
        key_tokens=["refund", "window"],
    )
    items = [
        RetrievedItem(source_section_id="other#a", content="noise"),
        RetrievedItem(
            source_section_id="returns_policy.md#return-window",
            content="refund within the return window",
        ),
    ]
    outcome = evaluate_query(query, items, arm="stack_a")

    assert outcome.query_id == "q-1"
    assert outcome.arm == "stack_a"
    assert outcome.stratum == Stratum("factoid", "high_overlap", "en")
    assert outcome.hit_at_1 == 0.0  # gold item is rank 2
    assert outcome.hit_at_3 == 1.0
    assert outcome.reciprocal_rank == pytest.approx(0.5)


def test_evaluate_query_unanswerable_scores_all_metrics_zero():
    query = Query(
        query_id="q-2",
        text="what is the CEO's home address?",
        scenario_stratum="unanswerable",
        overlap_stratum="low_overlap",
        language="en",
    )
    items = [RetrievedItem(source_section_id="other#a", content="anything")]
    outcome = evaluate_query(query, items, arm="stack_a")
    assert (outcome.hit_at_1, outcome.hit_at_3, outcome.reciprocal_rank) == (
        0.0,
        0.0,
        0.0,
    )


# ---------------------------------------------------------------------------
# arm_metric_means
# ---------------------------------------------------------------------------
def test_arm_metric_means_averages_per_arm():
    outcomes = [
        _outcome("q1", "factoid", "high_overlap", "en", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q2", "factoid", "high_overlap", "en", "stack_a", 0.0, 0.0, 0.0),
        _outcome("q1", "factoid", "high_overlap", "en", "stack_b", 1.0, 1.0, 1.0),
        _outcome("q2", "factoid", "high_overlap", "en", "stack_b", 1.0, 1.0, 1.0),
    ]
    means = arm_metric_means(outcomes, "hit_at_3")
    assert means == {"stack_a": 0.5, "stack_b": 1.0}


def test_arm_metric_means_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        arm_metric_means([], "hit_at_3")


# ---------------------------------------------------------------------------
# group_by_stratum — full triple vs single-axis grouping
# ---------------------------------------------------------------------------
def test_group_by_stratum_full_triple_separates_distinct_strata():
    outcomes = [
        _outcome("q1", "factoid", "high_overlap", "en", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q2", "cross_doc", "low_overlap", "zh", "stack_a", 0.0, 0.0, 0.0),
    ]
    groups = group_by_stratum(outcomes)
    assert set(groups) == {"factoid|high_overlap|en", "cross_doc|low_overlap|zh"}


def test_group_by_language_pools_across_scenario_and_overlap():
    outcomes = [
        _outcome("q1", "factoid", "high_overlap", "en", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q2", "cross_doc", "low_overlap", "en", "stack_a", 0.0, 0.0, 0.0),
        _outcome("q3", "factoid", "high_overlap", "zh", "stack_a", 1.0, 1.0, 1.0),
    ]
    groups = group_by_stratum(outcomes, key=BY_LANGUAGE)
    assert set(groups) == {"en", "zh"}
    assert len(groups["en"]) == 2
    assert len(groups["zh"]) == 1


def test_group_by_scenario_and_overlap_are_selectable_axes():
    outcomes = [
        _outcome("q1", "factoid", "high_overlap", "en", "stack_a", 1.0, 1.0, 1.0),
        _outcome(
            "q2", "version_conflict", "low_overlap", "en", "stack_a", 0.0, 0.0, 0.0
        ),
    ]
    assert set(group_by_stratum(outcomes, key=BY_SCENARIO)) == {
        "factoid",
        "version_conflict",
    }
    assert set(group_by_stratum(outcomes, key=BY_OVERLAP)) == {
        "high_overlap",
        "low_overlap",
    }


# ---------------------------------------------------------------------------
# stratified_metrics / macro_metrics — the headline AC fixture
# ---------------------------------------------------------------------------
def test_macro_winner_differs_from_a_stratum_winner():
    """Three single-query strata: stack_a wins two, stack_b wins the third.

    Per-stratum: stratum_1 -> stack_a, stratum_2 -> stack_a, stratum_3 ->
    stack_b. Macro (mean of per-arm per-stratum means, one stratum one vote):
    stack_a = mean(1, 1, 0) = 0.667, stack_b = mean(0, 0, 1) = 0.333 -> the
    MACRO winner is stack_a, even though stack_b is the winner within
    stratum_3 alone. A pooled headline number would report only "stack_a
    wins" and hide that stack_b actually wins one whole stratum outright.
    """
    outcomes = [
        _outcome("q1", "factoid", "high_overlap", "en", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q1", "factoid", "high_overlap", "en", "stack_b", 0.0, 0.0, 0.0),
        _outcome("q2", "cross_doc", "high_overlap", "en", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q2", "cross_doc", "high_overlap", "en", "stack_b", 0.0, 0.0, 0.0),
        _outcome(
            "q3", "version_conflict", "low_overlap", "zh", "stack_a", 0.0, 0.0, 0.0
        ),
        _outcome(
            "q3", "version_conflict", "low_overlap", "zh", "stack_b", 1.0, 1.0, 1.0
        ),
    ]

    per_stratum = stratified_metrics(outcomes, "hit_at_3")
    macro = macro_metrics(outcomes, "hit_at_3")

    stratum_3_label = "version_conflict|low_overlap|zh"
    assert winning_arm(per_stratum[stratum_3_label]) == "stack_b"
    assert winning_arm(macro) == "stack_a"
    assert macro == {"stack_a": pytest.approx(2 / 3), "stack_b": pytest.approx(1 / 3)}


def test_macro_metrics_differs_from_naive_pooled_mean_when_strata_are_uneven():
    """Macro (one-stratum-one-vote) vs a size-weighted pooled mean can disagree.

    stratum_1 has 1 query, unanimous for stack_b. stratum_2 has 3 queries,
    unanimous for stack_a. Pooled (query-count-weighted) mean favours
    stack_a 3:1. Macro gives each stratum equal weight -> a tie, which is
    already a different conclusion than the pooled "stack_a wins" — the
    aggregation module deliberately does not silently fall back to the
    pooled number.
    """
    outcomes = [
        _outcome("q1", "factoid", "high_overlap", "en", "stack_a", 0.0, 0.0, 0.0),
        _outcome("q1", "factoid", "high_overlap", "en", "stack_b", 1.0, 1.0, 1.0),
        _outcome("q2", "cross_doc", "low_overlap", "zh", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q2", "cross_doc", "low_overlap", "zh", "stack_b", 0.0, 0.0, 0.0),
        _outcome("q3", "cross_doc", "low_overlap", "zh", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q3", "cross_doc", "low_overlap", "zh", "stack_b", 0.0, 0.0, 0.0),
        _outcome("q4", "cross_doc", "low_overlap", "zh", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q4", "cross_doc", "low_overlap", "zh", "stack_b", 0.0, 0.0, 0.0),
    ]

    pooled = arm_metric_means(outcomes, "hit_at_3")
    macro = macro_metrics(outcomes, "hit_at_3")

    assert winning_arm(pooled) == "stack_a"
    assert macro == {"stack_a": pytest.approx(0.5), "stack_b": pytest.approx(0.5)}


def test_macro_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        macro_metrics([], "hit_at_3")


def test_hit_at_1_and_mrr_are_reported_alongside_hit_at_3():
    """Both HIT_AT_1 and HIT_AT_3 constants map to the metric names the
    aggregation functions accept, plus reciprocal_rank for MRR."""
    outcomes = [
        _outcome("q1", "factoid", "high_overlap", "en", "stack_a", 1.0, 1.0, 1.0),
        _outcome("q1", "factoid", "high_overlap", "en", "stack_b", 0.0, 1.0, 0.5),
    ]
    assert arm_metric_means(outcomes, "hit_at_1") == {"stack_a": 1.0, "stack_b": 0.0}
    assert arm_metric_means(outcomes, "hit_at_3") == {"stack_a": 1.0, "stack_b": 1.0}
    assert arm_metric_means(outcomes, "reciprocal_rank") == {
        "stack_a": 1.0,
        "stack_b": 0.5,
    }
    assert HIT_AT_1 == 1 and HIT_AT_3 == 3


# ---------------------------------------------------------------------------
# winning_arm
# ---------------------------------------------------------------------------
def test_winning_arm_picks_the_highest_score():
    assert winning_arm({"stack_a": 0.4, "stack_b": 0.6}) == "stack_b"


def test_winning_arm_breaks_ties_by_name():
    assert winning_arm({"stack_b": 0.5, "stack_a": 0.5}) == "stack_a"


def test_winning_arm_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        winning_arm({})
