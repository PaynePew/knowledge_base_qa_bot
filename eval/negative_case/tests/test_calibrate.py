"""Threshold-calibration tests (#253).

Pure sweep/recommend logic is tested with synthetic scores; ``collect_scores`` is
an integration check that in-scope queries retrieve (non-zero) and clearly-out-of-
scope queries do not. LLM-free; the autouse conftest isolates production paths.
"""

from __future__ import annotations

from eval.negative_case.calibrate import (
    ThresholdPoint,
    collect_scores,
    recommend,
    sweep,
)
from eval.negative_case.positive_cases import POSITIVE_CASES


def test_sweep_counts_scores_below_threshold():
    pos = [2.0, 2.0]
    neg = [0.0, 0.0, 1.5]
    points = sweep(pos, neg, thresholds=[1.0, 1.75])
    p1, p2 = points
    assert p1.threshold == 1.0
    assert p1.correct_refusal_rate == 2 / 3  # the two zeros are < 1.0
    assert p1.over_refusal_rate == 0.0  # no positive below 1.0
    assert p2.correct_refusal_rate == 1.0  # 0, 0, 1.5 all < 1.75
    assert p2.over_refusal_rate == 0.0


def test_youden_j_is_difference():
    assert abs(ThresholdPoint(1.0, 0.8, 0.1).youden_j - 0.7) < 1e-9


def test_recommend_keeps_current_default_when_optimal():
    """When the shipped 0.5 default is itself J-optimal, keep it (no churn)."""
    points = [
        ThresholdPoint(0.25, 0.87, 0.0),  # J = 0.87 (optimal plateau)
        ThresholdPoint(0.5, 0.87, 0.0),  # J = 0.87 — current default, optimal
        ThresholdPoint(1.5, 0.93, 0.10),  # J = 0.83
    ]
    assert recommend(points).threshold == 0.5


def test_recommend_picks_plateau_median_when_default_suboptimal():
    """When 0.5 is NOT optimal, pick the median of the optimal plateau."""
    points = [
        ThresholdPoint(0.5, 0.5, 0.0),  # current default, sub-optimal
        ThresholdPoint(1.0, 0.9, 0.0),  # optimal plateau
        ThresholdPoint(1.25, 0.9, 0.0),  # optimal plateau
        ThresholdPoint(1.5, 0.9, 0.0),  # optimal plateau
    ]
    # optimal = [1.0, 1.25, 1.5]; median (index 1) → 1.25
    assert recommend(points).threshold == 1.25


def test_collect_scores_separates_positive_from_clearly_oos():
    positive, negative = collect_scores()
    assert len(positive) == len(POSITIVE_CASES)
    # Every in-scope query retrieves something — none should be a false refusal at 0.
    assert all(s > 0.0 for s in positive)
    # The clearly-out-of-scope negatives have no overlap → some score exactly 0.
    assert any(s == 0.0 for s in negative)
