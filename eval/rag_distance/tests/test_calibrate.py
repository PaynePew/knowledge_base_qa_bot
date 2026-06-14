"""Distance-gate calibration tests (#257 / #258 follow-up).

Pure sweep/recommend logic is tested with synthetic distances; ``collect_distances``
is a shape check that the harness builds the RAG index and returns one min-distance
per query. The autouse conftest fakes embeddings + isolates paths, so the suite is
hermetic and never spends OpenAI quota (the real separation is measured by the
manual ``calibrate.main`` run).
"""

from __future__ import annotations

from eval.negative_case.cases import NEGATIVE_CASES
from eval.negative_case.positive_cases import POSITIVE_CASES
from eval.rag_distance.calibrate import (
    CeilingPoint,
    collect_distances,
    recommend,
    sweep,
)


def test_sweep_counts_distances_above_ceiling():
    """The gate refuses when min-distance > ceiling (mirror image of the BM25 sweep)."""
    pos = [0.5, 0.5]  # close → in-scope
    neg = [1.9, 1.9, 1.0]  # far → out-of-scope
    points = sweep(pos, neg, ceilings=[0.8, 1.5])
    p1, p2 = points
    assert p1.ceiling == 0.8
    assert p1.correct_refusal_rate == 1.0  # all 3 negatives are > 0.8
    assert p1.over_refusal_rate == 0.0  # neither positive is > 0.8
    assert p2.ceiling == 1.5
    assert p2.correct_refusal_rate == 2 / 3  # 1.9, 1.9 > 1.5; 1.0 is not
    assert p2.over_refusal_rate == 0.0


def test_youden_j_is_difference():
    assert abs(CeilingPoint(1.0, 0.9, 0.1).youden_j - 0.8) < 1e-9


def test_recommend_picks_plateau_median():
    """No production incumbent (gate is new/off) → pick the optimal-plateau median."""
    points = [
        CeilingPoint(0.8, 0.6, 0.0),
        CeilingPoint(1.0, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.2, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.4, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.6, 1.0, 0.4),
    ]
    # optimal = [1.0, 1.2, 1.4]; median (index 1) → 1.2
    assert recommend(points).ceiling == 1.2


def test_collect_distances_shape():
    """collect_distances builds the index and returns one min-distance per query."""
    positive, negative = collect_distances()
    assert len(positive) == len(POSITIVE_CASES)
    assert len(negative) == len(NEGATIVE_CASES)
    assert all(d >= 0.0 for d in positive + negative)
