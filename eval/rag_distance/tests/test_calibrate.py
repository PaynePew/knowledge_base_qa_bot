"""Distance-gate calibration tests (#257 / #258, spec parity #656).

Pure sweep/recommend logic is tested with synthetic distances; ``collect_distances``
is a shape check that the harness builds the RAG index and returns one min-distance
per query, exercised for both the English and zh (#656) negative-case specs. The
autouse conftest fakes embeddings + isolates paths, so the suite is hermetic and
never spends OpenAI quota (the real separation is measured by the manual
``calibrate.main`` run).
"""

from __future__ import annotations

import pytest

from eval.negative_case.cases import NEGATIVE_CASES
from eval.negative_case.lang import resolve_lang
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


def test_recommend_picks_plateau_median_when_default_not_on_plateau():
    """Current default (1.1) is not on this plateau → pick the optimal-plateau median."""
    points = [
        CeilingPoint(0.8, 0.6, 0.0),
        CeilingPoint(1.0, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.2, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.4, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.6, 1.0, 0.4),
    ]
    # optimal = [1.0, 1.2, 1.4]; median (index 1) → 1.2
    assert recommend(points).ceiling == 1.2


def test_recommend_keeps_current_default_when_optimal():
    """#656 parity fix (mirrors ``eval.negative_case.calibrate.recommend``): when the
    shipped 1.1 default is itself J-optimal, keep it (no churn) rather than the
    plateau median, even if the median would land elsewhere.
    """
    points = [
        CeilingPoint(1.0, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.1, 1.0, 0.0),  # optimal plateau — current default
        CeilingPoint(1.2, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.3, 1.0, 0.0),  # optimal plateau
        CeilingPoint(1.6, 1.0, 0.4),
    ]
    # plain plateau median (index 2) would be 1.2; the default-preference picks 1.1.
    assert recommend(points).ceiling == 1.1


def test_collect_distances_shape():
    """collect_distances builds the index and returns one min-distance per query."""
    positive, negative = collect_distances()
    assert len(positive) == len(POSITIVE_CASES)
    assert len(negative) == len(NEGATIVE_CASES)
    assert all(d >= 0.0 for d in positive + negative)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_collect_distances_shape_by_lang(lang):
    """#656: the same KB_EVAL_LANG selector Stack A uses also drives Stack B's sweep.

    Confirms the zh negative-case spec (corpus_zh + cases_zh) plumbs through
    ``collect_distances`` hermetically. The real zh separation still needs a live
    embeddings run (calibrate.main); this only checks the shape.
    """
    cfg = resolve_lang(lang)
    positive, negative = collect_distances(
        cfg.corpus_dir, cfg.positive_cases, cfg.negative_cases
    )
    assert len(positive) == len(cfg.positive_cases)
    assert len(negative) == len(cfg.negative_cases)
    assert all(d >= 0.0 for d in positive + negative)
