"""Weight-calibration tests (#578).

Pure ``recommend`` logic is tested with synthetic points (mirrors
``eval/negative_case/tests/test_calibrate.py``'s style for the #253
precedent); the synthetic-corpus builder and end-to-end ``sweep`` get one
integration check each. LLM-free; the autouse conftest restores
``markdown_kb.app.indexer`` module state after every test.
"""

from __future__ import annotations

from eval.qa_field_weight.calibrate import (
    WeightPoint,
    build_synthetic_corpus,
    recommend,
    sweep,
)


def test_weight_point_separation_is_the_difference():
    point = WeightPoint(weight=0.5, own_question_hit_rate=0.9, pollution_rate=0.2)
    assert abs(point.separation - 0.7) < 1e-9


def test_recommend_picks_plateau_median_among_floor_holding_points():
    """Weights that drop the own-question floor are disqualified outright."""
    points = [
        WeightPoint(
            weight=0.0, own_question_hit_rate=0.0, pollution_rate=0.0
        ),  # floor broken
        WeightPoint(
            weight=0.2, own_question_hit_rate=1.0, pollution_rate=0.1
        ),  # optimal plateau
        WeightPoint(
            weight=0.5, own_question_hit_rate=1.0, pollution_rate=0.1
        ),  # optimal plateau
        WeightPoint(
            weight=1.0, own_question_hit_rate=1.0, pollution_rate=0.1
        ),  # optimal plateau
    ]
    # optimal (floor-holding, best separation) = [0.2, 0.5, 1.0]; median → 0.5.
    assert recommend(points).weight == 0.5


def test_recommend_falls_back_when_no_weight_holds_the_floor():
    """If nothing holds the floor, fall back to the safest option instead of a broken pick."""
    points = [
        WeightPoint(weight=0.0, own_question_hit_rate=0.5, pollution_rate=0.0),
        WeightPoint(weight=1.0, own_question_hit_rate=0.5, pollution_rate=0.4),
    ]
    # Both tie at the max hit rate (0.5); among those, 0.0 has the better
    # (higher) separation (0.5 vs 0.1) so it wins even though it is "unsafe" —
    # there is no safe option to prefer here.
    assert recommend(points).weight == 0.0


def test_build_synthetic_corpus_shape():
    sections, real_file, own_question_id = build_synthetic_corpus(noise_count=5)
    # real + distractor + 5 noise + own-question page + filler.
    assert len(sections) == 9
    assert real_file == "配送地區"
    files = {s.file for s in sections}
    assert real_file in files
    assert any(s.id == own_question_id for s in sections)
    noise_files = [s.file for s in sections if s.file.startswith("qa-noise-zh-")]
    assert len(noise_files) == 5


def test_sweep_own_question_floor_breaks_only_at_zero():
    """Regression guard: the #570 own-question invariant survives every weight > 0."""
    sections, real_file, own_question_id = build_synthetic_corpus(noise_count=3)
    import markdown_kb.app.indexer as mk_indexer

    mk_indexer.sections = sections
    mk_indexer.rebuild_stats()

    points = sweep(real_file, own_question_id, weights=(1.0, 0.3, 0.05, 0.0))
    by_weight = {p.weight: p for p in points}
    assert by_weight[1.0].own_question_hit_rate == 1.0
    assert by_weight[0.3].own_question_hit_rate == 1.0
    assert by_weight[0.05].own_question_hit_rate == 1.0
    assert by_weight[0.0].own_question_hit_rate == 0.0
