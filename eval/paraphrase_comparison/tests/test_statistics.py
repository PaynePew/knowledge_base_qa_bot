"""Pure-function unit tests for McNemar, Wilson CI, and Holm correction.

All tests use hand-checked contingency inputs and do NOT call any LLM or
require an API key.  The statistics are a thin layer over the binomial CDF
(standard library ``math``), so every expected value below can be verified
by hand-calculation.
"""

from __future__ import annotations

import pytest

from eval.paraphrase_comparison.statistics import (
    TypeStatResult,
    compute_type_stats,
    holm_correct,
    mcnemar_exact_p,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# mcnemar_exact_p
# ---------------------------------------------------------------------------


def test_mcnemar_no_discordant_pairs_gives_one():
    """b=0, c=0 → no evidence of difference; p must be 1.0."""
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_symmetric_gives_one():
    """b == c → the two stacks are equivalent; p must be 1.0."""
    assert mcnemar_exact_p(5, 5) == 1.0


def test_mcnemar_hand_checked_b3_c9():
    """b=3, c=9: hand-computed two-sided exact p ≈ 0.1460.

    Under H0, X ~ Binomial(12, 0.5).
    P(X ≤ 3) = sum_{k=0}^{3} C(12,k)/2^12 = (1+12+66+220)/4096 = 299/4096
    Two-sided p = 2 * 299/4096 = 598/4096 ≈ 0.14600585...
    """
    expected = 2 * 299 / 4096  # ≈ 0.14600585...
    assert abs(mcnemar_exact_p(3, 9) - expected) < 1e-9


def test_mcnemar_hand_checked_b1_c9():
    """b=1, c=9: hand-computed two-sided exact p ≈ 0.02148.

    Under H0, X ~ Binomial(10, 0.5).
    P(X ≤ 1) = (1 + 10)/1024 = 11/1024
    Two-sided p = 2 * 11/1024 = 22/1024 ≈ 0.021484375
    """
    expected = 22 / 1024  # ≈ 0.021484375
    assert abs(mcnemar_exact_p(1, 9) - expected) < 1e-9


def test_mcnemar_p_value_is_clamped_to_one():
    """p-value must never exceed 1.0 even when computed sum exceeds 1 due to overlap."""
    p = mcnemar_exact_p(5, 5)
    assert 0.0 <= p <= 1.0


def test_mcnemar_result_is_always_between_zero_and_one():
    for b, c in [(0, 0), (0, 1), (1, 0), (2, 8), (10, 0), (0, 10), (7, 7)]:
        p = mcnemar_exact_p(b, c)
        assert 0.0 <= p <= 1.0, f"p out of range for b={b}, c={c}"


def test_mcnemar_is_symmetric_in_b_and_c():
    """McNemar is symmetric: p(b,c) == p(c,b)."""
    assert mcnemar_exact_p(2, 8) == pytest.approx(mcnemar_exact_p(8, 2))
    assert mcnemar_exact_p(1, 9) == pytest.approx(mcnemar_exact_p(9, 1))


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


def test_wilson_ci_all_hits_high_lower_bound():
    """k=10, n=10: lower bound must be well above 0.5."""
    lo, hi = wilson_ci(10, 10)
    assert lo > 0.69  # Wilson shrinks toward centre; exactly 1.0 is impossible
    assert hi == pytest.approx(1.0)


def test_wilson_ci_zero_hits_low_upper_bound():
    """k=0, n=10: upper bound must be well below 0.5."""
    lo, hi = wilson_ci(0, 10)
    assert lo == pytest.approx(0.0)
    assert hi < 0.31


def test_wilson_ci_hand_checked_k7_n10():
    """k=7, n=10, z=1.96: hand-verified centre ≈ 0.6444, half-width ≈ 0.2477.

    p̂=0.7, z²=3.8416
    center = (0.7 + 1.9208/10) / 1.38416 ≈ 0.6444
    half_width = 1.96 * sqrt(0.021 + 0.009604) / 1.38416 ≈ 0.2477
    lo ≈ 0.3967, hi ≈ 0.8921
    """
    lo, hi = wilson_ci(7, 10)
    assert abs(lo - 0.3967) < 0.002
    assert abs(hi - 0.8921) < 0.002


def test_wilson_ci_bounds_are_in_zero_one():
    for k, n in [(0, 1), (1, 1), (5, 10), (0, 50), (50, 50)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0, f"CI out of [0,1] for k={k}, n={n}"


def test_wilson_ci_interval_contains_sample_proportion():
    """The Wilson CI must bracket p̂ for reasonable inputs."""
    for k, n in [(3, 10), (7, 10), (1, 5), (4, 5)]:
        lo, hi = wilson_ci(k, n)
        p_hat = k / n
        assert lo <= p_hat <= hi, f"CI does not contain p̂={p_hat} for k={k}, n={n}"


def test_wilson_ci_n_zero_raises_value_error():
    """n=0 is undefined; raise ValueError."""
    with pytest.raises(ValueError):
        wilson_ci(0, 0)


# ---------------------------------------------------------------------------
# holm_correct
# ---------------------------------------------------------------------------


def test_holm_empty_list_returns_empty():
    assert holm_correct([]) == []


def test_holm_single_p_value_unchanged():
    assert holm_correct([0.03]) == pytest.approx([0.03])


def test_holm_hand_checked_five_p_values():
    """Hand-verified for [0.02, 0.04, 0.06, 0.10, 0.15] (sorted input).

    Holm step-down (step-up cummax):
      rank 1 (0.02): adj = min(1, 0.02*5)       = 0.10
      rank 2 (0.04): adj = min(1, max(0.04*4, 0.10)) = 0.16
      rank 3 (0.06): adj = min(1, max(0.06*3, 0.16)) = 0.18
      rank 4 (0.10): adj = min(1, max(0.10*2, 0.18)) = 0.20
      rank 5 (0.15): adj = min(1, max(0.15*1, 0.20)) = 0.20
    Output order follows input order.
    """
    raw = [0.02, 0.04, 0.06, 0.10, 0.15]
    corrected = holm_correct(raw)
    assert len(corrected) == 5
    assert corrected == pytest.approx([0.10, 0.16, 0.18, 0.20, 0.20], abs=1e-9)


def test_holm_unsorted_input_preserves_position():
    """Input order is preserved: the returned list aligns position-for-position."""
    # Same 5 values as above but shuffled; expected output aligns with input positions.
    raw = [0.15, 0.04, 0.10, 0.02, 0.06]
    expected_map = {0.02: 0.10, 0.04: 0.16, 0.06: 0.18, 0.10: 0.20, 0.15: 0.20}
    corrected = holm_correct(raw)
    for raw_p, adj_p in zip(raw, corrected):
        assert adj_p == pytest.approx(expected_map[raw_p], abs=1e-9)


def test_holm_all_significant_low_p_values():
    """Very low p-values should all remain below alpha=0.05 after correction."""
    raw = [0.001, 0.002, 0.003, 0.004, 0.005]
    corrected = holm_correct(raw)
    assert all(p < 0.05 for p in corrected)


def test_holm_capped_at_one():
    """Corrected p-values must never exceed 1.0."""
    raw = [0.40, 0.50, 0.60, 0.70, 0.80]
    corrected = holm_correct(raw)
    assert all(p <= 1.0 for p in corrected)


# ---------------------------------------------------------------------------
# compute_type_stats (integration of the three primitives)
# ---------------------------------------------------------------------------


def test_compute_type_stats_returns_dataclass():
    """compute_type_stats returns a TypeStatResult for each type."""
    # hits_a[i]=1 means Stack A hit on paraphrase i; hits_b similarly.
    hits_a = [1, 1, 0, 0, 1, 1, 0, 1]
    hits_b = [1, 0, 1, 0, 1, 0, 1, 1]
    result = compute_type_stats(hits_a, hits_b)
    assert isinstance(result, TypeStatResult)


def test_compute_type_stats_p_value_in_range():
    hits_a = [1, 1, 0, 0, 1, 1, 0, 1]
    hits_b = [1, 0, 1, 0, 1, 0, 1, 1]
    result = compute_type_stats(hits_a, hits_b)
    assert 0.0 <= result.mcnemar_p <= 1.0


def test_compute_type_stats_wilson_ci_brackets_hit_rate():
    hits_a = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]  # k=5, n=10
    hits_b = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0]  # k=4, n=10
    result = compute_type_stats(hits_a, hits_b)
    rate_a = sum(hits_a) / len(hits_a)
    rate_b = sum(hits_b) / len(hits_b)
    lo_a, hi_a = result.ci_a
    lo_b, hi_b = result.ci_b
    assert lo_a <= rate_a <= hi_a
    assert lo_b <= rate_b <= hi_b


def test_compute_type_stats_all_concordant_no_difference():
    """If both stacks always agree (all hit or all miss), p must be 1.0."""
    hits_a = [1, 1, 1, 0, 0]
    hits_b = [1, 1, 1, 0, 0]  # identical outcomes
    result = compute_type_stats(hits_a, hits_b)
    assert result.mcnemar_p == 1.0
    assert result.b == 0 and result.c == 0


def test_compute_type_stats_discordant_pairs_match_hand_count():
    """b = #(A-hit, B-miss), c = #(A-miss, B-hit) must be counted correctly."""
    # A: 1 0 1 0   B: 0 1 0 1   → b=2 (positions 0,2), c=2 (positions 1,3)
    hits_a = [1, 0, 1, 0]
    hits_b = [0, 1, 0, 1]
    result = compute_type_stats(hits_a, hits_b)
    assert result.b == 2
    assert result.c == 2


def test_compute_type_stats_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        compute_type_stats([1, 0], [1])


def test_compute_type_stats_raises_on_empty():
    with pytest.raises(ValueError):
        compute_type_stats([], [])
