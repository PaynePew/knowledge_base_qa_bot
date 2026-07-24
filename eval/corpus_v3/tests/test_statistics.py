"""Paired statistical test wrapper tests — external behaviour only
(CODING_STANDARD §0.2).

McNemar's exact p-value is a thin layer over the binomial CDF, so every
hand-checked expected value below is verifiable by hand. The bootstrap
wrapper is stochastic in general, but the fixtures here use constant paired
differences, which collapse EVERY resample to the same mean regardless of
seed or draw — a genuinely "known answer", not just a reproducibility check.
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.statistics import (
    BootstrapResult,
    McNemarResult,
    bootstrap_test,
    mcnemar_exact_p,
    mcnemar_test,
)

# ---------------------------------------------------------------------------
# mcnemar_exact_p
# ---------------------------------------------------------------------------


def test_mcnemar_no_discordant_pairs_gives_one():
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_symmetric_gives_one():
    assert mcnemar_exact_p(5, 5) == 1.0


def test_mcnemar_hand_checked_b3_c9():
    """b=3, c=9: X ~ Binomial(12, 0.5). P(X<=3) = (1+12+66+220)/4096 = 299/4096.
    Two-sided p = 2 * 299/4096 ~= 0.14600585."""
    expected = 2 * 299 / 4096
    assert abs(mcnemar_exact_p(3, 9) - expected) < 1e-9


def test_mcnemar_hand_checked_b1_c9():
    """b=1, c=9: X ~ Binomial(10, 0.5). P(X<=1) = 11/1024. Two-sided p = 22/1024."""
    expected = 22 / 1024
    assert abs(mcnemar_exact_p(1, 9) - expected) < 1e-9


def test_mcnemar_is_symmetric_in_b_and_c():
    assert mcnemar_exact_p(2, 8) == pytest.approx(mcnemar_exact_p(8, 2))


@pytest.mark.parametrize("b,c", [(0, 0), (0, 1), (1, 0), (2, 8), (10, 0), (7, 7)])
def test_mcnemar_result_always_in_unit_interval(b, c):
    assert 0.0 <= mcnemar_exact_p(b, c) <= 1.0


# ---------------------------------------------------------------------------
# mcnemar_test wrapper
# ---------------------------------------------------------------------------


def test_mcnemar_test_computes_b_c_and_p_from_raw_hit_vectors():
    # 3 queries where A hit / B missed (b), 9 where A missed / B hit (c),
    # padded with concordant pairs that must NOT affect b/c.
    hits_a = [1, 1, 1] + [0] * 9 + [1, 1]
    hits_b = [0, 0, 0] + [1] * 9 + [1, 1]
    result = mcnemar_test(hits_a, hits_b)
    assert isinstance(result, McNemarResult)
    assert (result.b, result.c) == (3, 9)
    assert result.p_value == pytest.approx(2 * 299 / 4096)


def test_mcnemar_test_identical_vectors_gives_p_one():
    hits = [1, 0, 1, 1, 0]
    result = mcnemar_test(hits, hits)
    assert (result.b, result.c) == (0, 0)
    assert result.p_value == 1.0


def test_mcnemar_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        mcnemar_test([1, 0], [1, 0, 1])


def test_mcnemar_test_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        mcnemar_test([], [])


# ---------------------------------------------------------------------------
# bootstrap_test wrapper — known-answer fixtures (constant paired diffs)
# ---------------------------------------------------------------------------


def test_bootstrap_test_constant_zero_difference_is_a_known_answer():
    """Every paired diff is 0 -> every resample mean is 0, regardless of seed."""
    values_a = [0.5, 0.5, 0.5, 0.5]
    values_b = [0.5, 0.5, 0.5, 0.5]
    result = bootstrap_test(values_a, values_b, n_resamples=500, seed=42)
    assert isinstance(result, BootstrapResult)
    assert result.observed_diff == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.p_value == 1.0


def test_bootstrap_test_constant_positive_difference_is_a_known_answer():
    """Every paired diff is exactly 1.0 -> every resample mean is 1.0."""
    values_a = [1.0, 1.0, 1.0]
    values_b = [0.0, 0.0, 0.0]
    result = bootstrap_test(values_a, values_b, n_resamples=500, seed=7)
    assert result.observed_diff == 1.0
    assert result.ci_low == pytest.approx(1.0)
    assert result.ci_high == pytest.approx(1.0)
    assert result.p_value == 0.0


def test_bootstrap_test_constant_negative_difference_is_a_known_answer():
    values_a = [0.0, 0.0, 0.0]
    values_b = [1.0, 1.0, 1.0]
    result = bootstrap_test(values_a, values_b, n_resamples=500, seed=7)
    assert result.observed_diff == -1.0
    assert result.ci_low == pytest.approx(-1.0)
    assert result.ci_high == pytest.approx(-1.0)
    assert result.p_value == 0.0


def test_bootstrap_test_is_deterministic_given_a_seed():
    values_a = [1.0, 0.0, 1.0, 0.0, 1.0]
    values_b = [0.0, 0.0, 1.0, 1.0, 0.0]
    first = bootstrap_test(values_a, values_b, n_resamples=200, seed=123)
    second = bootstrap_test(values_a, values_b, n_resamples=200, seed=123)
    assert first == second


def test_bootstrap_test_different_seeds_can_disagree_but_stay_in_range():
    values_a = [1.0, 0.0, 1.0, 0.0, 1.0]
    values_b = [0.0, 0.0, 1.0, 1.0, 0.0]
    result = bootstrap_test(values_a, values_b, n_resamples=1000, seed=1)
    assert -1.0 <= result.ci_low <= result.ci_high <= 1.0
    assert 0.0 <= result.p_value <= 1.0


def test_bootstrap_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        bootstrap_test([1.0, 0.0], [1.0])


def test_bootstrap_test_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_test([], [])
