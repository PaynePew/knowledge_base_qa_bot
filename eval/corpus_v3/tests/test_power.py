"""Power analysis tests — external behaviour only (CODING_STANDARD §0.2).

Covers issue #660 AC-1: the power calculation is pure and its properties are
verifiable without an external stats-package cross-check (no scipy/statsmodels
dependency, per §7 "Do NOT add a requirements.txt" / prefer stdlib): a chosen
``n`` must achieve at least the target power (round-trip against the inverse
function), and required ``n`` must move in the statistically correct direction
as each input tightens.
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.power import (
    PowerInputs,
    achieved_power_paired_proportions,
    discordant_proportion_under_independence,
    per_stratum_requirements,
    required_n_paired_proportions,
)


def _inputs(**overrides) -> PowerInputs:
    fields = dict(alpha=0.05, power=0.80, mdd=0.05, p_baseline=0.88)
    fields.update(overrides)
    return PowerInputs(**fields)


# ---------------------------------------------------------------------------
# discordant_proportion_under_independence
# ---------------------------------------------------------------------------
def test_discordant_proportion_is_symmetric_under_independence():
    # p01 = p*(1-(p-d)), p10 = (p-d)*(1-p) — swapping which arm is "baseline"
    # (p vs p-d) changes which term is p01 vs p10 but not their sum.
    psi_from_high = discordant_proportion_under_independence(0.8, 0.1)
    psi_from_low = discordant_proportion_under_independence(0.7, -0.1)
    assert psi_from_high == pytest.approx(psi_from_low)


def test_discordant_proportion_is_zero_at_the_extremes():
    # p_baseline=1.0, mdd=0 -> both arms certain -> no discordant pairs possible.
    assert discordant_proportion_under_independence(1.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# required_n_paired_proportions — validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field, value",
    [
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("power", 0.0),
        ("power", 1.0),
        ("mdd", 0.0),
        ("mdd", 1.0),
        ("p_baseline", -0.1),
        ("p_baseline", 1.1),
    ],
)
def test_power_inputs_rejects_out_of_range_field(field, value):
    with pytest.raises(ValueError, match=field):
        _inputs(**{field: value})


def test_power_inputs_rejects_mdd_that_pushes_other_arm_out_of_range():
    with pytest.raises(ValueError, match="p_baseline - mdd"):
        _inputs(p_baseline=0.02, mdd=0.05)


# ---------------------------------------------------------------------------
# required_n_paired_proportions — monotonicity (the statistically-correct
# direction of change is the property a pure closed-form calc must satisfy)
# ---------------------------------------------------------------------------
def test_smaller_mdd_requires_more_queries():
    n_loose = required_n_paired_proportions(_inputs(mdd=0.10)).required_n
    n_tight = required_n_paired_proportions(_inputs(mdd=0.05)).required_n
    assert n_tight > n_loose


def test_higher_power_requires_more_queries():
    n_lo = required_n_paired_proportions(_inputs(power=0.70)).required_n
    n_hi = required_n_paired_proportions(_inputs(power=0.95)).required_n
    assert n_hi > n_lo


def test_tighter_alpha_requires_more_queries():
    n_loose = required_n_paired_proportions(_inputs(alpha=0.10)).required_n
    n_tight = required_n_paired_proportions(_inputs(alpha=0.01)).required_n
    assert n_tight > n_loose


def test_required_n_is_a_positive_integer():
    result = required_n_paired_proportions(_inputs())
    assert isinstance(result.required_n, int)
    assert result.required_n > 0


def test_discordant_proportion_always_exceeds_mdd_squared_for_valid_inputs():
    # psi - mdd**2 == a*(1-a) + b*(1-b) (a=p_baseline, b=p_baseline-mdd), which
    # is strictly positive over the whole PowerInputs-validated domain (see
    # required_n_paired_proportions docstring) — checked at a boundary-ish
    # value (p_baseline=1.0) where the margin is smallest.
    inputs = _inputs(p_baseline=1.0, mdd=0.05)
    psi = discordant_proportion_under_independence(inputs.p_baseline, inputs.mdd)
    assert psi > inputs.mdd**2


# ---------------------------------------------------------------------------
# achieved_power_paired_proportions — inverse round-trip
# ---------------------------------------------------------------------------
def test_achieved_power_at_required_n_meets_or_exceeds_target():
    inputs = _inputs()
    result = required_n_paired_proportions(inputs)
    achieved = achieved_power_paired_proportions(result.required_n, inputs)
    assert achieved >= inputs.power - 1e-9


def test_achieved_power_at_one_fewer_query_is_lower():
    inputs = _inputs()
    result = required_n_paired_proportions(inputs)
    achieved_at_n = achieved_power_paired_proportions(result.required_n, inputs)
    achieved_below = achieved_power_paired_proportions(result.required_n - 1, inputs)
    assert achieved_below < achieved_at_n


def test_achieved_power_rejects_non_positive_n():
    with pytest.raises(ValueError, match="n must be positive"):
        achieved_power_paired_proportions(0, _inputs())


# ---------------------------------------------------------------------------
# per_stratum_requirements
# ---------------------------------------------------------------------------
def test_per_stratum_requirements_applies_base_inputs_to_every_stratum():
    strata = ["factoid", "cross_doc", "version_conflict", "unanswerable"]
    results = per_stratum_requirements(_inputs(), strata)
    assert set(results) == set(strata)
    ns = {result.required_n for result in results.values()}
    assert len(ns) == 1  # identical inputs -> identical n


def test_per_stratum_requirements_honours_per_stratum_override():
    strata = ["factoid", "zh_factoid"]
    zh_inputs = _inputs(power=0.60, mdd=0.10)  # relaxed zh gate
    results = per_stratum_requirements(
        _inputs(), strata, overrides={"zh_factoid": zh_inputs}
    )
    assert results["zh_factoid"].inputs == zh_inputs
    assert results["zh_factoid"].required_n < results["factoid"].required_n
