"""Deep module per Ousterhout. Public surface: ``PowerInputs``,
``PowerAnalysisResult``, ``discordant_proportion_under_independence``,
``required_n_paired_proportions``, ``achieved_power_paired_proportions``,
``per_stratum_requirements``.

Prospective paired-proportions power analysis for the corpus v3 query-set
size (issue #660, ADR-0045 Prerequisite 4: "n derived from a prospective
power analysis so the kill threshold exceeds the minimal detectable
difference"). The v2 eval's n=260 could not detect its own observed A-B gap
(minimal detectable difference ~6-7 hit@3 points vs the observed 5.6,
`eval/fairness_review/literature.md` §2) — "not significant" meant
"underpowered", not "no difference". This module makes the query-set size a
calculation, not a guess, for every paired binary content-axis metric the
verdict report tests with McNemar (``eval.corpus_v3.statistics.mcnemar_test``).

Sample size follows the standard closed-form for a paired-proportions
(McNemar) test (Connor 1987 — the same normal-approximation family Sakai's
topic-set-size design uses for IR test-collection sizing): given the two
arms' discordant-pair proportion ``psi`` and the marginal difference ``d`` to
detect (the MDD), the required number of paired observations is

    n = (z_(alpha/2) * sqrt(psi) + z_power * sqrt(psi - d^2))^2 / d^2

``psi`` is not directly observable before any data exists, so
:func:`discordant_proportion_under_independence` estimates it from a single
baseline proportion under an independence assumption between the two arms'
per-query outcomes. Independence is the conservative choice here: any
positive correlation between the two arms (which paired same-query
comparisons typically have) would only shrink ``psi`` and therefore the
required ``n``, so this module never *under*-estimates the query set needed
— consistent with ADR-0045's "burden of proof is on the wiki" stance.

Uses ``statistics.NormalDist`` (stdlib) for the z-values — no new dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()


def _z(p: float) -> float:
    """The z-value (inverse standard-normal CDF) at cumulative probability ``p``."""
    return _NORMAL.inv_cdf(p)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerInputs:
    """The four design inputs a prospective power analysis must fix in advance.

    ``alpha`` is the two-sided significance level (ADR-0045: p < 0.05, so
    ``alpha=0.05``). ``power`` is the target probability of detecting a real
    effect of size ``mdd`` (minimal detectable difference, a proportion in
    [0, 1], e.g. ``0.05`` for 5 percentage points). ``p_baseline`` anchors the
    discordant-proportion estimate — one arm's expected proportion on the
    metric under test (e.g. the wiki arm's v2 grounding-pass-rate analogue);
    the other arm is implicitly ``p_baseline - mdd``.
    """

    alpha: float
    power: float
    mdd: float
    p_baseline: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        if not (0.0 < self.power < 1.0):
            raise ValueError(f"power must be in (0, 1), got {self.power!r}")
        if not (0.0 < self.mdd < 1.0):
            raise ValueError(f"mdd must be in (0, 1), got {self.mdd!r}")
        if not (0.0 <= self.p_baseline <= 1.0):
            raise ValueError(f"p_baseline must be in [0, 1], got {self.p_baseline!r}")
        other = self.p_baseline - self.mdd
        if not (0.0 <= other <= 1.0):
            raise ValueError(
                f"p_baseline - mdd ({other!r}) must stay within [0, 1]; "
                f"p_baseline={self.p_baseline!r}, mdd={self.mdd!r}"
            )


@dataclass(frozen=True)
class PowerAnalysisResult:
    """The output of one :func:`required_n_paired_proportions` call.

    ``required_n`` is the number of paired query observations needed — the
    per-stratum query count when ``inputs`` describes that stratum's test.
    """

    inputs: PowerInputs
    discordant_proportion: float
    required_n: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def discordant_proportion_under_independence(p_baseline: float, mdd: float) -> float:
    """Estimate ``psi`` (the discordant-pair proportion) from one baseline proportion.

    Assumes the two arms' per-query binary outcomes are independent draws
    with marginal probabilities ``p_baseline`` and ``p_baseline - mdd``. Under
    independence: ``p01 = p_baseline * (1 - p_other)`` (baseline hits, other
    misses) and ``p10 = p_other * (1 - p_baseline)`` (baseline misses, other
    hits); ``psi = p01 + p10``. See module docstring for why independence is
    the conservative (never-under-estimates-n) choice.
    """
    p_other = p_baseline - mdd
    p01 = p_baseline * (1.0 - p_other)
    p10 = p_other * (1.0 - p_baseline)
    return p01 + p10


def required_n_paired_proportions(inputs: PowerInputs) -> PowerAnalysisResult:
    """The paired query-observation count needed to detect ``inputs.mdd`` (Connor 1987).

    ``psi > mdd**2`` always holds for any ``inputs`` that passed
    ``PowerInputs`` validation: writing ``a = p_baseline``, ``b = p_baseline -
    mdd``, algebra gives ``psi - mdd**2 = a*(1-a) + b*(1-b)``, which is
    strictly positive for any ``a, b`` in ``[0, 1]`` except the impossible
    combination excluded by ``mdd`` being open-interval-bounded in ``(0, 1)``.
    So the square root below never hits a negative radicand.
    """
    d = inputs.mdd
    psi = discordant_proportion_under_independence(inputs.p_baseline, d)
    z_alpha = _z(1.0 - inputs.alpha / 2.0)
    z_power = _z(inputs.power)
    n_raw = (z_alpha * math.sqrt(psi) + z_power * math.sqrt(psi - d * d)) ** 2 / (d * d)
    return PowerAnalysisResult(
        inputs=inputs,
        discordant_proportion=psi,
        required_n=math.ceil(n_raw),
    )


def achieved_power_paired_proportions(n: int, inputs: PowerInputs) -> float:
    """The achieved power at sample size ``n`` for ``inputs`` (inverse of the sizing formula).

    A sanity-check companion to :func:`required_n_paired_proportions`: solves
    the same closed-form for ``power`` given a fixed ``n`` instead of solving
    for ``n`` given a fixed ``power``. Used to confirm a chosen (rounded, or
    resource-constrained) ``n`` still meets or exceeds the target power.
    Raises ``ValueError`` if ``n <= 0``. See :func:`required_n_paired_proportions`
    for why ``psi > mdd**2`` always holds for validated ``inputs``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}")
    d = inputs.mdd
    psi = discordant_proportion_under_independence(inputs.p_baseline, d)
    z_alpha = _z(1.0 - inputs.alpha / 2.0)
    z_power = (math.sqrt(n) * d - z_alpha * math.sqrt(psi)) / math.sqrt(psi - d * d)
    return _NORMAL.cdf(z_power)


def per_stratum_requirements(
    base_inputs: PowerInputs,
    strata: list[str],
    overrides: dict[str, PowerInputs] | None = None,
) -> dict[str, PowerAnalysisResult]:
    """The required ``n`` for each of ``strata``, each stratum's own paired test.

    Every stratum in ``strata`` gets ``base_inputs`` unless it has an entry in
    ``overrides`` (e.g. a zh slice sized to a relaxed ``power`` / ``mdd`` —
    ADR-0045 Prerequisite 4's "zh query slice with its own gates", PRD #654
    user story 7). Each scenario/language stratum is aggregated and tested
    independently (``eval.corpus_v3.aggregation``'s per-stratum-first design),
    so each needs its own adequately powered ``n`` rather than a slice of one
    pooled total.
    """
    overrides = overrides or {}
    return {
        stratum: required_n_paired_proportions(overrides.get(stratum, base_inputs))
        for stratum in strata
    }
