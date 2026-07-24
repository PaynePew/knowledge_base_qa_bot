"""Deep module per Ousterhout. Public surface: ``McNemarResult``, ``mcnemar_exact_p``,
``mcnemar_test``, ``BootstrapResult``, ``bootstrap_test``.

Paired statistical test wrappers for the corpus v3 verdict report (issue
#659, ADR-0045: "Significance means a paired test on the shared query set
(McNemar or bootstrap over per-query outcomes) at p < 0.05"). Both wrappers
take raw paired per-query values and return a result object the report can
cite directly — no LLM calls, pure standard-library math, deterministic given
a seed.

* **McNemar** (:func:`mcnemar_test`) is the paired test for a **binary**
  per-query metric (hit@1, hit@3): each query is a discordant/concordant pair
  between two arms, and the exact two-sided p-value comes from the binomial
  CDF over the discordant count.
* **Bootstrap** (:func:`bootstrap_test`) is the paired test for a **rate**
  metric that need not be binary (reciprocal rank, or any per-query score in
  [0, 1] more generally): it resamples the paired per-query differences with
  replacement and reports the resample-mean confidence interval plus a
  percentile-based two-sided p-value for "the true mean difference is 0".

This module is intentionally independent of ``eval.paraphrase_comparison
.statistics`` (which already implements ``mcnemar_exact_p`` for the v2 eval)
rather than importing it — PRD #654 specifies corpus v3 as "a new eval
package, sibling to the existing paraphrase-comparison eval ... with its own
... production isolation", and ``eval/corpus_v3/models.py`` set the same
precedent (shape mirrored, not imported) for exactly this reason.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# McNemar exact two-sided p-value (paired binary metrics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    """Paired McNemar result over one binary metric's per-query hit vectors.

    ``b`` = queries where arm A hit and arm B missed; ``c`` = queries where
    arm A missed and arm B hit. Concordant queries (both hit or both missed)
    carry no information about a *difference* and are excluded from ``b``/``c``
    by construction.
    """

    b: int
    c: int
    p_value: float


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant-pair counts ``b`` and ``c``.

    Under H0 the discordant total ``n_d = b + c`` is Binomial(n_d, 0.5)
    distributed; the exact two-sided p-value is ``2 * P(X <= min(b, c))``,
    clamped to 1.0. ``n_d == 0`` (no discordant pairs — the arms never
    disagreed) returns 1.0: no evidence of a difference.
    """
    n_d = b + c
    if n_d == 0:
        return 1.0
    smaller = min(b, c)
    cdf = sum(math.comb(n_d, k) / (2**n_d) for k in range(smaller + 1))
    return min(1.0, 2.0 * cdf)


def mcnemar_test(hits_a: Sequence[int], hits_b: Sequence[int]) -> McNemarResult:
    """Paired McNemar test wrapper: raw per-query 0/1 hit vectors in, result out.

    ``hits_a[i]`` and ``hits_b[i]`` must be the two arms' hit/miss (1/0) for
    the SAME i-th query. Raises ``ValueError`` if the vectors differ in
    length or are empty.
    """
    n = len(hits_a)
    if n == 0:
        raise ValueError("mcnemar_test needs at least one paired observation")
    if len(hits_b) != n:
        raise ValueError(f"hits_a length {n} != hits_b length {len(hits_b)}")
    b = sum(1 for a, bv in zip(hits_a, hits_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(hits_a, hits_b) if a == 0 and bv == 1)
    return McNemarResult(b=b, c=c, p_value=mcnemar_exact_p(b, c))


# ---------------------------------------------------------------------------
# Paired bootstrap (rate metrics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Paired bootstrap result for a rate metric's mean difference (A − B).

    ``ci_low`` / ``ci_high`` is the percentile confidence interval (width set
    by ``confidence``) of the resampled mean difference. ``p_value`` is the
    two-sided percentile-method p-value for the null hypothesis that the true
    mean difference is 0: twice the fraction of resample means that landed on
    the opposite side of 0 from the observed difference, clamped to 1.0.
    """

    observed_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_resamples: int


def bootstrap_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Paired bootstrap test wrapper for a rate metric (e.g. reciprocal rank).

    ``values_a[i]`` and ``values_b[i]`` are the two arms' per-query scores for
    the SAME i-th query. Resamples the ``n`` paired differences ``n_resamples``
    times (with replacement, a fixed ``seed`` for determinism) and reports the
    resample-mean confidence interval and a two-sided p-value.

    Raises ``ValueError`` if the vectors differ in length or are empty.
    """
    n = len(values_a)
    if n == 0:
        raise ValueError("bootstrap_test needs at least one paired observation")
    if len(values_b) != n:
        raise ValueError(f"values_a length {n} != values_b length {len(values_b)}")

    diffs = [a - b for a, b in zip(values_a, values_b)]
    observed_diff = sum(diffs) / n

    rng = random.Random(seed)
    resample_means = [
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    ]
    resample_means.sort()

    alpha = 1.0 - confidence
    lo_idx = max(0, min(n_resamples - 1, round((alpha / 2) * (n_resamples - 1))))
    hi_idx = max(0, min(n_resamples - 1, round((1 - alpha / 2) * (n_resamples - 1))))
    ci_low = resample_means[lo_idx]
    ci_high = resample_means[hi_idx]

    if observed_diff >= 0:
        one_sided = sum(1 for m in resample_means if m <= 0) / n_resamples
    else:
        one_sided = sum(1 for m in resample_means if m >= 0) / n_resamples
    p_value = min(1.0, 2.0 * one_sided)

    return BootstrapResult(
        observed_diff=observed_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_resamples=n_resamples,
    )
