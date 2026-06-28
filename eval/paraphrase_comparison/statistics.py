"""Deep module per Ousterhout. Public surface: ``mcnemar_exact_p``, ``wilson_ci``,
``holm_correct``, ``cochran_q``, ``CochranQResult``, ``TypeStatResult``,
``compute_type_stats``.

Pure statistical functions for the Phase 8.5 McNemar / Wilson CI / Holm report
upgrade (issue #140, PRD #137) and the Phase 13 three-arm Cochran's Q omnibus
(issue #316, PRD #309). All functions are deterministic, require no LLM, and
carry zero external dependencies beyond the Python standard library ``math``.
This makes them unit-testable offline against hand-checked contingency inputs.

Statistical background
----------------------
* **McNemar exact test** — a paired binary comparison for two classifiers on the
  same items.  For a per-type contingency table (b = A-hit B-miss, c = A-miss
  B-hit), the exact two-sided p-value is 2 * CDF_Binomial(min(b,c); b+c, 0.5),
  clamped to 1.0.  When b+c == 0 there are no discordant pairs; p = 1.0.

* **Wilson score interval** — the frequentist 95% confidence interval for a
  proportion that stays within [0,1] even at the extremes (unlike Wald).  With
  z = 1.96, n observations and k successes:
    center    = (p̂ + z²/2n) / (1 + z²/n)
    half_width = z * sqrt(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)

* **Holm correction** — a step-down multiple-comparison correction more powerful
  than Bonferroni.  Adjusted p-values are computed with a step-up cumulative
  maximum so the output aligns with the *input* order (unsorted).

* **Cochran's Q** — the omnibus test for ``k`` (≥ 2) related binary samples
  measured on the same blocks (here: the same Paraphrases retrieved by each of
  the three arms).  It generalises McNemar to 3+ paired classifiers and asks
  whether the arms share one common success proportion.  With column totals
  ``C_j`` (successes per arm), row totals ``R_i`` (arms that hit on block i) and
  grand total ``N = Σ C_j = Σ R_i``::

      Q = (k-1) * (k * Σ C_j² − N²) / (k * N − Σ R_i²)

  Q is distributed χ² with ``k-1`` degrees of freedom; the upper-tail p-value is
  ``P(χ²_{k-1} > Q)``.  When the denominator collapses to 0 (every block is all
  hit or all miss — no discordant evidence) Q is reported 0.0 with p = 1.0, the
  same convention McNemar uses for b+c == 0.  A significant Q is the gate for the
  post-hoc pairwise McNemar comparisons (composed in the runner, Holm-corrected).
  The χ² survival function is computed from the regularised upper incomplete
  gamma function in pure ``math`` (no scipy), so the omnibus stays as offline and
  hand-verifiable as the other primitives — for df=2 it reduces to the closed
  form ``exp(-Q/2)``, which the unit tests pin.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# McNemar exact two-sided p-value
# ---------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant-pair counts ``b`` and ``c``.

    ``b`` = number of paraphrases where Stack A hit and Stack B missed;
    ``c`` = number where Stack A missed and Stack B hit.  Under H0 the total
    discordant count n_d = b + c comes from Binomial(n_d, 0.5).  The exact
    two-sided p-value is 2 * P(X ≤ min(b, c)) clamped to [0, 1].
    """
    n_d = b + c
    if n_d == 0:
        return 1.0
    smaller = min(b, c)
    # CDF at the smaller tail
    cdf = sum(math.comb(n_d, k) / (2**n_d) for k in range(smaller + 1))
    return min(1.0, 2.0 * cdf)


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    ``k`` successes out of ``n`` trials.  ``z`` is the standard-normal quantile
    (default 1.96 for 95%).  Raises ``ValueError`` if n == 0.

    Returns ``(lower, upper)`` both clamped to [0, 1].
    """
    if n == 0:
        raise ValueError("n must be > 0")
    p_hat = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denominator
    half_width = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n)) / denominator
    )
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return lo, hi


# ---------------------------------------------------------------------------
# Holm–Bonferroni correction
# ---------------------------------------------------------------------------


def holm_correct(p_values: list[float]) -> list[float]:
    """Return Holm-corrected p-values in the same order as the input.

    Each corrected p̃ = min(1, max(p_i * (m − rank + 1), p̃_{prev})) where
    rank is the ascending rank of the raw p-value (1 = smallest).  The running
    maximum enforces step-up monotonicity so the corrected sequence is
    non-decreasing in rank order; values are clamped to 1.0.
    """
    m = len(p_values)
    if m == 0:
        return []
    # Argsort: indices sorted by ascending p-value.
    order = sorted(range(m), key=lambda i: p_values[i])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):  # rank 0 = smallest
        factor = m - rank  # Holm factor: m, m-1, ..., 1
        adj = min(1.0, max(p_values[idx] * factor, running_max))
        corrected[idx] = adj
        running_max = adj
    return corrected


# ---------------------------------------------------------------------------
# Chi-square survival function (pure math — no scipy)
# ---------------------------------------------------------------------------


def _gammq(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) = 1 − P(a, x).

    Numerical-Recipes split: the series representation converges for x < a+1,
    the continued fraction for x ≥ a+1. Standard-library ``math`` only (lgamma /
    exp / log), so the χ² tail stays dependency-free and deterministic.
    """
    if x < 0.0 or a <= 0.0:
        raise ValueError("a must be > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cont_frac(a, x)


def _gamma_series(a: float, x: float) -> float:
    """Lower regularised incomplete gamma P(a, x) via its series expansion."""
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gamma_cont_frac(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a, x) via the Lentz continued fraction."""
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi_square_sf(x: float, df: int) -> float:
    """Upper-tail χ² probability P(χ²_df > x).

    The p-value of a χ² statistic. For df=2 this equals the closed form
    ``exp(-x/2)`` (the unit tests pin this), and the general case is the
    regularised upper incomplete gamma Q(df/2, x/2).
    """
    if df < 1:
        raise ValueError("df must be >= 1")
    if x <= 0.0:
        return 1.0
    return _gammq(df / 2.0, x / 2.0)


# ---------------------------------------------------------------------------
# Cochran's Q — omnibus for k (>= 2) related binary samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CochranQResult:
    """Cochran's Q omnibus result over ``k`` related binary arms.

    ``q`` — the Q statistic (0.0 when every block is concordant, i.e. no
    discordant evidence). ``df`` — degrees of freedom = ``k - 1``. ``p_value`` —
    the upper-tail χ² probability P(χ²_df > q); 1.0 in the all-concordant case.
    A significant ``p_value`` (< 0.05) is the gate for the post-hoc pairwise
    McNemar comparisons (composed and Holm-corrected in the runner).
    """

    q: float
    df: int
    p_value: float


def cochran_q(*arms: Sequence[int]) -> CochranQResult:
    """Cochran's Q omnibus test across ``k`` (>= 2) related binary arms.

    Each ``arm`` is an aligned 0/1 hit vector over the SAME blocks (Paraphrases),
    so ``arms[j][i]`` is whether arm ``j`` hit on block ``i``. Generalises McNemar
    to 3+ paired classifiers: H0 is that all arms share one success proportion.

    Q = (k-1)*(k*Σ C_j² − N²)/(k*N − Σ R_i²) with column totals ``C_j``, row
    totals ``R_i`` and grand total ``N``; distributed χ² with ``k-1`` df. When the
    denominator collapses to 0 (every block all-hit or all-miss) Q is undefined
    and reported as 0.0 / p=1.0 — the McNemar convention for no discordant pairs.

    Raises ``ValueError`` if fewer than two arms are given, the arms differ in
    length, or the blocks are empty.
    """
    k = len(arms)
    if k < 2:
        raise ValueError("cochran_q needs at least two arms")
    n_blocks = len(arms[0])
    if n_blocks == 0:
        raise ValueError("arms must be non-empty")
    if any(len(arm) != n_blocks for arm in arms):
        raise ValueError("all arms must have the same length")

    col_totals = [sum(arm) for arm in arms]  # C_j: successes per arm
    row_totals = [sum(arm[i] for arm in arms) for i in range(n_blocks)]  # R_i
    grand_total = sum(col_totals)  # N

    denominator = k * grand_total - sum(r * r for r in row_totals)
    df = k - 1
    if denominator == 0:
        # Every block is all-hit or all-miss → no discordant evidence.
        return CochranQResult(q=0.0, df=df, p_value=1.0)

    numerator = (k - 1) * (k * sum(c * c for c in col_totals) - grand_total**2)
    q = numerator / denominator
    return CochranQResult(q=q, df=df, p_value=chi_square_sf(q, df))


# ---------------------------------------------------------------------------
# Per-type combined result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeStatResult:
    """Statistical summary for one Paraphrase Type's paired comparison.

    ``b``, ``c`` — discordant pair counts (b = A-hit B-miss, c = A-miss B-hit).
    ``mcnemar_p`` — raw exact McNemar two-sided p-value (before Holm).
    ``ci_a``, ``ci_b`` — 95% Wilson CIs for Stack A and Stack B's hit_rate.
    ``n`` — total paraphrase count for this type.
    ``hit_rate_a``, ``hit_rate_b`` — empirical hit rates.
    """

    b: int
    c: int
    mcnemar_p: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    n: int
    hit_rate_a: float
    hit_rate_b: float


def compute_type_stats(
    hits_a: list[int],
    hits_b: list[int],
) -> TypeStatResult:
    """Compute paired McNemar + Wilson CI stats for one Paraphrase Type.

    ``hits_a[i]`` and ``hits_b[i]`` are 1 (hit) or 0 (miss) for the i-th
    Paraphrase.  Both lists must be the same non-zero length.

    Raises ``ValueError`` if lengths differ or lists are empty.
    """
    n = len(hits_a)
    if n == 0:
        raise ValueError("hits_a and hits_b must be non-empty")
    if len(hits_b) != n:
        raise ValueError(f"hits_a length {n} != hits_b length {len(hits_b)}")
    b = sum(1 for a, bv in zip(hits_a, hits_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(hits_a, hits_b) if a == 0 and bv == 1)
    p = mcnemar_exact_p(b, c)
    k_a = sum(hits_a)
    k_b = sum(hits_b)
    return TypeStatResult(
        b=b,
        c=c,
        mcnemar_p=p,
        ci_a=wilson_ci(k_a, n),
        ci_b=wilson_ci(k_b, n),
        n=n,
        hit_rate_a=k_a / n,
        hit_rate_b=k_b / n,
    )
