"""Deep module per Ousterhout. Public surface: ``mcnemar_exact_p``, ``wilson_ci``,
``holm_correct``, ``TypeStatResult``, ``compute_type_stats``.

Pure statistical functions for the Phase 8.5 McNemar / Wilson CI / Holm report
upgrade (issue #140, PRD #137). All functions are deterministic, require no LLM,
and carry zero external dependencies beyond the Python standard library ``math``.
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
"""

from __future__ import annotations

import math
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
