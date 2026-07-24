"""Deep module per Ousterhout. Public surface: ``Stratum``, ``QueryOutcome``,
``evaluate_query``, ``group_by_stratum``, ``arm_metric_means``,
``stratified_metrics``, ``macro_metrics``, ``winning_arm``, ``FULL_STRATUM``,
``BY_SCENARIO``, ``BY_OVERLAP``, ``BY_LANGUAGE``.

Per-stratum-first, macro-second aggregation for the corpus v3 fair experiment
(issue #659, PRD #654 user stories 6-7, 9, 13: "Aggregation is per stratum
first, macro second, so the overlap-predicts-winner effect and per-language
behavior are visible instead of hidden in aggregates").

A MACRO aggregate here means the mean of each stratum's own per-arm mean,
weighting every stratum equally regardless of how many queries fall in it.
That is a deliberate choice, not an approximation of a pooled (query-count
-weighted) mean: it is what lets the macro winner across all strata differ
from the winner within any single stratum (a few small strata unanimous for
one arm can outvote one large stratum unanimous for the other). The verdict
report (a later slice) is expected to read the per-stratum table AND the
macro number together — never only the pooled headline — so a scenario- or
language-specific loss is never hidden inside a win on average.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from .metrics import HIT_AT_1, HIT_AT_3, hit_at_k, reciprocal_rank_at_k
from .models import RetrievedItem
from .query_schema import Query

MetricName = Literal["hit_at_1", "hit_at_3", "reciprocal_rank"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stratum:
    """The three ADR-0045 Prerequisite 4 query labels, carried as one unit."""

    scenario: str
    overlap: str
    language: str


@dataclass(frozen=True)
class QueryOutcome:
    """One retrieval arm's scored outcome for one query, tagged with its stratum."""

    query_id: str
    stratum: Stratum
    arm: str
    hit_at_1: float
    hit_at_3: float
    reciprocal_rank: float


def evaluate_query(
    query: Query, items: Sequence[RetrievedItem], arm: str
) -> QueryOutcome:
    """Score ``arm``'s retrieved ``items`` for ``query`` into a stratum-tagged outcome."""
    gold = query.gold_section_ids
    tokens = query.key_tokens
    return QueryOutcome(
        query_id=query.query_id,
        stratum=Stratum(query.scenario_stratum, query.overlap_stratum, query.language),
        arm=arm,
        hit_at_1=hit_at_k(items, gold, tokens, HIT_AT_1),
        hit_at_3=hit_at_k(items, gold, tokens, HIT_AT_3),
        reciprocal_rank=reciprocal_rank_at_k(items, gold, tokens, HIT_AT_3),
    )


# ---------------------------------------------------------------------------
# Stratum grouping keys — the axes the report groups by (full triple, or one
# label alone so "overlap-predicts-winner" and "per-language behavior" are
# each visible on their own, not only inside the full-triple breakdown).
# ---------------------------------------------------------------------------
StratumKeyFn = Callable[[Stratum], str]


def _full_stratum_key(stratum: Stratum) -> str:
    return f"{stratum.scenario}|{stratum.overlap}|{stratum.language}"


def _scenario_key(stratum: Stratum) -> str:
    return stratum.scenario


def _overlap_key(stratum: Stratum) -> str:
    return stratum.overlap


def _language_key(stratum: Stratum) -> str:
    return stratum.language


FULL_STRATUM: StratumKeyFn = _full_stratum_key
BY_SCENARIO: StratumKeyFn = _scenario_key
BY_OVERLAP: StratumKeyFn = _overlap_key
BY_LANGUAGE: StratumKeyFn = _language_key


# ---------------------------------------------------------------------------
# Public API — aggregation
# ---------------------------------------------------------------------------
def group_by_stratum(
    outcomes: Sequence[QueryOutcome], key: StratumKeyFn = FULL_STRATUM
) -> dict[str, list[QueryOutcome]]:
    """Group ``outcomes`` by ``key(outcome.stratum)``, preserving encounter order."""
    groups: dict[str, list[QueryOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault(key(outcome.stratum), []).append(outcome)
    return groups


def arm_metric_means(
    outcomes: Sequence[QueryOutcome], metric: MetricName
) -> dict[str, float]:
    """Mean of ``metric`` per arm over ``outcomes`` (a flat, unstratified mean)."""
    if not outcomes:
        raise ValueError("arm_metric_means needs at least one outcome")
    by_arm: dict[str, list[float]] = {}
    for outcome in outcomes:
        by_arm.setdefault(outcome.arm, []).append(getattr(outcome, metric))
    return {arm: mean(values) for arm, values in by_arm.items()}


def stratified_metrics(
    outcomes: Sequence[QueryOutcome],
    metric: MetricName,
    key: StratumKeyFn = FULL_STRATUM,
) -> dict[str, dict[str, float]]:
    """Per-stratum, per-arm mean of ``metric`` — the "per stratum first" table."""
    return {
        label: arm_metric_means(group, metric)
        for label, group in group_by_stratum(outcomes, key).items()
    }


def macro_metrics(
    outcomes: Sequence[QueryOutcome],
    metric: MetricName,
    key: StratumKeyFn = FULL_STRATUM,
) -> dict[str, float]:
    """Macro aggregate: mean of each stratum's per-arm mean, one stratum one vote.

    See module docstring — this is NOT a query-count-weighted pooled mean, and
    can therefore disagree with both the pooled mean and any single stratum's
    winner.
    """
    per_stratum = stratified_metrics(outcomes, metric, key)
    if not per_stratum:
        raise ValueError("macro_metrics needs at least one outcome")
    arms = sorted({outcome.arm for outcome in outcomes})
    return {
        arm: mean(cell[arm] for cell in per_stratum.values() if arm in cell)
        for arm in arms
    }


def winning_arm(arm_scores: dict[str, float]) -> str:
    """The arm with the highest score; ties broken by arm name for determinism."""
    if not arm_scores:
        raise ValueError("winning_arm needs at least one arm score")
    return max(sorted(arm_scores), key=lambda arm: arm_scores[arm])
