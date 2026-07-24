"""Deep module per Ousterhout. Public surface: ``BUDGET_USD_CAP``,
``CostProjection``, ``GuardResult``, ``project_spend``, ``check_cost_guard``.

Pre-live-run cost guard for the corpus v3 verdict run (issue #662 AC 2, PRD
#654's four-part cost model). Before any live LLM call executes, the runner
must project the FULL run's total spend from the cost ledger's
already-recorded per-call averages and refuse to run if the projection
exceeds the pre-registered budget cap -- ADR-0045's burden-of-proof stance
(the wiki carries the extra build cost, so it must prove its value, not be
assumed safe) extends here to the verdict run itself: an unbounded live run
is never launched on an unverified estimate.

Primary units stay calls/tokens (``eval.cost_ledger``'s convention); USD is
the secondary, budget-facing unit this guard actually gates on, since the
$10 cap is a dollar figure, not a call-count figure.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.cost_ledger.ledger import CostLedger

# Pre-registered budget cap (issue #662 AC 2). Not a runtime flag a caller can
# quietly override at the call site that matters (``check_cost_guard`` reads
# it as the default) -- raising it is a deliberate, reviewed code change.
BUDGET_USD_CAP = 10.0


@dataclass(frozen=True)
class CostProjection:
    """Projected USD spend for a planned run, derived from a recorded sample.

    ``sample_calls`` / ``sample_usd`` are the ledger's ALREADY-RECORDED basis
    (e.g. a small pilot batch) this projection scales from; ``planned_calls``
    is the full run's expected call count. ``avg_usd_per_call`` /
    ``projected_usd`` are ``None`` when the sample has no priced calls (see
    ``StackPhaseTotals.usd``) -- the projection cannot then be trusted, and
    the guard must fail closed (see :func:`check_cost_guard`).
    """

    sample_calls: int
    sample_usd: float | None
    planned_calls: int
    avg_usd_per_call: float | None
    projected_usd: float | None


@dataclass(frozen=True)
class GuardResult:
    """The guard's verdict on one projection: proceed with the live run, or halt."""

    proceed: bool
    projection: CostProjection
    message: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def project_spend(
    ledger: CostLedger, *, phase: str, planned_calls: int
) -> CostProjection:
    """Project ``planned_calls`` worth of ``phase`` spend from the ledger's
    ALREADY-RECORDED per-call average for that phase (issue #662 AC 2:
    "project total spend from the ledger's per-query averages").

    Raises ``ValueError`` if ``planned_calls`` is negative, or if the ledger
    has NO recorded calls for ``phase`` at all -- there is no average to
    scale from, and this function must never silently treat "no sample" as
    "zero cost".
    """
    if planned_calls < 0:
        raise ValueError(f"planned_calls must be >= 0, got {planned_calls!r}")
    totals = ledger.totals(phase=phase)
    if totals.calls == 0:
        raise ValueError(
            f"no recorded calls for phase={phase!r} -- cannot project spend "
            "without a sample to average from; record a pilot batch first"
        )
    avg = totals.usd / totals.calls if totals.usd is not None else None
    projected = avg * planned_calls if avg is not None else None
    return CostProjection(
        sample_calls=totals.calls,
        sample_usd=totals.usd,
        planned_calls=planned_calls,
        avg_usd_per_call=avg,
        projected_usd=projected,
    )


def check_cost_guard(
    projection: CostProjection, *, cap_usd: float = BUDGET_USD_CAP
) -> GuardResult:
    """Gate a live run on ``projection`` (issue #662 AC 2: "if projected spend
    exceeds $10 USD, stop ... instead of running").

    Fails closed: an unpriced projection (``projected_usd is None`` -- e.g.
    the sample's model has no pinned unit price in ``eval.cost_ledger
    .unit_prices``) halts exactly like an over-budget one, since "unknown"
    must never read as "safe to run".
    """
    if projection.projected_usd is None:
        return GuardResult(
            proceed=False,
            projection=projection,
            message=(
                "cost projection unavailable (sampled calls have no pinned "
                "unit price) -- halting; mark the issue ready-for-human with "
                "this projection instead of running"
            ),
        )
    if projection.projected_usd > cap_usd:
        return GuardResult(
            proceed=False,
            projection=projection,
            message=(
                f"projected spend ${projection.projected_usd:.2f} exceeds the "
                f"${cap_usd:.2f} cap -- halting; mark the issue ready-for-human "
                "with this projection instead of running"
            ),
        )
    return GuardResult(
        proceed=True,
        projection=projection,
        message=(
            f"projected spend ${projection.projected_usd:.2f} is within the "
            f"${cap_usd:.2f} cap -- proceeding"
        ),
    )
