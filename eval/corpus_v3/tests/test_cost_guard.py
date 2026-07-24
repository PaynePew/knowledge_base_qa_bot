"""Cost guard tests — external behaviour only (CODING_STANDARD §0.2).

Pure logic, no LLM calls: every projection is scaled from a hand-scripted
``CostLedger`` sample (CODING_STANDARD §6.5 "fixtures are hand-written,
deterministic ... never LLM-generated"), never real spend.
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.cost_guard import (
    BUDGET_USD_CAP,
    check_cost_guard,
    project_spend,
)
from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata


def _ledger_with_priced_sample(n_calls: int, input_tokens: int) -> CostLedger:
    """A ledger with ``n_calls`` identical, pinned-price (gpt-4o-mini) query calls."""
    ledger = CostLedger()
    for _ in range(n_calls):
        ledger.record(
            stack="wiki",
            phase="query",
            model="gpt-4o-mini",
            usage=UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=50,
                total_tokens=input_tokens + 50,
            ),
        )
    return ledger


def _ledger_with_unpriced_sample(n_calls: int) -> CostLedger:
    """A ledger whose calls use a model absent from ``unit_prices.PINNED_PRICES``."""
    ledger = CostLedger()
    for _ in range(n_calls):
        ledger.record(
            stack="wiki",
            phase="query",
            model="some-unpriced-model",
            usage=UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150),
        )
    return ledger


# ---------------------------------------------------------------------------
# project_spend
# ---------------------------------------------------------------------------
def test_project_spend_scales_the_sample_average_to_the_planned_call_count():
    ledger = _ledger_with_priced_sample(n_calls=10, input_tokens=1000)
    sample_usd = ledger.totals(phase="query").usd
    assert sample_usd is not None

    projection = project_spend(ledger, phase="query", planned_calls=100)

    assert projection.sample_calls == 10
    assert projection.planned_calls == 100
    assert projection.avg_usd_per_call == pytest.approx(sample_usd / 10)
    assert projection.projected_usd == pytest.approx(sample_usd / 10 * 100)


def test_project_spend_raises_on_negative_planned_calls():
    ledger = _ledger_with_priced_sample(n_calls=1, input_tokens=100)
    with pytest.raises(ValueError, match="planned_calls"):
        project_spend(ledger, phase="query", planned_calls=-1)


def test_project_spend_raises_when_ledger_has_no_sample_for_the_phase():
    ledger = _ledger_with_priced_sample(n_calls=5, input_tokens=100)  # phase="query"
    with pytest.raises(ValueError, match="no recorded calls"):
        project_spend(ledger, phase="update", planned_calls=10)


def test_project_spend_reports_none_when_the_sample_has_no_pinned_price():
    ledger = _ledger_with_unpriced_sample(n_calls=5)
    projection = project_spend(ledger, phase="query", planned_calls=100)
    assert projection.sample_calls == 5
    assert projection.avg_usd_per_call is None
    assert projection.projected_usd is None


# ---------------------------------------------------------------------------
# check_cost_guard
# ---------------------------------------------------------------------------
def test_check_cost_guard_proceeds_when_projection_is_within_the_cap():
    # A tiny sample and a tiny planned run -> well under $10.
    ledger = _ledger_with_priced_sample(n_calls=10, input_tokens=1000)
    projection = project_spend(ledger, phase="query", planned_calls=50)
    assert projection.projected_usd < BUDGET_USD_CAP

    result = check_cost_guard(projection)

    assert result.proceed is True
    assert "proceeding" in result.message


def test_check_cost_guard_halts_when_projection_exceeds_the_cap():
    # A high per-call cost scaled to a huge planned run -> well over $10.
    ledger = _ledger_with_priced_sample(n_calls=10, input_tokens=1_000_000)
    projection = project_spend(ledger, phase="query", planned_calls=1_000_000)
    assert projection.projected_usd > BUDGET_USD_CAP

    result = check_cost_guard(projection)

    assert result.proceed is False
    assert "exceeds" in result.message
    assert "ready-for-human" in result.message


def test_check_cost_guard_halts_when_projection_is_unpriced_fail_closed():
    ledger = _ledger_with_unpriced_sample(n_calls=5)
    projection = project_spend(ledger, phase="query", planned_calls=100)
    assert projection.projected_usd is None

    result = check_cost_guard(projection)

    assert result.proceed is False
    assert "unavailable" in result.message
    assert "ready-for-human" in result.message


def test_check_cost_guard_respects_a_custom_cap():
    ledger = _ledger_with_priced_sample(n_calls=10, input_tokens=1000)
    projection = project_spend(ledger, phase="query", planned_calls=100)
    # Whatever this projects to, a cap of 0.0 must always halt.
    result = check_cost_guard(projection, cap_usd=0.0)
    assert result.proceed is False
