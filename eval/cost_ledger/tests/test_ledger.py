"""CostLedger tests — asserted against scripted fake usage metadata (PRD #654
Testing Decisions: "Cost accounting is tested by asserting call/token ledger
entries against scripted fake usage metadata, not real spend"). No LLM calls
anywhere in this file.
"""

from __future__ import annotations

import pytest

from eval.cost_ledger.ledger import CostLedger
from eval.cost_ledger.models import UsageMetadata


def _usage(input_tokens: int, output_tokens: int) -> UsageMetadata:
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def test_record_rejects_a_phase_outside_build_query_update():
    ledger = CostLedger()
    with pytest.raises(ValueError, match="phase"):
        ledger.record(
            stack="A", phase="ingest", model="gpt-4o-mini", usage=_usage(1, 1)
        )


def test_calls_returns_every_recorded_call_in_order():
    ledger = CostLedger()
    ledger.record(stack="A", phase="build", model="gpt-4o-mini", usage=_usage(10, 5))
    ledger.record(stack="B", phase="query", model="gpt-4o-mini", usage=_usage(20, 8))

    calls = ledger.calls
    assert [c.stack for c in calls] == ["A", "B"]
    assert [c.phase for c in calls] == ["build", "query"]


def test_calls_property_is_a_copy():
    ledger = CostLedger()
    ledger.record(stack="A", phase="build", model="gpt-4o-mini", usage=_usage(1, 1))
    snapshot = ledger.calls
    snapshot.append(snapshot[0])  # mutate the returned copy
    assert len(ledger.calls) == 1, (
        "mutating the returned list must not affect the ledger"
    )


def test_totals_aggregates_calls_and_tokens_for_one_stack_and_phase():
    ledger = CostLedger()
    ledger.record(stack="A", phase="build", model="gpt-4o-mini", usage=_usage(100, 50))
    ledger.record(stack="A", phase="build", model="gpt-4o-mini", usage=_usage(200, 75))
    ledger.record(stack="B", phase="build", model="gpt-4o-mini", usage=_usage(999, 999))

    totals = ledger.totals(stack="A", phase="build")
    assert totals.stack == "A"
    assert totals.phase == "build"
    assert totals.calls == 2
    assert totals.input_tokens == 300
    assert totals.output_tokens == 125
    assert totals.total_tokens == 425


def test_totals_with_no_filter_aggregates_everything():
    ledger = CostLedger()
    ledger.record(stack="A", phase="build", model="gpt-4o-mini", usage=_usage(10, 10))
    ledger.record(stack="B", phase="query", model="gpt-4o-mini", usage=_usage(20, 20))

    totals = ledger.totals()
    assert totals.stack == "*"
    assert totals.phase == "*"
    assert totals.calls == 2
    assert totals.total_tokens == 60


def test_totals_on_empty_ledger_is_zero_calls_and_none_usd():
    totals = CostLedger().totals()
    assert totals.calls == 0
    assert totals.input_tokens == 0
    assert totals.output_tokens == 0
    assert totals.total_tokens == 0
    assert totals.usd is None


def test_totals_usd_sums_only_priced_calls():
    ledger = CostLedger()
    ledger.record(
        stack="A", phase="build", model="gpt-4o-mini", usage=_usage(1_000_000, 0)
    )
    ledger.record(
        stack="A",
        phase="build",
        model="some-unpinned-finetune",
        usage=_usage(1_000_000, 0),
    )

    totals = ledger.totals(stack="A", phase="build")
    # gpt-4o-mini pinned at $0.15 / 1M input tokens (unit_prices.py); the
    # unpinned model's tokens are excluded from usd but still counted in
    # calls/total_tokens.
    assert totals.calls == 2
    assert totals.input_tokens == 2_000_000
    assert totals.usd == pytest.approx(0.15)


def test_totals_usd_is_none_when_no_matching_call_is_priced():
    ledger = CostLedger()
    ledger.record(
        stack="A", phase="build", model="some-unpinned-finetune", usage=_usage(10, 10)
    )
    assert ledger.totals(stack="A").usd is None


def test_totals_by_stack_phase_returns_one_entry_per_distinct_pair():
    ledger = CostLedger()
    ledger.record(stack="A", phase="build", model="gpt-4o-mini", usage=_usage(10, 10))
    ledger.record(stack="A", phase="query", model="gpt-4o-mini", usage=_usage(20, 20))
    ledger.record(stack="B", phase="build", model="gpt-4o-mini", usage=_usage(30, 30))

    grouped = ledger.totals_by_stack_phase()
    assert set(grouped) == {("A", "build"), ("A", "query"), ("B", "build")}
    assert grouped[("A", "build")].calls == 1
    assert grouped[("A", "build")].total_tokens == 20
    assert grouped[("B", "build")].total_tokens == 60
