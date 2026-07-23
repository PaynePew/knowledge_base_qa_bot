"""Pure unit-price tests — the pinned appendix, no live pricing lookup."""

from __future__ import annotations

from eval.cost_ledger.models import UsageMetadata
from eval.cost_ledger.unit_prices import estimate_usd


def test_estimate_usd_known_model():
    usage = UsageMetadata(
        input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000
    )
    # gpt-4o-mini pinned: $0.15/1M input + $0.60/1M output.
    assert estimate_usd("gpt-4o-mini", usage) == 0.75


def test_estimate_usd_zero_usage_is_zero_cost():
    assert estimate_usd("gpt-4o-mini", UsageMetadata()) == 0.0


def test_estimate_usd_unknown_model_returns_none():
    assert (
        estimate_usd(
            "some-unpinned-finetune", UsageMetadata(input_tokens=1, output_tokens=1)
        )
        is None
    )
