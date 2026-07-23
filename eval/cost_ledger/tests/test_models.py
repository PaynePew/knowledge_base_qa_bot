"""Pure model tests — no LLM, no filesystem."""

from __future__ import annotations

from eval.cost_ledger.models import UsageMetadata


def test_usage_metadata_from_raw_reads_all_three_fields():
    usage = UsageMetadata.from_raw(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    assert usage == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_usage_metadata_from_raw_none_is_all_zero():
    assert UsageMetadata.from_raw(None) == UsageMetadata()


def test_usage_metadata_from_raw_empty_dict_is_all_zero():
    assert UsageMetadata.from_raw({}) == UsageMetadata()


def test_usage_metadata_from_raw_missing_keys_default_to_zero():
    usage = UsageMetadata.from_raw({"input_tokens": 3})
    assert usage == UsageMetadata(input_tokens=3, output_tokens=0, total_tokens=0)
