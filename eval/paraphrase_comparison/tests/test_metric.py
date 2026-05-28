"""C5c hit-metric verdict tests (external behaviour only, CODING_STANDARD §0.2).

The metric is deterministic, so these assert the verdict directly — including
the correct-metadata-wrong-content miss case the AC calls out explicitly.
"""

from __future__ import annotations

from deepeval.test_case import LLMTestCase

from eval.paraphrase_comparison.metric import (
    HitRateAtK,
    hit_at_k,
    is_hit,
    reciprocal_rank_at_k,
)
from eval.paraphrase_comparison.models import RetrievedItem

GOLD = "returns_policy.md#return-window"
KEY_TOKENS = ["refund", "packaging", "receipt"]


def _item(source_id: str, content: str) -> RetrievedItem:
    return RetrievedItem(source_section_id=source_id, content=content)


def test_hit_requires_both_source_match_and_token_overlap():
    item = _item(GOLD, "Items returned in original packaging get a full refund.")
    assert is_hit(item, GOLD, KEY_TOKENS) is True


def test_wrong_source_is_a_miss_even_with_token_overlap():
    item = _item("shipping_options.md#standard-delivery", "full refund and packaging")
    assert is_hit(item, GOLD, KEY_TOKENS) is False


def test_correct_metadata_wrong_content_is_a_miss():
    # Source id matches the gold section, but the content shares NO Key Token —
    # the metric must score this a miss (the AC's key edge case).
    item = _item(GOLD, "This paragraph is about something entirely unrelated.")
    assert is_hit(item, GOLD, KEY_TOKENS) is False


def test_empty_key_tokens_is_a_miss():
    item = _item(GOLD, "full refund packaging receipt")
    assert is_hit(item, GOLD, []) is False


def test_token_overlap_is_case_insensitive():
    item = _item(GOLD, "REFUND issued once PACKAGING is verified")
    assert is_hit(item, GOLD, KEY_TOKENS) is True


def test_hit_at_k_only_considers_top_k():
    items = [
        _item("other#a", "noise"),
        _item("other#b", "noise"),
        _item("other#c", "noise"),
        _item(GOLD, "refund and packaging"),  # 4th — outside k=3
    ]
    assert hit_at_k(items, GOLD, KEY_TOKENS, k=3) == 0.0
    assert hit_at_k(items, GOLD, KEY_TOKENS, k=4) == 1.0


def test_metric_scores_test_case_as_hit():
    metric = HitRateAtK(k=3)
    case = LLMTestCase(
        input="how long to return",
        actual_output="",
        expected_output=GOLD,
        retrieval_context=[GOLD],
        metadata={
            "retrieved_items": [_item(GOLD, "refund within thirty days, packaging")],
            "key_tokens": KEY_TOKENS,
        },
    )
    score = metric.measure(case)
    assert score == 1.0
    assert metric.is_successful() is True


def test_metric_scores_correct_id_wrong_content_as_miss():
    metric = HitRateAtK(k=3)
    case = LLMTestCase(
        input="how long to return",
        actual_output="",
        expected_output=GOLD,
        retrieval_context=[GOLD],
        metadata={
            "retrieved_items": [_item(GOLD, "completely unrelated text body")],
            "key_tokens": KEY_TOKENS,
        },
    )
    assert metric.measure(case) == 0.0
    assert metric.is_successful() is False


# ---------------------------------------------------------------------------
# Reciprocal rank (MRR-at-k) — PRD #100
# ---------------------------------------------------------------------------
def test_reciprocal_rank_is_one_when_first_item_hits():
    items = [_item(GOLD, "refund and packaging")]
    assert reciprocal_rank_at_k(items, GOLD, KEY_TOKENS, k=3) == 1.0


def test_reciprocal_rank_is_half_when_first_hit_is_at_rank_two():
    items = [
        _item("other#a", "noise"),
        _item(GOLD, "refund issued with packaging"),  # rank 2 -> RR=0.5
        _item(GOLD, "refund and packaging"),
    ]
    assert reciprocal_rank_at_k(items, GOLD, KEY_TOKENS, k=3) == 0.5


def test_reciprocal_rank_is_third_when_first_hit_is_at_rank_three():
    items = [
        _item("other#a", "noise"),
        _item("other#b", "noise"),
        _item(GOLD, "refund and packaging"),  # rank 3 -> RR=1/3
    ]
    rr = reciprocal_rank_at_k(items, GOLD, KEY_TOKENS, k=3)
    assert abs(rr - (1.0 / 3.0)) < 1e-9


def test_reciprocal_rank_is_zero_when_no_top_k_item_hits():
    items = [
        _item("other#a", "noise"),
        _item("other#b", "noise"),
        _item("other#c", "noise"),
        _item(GOLD, "refund and packaging"),  # 4th — outside k=3
    ]
    assert reciprocal_rank_at_k(items, GOLD, KEY_TOKENS, k=3) == 0.0
    # k=4 brings the hit into view at rank 4 -> RR=0.25.
    assert reciprocal_rank_at_k(items, GOLD, KEY_TOKENS, k=4) == 0.25


def test_reciprocal_rank_uses_first_hit_not_a_later_one():
    items = [
        _item(GOLD, "refund and packaging"),  # rank 1 hit wins
        _item(GOLD, "refund and packaging"),
    ]
    assert reciprocal_rank_at_k(items, GOLD, KEY_TOKENS, k=3) == 1.0


def test_metric_exposes_reciprocal_rank_alongside_hit():
    metric = HitRateAtK(k=3)
    case = LLMTestCase(
        input="how long to return",
        actual_output="",
        expected_output=GOLD,
        retrieval_context=["other#a", GOLD],
        metadata={
            "retrieved_items": [
                _item("other#a", "noise"),
                _item(GOLD, "refund within thirty days, packaging"),  # rank 2
            ],
            "key_tokens": KEY_TOKENS,
        },
    )
    assert metric.measure(case) == 1.0  # hit_rate@3
    assert metric.reciprocal_rank == 0.5  # first hit at rank 2
