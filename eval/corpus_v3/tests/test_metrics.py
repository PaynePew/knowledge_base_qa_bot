"""Stratum-agnostic hit-metric tests — external behaviour only (CODING_STANDARD §0.2).

Table-driven per the AC: the metric is deterministic, so these assert the
verdict directly, including the 1:N gold-Section-id case (ADR-0045
Prerequisite 3) and the hit@1 / hit@3 / MRR cutoffs (Prerequisite 4).
"""

from __future__ import annotations

import pytest

from eval.corpus_v3.metrics import (
    HIT_AT_1,
    HIT_AT_3,
    hit_at_k,
    is_hit,
    reciprocal_rank_at_k,
)
from eval.corpus_v3.models import RetrievedItem

GOLD = "returns_policy.md#return-window"
KEY_TOKENS = ["refund", "packaging", "receipt"]


def _item(source_id: str, content: str) -> RetrievedItem:
    return RetrievedItem(source_section_id=source_id, content=content)


# ---------------------------------------------------------------------------
# is_hit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source_id,content,gold_ids,key_tokens,expected",
    [
        # source matches, content overlaps -> hit
        (
            GOLD,
            "Items in original packaging get a full refund.",
            [GOLD],
            KEY_TOKENS,
            True,
        ),
        # wrong source, content overlaps -> miss
        ("other#a", "full refund and packaging", [GOLD], KEY_TOKENS, False),
        # correct source, content shares no Key Token -> miss (AC edge case)
        (GOLD, "This paragraph is unrelated.", [GOLD], KEY_TOKENS, False),
        # empty key tokens -> miss even with a source match
        (GOLD, "full refund packaging receipt", [GOLD], [], False),
        # empty gold set (unanswerable query) -> always a miss
        (GOLD, "full refund packaging receipt", [], KEY_TOKENS, False),
        # case-insensitive token overlap
        (GOLD, "REFUND issued once PACKAGING is verified", [GOLD], KEY_TOKENS, True),
        # 1:N gold set (entity-page mapping, ADR-0045 Prerequisite 3): a hit
        # against the SECOND mapped Section id still counts.
        ("other#b", "full refund after receipt", [GOLD, "other#b"], KEY_TOKENS, True),
    ],
)
def test_is_hit_table(source_id, content, gold_ids, key_tokens, expected):
    assert is_hit(_item(source_id, content), gold_ids, key_tokens) is expected


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------
def test_hit_at_k_only_considers_top_k():
    items = [
        _item("other#a", "noise"),
        _item("other#b", "noise"),
        _item("other#c", "noise"),
        _item(GOLD, "refund and packaging"),  # 4th — outside k=3
    ]
    assert hit_at_k(items, [GOLD], KEY_TOKENS, HIT_AT_3) == 0.0
    assert hit_at_k(items, [GOLD], KEY_TOKENS, k=4) == 1.0


def test_hit_at_1_is_stricter_than_hit_at_3():
    items = [
        _item("other#a", "noise"),
        _item(GOLD, "refund and packaging"),  # rank 2
    ]
    assert hit_at_k(items, [GOLD], KEY_TOKENS, HIT_AT_1) == 0.0
    assert hit_at_k(items, [GOLD], KEY_TOKENS, HIT_AT_3) == 1.0


def test_hit_at_k_empty_items_is_a_miss():
    assert hit_at_k([], [GOLD], KEY_TOKENS, HIT_AT_3) == 0.0


# ---------------------------------------------------------------------------
# reciprocal_rank_at_k (MRR building block)
# ---------------------------------------------------------------------------
def test_reciprocal_rank_at_rank_one():
    items = [_item(GOLD, "refund and packaging")]
    assert reciprocal_rank_at_k(items, [GOLD], KEY_TOKENS, HIT_AT_3) == 1.0


def test_reciprocal_rank_at_rank_two_is_one_half():
    items = [_item("other#a", "noise"), _item(GOLD, "refund and packaging")]
    assert reciprocal_rank_at_k(items, [GOLD], KEY_TOKENS, HIT_AT_3) == pytest.approx(
        0.5
    )


def test_reciprocal_rank_at_rank_three_is_one_third():
    items = [
        _item("other#a", "noise"),
        _item("other#b", "noise"),
        _item(GOLD, "refund and packaging"),
    ]
    assert reciprocal_rank_at_k(items, [GOLD], KEY_TOKENS, HIT_AT_3) == pytest.approx(
        1 / 3
    )


def test_reciprocal_rank_no_hit_in_top_k_is_zero():
    items = [_item("other#a", "noise"), _item("other#b", "noise")]
    assert reciprocal_rank_at_k(items, [GOLD], KEY_TOKENS, HIT_AT_3) == 0.0
