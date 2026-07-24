"""Dual-corpus hit-metric tests — external behaviour only (CODING_STANDARD §0.2).

ADR-0045 Prerequisite 3 / PRD #654 user story 4: the Key-Token hit condition
must draw from the UNION of both corpora's Key Tokens, and (per user story 3)
id-match must be membership in a gold Section SET rather than equality with a
single docs-native id. Pure-function, table-driven, synthetic fixtures — no
LLM calls.
"""

from __future__ import annotations

from eval.corpus_v3 import gold, metric
from eval.corpus_v3.models import RetrievedItem


def _item(source_section_id: str, content: str) -> RetrievedItem:
    return RetrievedItem(source_section_id=source_section_id, content=content)


# ---------------------------------------------------------------------------
# Symmetric gold-set membership (v2 would score these first two as misses)
# ---------------------------------------------------------------------------
def test_entity_page_hit_that_v2_would_score_as_a_guaranteed_miss():
    """v2's ``is_hit`` compares against a SINGLE docs-native id; an entity
    page's own wiki id can never equal it, so this is a guaranteed miss under
    v2 regardless of content. Here the gold SET (resolved via ``gold``)
    includes the entity id, so a content-matching hit is scored correctly.
    """
    gold_set = gold.resolve_gold_sections(
        {"acme-shop": frozenset({"shipping_policy.md#international-shipping"})},
        "shipping_policy.md#international-shipping",
    )
    item = _item("acme-shop", "Acme Shop ships internationally to over 40 countries.")

    assert metric.is_hit(item, gold_set, ["international", "ships"], [])


def test_docs_native_item_still_hits_its_own_gold_id():
    """The set-membership condition subsumes plain equality (Stack B's case)."""
    gold_set = gold.resolve_gold_sections(
        {"return-window": frozenset({"refund_policy.md#return-window"})},
        "refund_policy.md#return-window",
    )
    item = _item(
        "refund_policy.md#return-window", "Items may be returned within 30 days."
    )

    assert metric.is_hit(item, gold_set, ["30", "days"], [])


def test_item_outside_the_gold_set_is_a_miss_even_with_key_token_overlap():
    gold_set = frozenset({"refund_policy.md#return-window"})
    item = _item("warranty.md#coverage-period", "Refunds within 30 days.")

    assert not metric.is_hit(item, gold_set, ["30", "days"], [])


# ---------------------------------------------------------------------------
# Dual-corpus Key-Token union (v2 drew only from key_tokens_docs)
# ---------------------------------------------------------------------------
def test_reworded_wiki_section_passes_via_wiki_side_key_tokens():
    """The docs body says "refund"; the curated wiki Section reworded it to
    "reimbursement" — a real paraphrase, not a synonym present in the docs
    Key Tokens. v2 (docs-body IDF only) would score this a miss even though
    the retrieved content answers the question; the union with the wiki-side
    Key Tokens recovers the hit.
    """
    gold_set = frozenset({"return-window"})
    item = _item(
        "return-window",
        "Your reimbursement is issued within 30 days of the return being received.",
    )

    key_tokens_docs = ["refund", "processing"]  # docs-body wording; absent from content
    key_tokens_wiki = [
        "reimbursement"
    ]  # wiki's own reworded wording; present in content

    assert not metric.is_hit(item, gold_set, key_tokens_docs, [])  # docs-only: miss
    assert metric.is_hit(item, gold_set, key_tokens_docs, key_tokens_wiki)  # union: hit


def test_no_key_token_overlap_from_either_corpus_is_a_miss():
    gold_set = frozenset({"return-window"})
    item = _item("return-window", "Completely unrelated content.")

    assert not metric.is_hit(item, gold_set, ["refund"], ["reimbursement"])


def test_empty_key_tokens_from_both_corpora_is_a_miss():
    gold_set = frozenset({"return-window"})
    item = _item("return-window", "Refunds within 30 days.")

    assert not metric.is_hit(item, gold_set, [], [])


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------
def test_hit_at_k_true_when_any_top_k_item_hits():
    gold_set = frozenset({"return-window"})
    items = [
        _item("warranty.md#coverage-period", "unrelated"),
        _item("return-window", "Refunds within 30 days."),
    ]

    assert metric.hit_at_k(items, gold_set, ["refund"], ["days"], k=3) == 1.0


def test_hit_at_k_false_when_hit_falls_outside_the_cutoff():
    gold_set = frozenset({"return-window"})
    items = [
        _item("warranty.md#coverage-period", "unrelated"),
        _item("return-window", "Refunds within 30 days."),
    ]

    assert metric.hit_at_k(items, gold_set, ["refund"], ["days"], k=1) == 0.0
