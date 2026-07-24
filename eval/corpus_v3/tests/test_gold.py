"""Gold-mapping tests — external behaviour only (CODING_STANDARD §0.2).

ADR-0045 Prerequisite 3 / PRD #654 user story 3: gold-label mapping must be
symmetric across a docs-native id and a wiki id (concept OR entity, 1:N
allowed), through the SAME code path. These are pure-function, table-driven
tests over the committed corpus v3 wiki fixtures — no LLM calls.
"""

from __future__ import annotations

from pathlib import Path

from eval.corpus_v3 import gold

WIKI_DIR = Path(__file__).resolve().parents[1] / "wiki"


def test_build_gold_map_includes_concept_and_entity_pages():
    """v2's ``_wiki_slug_to_gold_section`` only scanned ``concepts/``; this scans both."""
    gold_map = gold.build_gold_map(WIKI_DIR)

    assert gold_map["return-window"] == frozenset({"refund_policy.md#return-window"})
    assert gold_map["acme-shop"] == frozenset(
        {
            "shipping_policy.md#international-shipping",
            "payment_methods.md#accepted-cards",
        }
    )


def test_build_gold_map_keeps_the_full_sources_list_not_just_the_first():
    """v2 kept ``sources[0]`` only; a 1:N entity page needs every source kept."""
    gold_map = gold.build_gold_map(WIKI_DIR)

    assert len(gold_map["acme-shop"]) == 2


def test_resolve_gold_sections_from_a_concept_wiki_id():
    gold_map = gold.build_gold_map(WIKI_DIR)

    resolved = gold.resolve_gold_sections(gold_map, "return-window")

    assert resolved == frozenset({"return-window", "refund_policy.md#return-window"})


def test_resolve_gold_sections_from_the_equivalent_docs_native_id():
    """The SAME function, called with the docs-native id, resolves to the same class."""
    gold_map = gold.build_gold_map(WIKI_DIR)

    resolved = gold.resolve_gold_sections(gold_map, "refund_policy.md#return-window")

    assert resolved == frozenset({"return-window", "refund_policy.md#return-window"})


def test_resolve_gold_sections_from_an_entity_wiki_id_covers_all_its_sources():
    """The v2-guaranteed-miss case: an entity page's own id now resolves to its full 1:N set."""
    gold_map = gold.build_gold_map(WIKI_DIR)

    resolved = gold.resolve_gold_sections(gold_map, "acme-shop")

    assert resolved == frozenset(
        {
            "acme-shop",
            "shipping_policy.md#international-shipping",
            "payment_methods.md#accepted-cards",
        }
    )


def test_resolve_gold_sections_from_a_docs_native_id_covered_by_an_entity_page():
    """The reverse direction: a docs id whose only wiki coverage is an entity page."""
    gold_map = gold.build_gold_map(WIKI_DIR)

    resolved = gold.resolve_gold_sections(gold_map, "payment_methods.md#accepted-cards")

    assert resolved == frozenset({"payment_methods.md#accepted-cards", "acme-shop"})


def test_resolve_gold_sections_unmapped_id_falls_back_to_itself():
    """An id absent from the table (e.g. Stack B's native docs chunk id) still resolves."""
    gold_map = gold.build_gold_map(WIKI_DIR)

    resolved = gold.resolve_gold_sections(
        gold_map, "unrelated_docs.md#unrelated-heading"
    )

    assert resolved == frozenset({"unrelated_docs.md#unrelated-heading"})
