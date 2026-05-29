"""Tests for auto-derived Gold Section inventory (AC2, issue #142).

Gold Sections are now derived by parsing the corpus into heading-Sections
rather than reading a hand-maintained YAML. The ``multi_sub_fact`` flag is
dropped — sub-fact narrowing is covered downstream by the Synthesizer's
context-bound evolutions (PRD #137).

All tests are offline and deterministic: the derivation is a pure parse of
committed corpus files with no LLM calls.
"""

from __future__ import annotations

from pathlib import Path

# The corpus used by Phase 8 (committed snapshot under eval package).
_PKG_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_DIR = _PKG_ROOT / "corpus"

# Entity sources are excluded from the Gold Section pool (they collapse into
# a single entity wiki page, not concept pages, and are not Paraphrase sources).
_ENTITY_SOURCES = {"warranty.md"}


def test_derive_gold_sections_parses_corpus():
    """derive_gold_sections returns at least 39 GoldSections from the corpus."""
    from eval.paraphrase_comparison.generation.sampling import derive_gold_sections

    sections = derive_gold_sections(_CORPUS_DIR, entity_sources=_ENTITY_SOURCES)
    assert len(sections) >= 39, (
        f"Expected ≥39 Gold Sections from corpus, got {len(sections)}"
    )


def test_derive_gold_sections_excludes_entity_sources():
    """Entity-sourced Sections must not appear in the derived Gold Section pool."""
    from eval.paraphrase_comparison.generation.sampling import derive_gold_sections

    sections = derive_gold_sections(_CORPUS_DIR, entity_sources=_ENTITY_SOURCES)
    entity_ids = {
        s.section_id for s in sections if s.section_id.startswith("warranty.md")
    }
    assert not entity_ids, f"Entity sections must be excluded: {entity_ids}"


def test_derive_gold_sections_ids_have_correct_format():
    """Every derived section_id is 'filename.md#slug' with exactly one '#'."""
    from eval.paraphrase_comparison.generation.sampling import derive_gold_sections

    sections = derive_gold_sections(_CORPUS_DIR, entity_sources=_ENTITY_SOURCES)
    for s in sections:
        assert s.section_id.count("#") == 1, (
            f"section_id must be 'file#slug', got: {s.section_id!r}"
        )


def test_derive_gold_sections_concept_slug_matches_section_id():
    """Each derived GoldSection's concept_slug equals the heading-slug part of its id."""
    from eval.paraphrase_comparison.generation.sampling import derive_gold_sections

    sections = derive_gold_sections(_CORPUS_DIR, entity_sources=_ENTITY_SOURCES)
    for s in sections:
        _, slug = s.section_id.split("#", 1)
        assert s.concept_slug == slug, (
            f"concept_slug={s.concept_slug!r} must equal id-slug={slug!r}"
        )


def test_derive_gold_sections_is_deterministic():
    """Calling derive_gold_sections twice yields the same ordered list."""
    from eval.paraphrase_comparison.generation.sampling import derive_gold_sections

    first = [
        s.section_id
        for s in derive_gold_sections(_CORPUS_DIR, entity_sources=_ENTITY_SOURCES)
    ]
    second = [
        s.section_id
        for s in derive_gold_sections(_CORPUS_DIR, entity_sources=_ENTITY_SOURCES)
    ]
    assert first == second


def test_load_gold_sections_derives_from_corpus():
    """load_gold_sections() with corpus_dir derives sections rather than reading YAML."""
    from eval.paraphrase_comparison.generation.sampling import load_gold_sections

    sections = load_gold_sections(
        corpus_dir=_CORPUS_DIR, entity_sources=_ENTITY_SOURCES
    )
    assert len(sections) >= 39
    # No section should have warranty.md as its source
    assert all(not s.section_id.startswith("warranty.md") for s in sections)


def test_gold_section_has_no_multi_sub_fact_field():
    """GoldSection dataclass has no multi_sub_fact field (dropped in issue #142)."""
    from eval.paraphrase_comparison.generation.sampling import GoldSection

    s = GoldSection(
        section_id="returns_policy.md#return-window", concept_slug="return-window"
    )
    assert not hasattr(s, "multi_sub_fact"), (
        "multi_sub_fact was dropped (issue #142); GoldSection must not have this field"
    )
