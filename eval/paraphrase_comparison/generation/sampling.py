"""Deep module per Ousterhout. Public surface: ``GoldSection``, ``load_gold_sections``, ``derive_gold_sections``, ``sha256_order``, ``sample_sections``, ``GOLD_SECTIONS_PATH``, ``CORPUS_ENTITY_SOURCES``.

Deterministic Gold Section sampling for Paraphrase generation (PRD #100, #102).

The generator must pick which docs Gold Sections each Paraphrase Type targets
in a way that is **reproducible across machines and Python runs**. Python's
built-in ``hash()`` is salted per-process (PYTHONHASHSEED), so it is unusable as
a stable ordering key. This module orders sections by ``sha256(seed:section_id)``
instead, which is byte-stable everywhere.

Gold Sections are auto-derived by parsing the committed corpus into heading-Sections
(issue #142). The former hand-maintained ``gold_sections.yaml`` and its
``multi_sub_fact`` flag have been dropped — sub-fact narrowing is covered downstream
by the Synthesizer's context-bound evolutions (PRD #137).

The entity Source list (``CORPUS_ENTITY_SOURCES``) is the one remaining piece of
hand-knowledge: entity sources collapse into a single entity wiki page and are
never Paraphrase sources.

Two sampling rules from the issue are encoded here:

  1. Cross-type reuse is allowed: each Paraphrase Type seeds its own ordering, so
     the same section may be drawn for several types. The seed is the type name,
     so a type's selection is stable regardless of which other types ran.
  2. ``sample_sections`` draws up to ``count`` sections in sha256-keyed order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from markdown_kb.app.indexer import parse_markdown, slugify

_PKG_ROOT = Path(__file__).resolve().parent.parent
GOLD_SECTIONS_PATH = _PKG_ROOT / "gold_sections.yaml"

# Entity sources are excluded from the Gold Section pool (they collapse into
# a single entity wiki page, not concept pages, and are not Paraphrase sources).
# This is the one piece of hand-knowledge that cannot be auto-derived without
# the LLM classifier that distinguishes entity from concept Sources.
# Mirrors corpus_generator.CORPUS_ENTITY_SOURCES (issue #143: acme_shop_about is
# an entity "about" page, not a customer-support retrieval target). The online
# /ingest fixtures use the production LLM classifier (classify_source, #106), so
# this static set is the gold-derivation contract — the post-build --qc-only
# check verifies the fixtures actually cover every Gold Section before spend.
CORPUS_ENTITY_SOURCES: frozenset[str] = frozenset({"warranty.md", "acme_shop_about.md"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoldSection:
    """A docs Gold Section eligible to source a Paraphrase.

    ``section_id`` is the ``{source-filename}#{heading-slug}`` docs id (the same
    form ``Paraphrase.gold_docs_section_id`` carries). ``concept_slug`` is the
    1:1 concept Wiki Page slug ``/ingest`` synthesised for it (the Paraphrase
    source surface — entity pages are excluded).

    Note: ``multi_sub_fact`` was removed in issue #142. Sub-fact narrowing for
    ``specificity_narrowing`` is now handled by the Synthesizer's context-bound
    evolutions rather than a hand-maintained per-section flag (PRD #137).
    """

    section_id: str
    concept_slug: str


# ---------------------------------------------------------------------------
# Inventory derivation
# ---------------------------------------------------------------------------
def derive_gold_sections(
    corpus_dir: Path,
    entity_sources: frozenset[str] | set[str] = CORPUS_ENTITY_SOURCES,
) -> list[GoldSection]:
    """Derive the Gold Section inventory by parsing the corpus into heading-Sections.

    Replaces the former hand-maintained ``gold_sections.yaml`` (issue #142).
    For each ``*.md`` file in ``corpus_dir`` that is NOT in ``entity_sources``,
    parse all body-bearing Sections and emit one ``GoldSection`` per Section.

    ``entity_sources`` is the set of filenames (basenames) whose sections should
    be excluded — they collapse into entity wiki pages, not concept pages, and
    are never Paraphrase sources (e.g. ``{"warranty.md"}``).

    Returns sections sorted by ``{filename}#{slug}`` so the order is stable and
    auditable (matches the order produced by the former YAML). Raises on a missing
    or unreadable corpus directory (fail-fast per CODING_STANDARD §4.1).
    """
    sections: list[GoldSection] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        if md_file.name in entity_sources:
            continue
        for section in parse_markdown(md_file, source_id=None):
            if not section.content.strip():
                continue
            slug = slugify(section.heading)
            sections.append(
                GoldSection(
                    section_id=f"{md_file.name}#{slug}",
                    concept_slug=slug,
                )
            )
    return sections


# ---------------------------------------------------------------------------
# Inventory loading (legacy YAML path preserved for backward compat)
# ---------------------------------------------------------------------------
def load_gold_sections(
    path: Path = GOLD_SECTIONS_PATH,
    *,
    corpus_dir: Path | None = None,
    entity_sources: frozenset[str] | set[str] = CORPUS_ENTITY_SOURCES,
) -> list[GoldSection]:
    """Return the Gold Section inventory.

    With ``corpus_dir`` provided: auto-derives the inventory by parsing the
    corpus (issue #142; preferred path). With ``corpus_dir=None`` (default):
    falls back to the legacy ``gold_sections.yaml`` for callers that have not
    yet been migrated — the YAML path is preserved for backward compatibility
    with the committed queries.yaml whose Gold Section ids were derived from
    the YAML inventory.

    Raises on a missing required field (YAML path) or an unreadable corpus
    (corpus path) rather than silently dropping a section (CODING_STANDARD §4.1).
    """
    if corpus_dir is not None:
        return derive_gold_sections(corpus_dir, entity_sources=entity_sources)

    # Legacy YAML path (backward compat — migrate callers to corpus_dir=).
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections: list[GoldSection] = []
    for entry in data.get("gold_sections", []):
        sections.append(
            GoldSection(
                section_id=entry["section_id"],
                concept_slug=entry["concept_slug"],
            )
        )
    return sections


# ---------------------------------------------------------------------------
# Deterministic ordering + sampling
# ---------------------------------------------------------------------------
def sha256_order(sections: list[GoldSection], seed: str) -> list[GoldSection]:
    """Return ``sections`` ordered by ``sha256(seed + ":" + section_id)``.

    The digest is a hex string; lexicographic ordering of equal-length hex
    digests is a stable, uniform shuffle keyed by ``seed``. Using sha256 (not
    Python ``hash()``) makes the order identical across machines and runs
    regardless of ``PYTHONHASHSEED`` — the determinism the generator relies on
    so a re-run reproduces the same ``queries.yaml`` selection.
    """
    return sorted(sections, key=lambda s: _digest(seed, s.section_id))


def sample_sections(
    sections: list[GoldSection],
    seed: str,
    count: int,
) -> list[GoldSection]:
    """Deterministically draw up to ``count`` Gold Sections for one Paraphrase Type.

    ``seed`` is the Paraphrase Type name, so each type gets its own stable
    ordering and cross-type reuse falls out naturally (the same section may be
    the top draw for several types).

    Returns fewer than ``count`` only when the pool is smaller than ``count``;
    never raises on an over-large ``count``.

    Note: the former ``multi_sub_fact_only`` parameter was removed in issue #142
    (the multi_sub_fact flag was dropped; sub-fact narrowing is now handled by
    the Synthesizer's context-bound evolutions, PRD #137).
    """
    return sha256_order(list(sections), seed)[:count]


def _digest(seed: str, section_id: str) -> str:
    """sha256 hex digest of ``"{seed}:{section_id}"`` (stable cross-process key)."""
    return hashlib.sha256(f"{seed}:{section_id}".encode()).hexdigest()
