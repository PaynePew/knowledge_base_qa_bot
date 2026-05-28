"""Deep module per Ousterhout. Public surface: ``GoldSection``, ``load_gold_sections``, ``sha256_order``, ``sample_sections``, ``GOLD_SECTIONS_PATH``.

Deterministic Gold Section sampling for Paraphrase generation (PRD #100, #102).

The generator must pick which docs Gold Sections each Paraphrase Type targets
in a way that is **reproducible across machines and Python runs**. Python's
built-in ``hash()`` is salted per-process (PYTHONHASHSEED), so it is unusable as
a stable ordering key. This module orders sections by ``sha256(seed:section_id)``
instead, which is byte-stable everywhere.

Two sampling rules from the issue are encoded here:

  1. ``specificity_narrowing`` may target ONLY multi-sub-fact Gold Sections (a
     section with a single fact has no high-distinctiveness sub-fact to narrow
     to). ``sample_sections(..., multi_sub_fact_only=True)`` enforces this.
  2. Cross-type reuse is allowed: each Paraphrase Type seeds its own ordering, so
     the same section may be drawn for several types. The seed is the type name,
     so a type's selection is stable regardless of which other types ran.

The Gold Section inventory (which docs sections exist and which are multi-sub-fact)
is committed as ``gold_sections.yaml`` so the sampling is auditable and does not
depend on re-parsing the corpus at generation time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

_PKG_ROOT = Path(__file__).resolve().parent.parent
GOLD_SECTIONS_PATH = _PKG_ROOT / "gold_sections.yaml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoldSection:
    """A docs Gold Section eligible to source a Paraphrase.

    ``section_id`` is the ``{source-filename}#{heading-slug}`` docs id (the same
    form ``Paraphrase.gold_docs_section_id`` carries). ``concept_slug`` is the
    1:1 concept Wiki Page slug ``/ingest`` synthesised for it (the Paraphrase
    source surface — entity pages are excluded). ``multi_sub_fact`` marks a
    section that carries several distinct retrievable facts, so a
    ``specificity_narrowing`` Paraphrase can narrow to one high-distinctiveness
    sub-fact.
    """

    section_id: str
    concept_slug: str
    multi_sub_fact: bool


# ---------------------------------------------------------------------------
# Inventory loading
# ---------------------------------------------------------------------------
def load_gold_sections(path: Path = GOLD_SECTIONS_PATH) -> list[GoldSection]:
    """Parse the committed ``gold_sections.yaml`` inventory into ``GoldSection`` objects.

    Raises on a missing required field rather than silently dropping a section —
    a corrupt inventory is a fail-fast condition for generation (mirrors the
    loader's fail-fast on a corrupt query set, CODING_STANDARD §4.1).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections: list[GoldSection] = []
    for entry in data.get("gold_sections", []):
        sections.append(
            GoldSection(
                section_id=entry["section_id"],
                concept_slug=entry["concept_slug"],
                multi_sub_fact=bool(entry["multi_sub_fact"]),
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
    *,
    multi_sub_fact_only: bool = False,
) -> list[GoldSection]:
    """Deterministically draw up to ``count`` Gold Sections for one Paraphrase Type.

    ``seed`` is the Paraphrase Type name, so each type gets its own stable
    ordering and cross-type reuse falls out naturally (the same section may be
    the top draw for several types). When ``multi_sub_fact_only`` is set, only
    multi-sub-fact sections are eligible — the ``specificity_narrowing`` rule
    (a single-fact section has no sub-fact to narrow to).

    Returns fewer than ``count`` only when the eligible pool is smaller than
    ``count``; never raises on an over-large ``count``.
    """
    pool = (
        [s for s in sections if s.multi_sub_fact]
        if multi_sub_fact_only
        else list(sections)
    )
    return sha256_order(pool, seed)[:count]


def _digest(seed: str, section_id: str) -> str:
    """sha256 hex digest of ``"{seed}:{section_id}"`` (stable cross-process key)."""
    return hashlib.sha256(f"{seed}:{section_id}".encode()).hexdigest()
