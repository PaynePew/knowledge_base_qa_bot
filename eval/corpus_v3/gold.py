"""Deep module per Ousterhout. Public surface: ``GoldMap``, ``build_gold_map``, ``resolve_gold_sections``.

Corpus-neutral gold-label mapping (ADR-0045 Prerequisite 3, PRD #654 user
story 3). The v2 harness's gold labels are docs-native ids
(``eval.paraphrase_comparison.models.Paraphrase.gold_docs_section_id``) and
its wiki-side resolution (``eval.paraphrase_comparison.stacks
._wiki_slug_to_gold_section``) only scans ``wiki/concepts/`` and keeps just
the FIRST ``sources:`` frontmatter entry — a wiki entity page (1:N Sources,
``type: entity``) has no entry in that map at all, so a correct entity-page
hit falls back to the page's own wiki id and can never equal a docs-native
gold id: a STRUCTURAL miss regardless of retrieval quality
(``eval/fairness_review/verdict.md`` § Query provenance).

This module removes that tilt with a single symmetric table: every wiki page
(concept OR entity) maps to the FULL set of docs-native Section ids its
``sources:`` frontmatter lists (1:N allowed). ``resolve_gold_sections`` looks
a retrieved item's id up in that table from EITHER direction — a docs-native
id or a wiki id, concept or entity — through the SAME public function, and
returns the full equivalence class of ids that count as a hit for that gold
answer. No stack's native id space needs special-casing.

Table format ("maintained with the fixtures", PRD #654): every wiki page
under ``wiki/{concepts,entities}/*.md`` already carries a ``sources:`` YAML
list in its frontmatter — the production ``/ingest`` output writes exactly
this (see ``wiki/entities/acme-shop.md`` for a 1:N example). ``build_gold_map``
is the sole reader of that convention, so the fixture markdown itself IS the
mapping table; there is no separate hand-maintained file to drift out of sync
with the corpus.
"""

from __future__ import annotations

import re
from pathlib import Path

# Frontmatter scan (mirrors eval.paraphrase_comparison.stacks's regex pair,
# duplicated rather than imported — PRD #654: corpus v3 stays isolated from
# the v2 eval package).
_SOURCES_BLOCK_RE = re.compile(
    r"^sources:\s*\n((?:[ \t]*-[ \t]*\S+.*\n)+)", re.MULTILINE
)
_SOURCE_ITEM_RE = re.compile(r"^[ \t]*-[ \t]*(\S+)", re.MULTILINE)

# Wiki page id -> the full set of docs-native Section ids its `sources:`
# frontmatter lists (1:N; entity pages included).
GoldMap = dict[str, frozenset[str]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_gold_map(wiki_dir: Path) -> GoldMap:
    """Scan every wiki page (concepts AND entities) into ``{wiki_id: docs_ids}``.

    Unlike v2's ``_wiki_slug_to_gold_section`` (concepts only, first source
    only), this keeps every page under EITHER subdir and its FULL ``sources:``
    list, so 1:N entity pages are represented instead of silently dropped. A
    page whose frontmatter carries no ``sources:`` block is skipped.
    """
    mapping: GoldMap = {}
    for subdir in ("concepts", "entities"):
        pages_dir = wiki_dir / subdir
        if not pages_dir.is_dir():
            continue
        for page in sorted(pages_dir.glob("*.md")):
            sources = _parse_sources(page.read_text(encoding="utf-8"))
            if sources:
                mapping[page.stem] = sources
    return mapping


def resolve_gold_sections(gold_map: GoldMap, section_id: str) -> frozenset[str]:
    """Return every id that counts as a hit for ``section_id``'s gold answer.

    ``section_id`` may be a docs-native id OR a wiki id (concept or entity) —
    the SAME function handles both directions:

      - if it names a wiki page in ``gold_map``, the result is that page's own
        id plus every docs-native Section id it sources from (1:N, entities
        included);
      - otherwise it is treated as a docs-native id: the result is that id
        plus every wiki page whose ``sources:`` list contains it (usually one
        concept page, but an entity page may ALSO cover it).

    Either direction always includes ``section_id`` itself, so a stack whose
    native id space already matches the gold id (e.g. Stack B's docs-native
    chunks, or an id absent from the table entirely) still resolves without
    the table adding or removing anything.
    """
    if section_id in gold_map:
        return frozenset({section_id}) | gold_map[section_id]
    covering_wiki_ids = {
        wiki_id for wiki_id, docs_ids in gold_map.items() if section_id in docs_ids
    }
    return frozenset({section_id}) | covering_wiki_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_sources(markdown: str) -> frozenset[str]:
    block = _SOURCES_BLOCK_RE.search(markdown)
    if not block:
        return frozenset()
    return frozenset(_SOURCE_ITEM_RE.findall(block.group(1)))
