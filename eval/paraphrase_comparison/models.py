"""Shallow module per Ousterhout. Public surface: ``Paraphrase``, ``RetrievedItem``, ``PARAPHRASE_TYPES``.

Domain data shapes for the Phase 8 retrieval comparison (CONTEXT.md § Phase 8
vocabulary, PRD #100). These are plain data carriers; all comparison logic
lives in the deep ``metric`` / ``stacks`` / ``runner`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

# CONTEXT.md § Phase 8 > Paraphrase Type — Core (LLM-generated) + Structural
# probes (hand-written). Slice 1 (tracer) exercises only synonym_swap; the full
# set is declared here so later slices add Paraphrases without re-introducing
# the vocabulary. ``ParaphraseType`` is the PRD #100 Literal; ``PARAPHRASE_TYPES``
# is its runtime tuple form for iteration/validation.
ParaphraseType = Literal[
    "synonym_swap",
    "word_reorder",
    "verbosity_expansion",
    "specificity_narrowing",
    "implicit_reference",
    "typo_fatfinger",
    "industry_jargon",
]
PARAPHRASE_TYPES: tuple[str, ...] = get_args(ParaphraseType)


@dataclass(frozen=True)
class Paraphrase:
    """A query variant probing retrieval robustness (CONTEXT.md § Phase 8 > Paraphrase).

    ``gold_docs_section_id`` is the docs Gold Section the Paraphrase should
    retrieve (``{source-filename}#{heading-slug}``). ``key_tokens_docs`` and
    ``key_tokens_wiki`` are the dual-side Key Tokens the C5c hit metric uses to
    confirm the retrieved *content* answers the Paraphrase (CONTEXT.md
    § Phase 8 > Key Tokens).
    """

    paraphrase_id: str
    paraphrase_type: ParaphraseType
    text: str
    gold_docs_section_id: str
    key_tokens_docs: list[str]
    key_tokens_wiki: list[str]
    generation_notes: str = ""

    @property
    def key_tokens(self) -> set[str]:
        """Union of the dual-side Key Tokens (CONTEXT.md § Phase 8 > Key Tokens)."""
        return {t.lower() for t in (*self.key_tokens_docs, *self.key_tokens_wiki)}


@dataclass(frozen=True)
class RetrievedItem:
    """A Retrieval Stack's hit, normalised to a docs-Section-granular shape.

    Both Stacks resolve their native unit to a common shape so the C5c metric is
    Stack-agnostic: ``source_section_id`` is the docs Gold Section id a hit maps
    to (Stack B's Chunk carries it directly; Stack A's Wiki Section resolves it
    via the page's ``sources`` frontmatter). ``content`` is the retrieved text
    whose tokens the metric overlaps against the Paraphrase's Key Tokens.
    """

    source_section_id: str
    content: str
    heading_path: list[str] = field(default_factory=list)
