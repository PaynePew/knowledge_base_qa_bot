"""Shallow module per Ousterhout. Public surface: ``ParaphraseDraft``, ``to_paraphrase``.

Structured-output schema for the Paraphrase generator (PRD #100, issue #102).

``ParaphraseDraft`` is the Pydantic schema bound to gpt-4o-mini via
``with_structured_output`` (ADR-0005 structured-output adapter pattern). The LLM
fills only the parts it can know from the prompt — the rewritten ``text`` and the
dual-side Key Tokens it judges distinctive. The generator owns the bookkeeping
fields (``paraphrase_id``, ``paraphrase_type``, ``gold_docs_section_id``) and
stitches them onto the draft via ``to_paraphrase`` to produce the canonical
``Paraphrase`` (whose exact field names ``loader.py`` reads).

This schema mirrors ``models.Paraphrase`` field names verbatim so there is no
translation drift (CONTEXT.md § Phase 8 > Paraphrase, Key Tokens).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Paraphrase, ParaphraseType


class ParaphraseDraft(BaseModel):
    """The LLM-supplied portion of a Paraphrase (the rest is generator-owned).

    Field names match ``Paraphrase`` exactly. ``key_tokens_docs`` are distinctive
    tokens drawn from the docs Gold Section wording; ``key_tokens_wiki`` from the
    concept Wiki Page wording — dual-side so either Stack's surface vocabulary can
    match (CONTEXT.md § Phase 8 > Key Tokens). ``generation_notes`` records the
    sub-fact targeted (specificity_narrowing) or the subject stripped
    (implicit_reference) for the human PR-review surface.
    """

    text: str = Field(description="The rewritten query variant.")
    key_tokens_docs: list[str] = Field(
        description="Distinctive tokens from the docs Gold Section wording.",
        min_length=1,
    )
    key_tokens_wiki: list[str] = Field(
        description="Distinctive tokens from the concept Wiki Page wording.",
        min_length=1,
    )
    generation_notes: str = Field(
        default="",
        description="Optional note: targeted sub-fact / stripped subject / etc.",
    )


def to_paraphrase(
    draft: ParaphraseDraft,
    *,
    paraphrase_id: str,
    paraphrase_type: ParaphraseType,
    gold_docs_section_id: str,
) -> Paraphrase:
    """Stitch a generator-owned id/type/gold onto an LLM ``ParaphraseDraft``.

    Produces the canonical ``Paraphrase`` the loader round-trips. Keeps the
    bookkeeping fields out of the LLM's structured output so the model cannot
    hallucinate a wrong Gold Section id or invent a Paraphrase Type.
    """
    return Paraphrase(
        paraphrase_id=paraphrase_id,
        paraphrase_type=paraphrase_type,
        text=draft.text,
        gold_docs_section_id=gold_docs_section_id,
        key_tokens_docs=list(draft.key_tokens_docs),
        key_tokens_wiki=list(draft.key_tokens_wiki),
        generation_notes=draft.generation_notes,
    )
