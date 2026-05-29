"""Medium module per Ousterhout. Public surface: ``classify_source``, ``generate_page``, ``generate_entity_page``.

LLM synthesis templates for wiki page generation.

Provides:
- `classify_source(content)` — LLM classifier returning SourceType
  ("entity" | "concept").
- `generate_page(section, source_type)` — concept-page synthesis for one
  Section.
- `generate_entity_page(sections, source_stem)` — entity-page synthesis
  collapsing an entire Source into a single page.

All LangChain types are confined to this module (CODING_STANDARD §2.4).
LLM calls use `with_structured_output` per ADR-0005.

See PRD #28 for the ingest pipeline design.
"""

from __future__ import annotations

import datetime
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .indexer import Section, slugify
from .schemas import SourceType, WikiPageDraft, WikiPageFrontmatter

# ---------------------------------------------------------------------------
# Ingest LLM singleton
# ---------------------------------------------------------------------------

_ingest_llm = None


def get_ingest_llm() -> ChatOpenAI:
    """Return a lazy singleton ChatOpenAI for ingest synthesis.

    Model resolution (two-layer fallback per Slice #1 AC):
        OPENAI_INGEST_MODEL  →  OPENAI_MODEL  →  gpt-4o-mini
    """
    global _ingest_llm
    if _ingest_llm is None:
        model_name = os.getenv(
            "OPENAI_INGEST_MODEL",
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
        _ingest_llm = ChatOpenAI(
            model=model_name,
            timeout=60,
            max_retries=1,
        )
    return _ingest_llm


# ---------------------------------------------------------------------------
# Structured output schemas (module-private)
# ---------------------------------------------------------------------------


class _PageSynthesisOutput(BaseModel):
    """Structured output schema for the concept/entity synthesis LLM call.

    Kept module-private — callers receive a `WikiPageDraft`, not this type.
    The `body` field holds the LLM-synthesised prose; `open_questions` captures
    any explicit open questions the LLM surfaces (may be empty).
    """

    body: str
    open_questions: list[str]


class _ClassifierOutput(BaseModel):
    """Structured output schema for the source-type classifier call.

    Kept module-private — callers receive a plain `SourceType` string.
    """

    type: SourceType


# ---------------------------------------------------------------------------
# Prompt templates — concept
# ---------------------------------------------------------------------------

_CONCEPT_SYSTEM_PROMPT = """\
You are a knowledge-base curator. Your task is to write a concise, accurate \
synthesis wiki page for the given Source section.

Rules:
- Write the page in the **same language as the Source**. If the Source is in \
Chinese, write the page in Chinese. If the Source is in English, write in \
English. Do not translate.
- Write a short, clear prose summary of the section content.
- Stay faithful to the source — do not add facts not present in the source.
- In `open_questions`, list any genuinely ambiguous or unanswered aspects \
apparent from the source. Leave the list empty if there are none.
- Keep the body to 2–4 sentences; quality over quantity.

Red link rule: When you mention a concept warranting its own page that the \
wiki may not yet cover, use Obsidian-style [[concept-slug]] wikilinks. The \
slug must use the same convention as wiki page filenames: for ASCII concepts \
use lowercase, hyphen-separated slugs (e.g. [[return-policy]]); for CJK / \
non-ASCII concepts keep the characters verbatim as they appear in the \
slugified heading (e.g. [[退款政策]]). do NOT verify whether the page actually \
exists — it is intentional that some links are unresolved ("red links"). \
Constraints: maximum 5 wikilinks per page; do NOT use for common terms \
(e.g. [[customer]], [[refund]] when the whole page is about refunds); only for \
concepts that warrant their own page (entity / concept).
"""


def _build_concept_user_message(section: Section) -> str:
    """Format a Section as the user message for the concept synthesis call."""
    heading = " > ".join(section.heading_path)
    return f"[Source: {section.id}]\nHeading: {heading}\n\n{section.content}"


# ---------------------------------------------------------------------------
# Prompt templates — entity
# ---------------------------------------------------------------------------

_ENTITY_SYSTEM_PROMPT = """\
You are a knowledge-base curator. Your task is to write a concise, accurate \
synthesis wiki page for the given Source, treating it as an *entity* (a \
product, person, place, or named thing that the KB describes as a whole).

Rules:
- Write the page in the **same language as the Source**. If the Source is in \
Chinese, write the page in Chinese. If the Source is in English, write in \
English. Do not translate.
- Write a short, clear prose summary covering the entire Source.
- Stay faithful to the source — do not add facts not present in the source.
- In `open_questions`, list any genuinely ambiguous or unanswered aspects \
apparent from the source. Leave the list empty if there are none.
- Keep the body to 3–6 sentences; quality over quantity.

Red link rule: When you mention a concept warranting its own page that the \
wiki may not yet cover, use Obsidian-style [[concept-slug]] wikilinks. The \
slug must use the same convention as wiki page filenames: for ASCII concepts \
use lowercase, hyphen-separated slugs (e.g. [[return-policy]]); for CJK / \
non-ASCII concepts keep the characters verbatim as they appear in the \
slugified heading (e.g. [[退款政策]]). do NOT verify whether the page actually \
exists — it is intentional that some links are unresolved ("red links"). \
Constraints: maximum 5 wikilinks per page; do NOT use for common terms \
(e.g. [[customer]], [[product]] when the whole page is about that product); \
only for concepts that warrant their own page (entity / concept).
"""


def _build_entity_user_message(sections: list[Section], source_stem: str) -> str:
    """Format all Sections of a Source as the user message for entity synthesis."""
    parts = [f"[Entity source: {source_stem}]"]
    for section in sections:
        heading = " > ".join(section.heading_path)
        parts.append(f"\n## {heading}\n\n{section.content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt templates — classifier
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a knowledge-base classifier. Given the full Markdown content of one \
Source document, decide whether it describes:
- "concept": a policy, process, how-to guide, or FAQ (multiple independent \
  topics, each section stands alone)
- "entity": a single named thing (product, service, person, organisation, \
  place) described from multiple angles

Respond with a single JSON object: {"type": "concept"} or {"type": "entity"}.
"""


def _build_classifier_user_message(content: str) -> str:
    """Format Source Markdown content as the user message for classification."""
    return f"Classify this Source document:\n\n{content}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_source(content: str) -> SourceType:
    """Classify a Source document as "entity" or "concept".

    Uses the ingest LLM with `with_structured_output` bound to
    `_ClassifierOutput`.  Returns the plain SourceType string so callers
    never touch LangChain types (CODING_STANDARD §2.4).

    Args:
        content: Full Markdown text of the Source document.

    Returns:
        "entity" or "concept".
    """
    llm = get_ingest_llm()
    chain = llm.with_structured_output(_ClassifierOutput)
    output: _ClassifierOutput = chain.invoke(
        [
            SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=_build_classifier_user_message(content)),
        ]
    )
    return output.type


def generate_page(
    section: Section,
    source_type: SourceType,
    *,
    now: datetime.datetime | None = None,
) -> WikiPageDraft:
    """Generate a WikiPageDraft for one Section (concept path).

    Uses LangChain `with_structured_output` bound to `_PageSynthesisOutput`
    so the body and open_questions arrive as structured fields (no brittle
    string parsing).  LangChain types are confined to this module per
    CODING_STANDARD §2.4.

    Args:
        section:     The Section to synthesise.
        source_type: "concept" (this function) or "entity" (use
                     `generate_entity_page` for that path).
        now:         Override UTC timestamp for testing.  If None, uses
                     ``datetime.datetime.now(datetime.UTC)``.

    Returns:
        A fully-populated WikiPageDraft ready for wiki_writer.write_pages_for_source.
    """
    if now is None:
        now = datetime.datetime.now(datetime.UTC)

    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    slug = slugify(section.heading)
    citation_id = f"{section.file}#{slug}"

    llm = get_ingest_llm()
    chain = llm.with_structured_output(_PageSynthesisOutput)

    output: _PageSynthesisOutput = chain.invoke(
        [
            SystemMessage(content=_CONCEPT_SYSTEM_PROMPT),
            HumanMessage(content=_build_concept_user_message(section)),
        ]
    )

    frontmatter = WikiPageFrontmatter(
        id=slug,
        type=source_type,
        created=ts,
        updated=ts,
        sources=[citation_id],
        status="live",
        open_questions=output.open_questions,
    )

    return WikiPageDraft(
        frontmatter=frontmatter,
        body=output.body,
        citation_line=f"[Source: {citation_id}]",
        slug=slug,
        heading=section.heading,
    )


def generate_entity_page(
    sections: list[Section],
    source_stem: str,
    source_filename: str,
    *,
    now: datetime.datetime | None = None,
) -> WikiPageDraft:
    """Generate a single WikiPageDraft collapsing all Sections of an entity Source.

    Called when `classify_source` returns "entity".  Produces one page at
    ``wiki/entities/<source_stem>.md`` (CODING_STANDARD §2.4 — no LangChain
    types escape this module).

    Args:
        sections:        All Sections parsed from the Source.
        source_stem:     Filename stem (no extension), used as the page slug.
        source_filename: Bare filename (e.g. "my_entity.md"), used in
                         frontmatter sources list and citation line.
        now:             Override UTC timestamp for testing.

    Returns:
        A WikiPageDraft with type="entity", slug=source_stem, and all section
        citation IDs in frontmatter.sources.
    """
    if now is None:
        now = datetime.datetime.now(datetime.UTC)

    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    slug = slugify(source_stem)

    # Build source citation list: one entry per section (same convention as concept)
    citation_ids = [f"{source_filename}#{slugify(s.heading)}" for s in sections]

    llm = get_ingest_llm()
    chain = llm.with_structured_output(_PageSynthesisOutput)

    output: _PageSynthesisOutput = chain.invoke(
        [
            SystemMessage(content=_ENTITY_SYSTEM_PROMPT),
            HumanMessage(content=_build_entity_user_message(sections, source_stem)),
        ]
    )

    frontmatter = WikiPageFrontmatter(
        id=slug,
        type="entity",
        created=ts,
        updated=ts,
        sources=citation_ids,
        status="live",
        open_questions=output.open_questions,
    )

    # Citation line lists the source file
    citation_line = f"[Source: {source_filename}]"

    # Entity pages derive their heading from the Source filename stem.
    # Underscores and hyphens map to spaces; title-case is applied so
    # ``acme_shop_about`` -> ``Acme Shop About``. This is still ASCII-biased
    # (the entity case has no Section heading to fall back on), but the
    # lossiness now lives where the input is known rather than in the writer.
    heading = source_stem.replace("_", " ").replace("-", " ").title()

    return WikiPageDraft(
        frontmatter=frontmatter,
        body=output.body,
        citation_line=citation_line,
        slug=slug,
        heading=heading,
    )
