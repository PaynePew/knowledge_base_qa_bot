"""Medium module per Ousterhout. Public surface: ``classify_source``, ``generate_page``, ``generate_entity_page``, ``generate_hub_page``, ``ingest_model_context_window``.

LLM synthesis templates for wiki page generation.

Provides:
- `classify_source(content)` — LLM classifier returning SourceType
  ("entity" | "concept").
- `generate_page(section, source_type)` — concept-page synthesis for one
  Section.
- `generate_entity_page(sections, source_stem)` — entity-page synthesis
  collapsing an entire Source into a single page.
- `generate_hub_page(sections, source_stem, source_filename, chapters)` —
  Hub Page synthesis for a Longform Source (ADR-0033 decision 3, issue #513):
  an entity-style "about this document" page with a programmatically
  appended wikilink list to every chapter page.

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

# Context-window token budgets (input limits) for known ingest models, taken from
# the provider model reference.  The per-Source ingest budget is *derived* from
# this (see ingest._max_ingest_tokens) so swapping OPENAI_INGEST_MODEL re-scales
# the budget instead of silently mismatching a frozen literal.  An unknown model
# falls back to _FALLBACK_CONTEXT_WINDOW — deliberately small, so an unrecognised
# model under-fills rather than overflows its real window.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-3.5-turbo": 16_385,
}
_FALLBACK_CONTEXT_WINDOW = 32_000


def _ingest_model_name() -> str:
    """Resolve the configured ingest model name — single source of truth.

    Resolution (two-layer fallback per Slice #1 AC):
        OPENAI_INGEST_MODEL  →  OPENAI_MODEL  →  gpt-4o-mini
    Read at call time so a restart-free env change takes effect on the next call.
    """
    return os.getenv("OPENAI_INGEST_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def ingest_model_context_window() -> int:
    """Return the context-window token budget for the configured ingest model.

    Looks the resolved model name up in ``_MODEL_CONTEXT_WINDOWS``; an unknown
    model returns ``_FALLBACK_CONTEXT_WINDOW`` (pessimistic — never over-estimate
    capacity, so an unrecognised model under-fills rather than overflows).
    Callers derive their token budgets from this; set ``KB_INGEST_MAX_TOKENS`` to
    override the derived budget outright.
    """
    return _MODEL_CONTEXT_WINDOWS.get(_ingest_model_name(), _FALLBACK_CONTEXT_WINDOW)


def get_ingest_llm() -> ChatOpenAI:
    """Return a lazy singleton ChatOpenAI for ingest synthesis.

    Model resolution delegates to ``_ingest_model_name`` (the single source of
    truth shared with ``ingest_model_context_window``).

    Pinned to ``temperature=0`` for deterministic, faithful curation: re-running
    ingest over the same Sources reproduces the same wiki layer (clean diffs,
    reproducible baked seed), and the Source-type classification step in
    particular stays stable. Mirrors the answer-path determinism fix (PR #283).
    """
    global _ingest_llm
    if _ingest_llm is None:
        _ingest_llm = ChatOpenAI(
            model=_ingest_model_name(),
            temperature=0,
            timeout=60,
            max_retries=int(os.getenv("KB_INGEST_MAX_RETRIES", "5")),
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
# Prompt templates — hub (ADR-0033 decision 3, issue #513)
# ---------------------------------------------------------------------------

_HUB_SYSTEM_PROMPT = """\
You are a knowledge-base curator. Your task is to write a concise "about \
this document" orientation page for a Longform Source (e.g. a book or long \
report), read here as a whole: what it is, its overall themes, and how its \
content is organised into chapters.

Rules:
- Write the page in the **same language as the Source**. If the Source is in \
Chinese, write the page in Chinese. If the Source is in English, write in \
English. Do not translate.
- Write a short, clear prose orientation covering the document as a whole — \
what it is about and how it is structured. This is a GLOBAL summary, not a \
chapter-by-chapter recap.
- Stay faithful to the source — do not add facts not present in the source.
- In `open_questions`, list any genuinely ambiguous or unanswered aspects \
apparent from the source. Leave the list empty if there are none.
- Keep the body to 3–6 sentences; quality over quantity.
- Do NOT use [[wikilinks]] yourself — the chapter list is appended after \
your prose by the pipeline, not by you.
"""


# ---------------------------------------------------------------------------
# Outline builder
# ---------------------------------------------------------------------------


def build_outline(content: str, *, max_tokens: int = 2000) -> str:
    """Build a compact outline from Source Markdown content.

    Returns ALL heading lines (lines that start with 1–6 ``#`` characters
    followed by a space) PLUS the first ``max_tokens * 3`` characters of
    non-heading body text, joined together.  This bounds the classifier's
    input while preserving structural context.

    LangChain-free — pure string work only (CODING_STANDARD §2.4).

    Args:
        content:    Full Markdown content of the Source (frontmatter stripped).
        max_tokens: Approximate token budget for body text.  The body char
                    limit is ``max_tokens * 3`` (inverse of ``_estimate_tokens``
                    //3 rounding, i.e. safe upper-bound for that estimate).

    Returns:
        A string with all headings preserved and body truncated to the char
        limit.  Headings always appear even when the body limit is 0.

    Note:
        Only ATX headings (1-6 ``#`` then a space) are recognised, matching the
        canonical Section parser (``indexer.HEADING_RE``).  Setext headings (a
        line underlined by ``===``/``---``) are not detected and their underline
        counts toward the body budget; real Sources in this KB use ATX, so
        classification is unaffected.
    """
    # Rule 1: every line starting with 1-6 # followed by a space is a heading.
    heading_lines: list[str] = []
    body_chars: list[str] = []
    body_budget = max_tokens * 3

    for line in content.splitlines(keepends=True):
        stripped = line.lstrip()
        # Rule 1: heading detection
        if stripped and stripped[0] == "#":
            # Count leading hashes
            i = 0
            while i < len(stripped) and stripped[i] == "#":
                i += 1
            if i <= 6 and len(stripped) > i and stripped[i] == " ":
                heading_lines.append(line.rstrip("\n"))
                continue
        # Rule 2: body text — collect up to body_budget chars
        if body_budget > 0:
            chunk = line[:body_budget]
            body_chars.append(chunk)
            body_budget -= len(chunk)

    # Headings first, then body
    parts = heading_lines + ["".join(body_chars)]
    return "\n".join(p for p in parts if p)


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

    The classifier receives ``build_outline(content)`` rather than the full
    Source text.  This bounds the LLM context usage: all headings are
    preserved (structural signal for classification) while body text is
    truncated to the first ~6 000 chars (2 000 tokens × 3 chars/token).
    Callers need not strip content before passing it here.

    Args:
        content: Full Markdown text of the Source document (frontmatter
                 already stripped by the caller in ingest.py).

    Returns:
        "entity" or "concept".
    """
    outline = build_outline(content)
    llm = get_ingest_llm()
    chain = llm.with_structured_output(_ClassifierOutput)
    output: _ClassifierOutput = chain.invoke(
        [
            SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=_build_classifier_user_message(outline)),
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


def generate_hub_page(
    sections: list[Section],
    source_stem: str,
    source_filename: str,
    chapters: list[tuple[str, str]],
    *,
    now: datetime.datetime | None = None,
) -> WikiPageDraft:
    """Generate the Hub Page for a Longform Source (ADR-0033 decision 3, issue #513).

    An entity-style "about this document" page — same synthesis shape as
    ``generate_entity_page`` (whole-Source prose covering all Sections) — with
    a chapter wikilink list appended AFTER the LLM prose, not generated by it.
    Appending the list programmatically (rather than trusting the LLM to
    mention every chapter within its own 5-wikilink-per-page style guidance)
    is what guarantees the acceptance criterion "the hub's wikilinks resolve
    to every chapter page (no Red Links among them)": every entry in
    ``chapters`` is a slug the caller has ALREADY collision-resolved for a
    chapter page in the same ingest batch, so every hub wikilink is
    resolvable by construction.

    The chapter list uses a bold label (``**Chapters:**``) rather than an ATX
    heading so the Hub Page stays a single Section when the wiki layer is
    itself re-indexed by ``indexer.parse_markdown`` (an ATX heading inside the
    body would split the page into two Sections at index time).

    Args:
        sections:        All Sections parsed from the Source (post Structure
                          Enrichment — each Section is one chapter).
        source_stem:      Filename stem (no extension), used as the page slug
                          basis (same convention as ``generate_entity_page``).
        source_filename:  Bare filename, used in frontmatter sources list and
                          citation line.
        chapters:         ``(slug, heading)`` pairs for every chapter page, in
                          Section order, using the FINAL (collision-resolved)
                          slug the caller wrote the chapter page under.
        now:              Override UTC timestamp for testing.

    Returns:
        A WikiPageDraft with type="entity" — the caller (ingest.py) assigns
        the collision-resolved slug the same way the entity route does.
    """
    if now is None:
        now = datetime.datetime.now(datetime.UTC)

    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    slug = slugify(source_stem)

    citation_ids = [f"{source_filename}#{slugify(s.heading)}" for s in sections]

    llm = get_ingest_llm()
    chain = llm.with_structured_output(_PageSynthesisOutput)

    output: _PageSynthesisOutput = chain.invoke(
        [
            SystemMessage(content=_HUB_SYSTEM_PROMPT),
            HumanMessage(content=_build_entity_user_message(sections, source_stem)),
        ]
    )

    if chapters:
        chapter_links = "\n".join(f"- [[{chapter_slug}]]" for chapter_slug, _heading in chapters)
        body = f"{output.body}\n\n**Chapters:**\n\n{chapter_links}"
    else:
        body = output.body

    frontmatter = WikiPageFrontmatter(
        id=slug,
        type="entity",
        created=ts,
        updated=ts,
        sources=citation_ids,
        status="live",
        open_questions=output.open_questions,
    )

    citation_line = f"[Source: {source_filename}]"
    heading = source_stem.replace("_", " ").replace("-", " ").title()

    return WikiPageDraft(
        frontmatter=frontmatter,
        body=body,
        citation_line=citation_line,
        slug=slug,
        heading=heading,
    )
