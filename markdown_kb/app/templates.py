"""LLM synthesis templates for wiki page generation.

Provides `generate_page(section, source_type)` which calls the ingest LLM
(via `get_ingest_llm`) with a `with_structured_output` binding to produce a
`WikiPageDraft`.  The LLM call is isolated here so `ingest.py` stays a
coordinator that never touches LangChain types directly (CODING_STANDARD §2.4).

This slice hard-codes `source_type="concept"` (no classifier yet — Slice #2
adds `classify_source` and the `entity` path).

See PRD #28 for the ingest pipeline design and Phase 3 Slice #1 issue for the
exact frontmatter schema requirement.
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
# Structured output schema used by the synthesis call
# ---------------------------------------------------------------------------


class _PageSynthesisOutput(BaseModel):
    """Structured output schema for the concept-synthesis LLM call.

    Kept module-private — callers receive a `WikiPageDraft`, not this type.
    The `body` field holds the LLM-synthesised prose; `open_questions` captures
    any explicit open questions the LLM surfaces (may be empty).
    """

    body: str
    open_questions: list[str]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CONCEPT_SYSTEM_PROMPT = """\
You are a knowledge-base curator. Your task is to write a concise, accurate \
synthesis wiki page for the given Source section.

Rules:
- Write a short, clear prose summary of the section content.
- Stay faithful to the source — do not add facts not present in the source.
- You may use [[wikilink]] syntax to reference related concepts you know from \
the source; prefer concrete links over vague ones.
- In `open_questions`, list any genuinely ambiguous or unanswered aspects \
apparent from the source. Leave the list empty if there are none.
- Keep the body to 2–4 sentences; quality over quantity.
"""


def _build_concept_user_message(section: Section) -> str:
    """Format a Section as the user message for the concept synthesis call."""
    heading = " > ".join(section.heading_path)
    return f"[Source: {section.id}]\nHeading: {heading}\n\n{section.content}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_page(
    section: Section,
    source_type: SourceType,  # noqa: ARG001  (only "concept" in Slice #1)
    *,
    now: datetime.datetime | None = None,
) -> WikiPageDraft:
    """Generate a WikiPageDraft for one Section using the ingest LLM.

    Uses LangChain `with_structured_output` bound to `_PageSynthesisOutput`
    so the body and open_questions arrive as structured fields (no brittle
    string parsing).  LangChain types are confined to this module per
    CODING_STANDARD §2.4.

    Args:
        section:     The Section to synthesise.
        source_type: Hardcoded to "concept" in this slice (Slice #2 adds
                     "entity" and the classifier call).
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
        type="concept",
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
    )
