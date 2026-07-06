"""Deep module per Ousterhout. Public surface: ``is_longform``, ``enrich_structure``, ``EnrichmentResult``, ``get_enrichment_llm``.

Structure Enrichment (ADR-0033 decision 2, issue #512) — the pass that gives a
Longform Source the heading structure its layout never carried.

Runs at the Import/Transcribe stage, AFTER the derived Markdown body is
assembled but BEFORE it is written to ``docs/`` (``importer._process_one_source``
and ``transcriber._force_transcribe`` are the two call sites — see their
module docstrings). Gated on the structural **longform predicate**
(``is_longform``): a Source whose mechanical structure is degenerate — zero
or one heading, a dominant preamble share, or any single Section over
``KB_INGEST_MAX_SECTION_TOKENS`` — gets ONE LLM call that proposes chapter
boundaries + titles, materialized as literal ATX headings directly into the
body text (never as side metadata — see ADR-0033 § "Materializing into docs/
is the load-bearing choice"). The same pass deterministically strips *page
furniture* (running headers/footers/page numbers repeating verbatim across
the document) before the LLM call, since furniture pollutes both the
proposal input and the eventual Section corpus.

The predicate never keys on filename, byte size, or page count (ADR-0033) —
a well-headed handbook of any length bypasses this module entirely, byte-
identically: ``is_longform`` returns ``False`` and ``enrich_structure`` never
calls the LLM.

Enrichment failure (LLM error, malformed proposal, unfindable boundary
anchor) fails SOFT: ``enrich_structure`` returns the ORIGINAL body unchanged
(``enriched=False``) — the caller writes the un-enriched transcript exactly
as Import/Transcribe would have without this module, per ADR-0033's
"enrichment failure fails soft" contract. Idempotency comes from the
existing hash-skip check in both callers (compares raw-bytes SHA-256 against
the docs frontmatter's ``content_sha256``, computed BEFORE this module is
even reached) — an unchanged raw file never re-enters this module, so it is
never re-enriched or re-billed.

This is the ONE new LLM call site this issue adds (ADR-0005 § "LLM-facing
surface enumeration"): ``ChatOpenAI.with_structured_output`` proposing a
chapter outline. LangChain types are confined to this module (CODING_STANDARD
§2.4). The single authorised ``@pytest.mark.live`` smoke test lives at
``markdown_kb/tests/test_structure_enrichment_live.py``.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from . import indexer as indexer_module
from . import ingest as ingest_module
from .logger import log_event

# ---------------------------------------------------------------------------
# Longform predicate knobs (ADR-0033: "thresholds are env-tunable with named
# defaults, following the existing _KB_INGEST_* knob discipline"). The third
# predicate leg (oversized Section) reuses ingest.py's own
# KB_INGEST_MAX_SECTION_TOKENS verbatim (ingest_module.max_section_tokens())
# rather than introducing a parallel cap.
# ---------------------------------------------------------------------------

_KB_LONGFORM_MAX_HEADINGS_DEFAULT = 1  # zero or one heading => longform
_KB_LONGFORM_PREAMBLE_SHARE_DEFAULT = 0.5  # preamble share strictly above this => longform

# ADR-0033 frames longform as structure "degenerate RELATIVE TO ITS SIZE" — a
# handful-of-characters Source with zero or one heading has nothing meaningful
# to segment into chapters (rule 7's whole-file Section already serves it
# fine); below this floor the heading-count and preamble-share legs never
# fire, regardless of their own thresholds. The oversized-section leg is
# unaffected: it only ever fires above KB_INGEST_MAX_SECTION_TOKENS (default
# 6000 tokens, far above this floor), so the floor never masks a genuine case.
_KB_LONGFORM_MIN_CHARS_DEFAULT = 2000

# Deterministic page-furniture repeat-detector: a line must repeat at least
# this many times (exact match, outside code fences, not a heading, not a
# horizontal rule) to be treated as running-header/footer/page-number
# residue and stripped. See _strip_page_furniture.
_KB_STRUCTURE_FURNITURE_MIN_REPEATS_DEFAULT = 3

# Same ATX heading pattern as indexer.HEADING_RE, reimplemented locally
# (precedent: templates.build_outline does the same) so this module stays
# decoupled from indexer.py's private regex — only parse_markdown_body is
# reused, for the parts that genuinely need real Section boundaries.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")


def _max_headings_threshold() -> int:
    """Heading count at/below which a Source is longform. Override: ``KB_LONGFORM_MAX_HEADINGS``."""
    return int(os.getenv("KB_LONGFORM_MAX_HEADINGS", str(_KB_LONGFORM_MAX_HEADINGS_DEFAULT)))


def _preamble_share_threshold() -> float:
    """Preamble-share ratio strictly above which a Source is longform.

    Override: ``KB_LONGFORM_PREAMBLE_SHARE``.
    """
    return float(os.getenv("KB_LONGFORM_PREAMBLE_SHARE", str(_KB_LONGFORM_PREAMBLE_SHARE_DEFAULT)))


def _min_chars_threshold() -> int:
    """Minimum stripped-body character count before longform detection even runs.

    Override: ``KB_LONGFORM_MIN_CHARS``.
    """
    return int(os.getenv("KB_LONGFORM_MIN_CHARS", str(_KB_LONGFORM_MIN_CHARS_DEFAULT)))


def _furniture_min_repeats() -> int:
    """Minimum exact-repeat count for a line to be treated as page furniture.

    Override: ``KB_STRUCTURE_FURNITURE_MIN_REPEATS``.
    """
    return int(
        os.getenv(
            "KB_STRUCTURE_FURNITURE_MIN_REPEATS",
            str(_KB_STRUCTURE_FURNITURE_MIN_REPEATS_DEFAULT),
        )
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    """Outcome of ``enrich_structure``.

    ``body`` is the (possibly enriched) Markdown body. ``enriched`` is True
    iff headings were materialized and ``docs/`` frontmatter should gain
    ``structure: enriched``. ``reason`` carries a truncated failure summary
    when enrichment was attempted (``is_longform`` fired) but failed —
    ``None`` when the Source bypassed enrichment (not longform) or when it
    succeeded.
    """

    body: str
    enriched: bool
    reason: str | None = None


# ---------------------------------------------------------------------------
# Longform predicate
# ---------------------------------------------------------------------------


def _heading_positions(body: str) -> list[tuple[int, int, str]]:
    """Return ``(char_offset, depth, heading_text)`` for every ATX heading outside fences."""
    positions: list[tuple[int, int, str]] = []
    in_fence = False
    offset = 0
    for line in body.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("```"):
            in_fence = not in_fence
            offset += len(line)
            continue
        if not in_fence:
            m = _HEADING_RE.match(stripped)
            if m:
                positions.append((offset, len(m.group(1)), m.group(2)))
        offset += len(line)
    return positions


def is_longform(body: str, *, filename: str = "source") -> bool:
    """Structural longform predicate (ADR-0033).

    Below ``KB_LONGFORM_MIN_CHARS`` (default 2000 stripped characters),
    always ``False`` — ADR-0033 frames longform as structure "degenerate
    relative to its SIZE"; a handful-of-characters Source has nothing
    meaningful to segment into chapters no matter its heading count.

    Otherwise True when ANY of the following holds over ``body`` (the
    assembled Import/Transcribe Markdown, frontmatter already stripped):

    1. Zero or one heading (heading count <= ``KB_LONGFORM_MAX_HEADINGS``,
       default 1).
    2. Dominant preamble: the non-whitespace text before the first heading
       is more than ``KB_LONGFORM_PREAMBLE_SHARE`` (default 0.5) of the
       whole stripped body.
    3. Any Section (per ``indexer.parse_markdown_body``'s real Section
       boundaries — the body-bearing rule, not a naive heading split)
       estimates over ``KB_INGEST_MAX_SECTION_TOKENS``.

    Never keyed on filename, byte size, or page count (ADR-0033) — a
    well-headed handbook of any length returns ``False`` here regardless of
    how long it is.
    """
    total_stripped = body.strip()
    if len(total_stripped) < _min_chars_threshold():
        return False

    headings = _heading_positions(body)
    if len(headings) <= _max_headings_threshold():
        return True

    first_pos = headings[0][0]
    preamble = body[:first_pos].strip()
    if (len(preamble) / len(total_stripped)) > _preamble_share_threshold():
        return True

    cap = ingest_module.max_section_tokens()
    sections = indexer_module.parse_markdown_body(body, source_prefix=filename)
    return any(ingest_module.estimate_tokens(sec.content) > cap for sec in sections)


# ---------------------------------------------------------------------------
# Deterministic page-furniture stripping
# ---------------------------------------------------------------------------


def _strip_page_furniture(body: str) -> tuple[str, int]:
    """Strip lines that repeat verbatim across the document (running headers/
    footers/page numbers — ADR-0033's "34 identical timestamp/URL/page-counter
    lines" case).

    Deterministic repeat-detection, NOT an LLM call: a line (outside code
    fences, non-blank, not a heading, not a horizontal rule) that recurs
    at least ``KB_STRUCTURE_FURNITURE_MIN_REPEATS`` times identically is
    treated as furniture and every occurrence is dropped. Headings and
    horizontal rules are excluded from candidacy so a legitimately repeated
    structural marker (e.g. ``---``) is never mistaken for furniture.

    Returns ``(stripped_body, lines_removed)``.
    """
    raw_lines = body.splitlines(keepends=False)

    in_fence = False
    candidacy: list[bool] = []
    for text in raw_lines:
        if text.startswith("```"):
            in_fence = not in_fence
            candidacy.append(False)
            continue
        key = text.strip()
        candidacy.append(
            not in_fence and bool(key) and not _HEADING_RE.match(text) and not _HR_RE.match(key)
        )

    counts: Counter[str] = Counter(
        text.strip() for text, candidate in zip(raw_lines, candidacy, strict=True) if candidate
    )
    threshold = _furniture_min_repeats()
    furniture_keys = {key for key, count in counts.items() if count >= threshold}

    kept: list[str] = []
    removed = 0
    for text, candidate in zip(raw_lines, candidacy, strict=True):
        if candidate and text.strip() in furniture_keys:
            removed += 1
            continue
        kept.append(text)

    new_body = "\n".join(kept)
    if body.endswith("\n") and not new_body.endswith("\n"):
        new_body += "\n"
    return new_body, removed


# ---------------------------------------------------------------------------
# LLM chapter proposal (the one new call site — LangChain confined here)
# ---------------------------------------------------------------------------

_ENRICH_SYSTEM_PROMPT = """\
You are segmenting a long, mechanically-unstructured document into chapters \
for a knowledge base.

Rules:
- Propose between 2 and 20 chapter-level boundaries that partition the \
document into coherent stretches, in the order they occur.
- For each chapter, give a short descriptive `title` (plain text, no leading \
`#` characters) and a `boundary_anchor`: the EXACT verbatim text of the \
first 8-15 words of that chapter's opening line, copied character-for-\
character from the document, so it can be located by an exact substring \
search. Do NOT paraphrase or summarize the anchor.
- Chapters must appear in the same order as they occur in the document; do \
not propose overlapping or out-of-order boundaries.
- Do not invent content — only propose boundaries for structure that is \
actually present in the document.
"""


class _ChapterProposal(BaseModel):
    """One proposed chapter boundary. Kept module-private — callers never see this type."""

    title: str
    boundary_anchor: str


class _ChapterOutline(BaseModel):
    """Structured-output schema for the enrichment LLM call."""

    chapters: list[_ChapterProposal]


_enrichment_llm: ChatOpenAI | None = None


def get_enrichment_llm() -> ChatOpenAI:
    """Return a lazy singleton ``ChatOpenAI`` for chapter-outline proposal.

    Model resolution mirrors the ingest-model config family (ADR-0033: "one
    structured-output LLM call, ingest-model config family") —
    ``OPENAI_INGEST_MODEL`` -> ``OPENAI_MODEL`` -> ``gpt-4o-mini`` — the same
    env vars ``templates._ingest_model_name`` resolves, so swapping the
    ingest model re-scales enrichment too without a second knob family.
    Pinned to ``temperature=0`` for reproducible chapter proposals.
    """
    global _enrichment_llm
    if _enrichment_llm is None:
        _enrichment_llm = ChatOpenAI(
            model=os.getenv("OPENAI_INGEST_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            temperature=0,
            timeout=60,
            max_retries=int(os.getenv("KB_INGEST_MAX_RETRIES", "5")),
        )
    return _enrichment_llm


def _propose_chapters(text: str) -> list[_ChapterProposal]:
    """Call the enrichment LLM for a chapter outline. Raises on empty/malformed output."""
    llm = get_enrichment_llm()
    chain = llm.with_structured_output(_ChapterOutline)
    output: _ChapterOutline = chain.invoke(
        [
            SystemMessage(content=_ENRICH_SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]
    )
    if not output.chapters:
        raise ValueError("enrichment LLM proposed zero chapters")
    return output.chapters


# ---------------------------------------------------------------------------
# Heading materialization + mechanical re-split fallback
# ---------------------------------------------------------------------------


def _emit_chapter(title: str, chunk: str, cap: int) -> str:
    """Render one ``## title`` heading + body.

    When ``chunk`` estimates over ``cap``, mechanically re-splits it at
    paragraph boundaries (blank-line-separated blocks) into additional
    ``## title (cont. N)`` headings, greedily packed under the cap (ADR-0033:
    "oversized proposals re-split mechanically at paragraph boundaries"). A
    single paragraph that alone exceeds the cap is emitted as its own
    (oversized) part — paragraphs are not split mid-sentence.
    """
    if ingest_module.estimate_tokens(chunk) <= cap:
        return f"## {title}\n\n{chunk.strip()}\n\n"

    paragraphs = [p for p in re.split(r"\n\s*\n", chunk.strip()) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = ingest_module.estimate_tokens(para)
        if current and current_len + para_len > cap:
            parts.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len
    if current:
        parts.append("\n\n".join(current))

    rendered = []
    for i, part in enumerate(parts, start=1):
        part_title = title if i == 1 else f"{title} (cont. {i})"
        rendered.append(f"## {part_title}\n\n{part.strip()}\n\n")
    return "".join(rendered)


def _materialize_headings(text: str, chapters: list[_ChapterProposal], *, filename: str) -> str:
    """Insert literal ATX headings at the proposed chapter boundaries.

    Locates each ``boundary_anchor`` by exact substring search, anchors the
    heading to the START of that anchor's line, and requires boundaries to
    be strictly increasing in document order. Raises ``ValueError`` (caught
    by ``enrich_structure``'s fail-soft wrapper) if an anchor cannot be
    found or boundaries are not monotonic — a malformed LLM proposal must
    never silently corrupt the document.

    Each chapter span is rendered via ``_emit_chapter``, which applies the
    mechanical per-section token-cap fallback.
    """
    positions: list[tuple[int, str]] = []
    for chapter in chapters:
        idx = text.find(chapter.boundary_anchor)
        if idx == -1:
            raise ValueError(f"boundary anchor not found in text: {chapter.boundary_anchor!r}")
        line_start = text.rfind("\n", 0, idx) + 1
        positions.append((line_start, chapter.title))

    positions.sort(key=lambda p: p[0])
    for i in range(1, len(positions)):
        if positions[i][0] <= positions[i - 1][0]:
            raise ValueError("proposed chapter boundaries are not strictly increasing")

    cap = ingest_module.max_section_tokens()
    pieces: list[str] = [text[: positions[0][0]]]
    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        pieces.append(_emit_chapter(title, text[start:end], cap))
    return "".join(pieces)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_structure(body: str, *, filename: str = "source") -> EnrichmentResult:
    """Run Structure Enrichment on ``body`` when the longform predicate fires.

    Gated on ``is_longform(body, filename=filename)``: a well-headed Source
    returns ``EnrichmentResult(body=body, enriched=False, reason=None)``
    unchanged, with NO LLM call and no log entry — the byte-identical bypass
    ADR-0033 requires.

    When longform, deterministically strips page furniture, calls the
    enrichment LLM for a chapter outline, and materializes the proposed
    headings (with mechanical re-split fallback for oversized chapters).
    Any failure along that path (LLM error, malformed proposal, unfindable
    boundary anchor) fails SOFT: returns the ORIGINAL, un-enriched ``body``
    with ``enriched=False`` and a truncated ``reason`` — import/transcribe
    still succeeds with the un-enriched transcript (ADR-0033).
    """
    if not is_longform(body, filename=filename):
        return EnrichmentResult(body=body, enriched=False, reason=None)

    try:
        stripped, furniture_removed = _strip_page_furniture(body)
        chapters = _propose_chapters(stripped)
        enriched_body = _materialize_headings(stripped, chapters, filename=filename)
    except Exception as exc:  # noqa: BLE001 — fail-soft boundary, any failure degrades gracefully
        reason = str(exc)[:200]
        log_event(
            "structure_enrichment_failed",
            f"source={filename} reason={reason!r}",
        )
        return EnrichmentResult(body=body, enriched=False, reason=reason)

    log_event(
        "structure_enrichment_applied",
        f"source={filename} chapters={len(chapters)} furniture_lines_removed={furniture_removed}",
    )
    return EnrichmentResult(body=enriched_body, enriched=True, reason=None)
