"""Deep module per Ousterhout. Public surface: ``query``.

Retrieval layer — query() for the grounded /chat endpoint.

Flow:
  1. tokenize query
  2. search top-3 Sections via BM25
  3. pre-LLM Cannot Confirm gate (ADR-0001)
  4. build_prompt with Citation markers
  5. call LLM (with error mapping for OpenAI exceptions)
  6. post-LLM Grounding Check via grounding.verify() (ADR-0004 layer 3)
  7. write chat log entry
  8. return {answer, sources, grounding_outcome}
"""

from __future__ import annotations

import os
from pathlib import Path

import openai
import yaml
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import grounding as grounding_module
from . import indexer
from .grounding import GroundingOutcome
from .logger import log_event
from .prompt_builder import SYSTEM_PROMPT, build_prompt

# Score threshold below which retrieval is treated as "no match".
# Default 0.5 — calibrated against the sample corpus. Override with
# KB_SCORE_THRESHOLD env var. Read at import time: a server restart picks
# up a new value; runtime changes do not (tests monkeypatch _SCORE_THRESHOLD
# directly).
_SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", "0.5"))

# Sentinel strings the system returns to /chat clients. Tests import these
# constants so a typo in production is caught instead of silently passing
# against a hardcoded test literal.
CANNOT_CONFIRM_PHRASE = "I cannot confirm from the knowledge base."
NOT_INDEXED_MESSAGE = "The knowledge base has not been indexed yet. Call POST /ingest to populate the wiki, then POST /index."

_llm = None
# Separate singleton for temperature=0 grounding retries.
# Tests monkeypatch both _llm and _retry_llm (or get_llm / get_retry_llm).
_retry_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout=20,
            max_retries=1,
        )
    return _llm


def get_retry_llm():
    """Return the temperature=0 LLM used for grounding retries."""
    global _retry_llm
    if _retry_llm is None:
        _retry_llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout=20,
            max_retries=1,
        )
    return _retry_llm


def query(question: str) -> dict:
    """Answer a question against the indexed corpus.

    Returns a dict with keys:
        answer            — grounded text (may be "I cannot confirm from the knowledge base.")
        sources           — list of {source, heading, score, content} dicts
        grounding_outcome — GroundingOutcome instance (always present, never None)

    Pre-LLM gates (ADR-0001 — never hand weak context to the LLM):
    1. If sections is empty → not indexed yet.
    2. If BM25 yields no results → Cannot Confirm (retrieval_empty).
    3. If top score < threshold → Cannot Confirm (below_threshold).

    Post-LLM gate (ADR-0004 layer 3):
    4. grounding.verify() validates every claim against cited Sections.
       Any unsupported claim → Cannot Confirm (claim_unsupported).
       Verifier failure after retry → Cannot Confirm (verifier_unavailable).
    """
    if not indexer.sections:
        log_event(
            "chat_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason=not_indexed',
        )
        return {
            "answer": NOT_INDEXED_MESSAGE,
            "sources": [],
            "grounding_outcome": GroundingOutcome(
                passed=False,
                reason="index_missing",
            ),
        }

    ranked = indexer.search(question, k=3)

    # Determine the effective top score (0.0 when no results were returned)
    top_score = ranked[0][1] if ranked else 0.0

    # Build sources list from whatever retrieval returned (even if below threshold).
    # sources is populated whenever retrieval ran — per ADR-0004 / PRD User Story 22.
    # Phase 6 Slice 6-3: each entry carries an optional ``derived_from`` chain
    # populated from the parent wiki page's ``frontmatter.sources``. This is
    # response-only audit data — see _derived_from_for_section and ADR-0006
    # §"PROMPT.md citation contract evolution" / PRD #78 Q4 for the W1 invariant
    # rationale (the chain must NEVER appear in the LLM CONTEXT or verifier prompt).
    sources = [
        {
            "source": sec.id,
            "heading": " > ".join(sec.heading_path),
            "score": round(score, 3),
            "content": sec.content[:240],
            "derived_from": _derived_from_for_section(sec),
        }
        for sec, score in ranked
    ]

    if top_score < _SCORE_THRESHOLD:
        # Cannot Confirm — pre-LLM gate, no LLM call (ADR-0001)
        # Log with reason=below_threshold regardless of whether search returned
        # results (score 0.0 is still below any positive threshold).
        truncated = question[:60].replace('"', "'")
        # Distinguish retrieval_empty (no results) from below_threshold (results but low score).
        # Both trigger pre-LLM gate; reason differs so callers can show appropriate fallback UX.
        gate_reason = "retrieval_empty" if not ranked else "below_threshold"
        if ranked:
            # Slice 4-5a: enrich below_threshold log with top BM25 hit id so
            # Phase 5 /lint can localise coverage gaps to specific sections.
            top_sec = ranked[0][0]
            log_event(
                "chat_fallback",
                f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}'
                f" top_section={top_sec.id}",
            )
        else:
            # retrieval_empty — no top hit exists, do not append top_section=
            log_event(
                "chat_fallback",
                f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}',
            )
        return {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": sources,
            "grounding_outcome": GroundingOutcome(
                passed=False,
                reason=gate_reason,
            ),
        }

    # Build sections list and prompt.
    # B3 page expansion (Slice 4-4): expand BM25 hits to full parent pages so
    # the LLM receives page-coherent context. The expanded list is used for
    # prompt construction and grounding verification; sources[] stays BM25 top-K.
    ranked_sections = [sec for sec, _score in ranked]
    expanded_sections = indexer.expand_to_pages(ranked_sections)
    prompt_text = build_prompt(question, expanded_sections)

    draft = _call_llm_with_error_handling(question, prompt_text)

    # Post-LLM Grounding Check (ADR-0004 layer 3).
    # Grounding verifier receives expanded_sections (all pages in LLM context)
    # so a claim citing a sibling section is correctly validated.
    # verify() never raises — all verifier failures map to
    # grounding_outcome.reason = "verifier_unavailable".
    outcome = grounding_module.verify(draft, expanded_sections)

    if outcome.passed:
        # Verifier approved — return the draft as-is.
        answer = draft
    else:
        # Verifier rejected or unavailable — fail-closed with Cannot Confirm.
        answer = CANNOT_CONFIRM_PHRASE
        # Slice 4-5a: enrich with cited= listing the post-B3-expansion Section ids.
        # Phase 5 /lint uses this to localise coverage gaps without replaying BM25.
        cited_ids = ",".join(sec.id for sec in expanded_sections)
        log_event(
            "chat_grounding_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason={outcome.reason}'
            f" cited={cited_ids}",
        )

    # Write chat log entry
    _write_chat_log(question, ranked)

    return {
        "answer": answer,
        "sources": sources,
        "grounding_outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derived_from_for_section(sec: indexer.Section) -> list[dict] | None:
    """Return the docs/ citation chain for a retrieved wiki Section.

    Phase 6 Slice 6-3 — closes ADR-0006 §"PROMPT.md citation contract evolution"
    deferred to Phase 6. Reads the parent wiki page's ``frontmatter.sources``
    and converts each ``"<docs-filename>#<heading-slug>"`` entry into a
    ``{source, heading}`` dict (matches ``schemas.CitationRef`` shape so Pydantic
    can validate at the route boundary without an explicit model conversion).

    Return contract:
        * Populated list — parent ``frontmatter.sources`` had entries.
        * ``None`` — parent had ``sources: []`` (or absent / non-list), the
          parent file is missing, or parsing parent frontmatter raised.
          ``None`` is semantically distinct from ``[]`` per PRD #78 Q4.

    Fail-soft contract: never raises. Parse failures degrade to ``None`` and
    emit a ``parse_warning`` log entry (reuses the existing kind; no new
    log_kinds entry per the slice constraints).

    Parent-page resolution: the wiki layout is ``<wiki>/<type-subdir>/<slug>.md``
    where ``<type-subdir>`` comes from ``Section.metadata["type"]`` (one of
    ``"entity"``, ``"concept"``, ``"qa"``). If ``type`` is absent or unknown
    we cannot construct the path -> degrade to ``None`` without a log entry
    (this is a structural issue surfaced elsewhere by C10 lint / parse_warning
    at index time).
    """
    metadata = sec.metadata or {}
    page_type = metadata.get("type")
    if page_type not in ("entity", "concept", "qa"):
        return None

    subdir_name = {"entity": "entities", "concept": "concepts", "qa": "qa"}[page_type]
    # Read indexer.WIKI_DIR via the module so tests monkeypatching
    # ``indexer.WIKI_DIR`` (the autouse conftest fixture does this) are
    # honoured at query time.
    parent_path: Path = indexer.WIKI_DIR / subdir_name / f"{sec.file}.md"

    if not parent_path.exists():
        log_event(
            "parse_warning",
            f"derived_from: parent wiki page missing for section={sec.id} path={parent_path.name}",
        )
        return None

    try:
        raw = parent_path.read_text(encoding="utf-8")
    except OSError as exc:
        log_event(
            "parse_warning",
            f"derived_from: parent read failed for section={sec.id} exc={type(exc).__name__}",
        )
        return None

    # Skip optional sentinel HTML comment preceding the frontmatter block.
    lines = raw.splitlines()
    dash_indices = [i for i, ln in enumerate(lines) if ln.strip() == "---"]
    if len(dash_indices) < 2:
        log_event(
            "parse_warning",
            f"derived_from: parent frontmatter fences missing for section={sec.id} "
            f"path={parent_path.name}",
        )
        return None

    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:
        log_event(
            "parse_warning",
            f"derived_from: parent frontmatter YAML invalid for section={sec.id} "
            f"path={parent_path.name} exc={type(exc).__name__}",
        )
        return None

    if not isinstance(parsed, dict):
        log_event(
            "parse_warning",
            f"derived_from: parent frontmatter not a mapping for section={sec.id} "
            f"path={parent_path.name}",
        )
        return None

    sources_field = parsed.get("sources")
    # Empty / absent / non-list -> semantic None per PRD #78 Q4.
    if not isinstance(sources_field, list) or not sources_field:
        return None

    # Convert each "docs-filename#heading-slug" entry to a CitationRef-shaped
    # dict. Entries that don't contain "#" or are not strings are skipped
    # silently — surfacing them belongs to C2 red-link lint, not retrieval.
    refs: list[dict] = []
    for entry in sources_field:
        if not isinstance(entry, str) or "#" not in entry:
            continue
        source, _, heading = entry.partition("#")
        refs.append({"source": source, "heading": heading})

    return refs or None


def _call_llm_with_error_handling(question: str, prompt_text: str) -> str:
    """Invoke the LLM and map OpenAI exceptions to HTTPExceptions.

    Error mapping:
      - APITimeoutError, RateLimitError → HTTP 503 (transient; caller should retry)
      - AuthenticationError            → HTTP 500 (bad API key)
      - Any other APIError             → HTTP 500 (unexpected service error)

    Each error is also logged to wiki/log.md via log_event with the
    appropriate kind tag (openai_transient | openai_auth | openai_api).
    """
    truncated = question[:60].replace('"', "'")
    try:
        response = get_llm().invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ]
        )
        return response.content
    except (openai.APITimeoutError, openai.RateLimitError) as exc:
        log_event(
            "chat_error",
            f'"{truncated}" kind=openai_transient exc={type(exc).__name__}',
        )
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable, please retry.",
        ) from exc
    except openai.AuthenticationError as exc:
        log_event(
            "chat_error",
            f'"{truncated}" kind=openai_auth exc={type(exc).__name__}',
        )
        raise HTTPException(
            status_code=500,
            detail="LLM service auth failed (check OPENAI_API_KEY).",
        ) from exc
    except openai.APIError as exc:
        log_event(
            "chat_error",
            f'"{truncated}" kind=openai_api exc={type(exc).__name__}',
        )
        raise HTTPException(
            status_code=500,
            detail=f"LLM service error: {exc!s}",
        ) from exc


def _write_chat_log(
    question: str,
    ranked: list[tuple[indexer.Section, float]],
) -> None:
    """Append a chat log entry to wiki/log.md.

    Format:
        ## [<ts>] chat | "<truncated_query>" top=<section_id>:<score>
    """
    truncated = question[:60].replace('"', "'")
    if ranked:
        top_sec, top_score = ranked[0]
        summary = f'"{truncated}" top={top_sec.id}:{round(top_score, 3)}'
    else:
        summary = f'"{truncated}" top=none'
    log_event("chat", summary)
