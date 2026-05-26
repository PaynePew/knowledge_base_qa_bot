"""Retrieval layer — query() for the grounded /chat endpoint.

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

import openai
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
NOT_INDEXED_MESSAGE = "The knowledge base has not been indexed yet. Call POST /index first."

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
    sources = [
        {
            "source": sec.id,
            "heading": " > ".join(sec.heading_path),
            "score": round(score, 3),
            "content": sec.content[:240],
        }
        for sec, score in ranked
    ]

    if top_score < _SCORE_THRESHOLD:
        # Cannot Confirm — pre-LLM gate, no LLM call (ADR-0001)
        # Log with reason=below_threshold regardless of whether search returned
        # results (score 0.0 is still below any positive threshold).
        truncated = question[:60].replace('"', "'")
        log_event(
            "chat_fallback",
            f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}',
        )
        # Distinguish retrieval_empty (no results) from below_threshold (results but low score).
        # Both trigger pre-LLM gate; reason differs so callers can show appropriate fallback UX.
        gate_reason = "retrieval_empty" if not ranked else "below_threshold"
        return {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": sources,
            "grounding_outcome": GroundingOutcome(
                passed=False,
                reason=gate_reason,
            ),
        }

    # Build sections list and prompt
    ranked_sections = [sec for sec, _score in ranked]
    prompt_text = build_prompt(question, ranked_sections)

    draft = _call_llm_with_error_handling(question, prompt_text)

    # Post-LLM Grounding Check (ADR-0004 layer 3).
    # Replaces the previous light [Source: heuristic.  verify() never raises —
    # all verifier failures map to grounding_outcome.reason = "verifier_unavailable".
    outcome = grounding_module.verify(draft, ranked_sections)

    if outcome.passed:
        # Verifier approved — return the draft as-is.
        answer = draft
    else:
        # Verifier rejected or unavailable — fail-closed with Cannot Confirm.
        answer = CANNOT_CONFIRM_PHRASE
        log_event(
            "chat_grounding_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason={outcome.reason}',
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
