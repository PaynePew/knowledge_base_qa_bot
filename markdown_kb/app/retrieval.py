"""Retrieval layer — happy-path query() for the grounded /chat endpoint.

Flow:
  1. tokenize query
  2. search top-3 Sections via BM25
  3. pre-LLM Cannot Confirm gate (ADR-0001)
  4. build_prompt with Citation markers
  5. call LLM (with error mapping for OpenAI exceptions)
  6. light grounding check (post-LLM heuristic)
  7. write chat log entry
  8. return {answer, sources}
"""
from __future__ import annotations

import os

import openai
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer
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
NOT_INDEXED_MESSAGE = (
    "The knowledge base has not been indexed yet. Call POST /index first."
)

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
        answer  — grounded text (may be "I cannot confirm from the knowledge base.")
        sources — list of {source, heading, score, content} dicts

    Pre-LLM gates (ADR-0001 — never hand weak context to the LLM):
    1. If sections is empty → not indexed yet.
    2. If BM25 yields no results → Cannot Confirm.
    3. If top score < threshold → Cannot Confirm.
    """
    if not indexer.sections:
        log_event(
            "chat_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason=not_indexed',
        )
        return {
            "answer": NOT_INDEXED_MESSAGE,
            "sources": [],
        }

    ranked = indexer.search(question, k=3)

    # Determine the effective top score (0.0 when no results were returned)
    top_score = ranked[0][1] if ranked else 0.0

    if top_score < _SCORE_THRESHOLD:
        # Cannot Confirm — pre-LLM gate, no LLM call (ADR-0001)
        # Log with reason=below_threshold regardless of whether search returned
        # results (score 0.0 is still below any positive threshold).
        truncated = question[:60].replace('"', "'")
        log_event(
            "chat_fallback",
            f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}',
        )
        return {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": [],
        }

    # Build sections list and prompt
    ranked_sections = [sec for sec, _score in ranked]
    prompt_text = build_prompt(question, ranked_sections)

    # Call LLM — map OpenAI exception classes to HTTP responses (issue #5)
    answer = _call_llm_with_error_handling(question, prompt_text)

    # Light grounding check (post-LLM heuristic — issue #5):
    # If the answer neither contains [Source: nor equals the exact Cannot Confirm
    # phrase, retry once at temperature=0. If still ungrounded, replace with
    # Cannot Confirm and clear sources.
    sources = [
        {
            "source": sec.id,
            "heading": " > ".join(sec.heading_path),
            "score": round(score, 3),
            "content": sec.content[:240],
        }
        for sec, score in ranked
    ]

    answer, sources = _apply_grounding_check(question, prompt_text, answer, sources)

    # Write chat log entry
    _write_chat_log(question, ranked)

    return {
        "answer": answer,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_llm_with_error_handling(question: str, prompt_text: str) -> str:
    """Invoke the LLM and map OpenAI exceptions to HTTPExceptions.

    Error mapping (issue #5):
      - APITimeoutError, RateLimitError → HTTP 503 (transient; caller should retry)
      - AuthenticationError            → HTTP 500 (bad API key)
      - Any other APIError             → HTTP 500 (unexpected service error)

    Each error is also logged to wiki/log.md via log_event with the
    appropriate kind tag (openai_transient | openai_auth | openai_api).
    """
    truncated = question[:60].replace('"', "'")
    try:
        response = get_llm().invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])
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


def _is_grounded(answer: str) -> bool:
    """Return True if the answer is grounded or is the exact Cannot Confirm phrase.

    An answer is considered grounded when it:
      - Equals the exact Cannot Confirm phrase (model followed the system prompt), OR
      - Contains at least one [Source: token (model cited a retrieved Section).
    """
    if answer == CANNOT_CONFIRM_PHRASE:
        return True
    return "[Source:" in answer


def _apply_grounding_check(
    question: str,
    prompt_text: str,
    answer: str,
    sources: list[dict],
) -> tuple[str, list[dict]]:
    """Post-LLM light grounding heuristic (issue #5).

    If the first answer is ungrounded (no [Source: and not Cannot Confirm):
      1. Retry the same prompt at temperature=0 (single retry).
      2. If the retry is still ungrounded, replace the answer with the exact
         Cannot Confirm phrase and empty the sources list.
      3. If the retry IS grounded, use it.

    If the first answer is already grounded, return it unchanged.
    """
    if _is_grounded(answer):
        return answer, sources

    # Ungrounded — retry at temperature=0
    truncated = question[:60].replace('"', "'")
    log_event(
        "chat_grounding_retry",
        f'"{truncated}" reason=ungrounded_first_response',
    )

    # Use get_retry_llm() (temperature=0) so tests can inject a stub via
    # monkeypatch; in production this is a fresh temperature=0 singleton.
    retry_response = get_retry_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ])
    retry_answer = retry_response.content

    if _is_grounded(retry_answer):
        return retry_answer, sources

    # Both calls ungrounded — replace with Cannot Confirm
    log_event(
        "chat_grounding_fallback",
        f'"{truncated}" reason=ungrounded_after_retry',
    )
    return CANNOT_CONFIRM_PHRASE, []


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
