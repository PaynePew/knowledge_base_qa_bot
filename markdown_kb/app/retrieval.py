"""Retrieval layer — happy-path query() for the grounded /chat endpoint.

Flow:
  1. tokenize query
  2. search top-3 Sections via BM25
  3. pre-LLM Cannot Confirm gate (ADR-0001)
  4. build_prompt with Citation markers
  5. call LLM
  6. write chat log entry
  7. return {answer, sources}
"""
from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer
from .logger import log_event
from .prompt_builder import SYSTEM_PROMPT, build_prompt

# Score threshold below which retrieval is treated as "no match".
# Default 0.5 — calibrated against the sample corpus. Override with
# KB_SCORE_THRESHOLD env var.
_SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", "0.5"))

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout=20,
            max_retries=1,
        )
    return _llm


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
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked = indexer.search(question, k=3)

    if not ranked or ranked[0][1] < _SCORE_THRESHOLD:
        # Cannot Confirm — pre-LLM gate, no LLM call (ADR-0001)
        _write_chat_log(question, ranked)
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    # Build sections list and prompt
    ranked_sections = [sec for sec, _score in ranked]
    prompt_text = build_prompt(question, ranked_sections)

    # Call LLM
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ])

    sources = [
        {
            "source": sec.id,
            "heading": " > ".join(sec.heading_path),
            "score": round(score, 3),
            "content": sec.content[:240],
        }
        for sec, score in ranked
    ]

    # Write chat log entry
    _write_chat_log(question, ranked)

    return {
        "answer": response.content,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_chat_log(
    question: str,
    ranked: list,
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
