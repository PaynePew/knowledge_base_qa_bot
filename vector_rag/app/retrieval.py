"""Deep module per Ousterhout. Public surface: ``query``, ``build_prompt``, ``SYSTEM_PROMPT``, ``get_llm``, ``CANNOT_CONFIRM_PHRASE``.

Vector RAG (Stack B) query path — retrieve Chunks, gate weak retrieval, build a
grounded prompt, call the LLM, and verify the draft against the cited Chunks.

This module owns Stack B's LLM call site, so it is an LLM-facing module and may
import LangChain (CODING_STANDARD §2.4). It consumes domain ``Chunk`` objects
from :mod:`vector_rag.app.indexer` (never LangChain ``Document``) so the
retrieval boundary stays framework-free.

Flow (mirrors markdown_kb's /chat per issue #103):
  1. index-missing gate → not-indexed message (HTTP 200)
  2. vector search top-k Chunks
  3. pre-LLM Cannot Confirm gate when retrieval is empty (ADR-0001)
  4. build_prompt with [Source: ...] / Heading: markers
  5. call LLM, mapping OpenAI exceptions to HTTP status (CODING_STANDARD §4.2)
  6. post-LLM Grounding Check via markdown_kb's grounding.verify() — adopted
     unchanged through its CitableContent Protocol (ADR-0004 Q9). Chunk
     satisfies the protocol, so verify() consumes it as-is.
  7. grounded answer or Cannot Confirm; always return a GroundingOutcome.

SYSTEM_PROMPT is Stack B's OWN literal of the ADR-0001 strict-grounded contract
— deliberately NOT imported from markdown_kb (the apps stay decoupled). A smoke
test guards against drift from markdown_kb's contract.
"""

from __future__ import annotations

import os

import openai
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# markdown_kb's grounding module is adopted unchanged (ADR-0004 Q9 / issue #103):
# vector_rag is the first real second consumer of the CitableContent Protocol.
# Chunk satisfies the protocol, so verify() needs no changes to grounding.py.
from markdown_kb.app import grounding as grounding_module
from markdown_kb.app.grounding import GroundingOutcome

from . import indexer
from .indexer import Chunk
from .logger import log_event

# ---------------------------------------------------------------------------
# Sentinel strings (CODING_STANDARD §3.3 — defined once, imported elsewhere)
# ---------------------------------------------------------------------------
# Stack B returns the SAME literal Cannot Confirm phrase as markdown_kb so the
# "KB cannot back this answer" surface is identical across both apps (ADR-0001).
# It is duplicated here (not imported) to keep the apps decoupled; the
# SYSTEM_PROMPT drift smoke test pins both literals against markdown_kb's.
CANNOT_CONFIRM_PHRASE = "I cannot confirm from the knowledge base."
NOT_INDEXED_MESSAGE = "The knowledge base has not been indexed yet. Call POST /index first."

# ---------------------------------------------------------------------------
# System prompt — Stack B's own literal of the ADR-0001 strict-grounded contract
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a strict knowledge-base assistant. Follow these rules exactly:

1. Answer ONLY using the information in the CONTEXT section below. Do not use outside world knowledge, training data, or inference beyond what is written.
2. Every factual claim in your answer MUST cite at least one source using the exact format: [Source: filename#heading]. Use the Citation ids as they appear in the CONTEXT headers.
3. If the CONTEXT does not contain enough information to answer the question, reply with the exact phrase: "I cannot confirm from the knowledge base." — nothing more, nothing less.
4. You may synthesize information across multiple cited Sections if needed, but every claim must still trace to a cited Section.
5. Never guess, never infer beyond the text, never complete gaps with general knowledge. "I cannot confirm from the knowledge base." is a good, expected answer — not a failure.
"""

# ---------------------------------------------------------------------------
# LLM singleton (lazy — CODING_STANDARD §2.7 / §10 lazy-singleton)
# ---------------------------------------------------------------------------
_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return the lazily-constructed chat LLM (CODING_STANDARD §10 lazy-singleton)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


# ---------------------------------------------------------------------------
# Prompt builder (mirrors markdown_kb structure — issue #103)
# ---------------------------------------------------------------------------
def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Build the [Human] message from retrieved Chunks.

    Mirrors markdown_kb's structure (CONTEXT before QUESTION, one
    ``[Source: ...]`` header + ``Heading:`` breadcrumb per block) so the two
    apps present the LLM an identical contract surface, filled exclusively with
    Stack B's own vector-retrieved Chunks.

    Structure:
        CONTEXT:

        [Source: filename#heading]
        Heading: parent > leaf
        <chunk content>

        (repeated for each chunk)

        QUESTION:
        <question>

    Scores are NOT included (PROMPT.md Q3: prevents the model reasoning
    "low score → guess").
    """
    parts: list[str] = ["CONTEXT:\n"]

    for chunk in chunks:
        breadcrumb = " > ".join(chunk.heading_path)
        block = f"[Source: {chunk.source}]\nHeading: {breadcrumb}\n{chunk.content}\n"
        parts.append(block)

    parts.append(f"\nQUESTION:\n{question}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public query path
# ---------------------------------------------------------------------------
def query(question: str) -> dict:
    """Answer a question against the persisted FAISS index.

    Returns a dict with keys:
        answer            — grounded text (may be the Cannot Confirm phrase)
        sources           — list of {source, heading, content} dicts
        grounding_outcome — GroundingOutcome (always present, never None)

    Pre-LLM gates (ADR-0001 — never hand weak context to the LLM):
    1. index not built → not-indexed message.
    2. vector search empty → Cannot Confirm (retrieval_empty).

    Post-LLM gate (ADR-0004 layer 3): grounding.verify() validates every claim
    against the cited Chunks. Any unsupported claim, or verifier unavailability
    after retry, → Cannot Confirm.
    """
    truncated = question[:60].replace('"', "'")

    if indexer.vectorstore is None:
        log_event("chat_fallback", f'"{truncated}" reason=not_indexed')
        return {
            "answer": NOT_INDEXED_MESSAGE,
            "sources": [],
            "grounding_outcome": GroundingOutcome(passed=False, reason="index_missing"),
        }

    chunks = indexer.search(question, k=3)

    if not chunks:
        # Pre-LLM Cannot Confirm gate — no LLM call (ADR-0001).
        log_event("chat_fallback", f'"{truncated}" reason=retrieval_empty')
        return {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": [],
            "grounding_outcome": GroundingOutcome(passed=False, reason="retrieval_empty"),
        }

    sources = [
        {
            "source": chunk.source,
            "heading": " > ".join(chunk.heading_path),
            "content": chunk.content[:240],
        }
        for chunk in chunks
    ]

    prompt_text = build_prompt(question, chunks)
    draft = _call_llm_with_error_handling(question, prompt_text)

    # Post-LLM Grounding Check (ADR-0004 layer 3). Chunk satisfies CitableContent,
    # so grounding.verify() consumes the retrieved Chunks unchanged. verify()
    # never raises — all verifier failures map to reason="verifier_unavailable".
    outcome = grounding_module.verify(draft, chunks)

    if outcome.passed:
        answer = draft
    else:
        answer = CANNOT_CONFIRM_PHRASE
        cited_ids = ",".join(chunk.source for chunk in chunks)
        log_event(
            "chat_grounding_fallback",
            f'"{truncated}" reason={outcome.reason} cited={cited_ids}',
        )

    _write_chat_log(question, chunks)

    return {"answer": answer, "sources": sources, "grounding_outcome": outcome}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _call_llm_with_error_handling(question: str, prompt_text: str) -> str:
    """Invoke the LLM and map OpenAI exceptions to HTTPExceptions.

    Error mapping (CODING_STANDARD §4.2):
      - APITimeoutError, RateLimitError → HTTP 503 (transient; caller retries)
      - AuthenticationError            → HTTP 500 (bad API key)
      - Any other APIError             → HTTP 500 (unexpected service error)

    Each branch emits a ``chat_error`` log entry tagged with the appropriate
    kind (openai_transient | openai_auth | openai_api).
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
        log_event("chat_error", f'"{truncated}" kind=openai_transient exc={type(exc).__name__}')
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable, please retry.",
        ) from exc
    except openai.AuthenticationError as exc:
        log_event("chat_error", f'"{truncated}" kind=openai_auth exc={type(exc).__name__}')
        raise HTTPException(
            status_code=500,
            detail="LLM service auth failed (check OPENAI_API_KEY).",
        ) from exc
    except openai.APIError as exc:
        log_event("chat_error", f'"{truncated}" kind=openai_api exc={type(exc).__name__}')
        raise HTTPException(
            status_code=500,
            detail=f"LLM service error: {exc!s}",
        ) from exc


def _write_chat_log(question: str, chunks: list[Chunk]) -> None:
    """Append a chat log entry to vector_rag/log.md.

    Format:
        ## [<ts>] chat | "<truncated query>" top=<chunk source> count=<N>
    """
    truncated = question[:60].replace('"', "'")
    if chunks:
        summary = f'"{truncated}" top={chunks[0].source} count={len(chunks)}'
    else:
        summary = f'"{truncated}" top=none count=0'
    log_event("chat", summary)
