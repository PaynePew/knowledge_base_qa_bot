"""Deep module per Ousterhout. Public surface: ``query``, ``stream_query``, ``build_prompt``, ``SYSTEM_PROMPT``, ``get_llm``, ``CANNOT_CONFIRM_PHRASE``.

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

Phase 9 (issue #120): query() is decomposed into two private helpers:
  _retrieve_and_gate() — vector search + pre-LLM Cannot Confirm gates
  _draft_and_verify()  — build_prompt + LLM draft + Grounding Check
Public query() composes them; contract is unchanged.
stream_query() uses the same decomposition to yield a sources-ready partial
before any LLM call (ADR-0009 verify-then-stream / sources-first).

RAG source objects carry ONLY citation id + heading + content — NO score,
NO derived_from (issue #120 spec; RAG serves raw docs/ Sources, not the
curated wiki layer that has frontmatter.sources chains).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# markdown_kb's grounding module is adopted unchanged (ADR-0004 Q9 / issue #103):
# vector_rag is the first real second consumer of the CitableContent Protocol.
# Chunk satisfies the protocol, so verify() needs no changes to grounding.py.
from markdown_kb.app import grounding as grounding_module
from markdown_kb.app.errors import LLMError
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
NOT_INDEXED_MESSAGE = (
    "The knowledge base has not been indexed yet. Call POST /index first."
)

# ---------------------------------------------------------------------------
# System prompt — Stack B's own literal of the ADR-0001 strict-grounded contract
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a strict knowledge-base assistant. Follow these rules exactly:

1. Answer ONLY using the information in the CONTEXT section below. Do not use outside world knowledge, training data, or inference beyond what is written.
2. Every factual claim in your answer MUST cite at least one source using the exact format: [Source: filename#heading]. Use the Citation ids as they appear in the CONTEXT headers.
3. If the CONTEXT does not contain enough information to answer the question, reply with the exact phrase: "I cannot confirm from the knowledge base." — nothing more, nothing less.
4. You may synthesize information across multiple cited Sections if needed, but every claim must still trace to a cited Section.
5. Never guess, never infer beyond the text, never complete gaps with general knowledge. "I cannot confirm from the knowledge base." is a good, expected answer — not a failure.
6. Answer in the same language as the QUESTION. Exception: if the CONTEXT does not contain enough information, always reply with the exact English phrase "I cannot confirm from the knowledge base." regardless of the question's language.
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

    Phase 9: composes _retrieve_and_gate() + _draft_and_verify().
    Public contract is unchanged; the split is behaviour-preserving.
    """
    gate = _retrieve_and_gate(question)
    if gate["early_exit"]:
        return {
            "answer": gate["answer"],
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }
    return _draft_and_verify(question, gate["chunks"], gate["sources"])


def stream_query(question: str) -> Iterator[dict]:
    """Generator that yields two dicts for use by the SSE streaming endpoint.

    Phase 9 — ADR-0009 (verify-then-stream / sources-first). Mirrors the
    markdown_kb stream_query() contract so the Gateway can reuse the shared
    serializer ``markdown_kb.app.sse.events_for_result`` for both stacks.

    Yields:
        1. A *partial* result dict immediately after retrieval (before any
           LLM call), so the gateway can emit the ``sources`` SSE event
           right away (~1 embedding round-trip for vector search).
           Shape: ``{sources, grounding_outcome, _phase: "sources_ready"}``.
        2. A *full* result dict after draft + Grounding Check complete.
           Shape: ``{answer, sources, grounding_outcome}`` — identical to
           what ``query()`` returns.

    RAG source objects carry ONLY citation id + heading + content — NO score,
    NO derived_from (issue #120 spec).

    The gateway endpoint must:
      a. Emit ``sources`` SSE event from yield 1.
      b. Emit ``token`` + ``done`` SSE events from yield 2.

    ADR-0009: only verified text is ever emitted as tokens.
    """
    gate = _retrieve_and_gate(question)

    # Yield the sources-ready partial so the gateway can emit the sources
    # event before making any LLM call (ADR-0009 sources-first invariant).
    yield {
        "_phase": "sources_ready",
        "sources": gate["sources"],
        "grounding_outcome": gate["grounding_outcome"],
        "early_exit": gate["early_exit"],
        "answer": gate.get("answer", ""),
        "chunks": gate.get("chunks", []),
    }

    if gate["early_exit"]:
        # Pre-LLM gate fired — the partial result IS the full result.
        # SSE uniformity (ADR-0009 / issue #138): all 5 CC reasons must stream
        # CANNOT_CONFIRM_PHRASE in the token events so the UI token stream is
        # identical regardless of which gate fired.  The ``index_missing`` path
        # in _retrieve_and_gate returns NOT_INDEXED_MESSAGE (the verbose
        # curator-targeted string); normalise to CANNOT_CONFIRM_PHRASE here so
        # stream callers never need to branch on the reason.  query() is
        # unaffected (it does not pass through this path).
        # The specific reason is preserved in grounding_outcome.reason for
        # machine consumers (done.reason in the SSE done event).
        yield {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }
        return

    # LLM phase — draft + Grounding Check.
    result = _draft_and_verify(question, gate["chunks"], gate["sources"])
    yield result


# ---------------------------------------------------------------------------
# Phase 9 private helpers — retrieve+gate and draft+verify
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pre-LLM distance-relevance gate config (issue #257)
# ---------------------------------------------------------------------------
# Calibrated default (eval/rag_distance, #257/#258 follow-up): with real
# text-embedding-3-small the FAISS L2 distance ceiling 1.1 perfectly separated the
# in-scope eval set (max distance 0.989) from the out-of-scope set (min 1.278) —
# a Youden-J-optimal plateau {1.0, 1.1, 1.2}, 1.1 = median (max margin). The model
# is unit-norm so L2² = 2 − 2·cos, i.e. this ceiling is equivalently a cosine floor.
# Enabled by default for parity with the wiki BM25 gate (CODING_STANDARD §4.3):
# both stacks refuse weak retrieval pre-LLM. Set KB_RAG_DISTANCE_THRESHOLD to
# override (e.g. a large value to disable). See eval/rag_distance/calibration_report.md.
_KB_RAG_DISTANCE_THRESHOLD_DEFAULT: float | None = 1.1


def _max_rag_distance() -> float | None:
    """Max distance for the closest chunk before a pre-LLM Cannot Confirm.

    Defaults to the calibrated ceiling (``_KB_RAG_DISTANCE_THRESHOLD_DEFAULT``);
    ``KB_RAG_DISTANCE_THRESHOLD`` overrides it (set a large value to disable). Read
    at call time so an env change takes effect on the next query without a restart —
    unlike the wiki ``KB_SCORE_THRESHOLD``, which is read once at import.
    """
    raw = os.getenv("KB_RAG_DISTANCE_THRESHOLD")
    return float(raw) if raw is not None else _KB_RAG_DISTANCE_THRESHOLD_DEFAULT


def _retrieve_and_gate(question: str) -> dict:
    """Vector search + all pre-LLM Cannot Confirm gates (ADR-0001).

    Returns a dict with:
        sources          — list of {source, heading, content} dicts
                           (NO score, NO derived_from — RAG source shape)
        grounding_outcome — provisional outcome (pre-LLM gate result)
        early_exit       — True when a pre-LLM gate fired (no LLM needed)
        answer           — set to the Cannot Confirm / not-indexed phrase on
                           early_exit paths; empty string otherwise
        chunks           — raw Chunk list for _draft_and_verify; only
                           meaningful when early_exit is False

    Callers (query() and stream_query()) use the early_exit flag to decide
    whether to call _draft_and_verify(). The sources list is always populated
    even on early_exit paths — the gateway emits a sources SSE event before
    checking early_exit.
    """
    truncated = question[:60].replace('"', "'")

    if indexer.vectorstore is None:
        # Lazy-load the persisted FAISS index from disk so a fresh Gateway
        # process can serve stack=rag without requiring a POST /index call in
        # the same process (the Gateway mounts only /wiki, not vector_rag's
        # /index — see issue #133). load_vector_index() returns (0, 0) and
        # leaves vectorstore=None when no persisted index exists on disk.
        indexer.load_vector_index()

    if indexer.vectorstore is None:
        log_event("chat_fallback", f'"{truncated}" reason=not_indexed')
        return {
            "sources": [],
            "grounding_outcome": GroundingOutcome(passed=False, reason="index_missing"),
            "early_exit": True,
            "answer": NOT_INDEXED_MESSAGE,
            "chunks": [],
        }

    results = indexer.search_with_distance(question, k=3)

    if not results:
        # Pre-LLM Cannot Confirm gate — no LLM call (ADR-0001).
        log_event("chat_fallback", f'"{truncated}" reason=retrieval_empty')
        return {
            "sources": [],
            "grounding_outcome": GroundingOutcome(
                passed=False, reason="retrieval_empty"
            ),
            "early_exit": True,
            "answer": CANNOT_CONFIRM_PHRASE,
            "chunks": [],
        }

    chunks = [chunk for chunk, _distance in results]

    # RAG source shape: citation id + heading + content ONLY.
    # NO score (prevents the model reasoning "low score → guess", PROMPT.md Q3).
    # NO derived_from (RAG serves raw docs/ Sources; frontmatter chains are a
    # wiki-layer concept — issue #120 spec). The distance stays a gate-only signal
    # and never enters sources.
    sources = [
        {
            "source": chunk.source,
            "heading": " > ".join(chunk.heading_path),
            "content": chunk.content[:240],
        }
        for chunk in chunks
    ]

    # Pre-LLM distance-relevance gate (#257). FAISS k-NN always returns k
    # neighbours, so "retrieved something" does NOT mean "relevant" — without this
    # an out-of-scope query forwards its least-far chunks to the LLM and relies on
    # the (expensive) post-LLM grounding net alone. When the CLOSEST chunk is still
    # farther than the configured ceiling, refuse before the LLM — symmetric with
    # the wiki BM25 gate (CODING_STANDARD §4.3 gate parity). Lives in this deep
    # module so Browser / MCP / CLI all inherit it. ON by default at the calibrated
    # ceiling (eval/rag_distance); KB_RAG_DISTANCE_THRESHOLD overrides it (set a
    # large value to disable).
    max_distance = _max_rag_distance()
    if max_distance is not None and min(d for _, d in results) > max_distance:
        log_event(
            "chat_fallback",
            f'"{truncated}" reason=below_threshold'
            f" rag_distance={round(min(d for _, d in results), 3)}",
        )
        return {
            "sources": sources,
            "grounding_outcome": GroundingOutcome(
                passed=False, reason="below_threshold"
            ),
            "early_exit": True,
            "answer": CANNOT_CONFIRM_PHRASE,
            "chunks": [],
        }

    return {
        "sources": sources,
        "grounding_outcome": GroundingOutcome(passed=True, reason="claim_supported"),
        "early_exit": False,
        "answer": "",
        "chunks": chunks,
    }


def _draft_and_verify(
    question: str,
    chunks: list[Chunk],
    sources: list[dict],
) -> dict:
    """LLM draft + post-LLM Grounding Check (ADR-0004 layer 3).

    Called only when the pre-LLM gates passed (early_exit is False).

    Args:
        question: The original user query.
        chunks:   Retrieved Chunk list from _retrieve_and_gate.
        sources:  Already-built sources list (passed through unchanged).

    Returns:
        Full result dict: {answer, sources, grounding_outcome}.
        Never returns unverified text — on grounding failure, answer is
        CANNOT_CONFIRM_PHRASE and grounding_outcome.passed is False.
    """
    truncated = question[:60].replace('"', "'")
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
    """Invoke the LLM and map OpenAI exceptions to a transport-agnostic LLMError.

    Error mapping (ADR-0015 — status table moves to the HTTP route):
      - APITimeoutError, RateLimitError → LLMError(retryable=True)
      - AuthenticationError            → LLMError(retryable=False)
      - Any other APIError             → LLMError(retryable=False)

    Each branch emits a ``chat_error`` log entry tagged with the appropriate
    kind (openai_transient | openai_auth | openai_api) BEFORE raising.
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
        raise LLMError(
            retryable=True,
            message="LLM service temporarily unavailable, please retry.",
        ) from exc
    except openai.AuthenticationError as exc:
        log_event(
            "chat_error", f'"{truncated}" kind=openai_auth exc={type(exc).__name__}'
        )
        raise LLMError(
            retryable=False,
            message="LLM service auth failed (check OPENAI_API_KEY).",
        ) from exc
    except openai.APIError as exc:
        log_event(
            "chat_error", f'"{truncated}" kind=openai_api exc={type(exc).__name__}'
        )
        raise LLMError(
            retryable=False,
            message=f"LLM service error: {exc!s}",
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
