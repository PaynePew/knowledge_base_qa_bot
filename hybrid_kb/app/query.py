"""Deep module per Ousterhout. Public surface: ``query``, ``stream_query``, ``get_llm``, ``warm_llm_client``, ``CANNOT_CONFIRM_PHRASE``, ``ensure_indexes_loaded``.

Hybrid Retrieval (Stack C) query path — slice S3 (ADR-0018 / #313) + the
streaming variant ``stream_query`` (slice S4 / #314) the Gateway dispatches to
for ``stack=hybrid``.

This is the LLM-facing surface of the Phase 13 Hybrid stack. It wraps S2's
retrieval core (``hybrid_kb.app.retrieval.retrieve_and_gate``) with the SAME
answer-synthesis + Grounding Check + Citation path the Wiki and RAG stacks use —
nothing downstream is reimplemented. Because the retrieval unit is ``Section``
(which satisfies the ``CitableContent`` protocol), every downstream concern is
reused unchanged by IMPORTING ``markdown_kb``'s leaf functions:

  * page expansion  → ``markdown_kb.app.indexer.expand_to_pages``
  * prompt building → ``markdown_kb.app.prompt_builder.build_prompt`` / ``SYSTEM_PROMPT``
  * Grounding Check → ``markdown_kb.app.grounding.verify`` (adopted unchanged via the
                      ``CitableContent`` protocol — ADR-0004 Q9)
  * Cannot Confirm  → ``markdown_kb.app.retrieval.CANNOT_CONFIRM_PHRASE`` (imported,
                      never paraphrased — the ADR-0001 contract string, trap #2)

Only the answer-synthesis LLM call site is owned HERE: ``hybrid_kb.query`` is a
new LLM-facing surface (ADR-0005 enumeration), so it owns its own lazy
``get_llm`` singleton and its own LangChain-confined call wrapper. LangChain
message/client types never leave this module (CODING_STANDARD §2.4 — no leak).

Flow (mirrors ``markdown_kb.query``'s composition):
  1. lazy-load both arms' indexes if cold (BM25 ``.kb/index.json`` + dense seed)
  2. ``retrieve_and_gate`` — overfetch both arms, per-arm OR-gate, RRF fuse (S2)
  3. pre-LLM OR-gate refused → Cannot Confirm sentinel, NO LLM call (AC3)
  4. page-expand the fused Sections, build the grounded prompt, call the LLM
  5. LLM self-refusal short-circuit (the model emitted the Cannot Confirm phrase)
  6. post-LLM Grounding Check; any unsupported claim → Cannot Confirm
  7. return ``{answer, sources, grounding_outcome}`` (same shape as Wiki / RAG)

The pre-LLM relevance gate lives in the S2 retrieval deep module, never here
(ADR-0018 §4.3 gate-parity). Filing is NOT invoked on the ``query()`` path — it
is a Gateway/route concern (parity with ``markdown_kb.query`` / ``vector_rag``'s
``query``, which do not file either).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from markdown_kb.app import grounding as grounding_module
from markdown_kb.app import indexer as bm25_indexer
from markdown_kb.app import retrieval as bm25_retrieval
from markdown_kb.app.errors import LLMError
from markdown_kb.app.grounding import GroundingOutcome
from markdown_kb.app.indexer import Section
from markdown_kb.app.prompt_builder import SYSTEM_PROMPT, build_prompt
from markdown_kb.app.schemas import qa_schema_lint_code

# CANNOT_CONFIRM_PHRASE is imported from the module that owns it (ADR-0001
# contract) — never paraphrased (trap #2 / CODING_STANDARD §3.3 define-once).
from markdown_kb.app.retrieval import CANNOT_CONFIRM_PHRASE

from . import dense_index
from . import retrieval as hybrid_retrieval
from .logger import log_event

__all__ = [
    "query",
    "stream_query",
    "get_llm",
    "warm_llm_client",
    "CANNOT_CONFIRM_PHRASE",
    "ensure_indexes_loaded",
]


# ---------------------------------------------------------------------------
# LLM singleton (lazy — CODING_STANDARD §2.7 / §10 lazy-singleton)
# ---------------------------------------------------------------------------
# Fixed seed for the answer-synthesis LLM (ADR-0038's C5-judge pattern,
# extended to the serving chain by ADR-0042 / issue #572, completed here for
# Stack C by issue #619). Per-module private constant. Best-effort only:
# OpenAI's seed is not a hard guarantee, but paired with temperature=0 it cuts
# run-to-run draft rewording that would otherwise flip the same question
# between a grounded answer and a false Cannot Confirm across calls.
_HYBRID_DRAFT_LLM_SEED = 7

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return the lazily-constructed answer-synthesis LLM (CODING_STANDARD §10).

    Hybrid owns its own call site (a new LLM-facing surface, ADR-0005), so this
    getter is the single seam hermetic tests monkeypatch — the deep retrieval /
    grounding modules are never mocked (CODING_STANDARD §6.3 / implement.md trap
    #1). ``temperature=0`` plus a fixed seed so a question never flip-flops
    between a grounded answer and a false Cannot Confirm across calls (parity
    with the Wiki/RAG stacks).
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            seed=_HYBRID_DRAFT_LLM_SEED,
            timeout=20,
            max_retries=1,
        )
    return _llm


def warm_llm_client() -> None:
    """Fire one tiny ping at the answer LLM client to prime its connection (issue #439).

    Mirrors ``markdown_kb.app.retrieval.warm_llm_client`` / ``vector_rag.app.
    retrieval.warm_llm_client`` for Stack C's own LLM singleton. Opt-in —
    called only from Gateway startup behind ``KB_WARMUP_PING`` (see
    ``gateway/app/warmup.py``). ``max_tokens=1`` keeps the completion itself a
    few tokens; the reply is discarded.

    Best-effort: any failure (auth, quota, network) is caught and logged, never
    raised — a failed ping degrades to the pre-issue-#439 behaviour (the client
    still lazily constructs + connects on the next real call) and never blocks
    Gateway startup.
    """
    try:
        get_llm().invoke("Hi", max_tokens=1)
        log_event("startup_warmup", "client=hybrid_llm status=ok")
    except Exception as exc:
        log_event(
            "startup_warmup",
            f"client=hybrid_llm status=failed exc={type(exc).__name__}",
        )


# ---------------------------------------------------------------------------
# Index lazy-load (both arms — cold-start parity with the other stacks)
# ---------------------------------------------------------------------------
def ensure_indexes_loaded() -> None:
    """Public cold-start warmup seam for the Hybrid (Stack C) two-arm index.

    The two-arm equivalent of ``markdown_kb.load_index_json`` (BM25 arm) and
    ``vector_rag``'s ``load_vector_index`` (dense arm): a fresh CLI / MCP /
    Gateway process can answer ``stack=hybrid`` straight from the committed
    seeds without an explicit build call.  Both seeds are committed
    (``.kb/index.json`` BM25, ``.kb/hybrid_dense/`` dense), so the common
    path is a pure load.  No-op (idempotent) when both arms are already warm.

    This is the PUBLIC seam external callers (e.g. ``kb_mcp``) MUST use to
    trigger a cold-start warmup — never import the old private name
    ``_ensure_indexes_loaded`` (CODING_STANDARD §2.4: recurring cross-package
    ``_private`` imports must be escalated to the owner's public API).
    """
    if not bm25_indexer.sections:
        bm25_indexer.load_index_json()
    if dense_index.vectorstore is None:
        dense_index.load_dense_index()


# ---------------------------------------------------------------------------
# Citation sources (wiki Section shape — parity with the Wiki stack)
# ---------------------------------------------------------------------------
def _build_sources(sections: list[Section]) -> list[dict]:
    """Build the response ``sources`` list from the fused wiki Sections.

    The common citation shape across stacks — ``source`` id + ``heading``
    breadcrumb + a content excerpt — plus an OPTIONAL resolvable wiki-page
    ``path`` so the reader UI renders a CLICKABLE citation that opens the source
    in-page (S4 / #314 AC3, parity with the Wiki stack's #266 viewer and the RAG
    stack's #307 docs path). Because Hybrid retrieves wiki ``Section`` objects,
    the page path is resolved by REFERENCE through the Wiki stack's own
    ``_wiki_page_path_for_section`` (ADR-0018 blessed cross-app reuse — the same
    pattern S2 uses for the calibrated thresholds): a pure ``metadata['type']`` →
    ``wiki/<subdir>/<file>.md`` mapping, no filesystem read. ``path`` is emitted
    only when it resolves (the Section carries a known wiki page ``type``), so a
    type-less test corpus degrades to a non-clickable citation rather than a
    clickable-but-404 one — keeping "a citation is clickable iff the viewer can
    open it" true at the source.

    The RRF fused score is deliberately NOT exposed: ADR-0018 fixes that it is not
    a calibrated relevance magnitude, so surfacing it as a citation "score" would
    mislead (and prevents the model reasoning "low score → guess", PROMPT.md Q3).
    ``sources`` is populated even on a Cannot Confirm verdict (the weak
    candidates), exactly as the existing stacks return their below-threshold
    sources.
    """
    result: list[dict] = []
    for sec in sections:
        entry: dict = {
            "source": sec.id,
            "heading": " > ".join(sec.heading_path),
            "content": sec.content[:240],
            # C10 coordinate (schema-invalid Filed Answer) for the reader teaching
            # tag — parity with the Wiki stack's source dict (markdown_kb.retrieval).
            "lint": qa_schema_lint_code(sec.metadata),
        }
        path = bm25_retrieval._wiki_page_path_for_section(sec)
        if path is not None:
            entry["path"] = path
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Public query path (mirrors markdown_kb.query composition — AC1)
# ---------------------------------------------------------------------------
def query(question: str) -> dict:
    """Answer a question with fused BM25 + dense retrieval over the wiki corpus.

    Returns a dict with keys (identical shape to the Wiki / RAG stacks so the CLI,
    MCP, and Gateway dispatch to all three uniformly):
        answer            — grounded text (may be the Cannot Confirm phrase)
        sources           — list of {source, heading, content} dicts
        grounding_outcome — GroundingOutcome (always present, never None)

    Pre-LLM gate (ADR-0018 per-arm OR-gate, enforced INSIDE the S2 retrieval deep
    module): when neither arm clears its calibrated native-score threshold the
    fused result is refused BEFORE any LLM call — the exact Cannot Confirm
    sentinel is returned with no synthesis (AC3).

    Post-LLM gate (ADR-0004 layer 3): ``grounding.verify`` validates every claim
    against the cited Sections; any unsupported claim → Cannot Confirm.

    Phase 13 S4 (#314): composes ``_retrieve_and_gate`` + ``_draft_and_verify``,
    the SAME two helpers ``stream_query`` composes, so the streaming and
    non-streaming paths can never diverge. Public contract unchanged.
    """
    gate = _retrieve_and_gate(question)
    if gate["early_exit"]:
        return {
            "answer": gate["answer"],
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }
    return _draft_and_verify(question, gate["sections"], gate["sources"])


def stream_query(question: str) -> Iterator[dict]:
    """Generator yielding the two dicts the SSE streaming endpoint consumes.

    Phase 13 S4 (#314) — the streaming variant of ``query``, mirroring
    ``markdown_kb.app.retrieval.stream_query`` and
    ``vector_rag.app.retrieval.stream_query`` so the Gateway reuses the shared
    serializer ``markdown_kb.app.sse.events_for_result`` for all three stacks
    (ADR-0009 verify-then-stream / sources-first; ADR-0010 the Gateway is the
    composition layer). The SSE machinery is NOT reimplemented here — this only
    surfaces the sources-then-answer split.

    Yields:
        1. A *partial* result dict immediately after retrieval (before any LLM
           call), so the Gateway can emit the ``sources`` SSE event right away.
           Shape: ``{_phase: "sources_ready", sources, grounding_outcome,
           early_exit, answer, sections}``. ``grounding_outcome`` here is the
           provisional pre-LLM OR-gate verdict.
        2. A *full* result dict after draft + Grounding Check.
           Shape: ``{answer, sources, grounding_outcome}`` — identical to what
           ``query()`` returns.

    ADR-0009: only verified text is ever emitted as tokens; on the pre-LLM gate
    the second yield IS the partial — the exact Cannot Confirm sentinel
    (imported, not paraphrased — trap #2) with NO LLM call (AC3).
    """
    gate = _retrieve_and_gate(question)

    # Yield the sources-ready partial so the Gateway can emit the sources event
    # before any LLM call (ADR-0009 sources-first invariant).
    yield {
        "_phase": "sources_ready",
        "sources": gate["sources"],
        "grounding_outcome": gate["grounding_outcome"],
        "early_exit": gate["early_exit"],
        "answer": gate.get("answer", ""),
        "sections": gate.get("sections", []),
    }

    if gate["early_exit"]:
        # Pre-LLM OR-gate refused — the partial IS the full result. Stream the
        # exact Cannot Confirm sentinel so the token stream is identical to the
        # other stacks' CC paths (SSE uniformity, ADR-0009 / §12.3). No LLM call.
        yield {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }
        return

    # LLM phase — draft + post-LLM Grounding Check (reuses query()'s composition).
    yield _draft_and_verify(question, gate["sections"], gate["sources"])


def _retrieve_and_gate(question: str) -> dict:
    """Lazy-load both arms, run S2 retrieve+gate, and build the citation sources.

    The single composition point shared by ``query`` and ``stream_query`` so the
    two paths can never drift (mirrors ``vector_rag``/``markdown_kb``, whose
    ``query``/``stream_query`` both compose a ``_retrieve_and_gate`` +
    ``_draft_and_verify`` pair). Returns a dict with:

        sources           — citation dicts (always populated, even on early_exit)
        grounding_outcome — the S2 pre-LLM OR-gate verdict
        early_exit        — True when the gate refused (callers skip the LLM)
        answer            — the Cannot Confirm sentinel on early_exit, else ""
        sections          — fused Sections for ``_draft_and_verify`` (empty on
                            early_exit — no synthesis is run)

    The ``chat_fallback`` log entry is written HERE on the pre-LLM gate (once),
    so both the streaming and non-streaming paths log the refusal identically.
    """
    ensure_indexes_loaded()

    gate = hybrid_retrieval.retrieve_and_gate(question)
    sections = gate["sections"]
    sources = _build_sources(sections)

    if gate["early_exit"]:
        # Pre-LLM OR-gate refused — Cannot Confirm with NO LLM call (AC3).
        truncated = question[:60].replace('"', "'")
        log_event(
            "chat_fallback",
            f'"{truncated}" reason={gate["grounding_outcome"].reason}',
        )
        return {
            "sources": sources,
            "grounding_outcome": gate["grounding_outcome"],
            "early_exit": True,
            "answer": CANNOT_CONFIRM_PHRASE,
            "sections": [],
        }

    return {
        "sources": sources,
        "grounding_outcome": gate["grounding_outcome"],
        "early_exit": False,
        "answer": "",
        "sections": sections,
    }


def _draft_and_verify(
    question: str,
    sections: list[Section],
    sources: list[dict],
) -> dict:
    """LLM draft + post-LLM Grounding Check over the fused wiki Sections.

    Called only when the pre-LLM OR-gate passed. Mirrors
    ``markdown_kb._draft_and_verify``: page expansion, the grounded prompt, the
    LLM self-refusal short-circuit, and ``grounding.verify`` are all reused by
    IMPORT (CODING_STANDARD §6.2 — nothing is synthesised here). Only the answer
    LLM call is hybrid's own (its LLM-facing surface).
    """
    # Page expansion: expand the fused hits to their full parent wiki pages so the
    # LLM receives page-coherent context. The expanded list drives prompt
    # construction and grounding; ``sources`` stays the fused top-k (the citations).
    expanded_sections = bm25_indexer.expand_to_pages(sections)
    prompt_text = build_prompt(question, expanded_sections)
    draft = _call_llm_with_error_handling(question, prompt_text)

    # LLM self-refusal short-circuit (parity with markdown_kb): the model emitted
    # the Cannot Confirm phrase verbatim per SYSTEM_PROMPT rules 3/6 (an
    # adjacent-absent query that cleared the gate but whose specific answer is
    # absent). The refusal carries no factual claim, so grounding.verify would have
    # nothing to refute and green-light a non-answer; treat it as Cannot Confirm
    # directly and skip the verifier (reason reuses claim_unsupported).
    if draft.strip() == CANNOT_CONFIRM_PHRASE:
        cited_ids = ",".join(sec.id for sec in expanded_sections)
        log_event(
            "chat_grounding_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason=claim_unsupported'
            f" cited={cited_ids}",
        )
        _write_chat_log(question, sections)
        return {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": sources,
            "grounding_outcome": GroundingOutcome(
                passed=False, reason="claim_unsupported"
            ),
        }

    # Post-LLM Grounding Check (ADR-0004 layer 3). Section satisfies CitableContent,
    # so verify() consumes the expanded Sections unchanged. verify() never raises —
    # all verifier failures map to reason="verifier_unavailable".
    outcome = grounding_module.verify(draft, expanded_sections)

    if outcome.passed:
        answer = draft
    else:
        answer = CANNOT_CONFIRM_PHRASE
        cited_ids = ",".join(sec.id for sec in expanded_sections)
        log_event(
            "chat_grounding_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason={outcome.reason}'
            f" cited={cited_ids}",
        )

    _write_chat_log(question, sections)
    return {"answer": answer, "sources": sources, "grounding_outcome": outcome}


# ---------------------------------------------------------------------------
# Internal helpers — hybrid's own LLM call site + log channel
# ---------------------------------------------------------------------------
def _call_llm_with_error_handling(question: str, prompt_text: str) -> str:
    """Invoke the synthesis LLM, mapping OpenAI exceptions to LLMError (ADR-0015).

    Hybrid's own call site (the new LLM-facing surface). The error mapping and the
    ``chat_error`` log kind mirror the Wiki/RAG wrappers so the three stacks render
    LLM failures identically, but the entry is written to hybrid_kb's OWN log
    channel (ADR-0018 additive invariant). LangChain message/client types are
    confined to this function (CODING_STANDARD §2.4).
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


def _write_chat_log(question: str, sections: list[Section]) -> None:
    """Append a chat log entry to hybrid_kb/log.md.

    Format:
        ## [<ts>] chat | "<truncated query>" top=<section id> count=N

    No score is logged — the RRF fused score is not a calibrated magnitude
    (ADR-0018), so the BM25 stack's ``top=<id>:<score>`` form is deliberately not
    mirrored here.
    """
    truncated = question[:60].replace('"', "'")
    if sections:
        summary = f'"{truncated}" top={sections[0].id} count={len(sections)}'
    else:
        summary = f'"{truncated}" top=none count=0'
    log_event("chat", summary)
