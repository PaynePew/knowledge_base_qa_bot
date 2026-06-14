"""Deep module per Ousterhout. Public surface: ``query``, ``stream_query``.

Retrieval layer — query() and stream_query() for the grounded /chat endpoint.

query() flow:
  1. tokenize query
  2. search top-3 Sections via BM25
  3. pre-LLM Cannot Confirm gate (ADR-0001)
  4. build_prompt with Citation markers
  5. call LLM (with error mapping for OpenAI exceptions)
  6. post-LLM Grounding Check via grounding.verify() (ADR-0004 layer 3)
  7. write chat log entry
  8. return {answer, sources, grounding_outcome}

stream_query() (Phase 9, ADR-0009) wraps query() to yield a partial result
dict immediately after retrieval (so the gateway can emit the sources SSE
event before the LLM calls run), then yields the full result after
draft + verify complete.

The retrieve / draft split is implemented via two private helpers:
  _retrieve_and_gate() — BM25 search + pre-LLM Cannot Confirm gates
  _draft_and_verify()  — build_prompt + LLM draft + Grounding Check
Public query() composes them; contract is unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import openai
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import grounding as grounding_module
from . import indexer
from .errors import LLMError
from .grounding import GroundingOutcome
from .logger import log_event
from .prompt_builder import SYSTEM_PROMPT, build_prompt

# Score threshold below which retrieval is treated as "no match" and the bot
# returns Cannot Confirm.
#
# The default is empirically justified by the #253 calibration sweep
# (eval/negative_case/calibration_report.md): every threshold in the [0.25, 1.25]
# plateau rejects all clearly-out-of-scope queries (~0 BM25 score) at 0%
# over-refusal of in-scope queries (min in-scope score ~1.41). 0.5 sits inside
# that plateau with margin on both sides, so it is kept (a working default is not
# churned to a different equally-good value). The residual adjacent-absent leaks
# score inside the in-scope range and are unreachable by ANY threshold without
# over-refusing real queries — they need semantic reranking (Phase 13 / FM2), not
# threshold tuning.
#
# Override with KB_SCORE_THRESHOLD env var. Read at import time: a server restart
# picks up a new value; runtime changes do not (tests monkeypatch _SCORE_THRESHOLD
# directly).
_KB_SCORE_THRESHOLD_DEFAULT = 0.5
_SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", str(_KB_SCORE_THRESHOLD_DEFAULT)))

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

    Phase 9: composes _retrieve_and_gate() + _draft_and_verify().
    Public contract is unchanged; the split is behaviour-preserving.
    """
    gate = _retrieve_and_gate(question)
    if gate["early_exit"]:
        # Pre-LLM gate fired — return early without calling the LLM.
        return {
            "answer": gate["answer"],
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }

    # LLM phase — draft + Grounding Check.
    result = _draft_and_verify(question, gate["ranked"], gate["sources"])
    return result


def stream_query(question: str) -> Iterator[dict]:
    """Generator that yields two dicts for use by the SSE streaming endpoint.

    Phase 9 — ADR-0009 (verify-then-stream / sources-first).

    Yields:
        1. A *partial* result dict immediately after retrieval (before any
           LLM call), so the gateway can emit the ``sources`` SSE event
           right away (~instant for BM25, ~1 embedding round-trip for RAG).
           Shape: ``{sources, grounding_outcome, _phase: "sources_ready"}``.
           ``grounding_outcome`` at this stage is provisional — it is the
           pre-LLM gate outcome (may be a Cannot Confirm if below threshold).
        2. A *full* result dict after draft + Grounding Check complete.
           Shape: ``{answer, sources, grounding_outcome}`` — identical to
           what ``query()`` returns.

    The gateway endpoint must:
      a. Emit ``sources`` SSE event from yield 1.
      b. Emit ``token`` + ``done`` SSE events from yield 2.

    ADR-0009: only verified text is ever emitted as tokens.  The generator
    never yields an unverified draft.
    """
    gate = _retrieve_and_gate(question)

    # Yield the sources-ready partial result so the gateway can emit the
    # sources event before making any LLM call.
    yield {
        "_phase": "sources_ready",
        "sources": gate["sources"],
        "grounding_outcome": gate["grounding_outcome"],
        "early_exit": gate["early_exit"],
        "answer": gate.get("answer", ""),
        "ranked": gate.get("ranked", []),
    }

    if gate["early_exit"]:
        # Pre-LLM gate fired — the partial result IS the full result.
        # Yield the final form so the caller always gets a full result on
        # the second yield.
        #
        # SSE uniformity (ADR-0009 / Phase 9 Slice 2): all five CC reasons must
        # stream CANNOT_CONFIRM_PHRASE in the token events so the UI token
        # stream is identical regardless of which gate fired.  The ``index_missing``
        # path in _retrieve_and_gate returns NOT_INDEXED_MESSAGE (the verbose
        # curator-targeted string); normalise to CANNOT_CONFIRM_PHRASE here so
        # stream callers never need to branch on the reason.  query() is
        # unaffected (it does not call this path via stream_query).
        # The specific reason is still preserved in grounding_outcome.reason for
        # machine consumers (done.reason in the SSE done event).
        yield {
            "answer": CANNOT_CONFIRM_PHRASE,
            "sources": gate["sources"],
            "grounding_outcome": gate["grounding_outcome"],
        }
        return

    # LLM phase — draft + Grounding Check.
    result = _draft_and_verify(question, gate["ranked"], gate["sources"])

    # Phase 9 Slice 4: file on grounding-pass — same behaviour as /chat.
    # Filing happens server-side at the post-verify point, independent of
    # client delivery (a disconnected client never causes partial/unfiled
    # state, and a partial answer is never filed). The existing SSE serializer
    # already reads result.get("filed") and populates done.filed.
    # Import is deferred to avoid a circular import with qa.py.
    from . import qa as _qa_module  # noqa: PLC0415

    filed = _qa_module.dispatch_filing(question, result)
    # Attach filing outcome to result so events_for_result() picks it up.
    result = {**result, "filed": filed}

    yield result


# ---------------------------------------------------------------------------
# Phase 9 private helpers — retrieve+gate and draft+verify
# ---------------------------------------------------------------------------


def _retrieve_and_gate(question: str) -> dict:
    """BM25 retrieval + all pre-LLM Cannot Confirm gates (ADR-0001).

    Returns a dict with:
        sources          — list of {source, heading, score, content, derived_from}
        grounding_outcome — provisional outcome (pre-LLM gate result)
        early_exit       — True when a pre-LLM gate fired (no LLM needed)
        answer           — set to the Cannot Confirm phrase on early_exit paths
        ranked           — raw (Section, score) pairs for _draft_and_verify;
                           only meaningful when early_exit is False

    Callers (query() and stream_query()) use the early_exit flag to decide
    whether to call _draft_and_verify().  The sources list is always
    populated even on early_exit paths — the gateway emits a sources SSE
    event before checking early_exit.
    """
    if not indexer.sections:
        # Lazy-load the persisted .kb/index.json from disk so a fresh Gateway
        # process can serve stack=wiki without requiring a POST /wiki/index call
        # in the same process (mirrors the RAG lazy-load fix, issue #133 / #148).
        # load_index_json() returns (0, 0) and leaves sections=[] when no
        # persisted index exists on disk.
        indexer.load_index_json()

    if not indexer.sections:
        log_event(
            "chat_fallback",
            f'"{question[:60].replace(chr(34), chr(39))}" reason=not_indexed',
        )
        return {
            "sources": [],
            "grounding_outcome": GroundingOutcome(passed=False, reason="index_missing"),
            "early_exit": True,
            "answer": NOT_INDEXED_MESSAGE,
            "ranked": [],
        }

    ranked = indexer.search(question, k=3)
    top_score = ranked[0][1] if ranked else 0.0

    # Phase 6 Slice 6-3: each entry carries an optional ``derived_from`` chain
    # populated from the parent wiki page's ``frontmatter.sources``. This is
    # response-only audit data — the chain must NEVER appear in the LLM CONTEXT
    # or verifier prompt (ADR-0006 / PRD #78 Q4 W1 invariant).
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
        truncated = question[:60].replace('"', "'")
        gate_reason = "retrieval_empty" if not ranked else "below_threshold"
        if ranked:
            top_sec = ranked[0][0]
            log_event(
                "chat_fallback",
                f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}'
                f" top_section={top_sec.id}",
            )
        else:
            log_event(
                "chat_fallback",
                f'"{truncated}" reason=below_threshold top_score={round(top_score, 3)}',
            )
        return {
            "sources": sources,
            "grounding_outcome": GroundingOutcome(passed=False, reason=gate_reason),
            "early_exit": True,
            "answer": CANNOT_CONFIRM_PHRASE,
            "ranked": [],
        }

    return {
        "sources": sources,
        "grounding_outcome": GroundingOutcome(passed=True, reason="claim_supported"),
        "early_exit": False,
        "answer": "",
        "ranked": ranked,
    }


def _draft_and_verify(
    question: str,
    ranked: list,
    sources: list[dict],
) -> dict:
    """LLM draft + post-LLM Grounding Check (ADR-0004 layer 3).

    Called only when the pre-LLM gates passed (early_exit is False).

    Args:
        question: The original user query.
        ranked:   (Section, score) pairs from BM25.
        sources:  Already-built sources list (passed through unchanged).

    Returns:
        Full result dict: {answer, sources, grounding_outcome}.
        Never returns unverified text — on grounding failure, answer is
        CANNOT_CONFIRM_PHRASE and grounding_outcome.passed is False.
    """
    # B3 page expansion (Slice 4-4): expand BM25 hits to full parent pages so
    # the LLM receives page-coherent context. The expanded list is used for
    # prompt construction and grounding verification; sources[] stays BM25 top-K.
    ranked_sections = [sec for sec, _score in ranked]
    expanded_sections = indexer.expand_to_pages(ranked_sections)
    prompt_text = build_prompt(question, expanded_sections)

    draft = _call_llm_with_error_handling(question, prompt_text)

    # Post-LLM Grounding Check (ADR-0004 layer 3).
    # verify() never raises — all verifier failures map to
    # grounding_outcome.reason = "verifier_unavailable".
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
    """Invoke the LLM and map OpenAI exceptions to a transport-agnostic LLMError.

    Error mapping (ADR-0015 — status table moves to the HTTP route):
      - APITimeoutError, RateLimitError → LLMError(retryable=True)
      - AuthenticationError            → LLMError(retryable=False)
      - Any other APIError             → LLMError(retryable=False)

    Each error is logged to wiki/log.md via log_event with the appropriate
    kind tag (openai_transient | openai_auth | openai_api) BEFORE raising,
    so the log always captures the failure regardless of the caller's
    exception-handling path.

    Callers that render via HTTP map retryable → 503, non-retryable → 500.
    Other adapters (SSE, MCP, CLI) render the LLMError in their own way.
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
            "chat_error",
            f'"{truncated}" kind=openai_auth exc={type(exc).__name__}',
        )
        raise LLMError(
            retryable=False,
            message="LLM service auth failed (check OPENAI_API_KEY).",
        ) from exc
    except openai.APIError as exc:
        log_event(
            "chat_error",
            f'"{truncated}" kind=openai_api exc={type(exc).__name__}',
        )
        raise LLMError(
            retryable=False,
            message=f"LLM service error: {exc!s}",
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
