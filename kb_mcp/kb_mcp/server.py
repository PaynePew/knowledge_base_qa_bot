"""FastMCP server exposing the knowledge base over stdio.

Phase 12 Slice 1 (ADR-0016).  Wraps ``markdown_kb`` and ``vector_rag`` deep
modules directly (NOT the Gateway).  Exposes six tools:
  - ``kb_ask_v1``    — grounded-answer tool (LLM draft + Grounding Check)
  - ``kb_search_v1`` — raw-evidence search with no LLM call
  - ``kb_read_hot_v1``  — read working-memory hot cache
  - ``kb_save_hot_v1``  — persist working-memory hot cache
  - ``kb_index_v1``  — rebuild the Section Index via build_index (Slice 231)
  - ``kb_lint_v1``   — run the Lint Pass via run_lint (Slice 231)

Launch via ``python -m kb_mcp`` (stdio transport, Claude Desktop compatible).

Server ``instructions`` (~200 tokens) guide the MCP host on:
  - tool-choice (kb_ask for grounded answers; kb_search for raw evidence)
  - ``stack`` default (always start with ``wiki``; only switch to ``rag`` on
    explicit user instruction)
  - Cannot-Confirm guidance (surface ``grounding.reason`` to the user; do NOT
    retry to force an answer)
  - LLMError guidance (isError=True means the LLM service is unavailable;
    code='LLM_UNAVAILABLE' is retryable, 'LLM_ERROR' is not)
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from .freshness import reload_if_stale
from .hot_cache import read_hot, save_hot
from .normalizer import normalize_rag_results, normalize_wiki_results

# ---------------------------------------------------------------------------
# Server-level instructions (~200 tokens)
# ---------------------------------------------------------------------------
_INSTRUCTIONS = (
    "You are connected to a grounded knowledge-base assistant.\n\n"
    "Available tools:\n"
    "- kb_ask_v1: Ask a question and receive a grounded answer (LLM synthesis "
    "with citation and grounding check).  Use this as the default for "
    "user questions.  Returns {stack, answer, citations, grounding}.\n"
    "- kb_search_v1: Retrieve raw Sections or Chunks from the KB index with no "
    "LLM synthesis.  Use this when the user wants to see the raw evidence or "
    "when you want to reason over sources yourself before composing an answer.\n"
    "- kb_read_hot_v1: Read working-memory hot cache (wiki/hot.md).  "
    "Call this at session start to recover where the previous session left off.  "
    "Returns empty string on the first session — that is normal, not an error.\n"
    "- kb_save_hot_v1: Persist a working-memory summary to the hot cache.  "
    "Call this at session end (or at a natural checkpoint) with a ~500-word "
    "summary composed by you.  The server only persists the bytes; you compose "
    "the summary.\n\n"
    "Stack guidance:\n"
    "- Always start with stack='wiki' (curated BM25 index).  "
    "Only switch to stack='rag' when the user explicitly asks to use the "
    "Vector RAG arm for comparison.\n\n"
    "Cannot-Confirm guidance:\n"
    "- When kb_ask returns grounding.passed=false, the KB cannot support the "
    "answer.  Surface grounding.reason to the user and do NOT retry.  "
    '"Cannot Confirm" is a valid, expected KB boundary — not a failure.\n\n'
    "LLM error guidance:\n"
    "- If kb_ask or kb_lint_v1 returns isError=true, the LLM service failed.  "
    "code='LLM_UNAVAILABLE' is transient — retry after a short wait.  "
    "code='LLM_ERROR' is non-recoverable — report the message to the user."
)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(name="kb_mcp", instructions=_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Tool: kb_ask_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_ask_v1",
    description=(
        "Ask a question and receive a grounded answer synthesised by the LLM "
        "with a post-LLM Grounding Check (ADR-0004).  Use this as the default "
        "tool for user questions.\n\n"
        "Parameters:\n"
        "  query  — the user's question (required)\n"
        "  stack  — 'wiki' (curated BM25, default) or 'rag' (Vector RAG)\n\n"
        "Returns on success: {stack, answer, citations, grounding}\n"
        "  answer      — grounded text, or the Cannot-Confirm phrase when the\n"
        "                KB cannot support the answer\n"
        "  citations   — list of {source, heading, score, content} dicts from\n"
        "                BM25 retrieval (the sections the LLM reasoned over)\n"
        "  grounding   — {passed: bool, reason: str}\n"
        "                passed=false means Cannot Confirm (a success result,\n"
        "                NOT isError — it is a valid KB boundary per ADR-0016)\n\n"
        "Returns isError=true on LLM failure:\n"
        "  {code, message} where code is 'LLM_UNAVAILABLE' (retryable) or\n"
        "  'LLM_ERROR' (non-retryable, report message to user)."
    ),
)
def kb_ask_v1(
    query: Annotated[str, Field(description="The question to answer.")],
    stack: Annotated[
        Literal["wiki", "rag"],
        Field(description="Retrieval stack: 'wiki' (BM25, default) or 'rag' (Vector RAG)."),
    ] = "wiki",
) -> Any:
    """Answer a question with LLM synthesis and a post-LLM Grounding Check.

    Routes to ``markdown_kb.app.retrieval.query()`` (wiki stack) or the
    vector_rag equivalent.  Returns a normalised dict on success.  On
    ``LLMError`` (ADR-0015), returns a ``CallToolResult`` with ``isError=True``
    so the MCP host receives a structured error payload instead of a raw
    exception.

    Cannot Confirm (``grounding.passed=False``) is ALWAYS a success result
    (not ``isError``).  The host must treat it as a KB boundary, not a failure.

    Defaults are enforced server-side — the MCP host MUST NOT rely on the model
    supplying default values (ADR-0016 strict schema).
    """
    from markdown_kb.app.errors import LLMError

    # Enforce defaults server-side regardless of what the host sends.
    if stack is None:
        stack = "wiki"

    try:
        if stack == "wiki":
            reload_if_stale()
            from markdown_kb.app.retrieval import query as wiki_query

            result = wiki_query(query)
        else:
            from vector_rag.app.retrieval import query as rag_query  # type: ignore[import-untyped]

            result = rag_query(query)

    except LLMError as exc:
        # ADR-0015 / ADR-0016: LLMError → structured MCP isError payload.
        # retryable=True  → code='LLM_UNAVAILABLE'
        # retryable=False → code='LLM_ERROR'
        code = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        payload = json.dumps({"code": code, "message": exc.message})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    # Map retrieval result to the MCP neutral shape.
    # result keys: answer, sources, grounding_outcome
    # sources: list of {source, heading, score, content, derived_from}
    # grounding_outcome: GroundingOutcome(passed, reason, ...)
    grounding_outcome = result["grounding_outcome"]
    citations = [
        {
            "source": src["source"],
            "heading": src["heading"],
            "score": src["score"],
            "content": src["content"],
        }
        for src in result.get("sources", [])
    ]
    return {
        "stack": stack,
        "answer": result["answer"],
        "citations": citations,
        "grounding": {
            "passed": grounding_outcome.passed,
            "reason": grounding_outcome.reason,
        },
    }


# ---------------------------------------------------------------------------
# Tool: kb_search_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_search_v1",
    description=(
        "Search the knowledge base and return raw Sections or Chunks with no "
        "LLM synthesis.  Use this tool to retrieve evidence you will reason "
        "over yourself, or when the user asks to see the raw KB content.\n\n"
        "Parameters:\n"
        "  query  — the search string (required)\n"
        "  stack  — 'wiki' (curated BM25, default) or 'rag' (Vector RAG)\n"
        "  k      — number of results to return (1–10, default 3)\n\n"
        "Returns: {stack, results: [{id, content, score|null}]}\n"
        "  score is a BM25 float for wiki; null for rag (no score exposed)."
    ),
)
def kb_search_v1(
    query: Annotated[str, Field(description="The search query string.")],
    stack: Annotated[
        Literal["wiki", "rag"],
        Field(description="Retrieval stack: 'wiki' (BM25, default) or 'rag' (Vector RAG)."),
    ] = "wiki",
    k: Annotated[
        int,
        Field(
            default=3,
            ge=1,
            le=10,
            description="Number of results to return (1–10).",
        ),
    ] = 3,
) -> dict:
    """Search the knowledge base index, returning normalized results.

    Defaults are enforced server-side — the MCP host MUST NOT rely on the model
    supplying default values (ADR-0016 strict schema).
    """
    # Enforce defaults server-side regardless of what the host sends.
    if stack is None:
        stack = "wiki"
    if k is None:
        k = 3
    k = max(1, min(10, k))

    if stack == "wiki":
        reload_if_stale()
        from markdown_kb.app.indexer import search as wiki_search

        hits = wiki_search(query, k=k)
        results = normalize_wiki_results(hits)
    else:
        from vector_rag.app.indexer import search as rag_search

        chunks = rag_search(query, k=k)
        results = normalize_rag_results(chunks)

    return {"stack": stack, "results": results}


# ---------------------------------------------------------------------------
# Tool: kb_read_hot_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_read_hot_v1",
    description=(
        "Read the working-memory hot cache (wiki/hot.md).  "
        "Call this at session start to recover where the previous session left off.\n\n"
        "Returns: {content: str}\n"
        "  content is the full text of wiki/hot.md, or '' on the first session "
        "(file absent is a normal state, not an error).\n\n"
        "This is a TOOL (agent-initiated), not a resource — the agent decides "
        "when to call it (L0 of the read-depth budget per ADR-0016)."
    ),
)
def kb_read_hot_v1() -> dict:
    """Return the hot-cache contents, or empty string when absent.

    Uses the module-level ``HOT_PATH`` which tests monkeypatch to a tmp dir.
    """
    content = read_hot()
    return {"content": content}


# ---------------------------------------------------------------------------
# Tool: kb_save_hot_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_save_hot_v1",
    description=(
        "Persist a working-memory summary to the hot cache (wiki/hot.md).\n\n"
        "Parameters:\n"
        "  summary — the ~500-word working-memory summary (required).  "
        "The host composes the summary; the server only persists the bytes.\n\n"
        "Returns: {ok: true} on success.\n\n"
        "Writes atomically (tmp-file + os.replace) so a crash mid-write never "
        "leaves a partial file.  Overwrites any previous hot.md."
    ),
)
def kb_save_hot_v1(
    summary: Annotated[str, Field(description="The working-memory summary to persist.")],
) -> dict:
    """Atomically persist ``summary`` to the hot-cache file.

    Uses the module-level ``HOT_PATH`` (from hot_cache) which tests monkeypatch.
    """
    save_hot(summary)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Tool: kb_index_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_index_v1",
    description=(
        "Rebuild the Section Index from the curated wiki (ADR-0003).  "
        "Call this after curating wiki pages (adding, editing, or removing pages) "
        "so the BM25 index reflects the latest content.\n\n"
        "Takes no parameters.\n\n"
        "Returns: {files_indexed, sections_indexed}\n"
        "  files_indexed   — number of source files indexed\n"
        "  sections_indexed — total Sections in the rebuilt index"
    ),
)
def kb_index_v1() -> dict:
    """Rebuild the Section Index by calling build_index() from the deep module.

    Thin wrapper: no defaults to enforce (zero-arg tool).  The deep module
    ``markdown_kb.app.indexer.build_index`` scans SOURCE_DIRS, persists
    ``.kb/index.json`` atomically, and returns ``(files_indexed, sections_indexed)``.

    No LLMError can arise here — build_index is a pure local operation.
    """
    from markdown_kb.app.indexer import build_index

    files_indexed, sections_indexed = build_index()
    return {"files_indexed": files_indexed, "sections_indexed": sections_indexed}


# ---------------------------------------------------------------------------
# Tool: kb_lint_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_lint_v1",
    description=(
        "Run the Lint Pass over the wiki and return structured findings "
        "(ADR-0003, Phase 5).  Use this to retrieve lint findings so you "
        "can reason over contradictions, stale pages, coverage gaps, and "
        "propose curator actions.\n\n"
        "Parameters:\n"
        "  include_c5 — whether to run the LLM-backed C5 page-pair "
        "contradiction check (default true).  Pass false to skip C5 and "
        "receive only the fast local checks.\n\n"
        "Returns on success: {report_path, findings, summary, check_errors}\n"
        "  findings   — structured per-check finding lists\n"
        "  summary    — {total_findings, findings_by_check, llm_calls, "
        "cost_usd, c5_pairs_capped, generated_at}\n"
        "  check_errors — dict of check_id → error string for any check that "
        "raised (other checks still ran — continue-on-error semantics)\n\n"
        "Returns isError=true when the C5 LLM call fails catastrophically:\n"
        "  {code, message} where code is 'LLM_UNAVAILABLE' (retryable) or\n"
        "  'LLM_ERROR' (non-retryable, report message to user).\n"
        "  Individual per-pair LLM errors within C5 are NOT isError — they "
        "are recorded in check_errors['c5'] and the check returns partial "
        "results (continue-on-error per the deep module contract)."
    ),
)
def kb_lint_v1(
    include_c5: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether to run C5 (LLM-backed page-pair contradiction check). "
                "Pass false to skip C5 and receive only fast local checks."
            ),
        ),
    ] = True,
) -> Any:
    """Run the Lint Pass via run_lint() from the deep module.

    Thin wrapper.  Defaults are enforced server-side (ADR-0016): ``include_c5``
    defaults to True so the full lint suite runs by default.

    On ``LLMError`` (ADR-0015), returns a ``CallToolResult`` with ``isError=True``
    so the MCP host receives a structured error payload.  This covers the case
    where the *entire* C5 check fails (e.g. LLM unreachable before any pair is
    judged).  Per-pair C5 failures are handled inside the deep module via
    continue-on-error semantics and appear in ``check_errors['c5']`` of the
    success payload — they do NOT trigger isError.
    """
    from markdown_kb.app.errors import LLMError
    from markdown_kb.app.lint import run_lint

    # Enforce default server-side
    if include_c5 is None:
        include_c5 = True

    try:
        response = run_lint(include_c5=include_c5)
    except LLMError as exc:
        # ADR-0015 / ADR-0016: LLMError → structured MCP isError payload.
        code = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        payload = json.dumps({"code": code, "message": exc.message})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    # Serialise the Pydantic LintResponse to a plain dict for MCP transport.
    # model_dump() converts nested Pydantic models to plain dicts/lists;
    # mode="json" ensures non-JSON-native types (e.g. datetime) are strings.
    return response.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Patch schema to add additionalProperties:false (ADR-0016 strict schema)
# ---------------------------------------------------------------------------
# FastMCP generates the tool parameters schema from the function signature but
# does not set additionalProperties:false by default.  We patch it here so MCP
# hosts (e.g. Claude Desktop) receive the strict schema and know not to send
# extra fields.  This does not affect FastMCP's internal argument validation.
def _add_strict_schema() -> None:
    """Patch tool parameter schemas to include additionalProperties:false."""
    for tool_name in (
        "kb_ask_v1",
        "kb_search_v1",
        "kb_read_hot_v1",
        "kb_save_hot_v1",
        "kb_index_v1",
        "kb_lint_v1",
    ):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters["additionalProperties"] = False


_add_strict_schema()
