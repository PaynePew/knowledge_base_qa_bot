"""FastMCP server exposing the knowledge base over stdio.

Phase 12 Slice 1 (ADR-0016).  Wraps ``markdown_kb`` and ``vector_rag`` deep
modules directly (NOT the Gateway).  Exposes ``kb_search_v1`` as the
tracer-bullet tool — raw-evidence search with no LLM call.

Launch via ``python -m kb_mcp`` (stdio transport, Claude Desktop compatible).

Server ``instructions`` (~150 tokens) guide the MCP host on:
  - tool-choice (when to call each tool)
  - ``stack`` default (always start with ``wiki``; only switch to ``rag`` on
    explicit user instruction)
  - Cannot-Confirm guidance (surface ``grounding.reason`` to the user; do NOT
    retry to force an answer)
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .freshness import reload_if_stale
from .hot_cache import read_hot, save_hot
from .normalizer import normalize_rag_results, normalize_wiki_results

# ---------------------------------------------------------------------------
# Server-level instructions (~150 tokens)
# ---------------------------------------------------------------------------
_INSTRUCTIONS = (
    "You are connected to a grounded knowledge-base assistant.\n\n"
    "Available tools:\n"
    "- kb_search_v1: Retrieve raw Sections or Chunks from the KB index.  "
    "Use this when the user wants to see the raw evidence or when you want to "
    "reason over sources yourself before composing an answer.\n"
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
    "- When the KB cannot support an answer, surface the grounding.reason to "
    "the user and do NOT retry to force an answer.  "
    '"Cannot Confirm" is a valid, expected KB boundary — not a failure.'
)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(name="kb_mcp", instructions=_INSTRUCTIONS)


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
# Patch schema to add additionalProperties:false (ADR-0016 strict schema)
# ---------------------------------------------------------------------------
# FastMCP generates the tool parameters schema from the function signature but
# does not set additionalProperties:false by default.  We patch it here so MCP
# hosts (e.g. Claude Desktop) receive the strict schema and know not to send
# extra fields.  This does not affect FastMCP's internal argument validation.
def _add_strict_schema() -> None:
    """Patch tool parameter schemas to include additionalProperties:false."""
    for tool_name in ("kb_search_v1", "kb_read_hot_v1", "kb_save_hot_v1"):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters["additionalProperties"] = False


_add_strict_schema()
