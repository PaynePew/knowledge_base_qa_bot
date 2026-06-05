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
from .normalizer import normalize_rag_results, normalize_wiki_results

# ---------------------------------------------------------------------------
# Server-level instructions (~150 tokens)
# ---------------------------------------------------------------------------
_INSTRUCTIONS = (
    "You are connected to a grounded knowledge-base assistant.\n\n"
    "Available tools:\n"
    "- kb_search_v1: Retrieve raw Sections or Chunks from the KB index.  "
    "Use this when the user wants to see the raw evidence or when you want to "
    "reason over sources yourself before composing an answer.\n\n"
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
# Patch schema to add additionalProperties:false (ADR-0016 strict schema)
# ---------------------------------------------------------------------------
# FastMCP generates the tool parameters schema from the function signature but
# does not set additionalProperties:false by default.  We patch it here so MCP
# hosts (e.g. Claude Desktop) receive the strict schema and know not to send
# extra fields.  This does not affect FastMCP's internal argument validation.
def _add_strict_schema() -> None:
    """Patch kb_search_v1's parameter schema to include additionalProperties:false."""
    tool = mcp._tool_manager.get_tool("kb_search_v1")
    if tool is not None:
        tool.parameters["additionalProperties"] = False


_add_strict_schema()
