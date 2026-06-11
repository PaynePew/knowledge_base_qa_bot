"""FastMCP server exposing the knowledge base over stdio.

Phase 12 Slice 1 (ADR-0016) + Slice 4 (capture, issue #230) + Slice 6 (import, issue #233).
Wraps ``markdown_kb`` and ``vector_rag`` deep modules directly (NOT the Gateway).
Exposes six tools:
  - ``kb_ask_v1``     — grounded-answer tool (LLM draft + Grounding Check)
  - ``kb_search_v1``  — raw-evidence search with no LLM call
  - ``kb_read_hot_v1``   — read working-memory hot cache
  - ``kb_save_hot_v1``   — persist working-memory hot cache
  - ``kb_capture_v1``    — author a Markdown Source from conversation to docs/
  - ``kb_import_v1``     — import a local file into docs/ via Import deep module

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
    "the summary.\n"
    "- kb_capture_v1: Author a Markdown Source from this conversation and "
    "persist it to docs/.  Use this to turn session reasoning into a permanent "
    "KB Source.  Capture skips Import — content is already canonical Markdown.  "
    "Provenance frontmatter is stamped automatically.  After capturing, run "
    "kb_ingest_v1 / kb_index_v1 to make it retrievable.  "
    "Returns {ok: true, path: str}.  Unsafe filenames return isError with "
    "code='CAPTURE_REJECTED'.\n"
    "- kb_import_v1: Import a local file from the filesystem into docs/ by "
    "reading it and converting it via the Import deep module.  Supported: "
    ".html, .txt, .md.  Returns {ok: true, source: str, status: str}.  "
    "Bad paths / unsafe basenames return isError with code='IMPORT_REJECTED'.\n\n"
    "Stack guidance:\n"
    "- Always start with stack='wiki' (curated BM25 index).  "
    "Only switch to stack='rag' when the user explicitly asks to use the "
    "Vector RAG arm for comparison.\n\n"
    "Cannot-Confirm guidance:\n"
    "- When kb_ask returns grounding.passed=false, the KB cannot support the "
    "answer.  Surface grounding.reason to the user and do NOT retry.  "
    '"Cannot Confirm" is a valid, expected KB boundary — not a failure.\n\n'
    "LLM error guidance:\n"
    "- If kb_ask returns isError=true, the LLM service failed.  "
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
# Tool: kb_capture_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_capture_v1",
    description=(
        "Author a Markdown Source directly from this conversation and persist it "
        "to docs/.  Use this when you want to turn session reasoning into a "
        "permanent KB Source without leaving the conversation.\n\n"
        "Parameters:\n"
        "  filename — plain basename for the new Source (required, e.g. 'my_note.md').\n"
        "             Must not contain path separators, '..', or control characters.\n"
        "  content  — the Markdown body of the Source (required).\n\n"
        "Returns on success: {ok: true, path: str}\n"
        "  path — absolute path of the written file in docs/.\n\n"
        "Returns isError=true on rejection:\n"
        "  {code: 'CAPTURE_REJECTED', message: str}\n"
        "  Filename validation failures (traversal, separators) produce this error.\n\n"
        "Capture skips Import — content is assumed to be canonical Markdown already.\n"
        "Mandatory provenance frontmatter (origin/created_at/authored_by) is stamped\n"
        "automatically by the server; the caller must NOT include it in content.\n\n"
        "The captured Source flows into the normal Ingest → Index lifecycle via "
        "kb_ingest_v1 / kb_index_v1 — Capture only writes the Source to disk."
    ),
)
def kb_capture_v1(
    filename: Annotated[
        str, Field(description="Plain basename for the new Source (e.g. 'note.md').")
    ],
    content: Annotated[str, Field(description="The Markdown body of the Source.")],
) -> Any:
    """Write a Markdown Source to docs/ with mandatory provenance frontmatter.

    Delegates to ``markdown_kb.app.capture.capture_source``.  On
    ``ValueError`` (unsafe filename), returns a ``CallToolResult`` with
    ``isError=True`` so the MCP host receives a structured error payload
    instead of a raw exception.
    """
    from markdown_kb.app.capture import capture_source

    try:
        target = capture_source(filename, content)
    except ValueError as exc:
        payload = json.dumps({"code": "CAPTURE_REJECTED", "message": str(exc)})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    return {"ok": True, "path": str(target)}


# ---------------------------------------------------------------------------
# Tool: kb_import_v1
# ---------------------------------------------------------------------------
@mcp.tool(
    name="kb_import_v1",
    description=(
        "Import a local file into the knowledge base by reading it from the "
        "filesystem and converting it to a docs/ Source.  Reuses the Import "
        "deep module (``import_path``) so format conversion is always "
        "programmatic — the same path the CLI ``kb import`` uses.\n\n"
        "Parameters:\n"
        "  path — absolute path to the local file to import (required).\n"
        "         Supported formats: .html, .txt, .md.\n"
        "         The basename must be traversal-safe (no '#', ':', separators,\n"
        "         control characters, or bidi override codepoints).\n\n"
        "Returns on success: {ok: true, source: str, status: str}\n"
        "  source — basename of the written docs/ Source (e.g. 'note.md').\n"
        "  status — 'created' (fresh write), 'updated' (hash-drift overwrite),\n"
        "           or 'skipped' (hash-match no-op).\n\n"
        "Returns isError=true on rejection:\n"
        "  {code: 'IMPORT_REJECTED', message: str}\n"
        "  Covers: file not found, traversal-unsafe basename, unsupported\n"
        "  extension, empty/oversized file, and other ImportPathError cases.\n\n"
        "Reading an arbitrary local path is safe under the single-operator\n"
        "posture (ADR-0017).  After importing, run kb_ingest_v1 / kb_index_v1\n"
        "to make the new Source retrievable."
    ),
)
def kb_import_v1(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to the local file to import "
                "(e.g. 'C:\\\\docs\\\\note.txt' or '/home/user/note.html')."
            )
        ),
    ],
) -> Any:
    """Read a local file and convert it to a docs/ Source via the Import deep module.

    Delegates to ``markdown_kb.app.importer.import_path``.  On
    ``ImportPathError``, returns a ``CallToolResult`` with ``isError=True``
    so the MCP host receives a structured error payload instead of a raw exception.

    Reading an arbitrary local path is safe under the single-operator posture
    (ADR-0017).  Defaults are enforced server-side (ADR-0016 strict schema).
    """
    from pathlib import Path as _Path

    from markdown_kb.app.importer import ImportPathError, import_path

    try:
        result = import_path(_Path(path))
    except ImportPathError as exc:
        payload = json.dumps({"code": "IMPORT_REJECTED", "message": exc.message})
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=True,
        )

    # Neutral dict shape — no raw importer types cross the MCP boundary.
    source_basename = _Path(result.docs_path).name
    return {"ok": True, "source": source_basename, "status": result.status}


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
        "kb_capture_v1",
        "kb_import_v1",
    ):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters["additionalProperties"] = False


_add_strict_schema()
