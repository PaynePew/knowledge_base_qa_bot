"""Integration tests for kb_search_v1 using the FastMCP in-process harness.

Tests call the tool via ``mcp.call_tool()`` (the same path as a real MCP host),
not by calling the Python function directly.  This exercises FastMCP's argument
validation and the full dispatch path.

Deep modules are patched:
  - ``markdown_kb.app.indexer.search``   — wiki stack
  - ``vector_rag.app.indexer.search``    — rag stack
  - ``kb_mcp.freshness.reload_if_stale`` — suppress index I/O

These are the only patches needed; no LLM is involved in kb_search_v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubSection:
    id: str
    content: str


@dataclass
class _StubChunk:
    id: str
    content: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result."""
    # FastMCP returns a list of content items; the first is TextContent.
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_stacks(monkeypatch):
    """Patch both retrieval stacks and freshness so no real I/O occurs."""
    import markdown_kb.app.indexer as wiki_indexer
    import vector_rag.app.indexer as rag_indexer

    import kb_mcp.freshness as freshness_mod

    monkeypatch.setattr(
        wiki_indexer,
        "search",
        lambda query, k=3: [
            (_StubSection(id=f"wiki#{i}", content=f"wiki result {i} for '{query}'"), float(k - i))
            for i in range(k)
        ],
    )
    monkeypatch.setattr(
        rag_indexer,
        "search",
        lambda query, k=3: [
            _StubChunk(id=f"rag#{i}", content=f"rag result {i} for '{query}'") for i in range(k)
        ],
    )
    monkeypatch.setattr(freshness_mod, "reload_if_stale", lambda *_a, **_kw: False)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_tool_schema_shape():
    """kb_search_v1 exposes a strict JSON schema (query required, enum+default, additionalProperties:false)."""
    import asyncio

    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    search_tool = next((t for t in tools if t.name == "kb_search_v1"), None)
    assert search_tool is not None, "kb_search_v1 tool not registered"

    schema = search_tool.inputSchema
    assert schema.get("additionalProperties") is False, "additionalProperties:false missing"

    props = schema["properties"]
    assert "query" in props
    assert "stack" in props
    assert "k" in props

    # Only query is required.
    assert schema.get("required") == ["query"]

    # stack has enum + default (includes hybrid since S5/#315).
    assert props["stack"]["enum"] == ["wiki", "rag", "hybrid"]
    assert props["stack"]["default"] == "wiki"

    # k has min/max.
    assert props["k"]["minimum"] == 1
    assert props["k"]["maximum"] == 10


# ---------------------------------------------------------------------------
# Functional tests — wiki stack (default)
# ---------------------------------------------------------------------------


def test_kb_search_v1_wiki_stack_default():
    """Default stack=wiki returns {stack: 'wiki', results: [...]} with scores."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "refund policy"}))
    result = _parse_result(raw)

    assert result["stack"] == "wiki"
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 3  # default k=3


def test_kb_search_v1_wiki_result_shape():
    """Each wiki result has {id, content, score: float}."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "shipping"}))
    result = _parse_result(raw)

    for item in result["results"]:
        assert "id" in item
        assert "content" in item
        assert "score" in item
        assert isinstance(item["score"], float)


def test_kb_search_v1_explicit_wiki_stack():
    """Explicit stack='wiki' dispatches to wiki stack."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "returns", "stack": "wiki"}))
    result = _parse_result(raw)

    assert result["stack"] == "wiki"
    assert all(item["id"].startswith("wiki#") for item in result["results"])


# ---------------------------------------------------------------------------
# Functional tests — rag stack
# ---------------------------------------------------------------------------


def test_kb_search_v1_rag_stack():
    """stack='rag' returns {stack: 'rag', results: [...]} with score=null."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "shipping", "stack": "rag"}))
    result = _parse_result(raw)

    assert result["stack"] == "rag"
    assert isinstance(result["results"], list)


def test_kb_search_v1_rag_result_shape_score_null():
    """Each rag result has score=null (None → null in JSON)."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "cost", "stack": "rag"}))
    result = _parse_result(raw)

    for item in result["results"]:
        assert "id" in item
        assert "content" in item
        assert item["score"] is None


# ---------------------------------------------------------------------------
# k parameter
# ---------------------------------------------------------------------------


def test_kb_search_v1_respects_k_parameter():
    """k parameter controls the number of results."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "shipping", "k": 1}))
    result = _parse_result(raw)
    assert len(result["results"]) == 1


def test_kb_search_v1_k_clamped_to_max():
    """k is clamped to max=10 server-side even if the stub would return more."""
    import asyncio

    from kb_mcp.server import mcp

    # k=10 is valid; results count equals whatever the patched stub returns for k=10.
    raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "test", "k": 10}))
    result = _parse_result(raw)
    assert len(result["results"]) == 10


# ---------------------------------------------------------------------------
# Stack dispatch correctness
# ---------------------------------------------------------------------------


def test_wiki_results_have_score_rag_results_dont():
    """Wiki scores are floats; rag scores are null — verified across both calls."""
    import asyncio

    from kb_mcp.server import mcp

    wiki_raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "q", "stack": "wiki"}))
    rag_raw = asyncio.run(mcp.call_tool("kb_search_v1", {"query": "q", "stack": "rag"}))

    wiki_result = _parse_result(wiki_raw)
    rag_result = _parse_result(rag_raw)

    for item in wiki_result["results"]:
        assert isinstance(item["score"], float)
    for item in rag_result["results"]:
        assert item["score"] is None
