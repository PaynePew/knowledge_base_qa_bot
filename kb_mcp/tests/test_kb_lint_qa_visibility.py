"""Tests for kb_lint_v1's C8/C9 visibility enhancement (issue #377 / ADR-0026).

ADR-0026 decision 3: the MCP surface sees everything and approves nothing.
This adds a 'path' (wiki/qa/<slug>.md) to every C8 promotion-candidate and C9
stale-filed-answer finding, and a 'question' to C9 findings (the only one of
the two whose Pydantic model has no question field), WITHOUT adding a
gate-resolving tool and WITHOUT touching markdown_kb's lint/schema modules
(kept parallel-safe with sibling tier-B slices editing those files).

Drives the REAL run_lint via kb_lint_v1 (no mocking of the deep module, per
CODING_STANDARD §6.3) against qa pages planted under the conftest-redirected
tmp wiki dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result."""
    from mcp.types import CallToolResult

    if isinstance(raw, CallToolResult):
        return json.loads(raw.content[0].text)
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


def _write_qa_page(
    wiki_dir: Path,
    slug: str,
    *,
    question: str,
    status: str,
    count: int = 1,
    sources: list[str] | None = None,
    created: str = "2026-01-01T00:00:00Z",
    updated: str = "2026-01-01T00:00:00Z",
    body: str = "Answer body text.",
) -> Path:
    """Write a real wiki/qa/<slug>.md fixture (mirrors qa._render_qa_page's shape)."""
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "id": slug,
        "type": "qa",
        "created": created,
        "updated": updated,
        "sources": sources or [],
        "status": status,
        "open_questions": [],
        "question": question,
        "count": count,
    }
    fm_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    content = f"<!-- Auto-filed by POST /chat. -->\n\n---\n{fm_yaml}---\n\n{body}\n"
    page_path = qa_dir / f"{slug}.md"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _write_entity_page(wiki_dir: Path, slug: str, *, updated: str) -> Path:
    """Write a minimal entity page (only ``updated`` matters for C9 drift)."""
    entities_dir = wiki_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "id": slug,
        "type": "entity",
        "created": "2025-01-01T00:00:00Z",
        "updated": updated,
        "sources": ["source.md"],
        "status": "live",
        "open_questions": [],
    }
    fm_yaml = yaml.dump(frontmatter, default_flow_style=False)
    content = f"---\n{fm_yaml}---\n\n# {slug}\n\nContent.\n"
    page_path = entities_dir / f"{slug}.md"
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ---------------------------------------------------------------------------
# C8 promotion candidates gain 'path'
# ---------------------------------------------------------------------------


def test_c8_promotion_candidate_gains_path(tmp_path):
    """A real draft qa page surfaces as a C8 finding carrying 'path'."""
    import asyncio

    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp.server import mcp

    _write_qa_page(
        indexer_mod.WIKI_DIR,
        "popular-question",
        question="How do refunds work?",
        status="draft",
        count=5,
    )

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    candidates = result["findings"]["promotion_candidates"]
    assert len(candidates) == 1, f"Expected one C8 finding, got: {candidates}"
    assert candidates[0]["slug"] == "popular-question"
    assert candidates[0]["path"] == "wiki/qa/popular-question.md"


def test_c8_promotion_candidate_question_already_present(tmp_path):
    """C8's existing 'question' field (PromotionCandidateFinding) is untouched."""
    import asyncio

    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp.server import mcp

    _write_qa_page(
        indexer_mod.WIKI_DIR,
        "popular-question",
        question="How do refunds work?",
        status="draft",
    )

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    candidate = result["findings"]["promotion_candidates"][0]
    assert candidate["question"] == "How do refunds work?"


# ---------------------------------------------------------------------------
# C9 stale filed answers gain 'path' and 'question'
# ---------------------------------------------------------------------------


def test_c9_stale_finding_gains_path_and_question(tmp_path):
    """A live qa page whose cited entity re-ingested later surfaces as C9 with path+question."""
    import asyncio

    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp.server import mcp

    _write_entity_page(indexer_mod.WIKI_DIR, "refund-policy", updated="2026-06-01T00:00:00Z")
    _write_qa_page(
        indexer_mod.WIKI_DIR,
        "stale-answer",
        question="What is the refund window?",
        status="live",
        sources=["refund-policy#cancellation-window"],
        updated="2026-01-01T00:00:00Z",  # older than the entity -> stale
    )

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    stale = result["findings"]["stale_filed_answers"]
    assert len(stale) == 1, f"Expected one C9 finding, got: {stale}"
    assert stale[0]["page_slug"] == "stale-answer"
    assert stale[0]["path"] == "wiki/qa/stale-answer.md"
    assert stale[0]["question"] == "What is the refund window?"


def test_c9_stale_finding_question_is_none_when_page_unreadable(tmp_path, monkeypatch):
    """If the C9 page cannot be re-read at render time, 'question' is null, not an error."""
    import asyncio

    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp import qa_view
    from kb_mcp.server import mcp

    _write_entity_page(indexer_mod.WIKI_DIR, "refund-policy", updated="2026-06-01T00:00:00Z")
    _write_qa_page(
        indexer_mod.WIKI_DIR,
        "stale-answer",
        question="What is the refund window?",
        status="live",
        sources=["refund-policy#cancellation-window"],
        updated="2026-01-01T00:00:00Z",
    )

    # Simulate the page becoming unreadable between the C9 scan and the
    # visibility-enrichment read (e.g. deleted concurrently).
    monkeypatch.setattr(qa_view, "read_qa_page", lambda slug: None)

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    stale = result["findings"]["stale_filed_answers"][0]
    assert stale["question"] is None
    assert stale["path"] == "wiki/qa/stale-answer.md"


def test_no_qa_pages_yields_empty_c8_c9_with_no_error(tmp_path):
    """An empty wiki produces empty C8/C9 lists — the enrichment loops are no-ops."""
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    assert result["findings"]["promotion_candidates"] == []
    assert result["findings"]["stale_filed_answers"] == []
