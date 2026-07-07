"""kb_lint_v1's C5 added Routed fix-the-Source hint (issue #534, ADR-0036
decisions 1-2).

C5 is the second check (after C3, issue #408/ADR-0029) carrying two
remediation classes: its existing Authored Reconcile tier stays wired
unchanged (a wiki-rooted contradiction still converges there), and it ALSO
gains a Routed navigation hint for the source-rooted case — both pages are
faithfully grounded, but their own Sources disagree, which Reconcile cannot
fix. This adds a 'fix_via' text hint to every page_pairs (C5) finding,
sourced from the SAME shared taxonomy (``remediation_for("C5").secondary_route``)
the CLI's ``kb lint`` hint reads — mirrors ``test_kb_lint_c3_fix_source.py``.

Drives the REAL run_lint via kb_lint_v1 (no mocking of the deep module, per
CODING_STANDARD §6.3) with ONLY the drafting LLM (``get_lint_llm``) stubbed
to return a fixed contradiction verdict — mirrors
``test_kb_index_lint.py::TestKbLintV1C5LLMError``'s mocking seam.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result."""
    from mcp.types import CallToolResult

    if isinstance(raw, CallToolResult):
        return json.loads(raw.content[0].text)
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


def _write_wiki_page(wiki_dir: Path, slug: str, sources: list[str], body: str) -> Path:
    """Write a minimal wiki page with YAML frontmatter (mirrors
    test_kb_index_lint.py's helper of the same name)."""
    page_dir = wiki_dir / "concepts"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": "concept",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": sources,
        "status": "live",
        "open_questions": [],
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _stub_contradicting_pair(monkeypatch) -> None:
    """Monkeypatch get_lint_llm so the C5 judge returns a fixed 'direct'
    contradiction verdict for whichever pair run_lint hands it — the
    sanctioned §6.3 mock seam (never mock _judge_page_pair itself)."""
    import markdown_kb.app.lint as lint_mod
    from markdown_kb.app.schemas import PagePairFinding

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = PagePairFinding(
        severity="direct",
        page_a="page-alpha",
        page_b="page-beta",
        page_a_claim="Refunds take 5 business days.",
        page_b_claim="Refunds take 14 business days.",
        summary="The two pages disagree about the refund window.",
        suggested_action="Reconcile the two pages.",
    )
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    monkeypatch.setattr(lint_mod, "get_lint_llm", lambda: fake_llm)


def test_c5_contradiction_gains_fix_via_hint(monkeypatch):
    """A real judged C5 pair surfaces carrying fix_via."""
    import markdown_kb.app.lint as lint_mod

    from kb_mcp.server import mcp

    wiki_dir = lint_mod.WIKI_DIR  # already redirected to tmp by conftest
    _write_wiki_page(
        wiki_dir, "page-alpha", ["shared-source.md#section"], "Refunds take 5 business days."
    )
    _write_wiki_page(
        wiki_dir, "page-beta", ["shared-source.md#section"], "Refunds take 14 business days."
    )
    _stub_contradicting_pair(monkeypatch)

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": True}))
    result = _parse_result(raw)
    page_pairs = result["findings"]["page_pairs"]
    assert len(page_pairs) == 1, f"Expected one C5 finding, got: {page_pairs}"
    assert page_pairs[0]["page_a"] == "page-alpha"
    assert "docs/" in page_pairs[0]["fix_via"]


def test_fix_via_hint_is_driven_by_the_shared_taxonomy():
    """Not a re-derived string — reads remediation_for("C5").secondary_route,
    the SAME value the CLI ("kb lint") reads."""
    from markdown_kb.app.lint import remediation_for

    assert remediation_for("C5").tier == "authored"
    assert remediation_for("C5").secondary_route == "fix-source"


def test_no_page_pairs_yields_empty_list_no_error():
    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    assert result["findings"]["page_pairs"] == []


def test_axis_groups_check_shape_unaffected_by_fix_source_hint(monkeypatch):
    """The hint lives on findings, not axis_groups — the existing
    {code, label, count}-only shape for every check entry stays exact."""
    import markdown_kb.app.lint as lint_mod

    from kb_mcp.server import mcp

    wiki_dir = lint_mod.WIKI_DIR
    _write_wiki_page(
        wiki_dir, "page-alpha", ["shared-source.md#section"], "Refunds take 5 business days."
    )
    _write_wiki_page(
        wiki_dir, "page-beta", ["shared-source.md#section"], "Refunds take 14 business days."
    )
    _stub_contradicting_pair(monkeypatch)

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": True}))
    result = _parse_result(raw)
    for group in result["axis_groups"]:
        for check in group["checks"]:
            assert set(check.keys()) == {"code", "label", "count"}, (
                f"unexpected check shape: {check}"
            )
