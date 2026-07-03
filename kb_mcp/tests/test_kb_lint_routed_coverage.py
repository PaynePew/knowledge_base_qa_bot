"""kb_lint_v1's C1/C2 Routed visibility enhancement (tier-B S7, #383, ADR-0027).

C1 coverage gaps and C2 red links have no gate-resolving tool because there
is nothing to gate: Routed remediation navigates an existing workflow
(Upload -> Import -> Ingest), it never drafts content for a human to
approve. This adds a 'fill_via' text hint to every coverage_gaps (C1) and
red_links (C2) finding, sourced from the SAME shared taxonomy
(``remediation_for(...).route``) the Console's "Fill via Import" control and
``kb lint``'s CLI hint both read — mirrors the C8/C9 enrichment pattern in
test_kb_lint_qa_visibility.py.

Drives the REAL run_lint via kb_lint_v1 (no mocking of the deep module, per
CODING_STANDARD §6.3) against fixtures planted under the conftest-redirected
tmp wiki dir / log path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    body: str,
    *,
    subdir: str = "concepts",
) -> Path:
    """Write a minimal wiki page with a body (mirrors test_kb_index_lint.py)."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": ["source.md"],
        "status": "live",
        "open_questions": [],
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def _write_log(log_path: Path, lines: list[str]) -> None:
    """Write pre-formatted log lines (mirrors markdown_kb's C1 test helper)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fallback_line(ts: str, summary: str) -> str:
    return f"## [{ts}] chat_fallback | {summary}"


# ---------------------------------------------------------------------------
# C2 red links gain 'fill_via'
# ---------------------------------------------------------------------------


def test_c2_red_link_gains_fill_via_hint():
    """A real unresolved [[wikilink]] surfaces as a C2 finding carrying fill_via."""
    import asyncio

    import markdown_kb.app.lint as lint_mod

    from kb_mcp.server import mcp

    _write_wiki_page(
        lint_mod.WIKI_DIR,
        "some-page",
        "See [[missing-target]] for details.",
    )

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    red_links = result["findings"]["red_links"]
    assert len(red_links) == 1, f"Expected one C2 finding, got: {red_links}"
    assert red_links[0]["slug"] == "missing-target"
    assert "kb import" in red_links[0]["fill_via"]


def test_no_red_links_yields_empty_list_no_error():
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    assert result["findings"]["red_links"] == []


# ---------------------------------------------------------------------------
# C1 coverage gaps gain 'fill_via'
# ---------------------------------------------------------------------------


def test_c1_coverage_gap_gains_fill_via_hint():
    """A real chat_fallback log entry surfaces as a C1 finding carrying fill_via."""
    import asyncio

    import markdown_kb.app.lint as lint_mod

    from kb_mcp.server import mcp

    _write_log(
        lint_mod.LOG_PATH,
        [
            _fallback_line(
                "2026-07-01T00:00:00.000000Z",
                '"what is the refund policy" reason=retrieval_empty top_score=0.0',
            )
        ],
    )

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    gaps = result["findings"]["coverage_gaps"]
    assert len(gaps) == 1, f"Expected one C1 finding, got: {gaps}"
    assert gaps[0]["query_canonical"] == "what is the refund policy"
    assert "kb import" in gaps[0]["fill_via"]


def test_no_coverage_gaps_yields_empty_list_no_error():
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    assert result["findings"]["coverage_gaps"] == []


# ---------------------------------------------------------------------------
# axis_groups is untouched (C8/C9's precedent already pins its exact shape)
# ---------------------------------------------------------------------------


def test_axis_groups_check_shape_unaffected_by_routed_hint():
    """The route hint lives on findings, not axis_groups — the existing
    {code, label, count}-only shape for every check entry stays exact."""
    import asyncio

    import markdown_kb.app.lint as lint_mod

    from kb_mcp.server import mcp

    _write_wiki_page(lint_mod.WIKI_DIR, "some-page", "See [[missing-target]] for details.")

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    for group in result["axis_groups"]:
        for check in group["checks"]:
            assert set(check.keys()) == {"code", "label", "count"}, (
                f"unexpected check shape: {check}"
            )
