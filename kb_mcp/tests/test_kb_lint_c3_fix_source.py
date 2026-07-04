"""kb_lint_v1's C3 added Routed fix-the-Source hint (issue #408, ADR-0029
decisions 2-4).

C3 is the first check carrying two remediation classes: it stays Direct-tier
(its 'reingest_retry' action is unaffected), and ALSO gains a Routed
navigation hint for its dominant failure mode (claim_unsupported) — the real
fix (amending what the Source says) is knowledge only the human can supply,
so there is no tool call that resolves it. This adds a 'fix_via' text hint to
every failed_grounding (C3) finding, sourced from the SAME shared taxonomy
(``remediation_for("C3").secondary_route``) the Console's "Fix Source"
control and ``kb lint``'s CLI hint both read — mirrors
test_kb_lint_routed_coverage.py's C1/C2 pattern.

Drives the REAL run_lint via kb_lint_v1 (no mocking of the deep module, per
CODING_STANDARD §6.3) against a fixture wiki page carrying
status: failed_grounding, planted under the conftest-redirected tmp wiki dir.
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


def _write_failed_grounding_page(
    wiki_dir: Path,
    slug: str,
    source: str,
    *,
    reason: str = "claim_unsupported",
    unsupported_claims: list[str] | None = None,
) -> Path:
    """Write a wiki page with status=failed_grounding and grounding_failure
    block (mirrors markdown_kb/tests/lint/test_c3_reason_split_suggested_action.py)."""
    page_dir = wiki_dir / "concepts"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    grounding_failure: dict = {"reason": reason, "unsupported_claims": unsupported_claims or []}

    frontmatter = {
        "id": slug,
        "type": "concept",
        "created": "2026-07-04T00:00:00Z",
        "updated": "2026-07-04T00:00:00Z",
        "sources": [source],
        "status": "failed_grounding",
        "open_questions": [],
        "grounding_failure": grounding_failure,
    }
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n# {slug}\n\nFailed content.\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


def test_c3_failed_grounding_gains_fix_via_hint():
    """A real status:failed_grounding page surfaces as a C3 finding carrying
    fix_via."""
    import asyncio

    import markdown_kb.app.lint as lint_mod

    from kb_mcp.server import mcp

    _write_failed_grounding_page(
        lint_mod.WIKI_DIR,
        "broken-page",
        "policy.md",
        unsupported_claims=["The refund window is 90 days."],
    )

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    failed_grounding = result["findings"]["failed_grounding"]
    assert len(failed_grounding) == 1, f"Expected one C3 finding, got: {failed_grounding}"
    assert failed_grounding[0]["page_slug"] == "broken-page"
    assert "docs/" in failed_grounding[0]["fix_via"]


def test_fix_via_hint_is_driven_by_the_shared_taxonomy():
    """Not a re-derived string — reads remediation_for("C3").secondary_route,
    the same value the CLI ("kb lint") and Console ("Fix Source") read."""
    from markdown_kb.app.lint import remediation_for

    assert remediation_for("C3").tier == "direct"
    assert remediation_for("C3").secondary_route == "fix-source"


def test_no_failed_grounding_yields_empty_list_no_error():
    import asyncio

    from kb_mcp.server import mcp

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    assert result["findings"]["failed_grounding"] == []


def test_axis_groups_check_shape_unaffected_by_fix_source_hint():
    """The hint lives on findings, not axis_groups — the existing
    {code, label, count}-only shape for every check entry stays exact."""
    import asyncio

    import markdown_kb.app.lint as lint_mod

    from kb_mcp.server import mcp

    _write_failed_grounding_page(lint_mod.WIKI_DIR, "broken-page", "policy.md")

    raw = asyncio.run(mcp.call_tool("kb_lint_v1", {"include_c5": False}))
    result = _parse_result(raw)
    for group in result["axis_groups"]:
        for check in group["checks"]:
            assert set(check.keys()) == {"code", "label", "count"}, (
                f"unexpected check shape: {check}"
            )
