"""Negative test guarding ADR-0026's MCP gate-visibility invariant (issue #377).

ADR-0026 decision 3: "MCP sees everything and approves nothing." No MCP tool
may resolve a Gated remediation (promote / discard / re-file / edit /
orphan-delete) — those verbs are Console/CLI-only, driven by a *present*
human who can distinguish "I read the draft and approved it" from "the agent
acted alone." This test pins that invariant mechanically so a future slice
cannot silently register a gate-resolving tool.
"""

from __future__ import annotations

import asyncio

# Verbs that would resolve a Gated remediation if exposed as an MCP tool name
# (ADR-0026 § Consequences Invariant: "no MCP tool resolves a Gated
# remediation (promote, discard, re-file, edit, orphan-delete are
# Console/CLI only)"). Matched as a substring of the lowercased, underscore-
# stripped tool name so both "kb_qa_promote_v1" and a hypothetical
# "kb_promote_qa_v1" are caught.
_GATE_RESOLVING_VERBS = (
    "promote",
    "discard",
    "refile",
    "editqa",
    "orphandelete",
    "deleteqa",
)


def _normalized_tool_names() -> list[str]:
    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    return [t.name.lower().replace("_", "") for t in tools]


def test_no_mcp_tool_resolves_a_gated_remediation():
    """No registered MCP tool name contains a gate-resolving verb (ADR-0026)."""
    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    normalized = _normalized_tool_names()

    offending = [
        t.name
        for t, norm in zip(tools, normalized, strict=True)
        if any(verb in norm for verb in _GATE_RESOLVING_VERBS)
    ]
    assert offending == [], (
        f"Found MCP tool(s) that appear to resolve a Gated remediation: {offending}. "
        "ADR-0026: gates resolve on human surfaces only (Console/CLI) — MCP sees "
        "everything and approves nothing. Promote/discard/re-file/edit/orphan-delete "
        "belong on kb_cli's `kb qa` group, never as an MCP tool."
    )


def test_qa_lifecycle_tools_are_absent_by_exact_name():
    """The specific tool names a naive port of the HTTP qa endpoints would use are absent."""
    from kb_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    forbidden = {
        "kb_qa_promote_v1",
        "kb_qa_discard_v1",
        "kb_qa_delete_v1",
        "kb_qa_refile_v1",
        "kb_qa_edit_v1",
    }
    present = names & forbidden
    assert present == set(), f"Forbidden gate-resolving MCP tool(s) registered: {present}"
