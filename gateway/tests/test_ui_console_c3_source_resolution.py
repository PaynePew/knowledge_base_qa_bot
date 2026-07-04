"""Structural tests for the C3 Fix-the-Source resolved-path flow (issue #445).

Following the pattern in ``test_ui_console_c3_routed_fix_source.py``, these
tests inspect the production ``gateway/static/console.html`` file's text —
no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- The Console never rebuilds a "docs/" + basename path itself — it always
  reads the finding's server-resolved ``source_path`` (§12.5, no business
  logic in the client; the bug this issue fixes was exactly this
  client-side reconstruction 404ing for nested Sources).
- ``c3SourceLabel`` renders the three distinct ``source_resolution`` states
  (``"resolved"``, ``"missing"``, ``"ambiguous"``) distinctly, never
  collapsing missing/ambiguous into a guessed path.
- The View Source button is disabled — a real no-false-affordance guard,
  not just distinct text — when the Source could not be resolved to
  exactly one file.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _function_body(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace-depth
    scanning — robust to nested braces inside the body (mirrors the sibling
    test files' helper)."""
    m = re.search(rf"function {name}\([^)]*\)\s*\{{", text)
    assert m is not None, f"console.html must define function {name}(...)"
    start = m.end() - 1  # index of the opening brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unbalanced braces scanning function {name}")


# ---------------------------------------------------------------------------
# No client-side path reconstruction (the exact bug issue #445 fixes)
# ---------------------------------------------------------------------------


def test_console_never_rebuilds_docs_path_for_c3():
    """The Console must not concatenate "docs/" with a bare basename
    anywhere in the C3 Fix-the-Source flow — that construction 404'd for
    Sources living in a nested docs/ subdirectory (issue #445)."""
    text = _console_text()
    for fn_name in ("fixSourceBannerText", "showFixSourceBanner", "c3SourceLabel"):
        body = _function_body(text, fn_name)
        assert '"docs/" + ' not in body, (
            f"{fn_name} must not rebuild a docs/ path client-side: {body}"
        )


def test_show_fix_source_banner_opens_the_findings_own_source_path():
    text = _console_text()
    body = _function_body(text, "showFixSourceBanner")
    assert "openFile(finding.source_path" in body


# ---------------------------------------------------------------------------
# c3SourceLabel: three distinct resolution states
# ---------------------------------------------------------------------------


def test_c3_source_label_prefers_resolved_source_path():
    text = _console_text()
    body = _function_body(text, "c3SourceLabel")
    assert 'finding.source_resolution === "resolved"' in body
    assert "finding.source_path" in body


def test_c3_source_label_renders_missing_and_ambiguous_distinctly():
    """Missing and ambiguous must map to two DIFFERENT chrome strings —
    never collapsed into one generic "not found" message."""
    text = _console_text()
    body = _function_body(text, "c3SourceLabel")
    assert '"ambiguous"' in body
    assert "c3SourceAmbiguous" in body
    assert "c3SourceMissing" in body


def test_missing_and_ambiguous_chrome_exist_bilingually():
    text = _console_text()
    en_match = re.search(r"en:\s*\{(.*?)\n  \},", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},", text, re.DOTALL)
    assert en_match is not None and zh_match is not None
    assert re.search(r'c3SourceMissing:\s*"[^"]+"', en_match.group(1))
    assert re.search(r'c3SourceAmbiguous:\s*"[^"]+"', en_match.group(1))
    assert re.search(r'c3SourceMissing:\s*"[^"]+"', zh_match.group(1))
    assert re.search(r'c3SourceAmbiguous:\s*"[^"]+"', zh_match.group(1))


# ---------------------------------------------------------------------------
# View Source: disabled (no false affordance) when unresolved
# ---------------------------------------------------------------------------


def test_view_source_button_disabled_when_not_resolved():
    """§12.5 'honest, not alarmist' — a missing/ambiguous Source has no real
    path to open, so the button must be disabled rather than silently
    opening a guessed (and possibly wrong) file."""
    text = _console_text()
    body = _function_body(text, "showFixSourceBanner")
    assert re.search(r'source_resolution === "resolved"[^{]*&&\s*finding\.source_path', body), (
        f"showFixSourceBanner must gate the click handler on a resolved source_path: {body}"
    )
    assert "viewBtn.disabled = true" in body
