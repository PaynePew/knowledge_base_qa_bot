"""Structural tests for the Operator Console's C3 unsupported_claims rendering
(issue #407, ADR-0017 four-surface parity).

Following the pattern in ``test_ui_console_lint_remediation.py`` /
``test_ui_console_lint_bilingual.py``, these tests inspect the production
``gateway/static/console.html`` file's text — no DOM, no fetch, no browser, no
OPENAI_API_KEY (§6.3 / §12.7).

- C3's ROW_RENDERERS entry renders the finding's ``unsupported_claims`` list
  (previously dropped — Markdown already rendered it in Slice 5-2).
- An empty ``unsupported_claims`` list degrades gracefully to an honest
  bilingual "not recorded" note (mirrors ``reingestAction``'s ``noSource``
  pattern) rather than a blank cell or a crash.
- The C3 force:true Re-ingest wiring (issue #363 AC, pinned by
  ``test_ui_console_lint_remediation.py``) is untouched.
- No ``innerHTML`` assignment is introduced (§12.4).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace matching."""
    marker = f"function {name}("
    start = text.index(marker)
    depth = 0
    started = False
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
            started = True
        elif text[i] == "}":
            depth -= 1
            if started and depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unterminated function body for {name}")


def test_c3_row_renders_unsupported_claims():
    """C3's ROW_RENDERERS entry must read f.unsupported_claims (issue #407 AC
    'all four surfaces show the claims for a claim_unsupported finding')."""
    text = _console_text()
    row_renderers = re.search(r"var ROW_RENDERERS = \{(.*?)\n  \};", text, re.DOTALL)
    assert row_renderers is not None
    c3_entry = re.search(
        r"C3:\s*function\(i, f\)\s*\{.*?\n    \},", row_renderers.group(1), re.DOTALL
    )
    assert c3_entry is not None, "console.html must define ROW_RENDERERS.C3"
    assert "unsupported_claims" in c3_entry.group(0) or "c3ClaimsSummary(f)" in c3_entry.group(0), (
        "C3 row must render f.unsupported_claims (directly or via a helper)"
    )


def test_c3_row_still_wires_force_true_reingest():
    """Regression guard — the claims rendering must not disturb the existing
    C3 force:true Re-ingest wiring (issue #363 AC, pinned by
    test_ui_console_lint_remediation.py)."""
    text = _console_text()
    assert "reingestAction(f.source, true)" in text


def test_empty_claims_degrades_to_honest_note():
    """An empty unsupported_claims list must render an honest note, not a
    blank cell or a crash (issue #407 AC)."""
    text = _console_text()
    fn = _extract_function(text, "c3ClaimsSummary")
    assert "claims.length > 0" in fn, "must branch on whether any claims were recorded"
    assert "claimsNotRecorded" in fn, (
        "the empty-claims branch must use the bilingual honest-note key"
    )


def test_claims_not_recorded_key_is_bilingual_in_lint_chrome():
    """The new honest-note chrome string lives in the ONE shared LINT_CHROME
    home (issue #407 AC 'shared taxonomy carries any new zh/en label
    strings', ADR-0017) — both en and zh variants must be present."""
    text = _console_text()
    assert re.search(r"claimsNotRecorded:\s*\n?\s*\"", text), (
        "LINT_CHROME must define an en/zh pair for claimsNotRecorded"
    )
    assert "Claims not recorded." in text
    assert "未記錄相關聲明。" in text


def test_dynamic_finding_text_still_not_wrapped_in_lint_chrome():
    """Regression guard mirroring test_ui_console_lint_bilingual.py's
    invariant — ROW_RENDERERS itself must still not reference
    LINT_CHROME[consoleLang] directly (the C3 claims helper lives outside
    the object literal, exactly like reingestAction's noSource)."""
    text = _console_text()
    row_renderers = re.search(r"var ROW_RENDERERS = \{(.*?)\n  \};", text, re.DOTALL)
    assert row_renderers is not None
    assert "LINT_CHROME[consoleLang]" not in row_renderers.group(1)


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
