"""Structural tests for the Operator Console zh/en language toggle (issue #365,
ADR-0023 tier-A S5).

Following the pattern in ``test_ui_console_lint_remediation.py`` (S3) and
``test_ui_bilingual_starters.py`` (the reader UI's earlier bilingual slice),
these tests inspect the production ``gateway/static/console.html`` file's
text to assert the structural invariants of issue #365:

- A header toggle button exists (``#lang-toggle``) and the choice is
  persisted via ``localStorage`` (AC "persists across reloads").
- The axis headers, check labels, remediation button verbs, and Lint-card
  section/empty-state chrome are ALL bilingual, sourced from the SAME zh
  strings as ``markdown_kb/app/lint.py``'s ``LINT_AXIS_LABEL_ZH`` /
  ``LintCheckMeta.label_zh`` taxonomy (issue #365 AC "single source, no
  per-interface duplication") — this file cross-checks against the live
  Python taxonomy rather than a second hardcoded expectation, so the two
  cannot silently drift.
- The dynamic per-finding text (``f.reason``, ``f.source``, etc. — values
  read straight off the ``LintResponse``) stays untouched / English (issue
  #365 AC "the dynamic suggested_action long text is unchanged").
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the actual toggle-click -> re-render loop is
verified manually per §12.7 (visual rendering is out of scope for unit
tests) — these assertions pin the static source text only.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown_kb.app.lint import LINT_AXIS_LABEL_ZH, LINT_CHECK_TAXONOMY

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the S4 batch test file's helper of the same name)."""
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


# ---------------------------------------------------------------------------
# Header toggle exists + persists via localStorage
# ---------------------------------------------------------------------------


def test_lang_toggle_button_exists_in_masthead():
    text = _console_text()
    assert 'id="lang-toggle"' in text, "Console header must offer a language toggle (issue #365 AC)"


def test_lang_choice_persisted_via_localstorage():
    text = _console_text()
    assert "localStorage.getItem(LANG_STORAGE_KEY)" in text, (
        "the persisted language must be read back on load (issue #365 AC 'persists across reloads')"
    )
    assert "localStorage.setItem(LANG_STORAGE_KEY, consoleLang)" in text, (
        "a toggle click must persist the new choice (issue #365 AC 'persists across reloads')"
    )


def test_toggle_click_rerenders_a_visible_lint_card():
    """A toggle click while a Lint card is already on screen must re-render
    it in place — otherwise the AC 'switches all structural chrome' is only
    true for the NEXT lint run, not the current view."""
    text = _console_text()
    init_fn = re.search(r"function initConsoleLangToggle\(\) \{.*?\n\}\)\(\);", text, re.DOTALL)
    assert init_fn is not None
    body = init_fn.group(0)
    assert "applyConsoleLang();" in body
    assert "lintResultEl.replaceChildren(renderLintCard(lastLintData, lintResultEl));" in body


def test_toggle_rerender_respects_in_flight_busy_state():
    """The toggle's Lint-card re-render must be gated on !remediationInFlight —
    re-rendering while a remediation is in flight would mint fresh enabled
    buttons and defeat the #364 no-double-submit busy-state guard (S5 verify
    finding). The static chrome still switches immediately via applyConsoleLang."""
    text = _console_text()
    init_fn = re.search(r"function initConsoleLangToggle\(\) \{.*?\n\}\)\(\);", text, re.DOTALL)
    assert init_fn is not None
    body = init_fn.group(0)
    guard = re.search(r"if \(lastLintData && lintResultEl && !remediationInFlight\) \{", body)
    assert guard is not None, "the toggle's card re-render must be guarded on !remediationInFlight"


# ---------------------------------------------------------------------------
# Bilingual taxonomy: cross-checked against the SAME source as
# markdown_kb/app/lint.py (issue #365 AC "single source, no per-interface
# duplication") — no independently hardcoded expectation here.
# ---------------------------------------------------------------------------


def test_axis_zh_labels_match_the_python_taxonomy():
    text = _console_text()
    for axis, zh_label in LINT_AXIS_LABEL_ZH.items():
        assert zh_label in text, (
            f"axis {axis!r}'s zh label {zh_label!r} (markdown_kb/app/lint.py "
            "LINT_AXIS_LABEL_ZH) is missing from console.html"
        )


def test_check_zh_labels_match_the_python_taxonomy():
    text = _console_text()
    for code, meta in LINT_CHECK_TAXONOMY.items():
        assert meta.label_zh in text, (
            f"{code}'s zh label {meta.label_zh!r} (markdown_kb/app/lint.py "
            "LintCheckMeta.label_zh) is missing from console.html"
        )


def test_axis_head_is_language_aware():
    # axisHead() was folded into a per-axis collapsible (makeCollapse); the
    # language-aware axis label now lives in the axis render loop, but the
    # invariant is unchanged: the header label switches zh/en off consoleLang.
    text = _console_text()
    assert 'consoleLang === "zh" ? LINT_AXIS_LABEL_ZH[axis] : axis' in text


def test_check_head_is_language_aware():
    text = _console_text()
    fn = _extract_function(text, "checkHead")
    assert 'consoleLang === "zh" ? LINT_CHECK_LABEL_ZH[code] : meta.label' in fn


# ---------------------------------------------------------------------------
# Remediation button verbs + section chrome + empty states are bilingual
# ---------------------------------------------------------------------------


def test_remediation_button_verbs_are_bilingual():
    text = _console_text()
    for key in ("reingest", "reingestRetry", "reingestAll", "discard", "discardAll", "tierB"):
        assert re.search(rf"\b{key}:\s*\n?\s*\"", text), (
            f"LINT_CHROME must define an en/zh pair for {key!r} (issue #365 AC 'button verbs')"
        )
    # zh strings for every verb above must actually be present.
    for zh in (
        "重新匯入",
        "重新匯入（重試）",
        "全部重新匯入",
        "捨棄",
        "全部捨棄",
        "需人工撰寫（Tier B）",
    ):
        assert zh in text, f"button-verb zh string {zh!r} missing from console.html"


def test_empty_state_and_section_chrome_are_bilingual():
    text = _console_text()
    for zh in ("沒有發現問題", "未記錄來源", "檢查錯誤"):
        assert zh in text, f"chrome zh string {zh!r} missing from console.html"
    # English defaults are unchanged (regression guard — S3/S4 already pin
    # "Authored (tier B)" and the discardAllAction literal separately).
    assert "No findings — wiki is clean." in text
    assert "No source recorded — review manually." in text
    assert "Check errors" in text


def test_reingest_action_picks_verb_by_language():
    text = _console_text()
    fn = _extract_function(text, "reingestAction")
    assert (
        "retry ? LINT_CHROME[consoleLang].reingestRetry : LINT_CHROME[consoleLang].reingest" in fn
    )


def test_discard_all_action_overrides_english_default_for_zh():
    """S4's structural test pins the literal English button text inside
    discardAllAction as a stable anchor — this slice must keep that literal
    intact AND make the zh case real by overwriting .textContent right
    after creation (issue #365 AC), not by replacing the pinned literal."""
    text = _console_text()
    fn = _extract_function(text, "discardAllAction")
    assert 'text: "Discard all (" + slugs.length + ")"' in fn, (
        "the S4-pinned English literal must remain (gateway/tests/"
        "test_ui_console_lint_remediation_batch.py::test_c10_check_group_gets_discard_all_button)"
    )
    assert 'btn.textContent = LINT_CHROME.zh.discardAll + " (" + slugs.length + ")";' in fn, (
        "discardAllAction must override to the zh label when consoleLang is zh (issue #365 AC)"
    )


# ---------------------------------------------------------------------------
# Dynamic per-finding text (suggested_action, f.reason, f.source, ...) is
# UNCHANGED — explicitly out of scope (issue #365 AC)
# ---------------------------------------------------------------------------


def test_dynamic_finding_text_is_not_wrapped_in_lint_chrome():
    """ROW_RENDERERS must still pass server-derived per-finding strings
    (f.reason, f.source, f.page_slug, ...) straight through — only the
    STATIC chrome (axis/check/button/section/empty-state strings) is
    bilingual (issue #365 AC)."""
    text = _console_text()
    row_renderers = re.search(r"var ROW_RENDERERS = \{(.*?)\n  \};", text, re.DOTALL)
    assert row_renderers is not None
    body = row_renderers.group(1)
    assert "LINT_CHROME[consoleLang]" not in body, (
        "per-finding row renderers must not localise dynamic finding text (issue #365 AC)"
    )
    assert "f.reason" in body and "f.source" in body


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
