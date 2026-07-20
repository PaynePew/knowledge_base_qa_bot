"""Structural tests for the Field Manual rail section (issue #637).

A curator-facing manual, rendered as an expandable-row rail section below
Source Trash: one entry per ``ROW_RENDERERS`` finding code with a
plain-language trigger rule, a glossary of the terms that code's own rows
use, and what its remediation button(s) actually change. Content lives in
the existing ``LINT_CHROME`` bilingual dictionary as flat top-level keys
(``manual<CODE>Header`` / ``Rule`` / ``Glossary`` / ``Effects``), and each
lint finding row (or, for the three Curation-Queue-owned codes, the queue
count line) gets a small "?" affordance that expands + scrolls to its entry.

Following the pattern in ``test_ui_console_lint_bilingual.py``, these tests
inspect the production ``gateway/static/console.html`` file's text — no DOM,
no fetch, no browser, no OPENAI_API_KEY (fully hermetic, §6.3/§12.7).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"

# The code set is C1-C6 + C8-C12 (11 codes) — there is no C7 (issue #637
# scope pin 1 / issue #406 scope item 5). Kept as a plain tuple here (not
# imported from console.html, which has no Python runtime) so this file's
# expectation is independent of the production ROW_RENDERERS extraction
# below — the two are cross-checked against each other, not against a
# shared source, so neither can silently drift without a test failure.
EXPECTED_CODES = ("C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10", "C11", "C12")


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling ``test_ui_console_*`` helper of the same
    name — robust to nested braces inside the body)."""
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


def _brace_matched_block(text: str, marker: str) -> str:
    """Return the ``{ ... }`` body immediately following ``marker`` (which
    must end just before the opening brace), by depth-scanning — robust to
    nested braces (mirrors ``_lint_chrome_en_zh``'s sub-object helper in
    ``test_ui_console_lint_bilingual.py``)."""
    start = text.index(marker)
    body_start = start + len(marker) - 1
    depth = 0
    for i in range(body_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[body_start + 1 : i]
    raise AssertionError(f"unterminated block for marker {marker!r}")


def _row_renderers_codes(text: str) -> list[str]:
    """The real finding codes ``ROW_RENDERERS`` defines, in source order —
    the authoritative code set (issue #637 scope pin 1: 'the structural test
    iterates the actual ROW_RENDERERS keys, not a hardcoded C1..C12
    range')."""
    block = _brace_matched_block(text, "var ROW_RENDERERS = {")
    return re.findall(r"^\s*(C\d+):\s*function\(i, f\)", block, re.MULTILINE)


def _lint_chrome_en_zh(text: str) -> tuple[str, str]:
    """Return the (en_block, zh_block) source text of LINT_CHROME's two
    language objects (mirrors ``test_ui_console_lint_bilingual.py``)."""
    full_body = _brace_matched_block(text, "var LINT_CHROME = {")

    def _sub_object(label: str) -> str:
        return _brace_matched_block(full_body, f"{label}: {{")

    return _sub_object("en"), _sub_object("zh")


def _manual_keys(block: str, code: str) -> dict[str, str]:
    """The four manual{code}{Field} string values present in a LINT_CHROME
    language block, keyed by field name."""
    out = {}
    for field in ("Header", "Rule", "Glossary", "Effects"):
        m = re.search(rf'manual{code}{field}:\s*"((?:[^"\\]|\\.)*)"', block)
        if m:
            out[field] = m.group(1)
    return out


# ---------------------------------------------------------------------------
# Code set: exactly the 11 real ROW_RENDERERS codes, no C7
# ---------------------------------------------------------------------------


def test_row_renderers_code_set_is_the_expected_eleven_codes():
    text = _console_text()
    codes = _row_renderers_codes(text)
    assert sorted(codes, key=lambda c: int(c[1:])) == sorted(
        EXPECTED_CODES, key=lambda c: int(c[1:])
    )
    assert "C7" not in codes


def test_field_manual_codes_array_matches_row_renderers_with_no_c7():
    text = _console_text()
    row_codes = set(_row_renderers_codes(text))
    m = re.search(r"var FIELD_MANUAL_CODES = .*?\}, \[\]\);", text, re.DOTALL)
    assert m is not None, "FIELD_MANUAL_CODES must be derived from AXIS_CHECK_CODES/LINT_AXIS_ORDER"
    axis_block = _brace_matched_block(text, "var AXIS_CHECK_CODES = {")
    axis_codes = set(re.findall(r'"(C\d+)"', axis_block))
    assert axis_codes == row_codes, (
        "AXIS_CHECK_CODES (which FIELD_MANUAL_CODES flattens) must cover exactly "
        "the same codes as ROW_RENDERERS"
    )
    assert "manualC7" not in text


# ---------------------------------------------------------------------------
# Every ROW_RENDERERS code has a manual entry in BOTH languages
# ---------------------------------------------------------------------------


def test_every_row_renderers_code_has_a_manual_entry_in_both_languages():
    text = _console_text()
    en_block, zh_block = _lint_chrome_en_zh(text)
    for code in _row_renderers_codes(text):
        en = _manual_keys(en_block, code)
        zh = _manual_keys(zh_block, code)
        for field in ("Header", "Rule", "Glossary", "Effects"):
            assert field in en, f"LINT_CHROME.en.manual{code}{field} missing"
            assert field in zh, f"LINT_CHROME.zh.manual{code}{field} missing"
            assert en[field].strip(), f"LINT_CHROME.en.manual{code}{field} is empty"
            assert zh[field].strip(), f"LINT_CHROME.zh.manual{code}{field} is empty"


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def test_manual_entry_zh_text_is_real_chinese_not_a_copy_of_english():
    text = _console_text()
    en_block, zh_block = _lint_chrome_en_zh(text)
    for code in _row_renderers_codes(text):
        zh = _manual_keys(zh_block, code)
        for field in ("Rule", "Glossary", "Effects"):
            assert _has_cjk(zh[field]), (
                f"LINT_CHROME.zh.manual{code}{field} must contain real Chinese text"
            )


# ---------------------------------------------------------------------------
# Generic chrome (section label, per-entry field labels, help aria-label)
# ---------------------------------------------------------------------------


def test_generic_field_manual_chrome_is_bilingual():
    text = _console_text()
    en_block, zh_block = _lint_chrome_en_zh(text)
    for key in (
        "fieldManualSectionLabel",
        "manualRuleLabel",
        "manualGlossaryLabel",
        "manualEffectsLabel",
        "manualHelpAriaLabelPrefix",
    ):
        assert re.search(rf'{key}:\s*"[^"]+"', en_block), f"LINT_CHROME.en.{key} missing"
        zh_match = re.search(rf'{key}:\s*"([^"]+)"', zh_block)
        assert zh_match is not None, f"LINT_CHROME.zh.{key} missing"
        assert _has_cjk(zh_match.group(1)), f"LINT_CHROME.zh.{key} must contain real Chinese text"


# ---------------------------------------------------------------------------
# The HTML placeholder sits in the right rail, below Source Trash
# ---------------------------------------------------------------------------


def test_field_manual_section_placeholder_sits_below_source_trash_in_the_rail():
    text = _console_text()
    aside_match = re.search(r'<aside class="console-rail"[^>]*>.*?</aside>', text, re.DOTALL)
    assert aside_match is not None
    aside = aside_match.group(0)
    trash_idx = aside.index('id="source-trash-section"')
    manual_idx = aside.index('id="field-manual-section"')
    assert trash_idx < manual_idx, (
        "field-manual-section must come after source-trash-section in the rail"
    )


# ---------------------------------------------------------------------------
# The "?" affordance: findingRow (per-row) and queueCountLine (queue-owned)
# ---------------------------------------------------------------------------


def test_finding_row_accepts_a_code_and_renders_a_manual_help_button():
    text = _console_text()
    fn = _extract_function(text, "findingRow")
    assert "function findingRow(code, idx, title, subtitle, labelText, labelClass, actionsEl)" in fn
    assert "manualHelpButton(code)" in fn


def test_every_row_renderers_call_site_passes_its_own_code_to_finding_row():
    text = _console_text()
    block = _brace_matched_block(text, "var ROW_RENDERERS = {")
    for code in _row_renderers_codes(text):
        assert re.search(rf'findingRow\("{code}",', block), (
            f'ROW_RENDERERS.{code} must call findingRow("{code}", ...) — issue #637'
        )


def test_check_errors_row_normalises_the_lowercase_internal_id_to_a_manual_code():
    """check_errors is keyed by the internal check id (lowercase; C4's is
    "c4a") rather than the UI-facing code — the "?" affordance must
    normalise it, or every check-error row's help button would silently
    open nothing (issue #637 follow-up finding)."""
    text = _console_text()
    fn = _extract_function(text, "renderLintCard")
    assert 'var manualCode = k === "c4a" ? "C4" : k.toUpperCase();' in fn
    assert "findingRow(manualCode, i + 1, k, checkErrors[k]" in fn


def test_queue_count_line_renders_a_manual_help_button_for_queue_owned_codes():
    """C8/C9/C10 never reach ROW_RENDERERS's per-item rows (they render via
    queueCountLine instead — issue #438) so their "?" affordance anchors
    there, not on a lint finding row (issue #637 scope pin 3)."""
    text = _console_text()
    fn = _extract_function(text, "queueCountLine")
    assert "manualHelpButton(code)" in fn


def test_manual_help_button_is_defined_outside_row_renderers():
    """Bilingual text used inside ROW_RENDERERS must route through a helper
    defined outside the object (issue #637 scope pin 5, following the
    existing pattern the sibling structural test already pins)."""
    text = _console_text()
    block = _brace_matched_block(text, "var ROW_RENDERERS = {")
    assert "LINT_CHROME[consoleLang]" not in block
    fn = _extract_function(text, "manualHelpButton")
    assert "LINT_CHROME[consoleLang].manualHelpAriaLabelPrefix" in fn
    assert "openFieldManualEntry(code)" in fn


# ---------------------------------------------------------------------------
# openFieldManualEntry: expand + scroll, opening the mobile rail drawer first
# ---------------------------------------------------------------------------


def test_open_field_manual_entry_expands_and_scrolls_the_matching_wrapper():
    text = _console_text()
    fn = _extract_function(text, "openFieldManualEntry")
    assert "fieldManualEntries[code]" in fn
    assert 'wrap.classList.contains("collapsed")' in fn
    assert "scrollSoftIntoView(wrap" in fn
    assert 'matchMedia("(max-width: 899px)")' in fn, (
        "must mirror the same off-canvas-rail breakpoint initResourceRailDrawer uses"
    )


def test_field_manual_entries_start_collapsed():
    text = _console_text()
    assert '"field-manual-entry collapsed"' in text


# ---------------------------------------------------------------------------
# Boot-time bindText wiring (issue #498 convention): every entry field is
# registered for language-toggle retro-switching, not built once and frozen.
# ---------------------------------------------------------------------------


def test_render_field_manual_entry_binds_all_four_fields_via_bind_text():
    text = _console_text()
    fn = _extract_function(text, "renderFieldManualEntry")
    for suffix in ("Header", "Rule", "Glossary", "Effects"):
        assert f'"manual" + code + "{suffix}"' in fn, (
            f'renderFieldManualEntry must bindText(..., "manual" + code + "{suffix}")'
        )
    assert fn.count("bindText(") >= 6  # header + 3x(term-label + text)


# ---------------------------------------------------------------------------
# Content accuracy pins (issue #637 scope pins 2, 6, 7) — derived from the
# live markdown_kb/app/lint.py predicates, not invented.
# ---------------------------------------------------------------------------


def test_c9_rule_states_the_grace_period_not_bare_newer():
    """Scope pin 2: the C9 rule must include the 3-day grace period (PR
    #640) — 'flag when merely newer' is explicitly superseded."""
    text = _console_text()
    en_block, _ = _lint_chrome_en_zh(text)
    rule = _manual_keys(en_block, "C9")["Rule"]
    assert "KB_LINT_C9_GRACE_DAYS" in rule
    assert "3.0" in rule
    assert "grace" in rule.lower()


def test_c5_rule_says_llm_judged_not_a_deterministic_predicate():
    """Scope pin 7: C5's rule text must say the pair is judged by the LLM
    judge, never invent a mechanical predicate."""
    text = _console_text()
    en_block, _ = _lint_chrome_en_zh(text)
    rule = _manual_keys(en_block, "C5")["Rule"]
    assert "LLM" in rule and "judge" in rule.lower()
    assert "KB_LINT_C5_MAX_PAIRS" in rule


def test_threshold_env_vars_are_stated_as_defaults_not_fixed_constants():
    """Scope pin 6: KB_LINT_MIN_HITS / KB_LINT_PROMOTION_TOP_N /
    KB_LINT_C5_MAX_PAIRS / KB_LINT_C9_GRACE_DAYS must each read as an
    operator-configurable default, never a bare fixed number."""
    text = _console_text()
    en_block, _ = _lint_chrome_en_zh(text)
    checks = {
        "C1": "KB_LINT_MIN_HITS",
        "C5": "KB_LINT_C5_MAX_PAIRS",
        "C8": "KB_LINT_PROMOTION_TOP_N",
        "C9": "KB_LINT_C9_GRACE_DAYS",
    }
    for code, env_var in checks.items():
        rule = _manual_keys(en_block, code)["Rule"]
        assert env_var in rule, f"manual{code}Rule must name {env_var}"
        # "default" must appear near the env var (either side — "env,
        # default N" or "defaults to N ... via <ENV_VAR>") so it reads as
        # operator-configurable rather than a bare fixed number.
        idx = rule.index(env_var)
        window = rule[max(0, idx - 60) : idx + 60]
        assert "default" in window.lower(), (
            f"manual{code}Rule must present {env_var} as a default, not a fixed constant: {window!r}"
        )


def test_c11_glossary_distinguishes_full_and_partial_orphans():
    text = _console_text()
    en_block, _ = _lint_chrome_en_zh(text)
    glossary = _manual_keys(en_block, "C11")["Glossary"]
    assert "full" in glossary and "partial" in glossary


def test_c6_glossary_explains_drift_as_source_mtime_vs_page_updated():
    text = _console_text()
    en_block, _ = _lint_chrome_en_zh(text)
    glossary = _manual_keys(en_block, "C6")["Glossary"]
    assert "hash" in glossary
    assert "drift" in glossary


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
