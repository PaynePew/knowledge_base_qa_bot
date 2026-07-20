"""Structural tests for the Field Manual lookup box (issue #643).

A search input at the top of the Field Manual rail section (issue #637)
filters its 11 entries by code, title, or body text. Matching entries stay
visible and auto-expand with matched substrings highlighted; non-matching
entries are hidden entirely (not merely collapsed). Clearing the box
restores the default state. The input/clear-button CHROME is a small shared
factory, ``railSearchInput(placeholderKey, onQuery)``, reusable by the
Browse quick-find sibling issue — the FILTER LOGIC itself
(``applyFieldManualFilter``) stays local to this section. Highlighting goes
through ``highlightMatches(containerEl, query)``, a module-scope,
textContent-safe span splitter (§12.4 bans ``innerHTML``).

Following the pattern in ``test_ui_console_field_manual.py``, these tests
inspect the production ``gateway/static/console.html`` file's text — no DOM,
no fetch, no browser, no OPENAI_API_KEY (fully hermetic, §6.3/§12.7).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


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
    must end just before the opening brace), by depth-scanning."""
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


def _lint_chrome_en_zh(text: str) -> tuple[str, str]:
    """Return the (en_block, zh_block) source text of LINT_CHROME's two
    language objects (mirrors ``test_ui_console_lint_bilingual.py``)."""
    full_body = _brace_matched_block(text, "var LINT_CHROME = {")

    def _sub_object(label: str) -> str:
        return _brace_matched_block(full_body, f"{label}: {{")

    return _sub_object("en"), _sub_object("zh")


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def _paren_matched_call(text: str, marker: str) -> str:
    """Return the ``( ... )`` argument list immediately following ``marker``
    (which must end just before the opening paren), by depth-scanning —
    the call-site analogue of ``_brace_matched_block`` for a function call
    rather than an object/block literal."""
    start = text.index(marker)
    body_start = start + len(marker) - 1
    depth = 0
    for i in range(body_start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[body_start + 1 : i]
    raise AssertionError(f"unterminated call for marker {marker!r}")


# ---------------------------------------------------------------------------
# New LINT_CHROME keys: bilingual, non-empty, real Chinese on the zh side
# ---------------------------------------------------------------------------


def test_search_chrome_keys_are_bilingual_and_zh_is_real_chinese():
    text = _console_text()
    en_block, zh_block = _lint_chrome_en_zh(text)
    for key in ("fieldManualSearchPlaceholder", "railSearchClearLabel", "fieldManualSearchEmpty"):
        en_match = re.search(rf'{key}:\s*"([^"]+)"', en_block)
        zh_match = re.search(rf'{key}:\s*"([^"]+)"', zh_block)
        assert en_match is not None, f"LINT_CHROME.en.{key} missing"
        assert zh_match is not None, f"LINT_CHROME.zh.{key} missing"
        assert en_match.group(1).strip(), f"LINT_CHROME.en.{key} is empty"
        assert _has_cjk(zh_match.group(1)), f"LINT_CHROME.zh.{key} must contain real Chinese text"


# ---------------------------------------------------------------------------
# railSearchInput(placeholderKey, onQuery): shared chrome factory
# ---------------------------------------------------------------------------


def test_rail_search_input_is_defined_at_module_scope_with_the_documented_signature():
    text = _console_text()
    assert re.search(
        r"^function railSearchInput\(placeholderKey, onQuery\) \{", text, re.MULTILINE
    ), (
        "railSearchInput must be a top-level function with signature "
        "(placeholderKey, onQuery) — issue #643 AC"
    )
    fn = _extract_function(text, "railSearchInput")
    assert "rail-search-clear" in fn
    assert "rail-search-input" in fn
    assert "railSearchClearLabel" in fn
    assert "onQuery(input.value)" in fn
    assert 'onQuery("")' in fn  # Clear button fires an empty query too


def test_rail_search_input_owns_no_filter_logic():
    """Scope pin (issue #643 AC: 'Keep FILTER LOGIC local to this section —
    only the input chrome is shared') — the shared factory must not
    reference Field-Manual-specific state."""
    text = _console_text()
    fn = _extract_function(text, "railSearchInput")
    assert "fieldManualEntries" not in fn
    assert "FIELD_MANUAL_CODES" not in fn
    assert "fieldManualSearchTargets" not in fn


def test_rail_search_input_registers_itself_for_lang_toggle_retro_switch():
    text = _console_text()
    assert "var RAIL_SEARCH_INPUTS = [];" in text
    fn = _extract_function(text, "railSearchInput")
    assert "RAIL_SEARCH_INPUTS.push(" in fn
    apply_fn = _extract_function(text, "applyConsoleLang")
    assert "RAIL_SEARCH_INPUTS.forEach(function(entry) { entry.refresh(); });" in apply_fn


# ---------------------------------------------------------------------------
# highlightMatches(containerEl, query): module-scope, textContent-safe
# ---------------------------------------------------------------------------


def test_highlight_matches_is_defined_at_module_scope_with_the_documented_signature():
    text = _console_text()
    assert re.search(r"^function highlightMatches\(containerEl, query\) \{", text, re.MULTILINE), (
        "highlightMatches must be a top-level function with signature "
        "(containerEl, query) — issue #643 AC"
    )


def test_highlight_matches_normalizes_nfc_and_lowercases_both_sides():
    text = _console_text()
    fn = _extract_function(text, "highlightMatches")
    assert fn.count('.normalize("NFC")') >= 2, (
        "must NFC-normalize both the container text and the query"
    )
    assert fn.count(".toLowerCase()") >= 2, "must lowercase both sides before comparing"


def test_highlight_matches_never_uses_inner_html():
    text = _console_text()
    fn = _extract_function(text, "highlightMatches")
    assert "innerHTML" not in fn
    assert 'el("mark"' in fn or "el('mark'" in fn, "must build highlight spans via the el() helper"
    assert "replaceChildren" in fn


def test_highlight_matches_reads_text_content_fresh_every_call():
    """Repeated calls (one per keystroke) must not accumulate stale <mark>
    spans from an earlier query — the source string is read from
    containerEl.textContent every time, not cached."""
    text = _console_text()
    fn = _extract_function(text, "highlightMatches")
    assert "var raw = containerEl.textContent;" in fn


# ---------------------------------------------------------------------------
# applyFieldManualFilter: local filter logic
# ---------------------------------------------------------------------------


def test_apply_field_manual_filter_is_defined_outside_row_renderers():
    text = _console_text()
    block = _brace_matched_block(text, "var ROW_RENDERERS = {")
    assert "applyFieldManualFilter" not in block
    assert re.search(r"^function applyFieldManualFilter\(rawQuery\) \{", text, re.MULTILINE)


def test_clearing_the_query_restores_every_entry_visible_and_collapsed():
    text = _console_text()
    fn = _extract_function(text, "applyFieldManualFilter")
    assert "if (!needle) {" in fn
    # Inside the blank-query branch: unhide + re-collapse every entry.
    blank_branch = fn[fn.index("if (!needle) {") : fn.index("if (!needle) {") + 700]
    assert 'classList.remove("filtered-out")' in blank_branch
    assert 'classList.add("collapsed")' in blank_branch
    assert 'setAttribute("aria-expanded", "false")' in blank_branch


def test_matching_entries_stay_visible_auto_expand_and_get_highlighted():
    text = _console_text()
    fn = _extract_function(text, "applyFieldManualFilter")
    assert "if (matched) {" in fn
    match_branch = fn[fn.index("if (matched) {") :]
    assert 'classList.remove("filtered-out")' in match_branch
    assert 'classList.remove("collapsed")' in match_branch
    assert 'setAttribute("aria-expanded", "true")' in match_branch
    assert "highlightMatches(target.headEl, fieldManualFilterQuery)" in match_branch


def test_non_matching_entries_are_hidden_not_merely_collapsed():
    text = _console_text()
    fn = _extract_function(text, "applyFieldManualFilter")
    assert 'wrap.classList.add("filtered-out")' in fn, (
        "non-matching entries must be hidden via .filtered-out (display:none), "
        "not just left in their .collapsed accordion state — issue #643 AC"
    )


def test_filter_matching_normalizes_nfc_and_lowercases_both_sides():
    text = _console_text()
    fn = _extract_function(text, "applyFieldManualFilter")
    assert fn.count('.normalize("NFC")') >= 2
    assert fn.count(".toLowerCase()") >= 2


def test_filtered_out_css_rule_hides_the_entry_entirely():
    text = _console_text()
    assert re.search(r"\.field-manual-entry\.filtered-out\s*\{[^}]*display:\s*none", text), (
        "the .filtered-out CSS rule must set display: none"
    )


# ---------------------------------------------------------------------------
# Lookup box sits at the top of the section (above the entries)
# ---------------------------------------------------------------------------


def test_search_widget_sits_above_the_entries_list_in_the_section():
    text = _console_text()
    block = _paren_matched_call(text, "fieldManualSection.append(")
    search_idx = block.index("fieldManualSearchWidget.wrap")
    entries_idx = block.index("fieldManualEntriesEl")
    assert search_idx < entries_idx, "the lookup box must sit above the entries list"


def test_search_widget_uses_the_shared_factory_with_the_field_manual_placeholder_key():
    text = _console_text()
    assert 'railSearchInput("fieldManualSearchPlaceholder", applyFieldManualFilter)' in text


# ---------------------------------------------------------------------------
# "?" jump clears any active filter first (issue #643 AC)
# ---------------------------------------------------------------------------


def test_open_field_manual_entry_clears_an_active_filter_before_expanding():
    text = _console_text()
    fn = _extract_function(text, "openFieldManualEntry")
    assert "fieldManualEntries[code]" in fn  # existing #637 behaviour still intact
    assert 'wrap.classList.contains("collapsed")' in fn
    assert "scrollSoftIntoView(wrap" in fn
    assert 'matchMedia("(max-width: 899px)")' in fn
    clear_idx = fn.index("if (fieldManualFilterQuery)")
    apply_idx = fn.index('applyFieldManualFilter("")')
    assert clear_idx < apply_idx
    # The clear must happen before the wrap's collapsed state is inspected.
    assert apply_idx < fn.index('wrap.classList.contains("collapsed")')


# ---------------------------------------------------------------------------
# Language toggle re-runs the filter (re-filter + re-highlight, not stale)
# ---------------------------------------------------------------------------


def test_apply_console_lang_reruns_the_field_manual_filter():
    text = _console_text()
    fn = _extract_function(text, "applyConsoleLang")
    assert "if (fieldManualSearchInput) {" in fn
    assert "applyFieldManualFilter(fieldManualSearchInput.value);" in fn


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
