"""Structural tests for the Browse rail quick-find box (issue #644).

A quick-find box at the top of the Browse rail (the read-only resource
browser, issue #171/#606) searches a FLAT, cross-tree listing fetched from
``GET /read/tree?recursive=true`` (issue #644) — a real backend addition,
since the client normally only ever holds one directory level at a time.
Reuses the SAME ``railSearchInput(placeholderKey, onQuery)`` factory and
``highlightMatches(containerEl, query)`` helper the #643 Field Manual lookup
box introduced (both module-scope, sibling-reusable by design).

Following the pattern in ``test_ui_console_field_manual_search.py``, these
tests inspect the production ``gateway/static/console.html`` file's text —
no DOM, no fetch, no browser, no OPENAI_API_KEY (fully hermetic, §6.3/§12.7).
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


def _paren_matched_call(text: str, marker: str) -> str:
    """Return the ``( ... )`` argument list immediately following ``marker``
    (which must end just before the opening paren), by depth-scanning."""
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


def _lint_chrome_en_zh(text: str) -> tuple[str, str]:
    """Return the (en_block, zh_block) source text of LINT_CHROME's two
    language objects (mirrors test_ui_console_lint_bilingual.py)."""
    full_body = _brace_matched_block(text, "var LINT_CHROME = {")

    def _sub_object(label: str) -> str:
        return _brace_matched_block(full_body, f"{label}: {{")

    return _sub_object("en"), _sub_object("zh")


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


# ---------------------------------------------------------------------------
# New LINT_CHROME keys: bilingual, non-empty, real Chinese on the zh side
# ---------------------------------------------------------------------------


def test_quick_find_chrome_keys_are_bilingual_and_zh_is_real_chinese():
    text = _console_text()
    en_block, zh_block = _lint_chrome_en_zh(text)
    keys = (
        "browseQuickFindPlaceholder",
        "browseQuickFindLoading",
        "browseQuickFindEmpty",
        "browseQuickFindTruncated",
        "browseQuickFindFailedToLoadPrefix",
    )
    for key in keys:
        en_match = re.search(rf'{key}:\s*"([^"]+)"', en_block)
        zh_match = re.search(rf'{key}:\s*"([^"]+)"', zh_block)
        assert en_match is not None, f"LINT_CHROME.en.{key} missing"
        assert zh_match is not None, f"LINT_CHROME.zh.{key} missing"
        assert en_match.group(1).strip(), f"LINT_CHROME.en.{key} is empty"
        assert _has_cjk(zh_match.group(1)), f"LINT_CHROME.zh.{key} must contain real Chinese text"


# ---------------------------------------------------------------------------
# Reuses the #643 shared factory/helper — no second implementation
# ---------------------------------------------------------------------------


def test_quick_find_widget_uses_the_shared_rail_search_input_factory():
    text = _console_text()
    assert 'railSearchInput("browseQuickFindPlaceholder", applyBrowseQuickFind)' in text, (
        "the Browse quick-find box must reuse the #643 railSearchInput() factory"
    )


def test_quick_find_row_reuses_highlight_matches_not_a_second_impl():
    """issue #644 AC: 'no second highlight implementation (must reuse
    highlightMatches)'."""
    text = _console_text()
    fn = _extract_function(text, "renderBrowseQuickFindRow")
    assert "highlightMatches(nameEl, browseQuickFindQuery)" in fn
    # No ad-hoc <mark> construction inside this function itself.
    assert 'el("mark"' not in fn


def test_apply_browse_quick_find_normalizes_nfc_and_lowercases_both_sides():
    text = _console_text()
    fn = _extract_function(text, "applyBrowseQuickFind")
    assert fn.count('.normalize("NFC")') >= 2, "must NFC-normalize both sides (issue #644 AC)"
    assert fn.count(".toLowerCase()") >= 2, "must lowercase both sides before comparing"


# ---------------------------------------------------------------------------
# Widget sits at the top of the Browse rail section
# ---------------------------------------------------------------------------


def test_quick_find_widget_sits_above_the_breadcrumb_and_entries():
    text = _console_text()
    block = _paren_matched_call(text, "browserSection.append(")
    search_idx = block.index("browseQuickFindWidget.wrap")
    breadcrumb_idx = block.index("breadcrumbEl")
    entries_idx = block.index("entriesEl")
    assert search_idx < breadcrumb_idx < entries_idx, (
        "the quick-find box must sit above both the breadcrumb and the entries list "
        "(issue #644 AC: 'quick-find box at the top of the Browse rail')"
    )


# ---------------------------------------------------------------------------
# Lazy-fetch-on-focus, once per search session (issue #644 AC)
# ---------------------------------------------------------------------------


def test_quick_find_fetches_the_recursive_tree_on_input_focus():
    text = _console_text()
    assert (
        'browseQuickFindWidget.input.addEventListener("focus", fetchBrowseQuickFindEntries);'
        in text
    ), "the recursive listing must be fetched lazily, on focus (issue #644 AC)"


def test_fetch_browse_quick_find_entries_calls_recursive_endpoint():
    text = _console_text()
    fn = _extract_function(text, "fetchBrowseQuickFindEntries")
    assert '"/read/tree?"' in fn
    assert "recursive" in fn and '"true"' in fn
    assert "browseQuickFindFetchInFlight" in fn, "must guard against overlapping fetches"


def test_fetch_browse_quick_find_entries_caches_truncated_flag():
    text = _console_text()
    fn = _extract_function(text, "fetchBrowseQuickFindEntries")
    assert "browseQuickFindTruncated = !!data.truncated;" in fn


# ---------------------------------------------------------------------------
# Typing >=1 char replaces the breadcrumb view; clearing restores it
# ---------------------------------------------------------------------------


def test_active_query_hides_breadcrumb_and_renders_flat_results():
    text = _console_text()
    fn = _extract_function(text, "applyBrowseQuickFind")
    assert 'breadcrumbEl.style.display = "none";' in fn
    assert "matches.map(renderBrowseQuickFindRow)" in fn


def test_clearing_the_query_restores_the_breadcrumb_view_from_cache_no_refetch():
    text = _console_text()
    fn = _extract_function(text, "applyBrowseQuickFind")
    blank_branch = fn[: fn.index('breadcrumbEl.style.display = "none";')]
    assert 'breadcrumbEl.style.display = "";' in blank_branch
    assert "renderEntries(lastBrowserEntries)" in blank_branch
    # No fetch() call in the blank-query branch — restoring is a pure re-render.
    assert "fetch(" not in blank_branch


def test_no_matches_shows_the_dedicated_empty_state():
    text = _console_text()
    fn = _extract_function(text, "applyBrowseQuickFind")
    assert "matches.length === 0" in fn
    assert "LINT_CHROME[consoleLang].browseQuickFindEmpty" in fn


def test_truncated_warning_toggles_with_the_cached_flag():
    text = _console_text()
    fn = _extract_function(text, "applyBrowseQuickFind")
    assert 'browseQuickFindTruncatedEl.classList.toggle("visible", browseQuickFindTruncated)' in fn


# ---------------------------------------------------------------------------
# Result-row click wiring (issue #644 AC)
# ---------------------------------------------------------------------------


def test_file_result_row_opens_like_a_normal_folder_listing_row():
    text = _console_text()
    fn = _extract_function(text, "renderBrowseQuickFindRow")
    assert "openFile(entry.relpath, entry.name)" in fn


def test_dir_result_row_navigates_in_and_clears_the_search():
    text = _console_text()
    fn = _extract_function(text, "renderBrowseQuickFindRow")
    clear_idx = fn.index("clearBrowseQuickFind()")
    nav_idx = fn.index("navigateTo(entry.relpath)")
    assert clear_idx < nav_idx, "must clear the search BEFORE navigating into the directory"


def test_clear_browse_quick_find_resets_input_and_reapplies_empty_query():
    text = _console_text()
    fn = _extract_function(text, "clearBrowseQuickFind")
    assert 'browseQuickFindWidget.input.value = "";' in fn
    assert 'applyBrowseQuickFind("");' in fn


# ---------------------------------------------------------------------------
# Language toggle re-runs the quick-find (re-filter + re-highlight)
# ---------------------------------------------------------------------------


def test_apply_console_lang_reruns_the_browse_quick_find():
    text = _console_text()
    fn = _extract_function(text, "applyConsoleLang")
    assert "if (browseQuickFindWidget) {" in fn
    assert "applyBrowseQuickFind(browseQuickFindWidget.input.value);" in fn


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
