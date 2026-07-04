"""Structural tests for file-viewer error/busy feedback in the Operator
Console (issue #444).

Following the pattern in ``test_ui_console_c3_routed_fix_source.py`` /
``test_ui_console_wikilink_linkify.py``, these tests inspect the production
``gateway/static/console.html`` file's text — no DOM, no fetch, no browser,
no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- A failing ``openFile`` scrolls the file viewer into view exactly like a
  success (previously only the success path scrolled).
- ``openFile`` accepts an OPTIONAL third argument (``{ onError }``) without
  changing its declared two-parameter signature — the existing
  ``test_ui_console_wikilink_linkify.py`` pins the exact literal
  ``"function openFile(relpath, name)"`` text, so the extra argument is read
  via ``arguments`` rather than a named parameter.
- Both file-open callers (the Fix Source banner's View Source button, and
  browse-tree file rows) wire an ``onError`` callback that renders an inline
  notice next to the control that was actually clicked.
- ``browserBusy`` writes are funnelled through one ``setBrowserBusy`` choke
  point that also flips a shared ``.file-open-busy`` body class, and CSS
  disables every file-open trigger (browse-tree rows, the View Source
  button, wikilinks) while it is set — so a click during an in-flight read
  is visibly refused, never a silent no-op.
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
# openFile: error path scrolls into view, exactly like success
# ---------------------------------------------------------------------------


def test_open_file_signature_is_unchanged_for_the_wikilink_test():
    """test_ui_console_wikilink_linkify.py pins the exact literal
    "function openFile(relpath, name)" — the third argument MUST stay
    optional/unnamed (read via `arguments`), never a declared parameter."""
    text = _console_text()
    assert "function openFile(relpath, name) {" in text


def test_open_file_error_path_scrolls_the_viewer_into_view():
    text = _console_text()
    body = _function_body(text, "openFile")
    catch_start = body.index(".catch(function(err)")
    catch_body = body[catch_start:]
    assert "scrollSoftIntoView(" in catch_body, (
        "a failing openFile must scroll the viewer into view, same as the success path"
    )


def test_open_file_success_and_error_both_scroll():
    """Guards against a future edit accidentally removing one of the two —
    both branches of the same Promise chain must scroll."""
    text = _console_text()
    body = _function_body(text, "openFile")
    assert body.count("scrollSoftIntoView(") >= 2, (
        "expected one scrollSoftIntoView call on the success path and one on the error path"
    )


def test_open_file_reads_an_optional_third_argument_via_arguments():
    text = _console_text()
    body = _function_body(text, "openFile")
    assert "arguments[2]" in body or "arguments.length" in body, (
        "the optional onError callback must be read via `arguments`, not a named parameter"
    )


def test_open_file_invokes_on_error_callback_when_present():
    text = _console_text()
    body = _function_body(text, "openFile")
    catch_start = body.index(".catch(function(err)")
    catch_body = body[catch_start:]
    assert "opts.onError" in catch_body, (
        "openFile's catch handler must invoke the caller-supplied onError callback"
    )


def test_open_file_busy_writes_go_through_the_shared_setter():
    """browserBusy must be set via setBrowserBusy (not a raw assignment)
    inside openFile, so the busy body class always tracks browserBusy."""
    text = _console_text()
    body = _function_body(text, "openFile")
    assert "setBrowserBusy(true)" in body
    assert "setBrowserBusy(false)" in body
    assert "browserBusy = true" not in body
    assert "browserBusy = false" not in body


def test_navigate_to_busy_writes_also_go_through_the_shared_setter():
    text = _console_text()
    body = _function_body(text, "navigateTo")
    assert "setBrowserBusy(true)" in body
    assert "setBrowserBusy(false)" in body
    assert "browserBusy = true" not in body
    assert "browserBusy = false" not in body


# ---------------------------------------------------------------------------
# setBrowserBusy: one choke point driving the shared busy body class
# ---------------------------------------------------------------------------


def test_set_browser_busy_toggles_the_shared_body_class():
    text = _console_text()
    body = _function_body(text, "setBrowserBusy")
    assert "browserBusy = busy" in body
    assert 'classList.toggle("file-open-busy", busy)' in body


def test_busy_css_disables_every_file_open_trigger():
    text = _console_text()
    assert "body.file-open-busy .entry-row" in text
    assert "body.file-open-busy .view-source-btn" in text
    assert "body.file-open-busy .wikilink" in text
    css_match = re.search(r"body\.file-open-busy[^{]*\{([^}]*)\}", text, re.DOTALL)
    assert css_match is not None
    css_body = css_match.group(1)
    assert "pointer-events: none" in css_body, (
        "the click must never reach the handler while busy — a real disable, not just a dim"
    )


# ---------------------------------------------------------------------------
# Browse-tree file rows: per-row notice slot wired to openFile's onError
# ---------------------------------------------------------------------------


def test_render_entries_gives_file_rows_a_notice_slot():
    text = _console_text()
    body = _function_body(text, "renderEntries")
    assert "entry-row-notice" in body
    assert "entry.is_dir ? null :" in body, (
        "only file rows (not directories) get a notice slot — a directory's "
        "failure already renders inline in entriesEl via navigateTo's own catch"
    )


def test_render_entries_wires_notice_to_open_file_on_error():
    text = _console_text()
    body = _function_body(text, "renderEntries")
    assert "onError: function(message)" in body
    assert 'noticeEl.classList.add("visible")' in body


def test_entry_row_notice_css_hidden_by_default_and_toggled_visible():
    text = _console_text()
    hidden_match = re.search(r"\.entry-row-notice\s*\{([^}]*)\}", text, re.DOTALL)
    assert hidden_match is not None
    assert "display: none" in hidden_match.group(1)
    visible_match = re.search(r"\.entry-row-notice\.visible\s*\{([^}]*)\}", text)
    assert visible_match is not None


# ---------------------------------------------------------------------------
# Fix Source banner's View Source button: inline notice next to the button
# ---------------------------------------------------------------------------


def test_show_fix_source_banner_wires_view_button_to_an_error_notice():
    text = _console_text()
    body = _function_body(text, "showFixSourceBanner")
    assert "onError: function(message)" in body
    assert '"card-status-msg err"' in body, (
        "the banner's View Source failure notice must reuse the existing "
        "card-status-msg err styling, not introduce a new error style"
    )


def test_show_fix_source_banner_still_never_fetches_directly():
    """ADR-0029 Invariant preserved: only openFile() (existing GET) is
    reached from inside this function; no new fetch call."""
    text = _console_text()
    body = _function_body(text, "showFixSourceBanner")
    assert "fetch(" not in body


def test_show_fix_source_banner_still_calls_open_file_with_docs_path():
    text = _console_text()
    body = _function_body(text, "showFixSourceBanner")
    assert "view-source-btn" in body
    assert "openFile(" in body
    assert '"docs/"' in body


# ---------------------------------------------------------------------------
# §12.4: still textContent/safe-DOM-construction only (no innerHTML introduced)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
