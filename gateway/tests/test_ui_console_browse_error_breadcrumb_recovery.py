"""Structural test for the Operator Console browse-rail error recovery
(issue #629).

Following the pattern in ``test_ui_console_file_viewer_error_feedback.py``,
this test inspects the production ``gateway/static/console.html`` file's
text — no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- ``navigateTo``'s ``.catch`` re-renders the breadcrumb for the attempted
  path. Without this, a failed directory load (e.g. clicking the ``.trash``
  root while it is empty) leaves the breadcrumb in whatever state it was in
  before the click. When that state is the root listing, the root crumb is
  a non-clickable ``crumb-current`` span, so nothing in the rail is
  clickable after the error — the only recovery was a full page reload.
  Calling ``renderBreadcrumb(relpath)`` in the catch renders the attempted
  path, whose root crumb is always a clickable button (see
  ``renderBreadcrumb``'s own branching), restoring navigation.
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


def test_navigate_to_error_path_rerenders_breadcrumb_for_attempted_path():
    text = _console_text()
    body = _function_body(text, "navigateTo")
    catch_start = body.index(".catch(function(err)")
    catch_body = body[catch_start:]
    assert "renderBreadcrumb(relpath)" in catch_body, (
        "a failed navigateTo must re-render the breadcrumb for the attempted "
        "path so the root crumb becomes a clickable button — otherwise a "
        "failure starting from the root listing leaves nothing clickable in "
        "the rail (issue #629)"
    )


def test_render_breadcrumb_root_crumb_is_clickable_for_a_nonempty_path():
    """Guards the invariant navigateTo's catch relies on: any non-root
    relpath renders the root crumb as a clickable button, never the
    non-clickable crumb-current span used only for relpath === ''."""
    text = _console_text()
    body = _function_body(text, "renderBreadcrumb")
    assert 'relpath === ""' in body
    else_branch = body[body.index('relpath === ""') :]
    assert '"crumb-btn"' in else_branch
