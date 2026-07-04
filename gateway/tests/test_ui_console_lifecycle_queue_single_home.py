"""Structural tests for issue #438 (Console: C8/C9/C10 render only in the
Curation Queue — remove Lint-card duplicate rows).

Before this slice, the Lint card's Lifecycle axis rendered a button-less row
per C8/C9/C10 finding (via ``ROW_RENDERERS``) IN ADDITION to the same
findings' actionable cards in the dedicated Curation Queue block — the same
finding shown twice on one page (issue #438). This slice makes the Curation
Queue the single home: the Lint card's axis loop (``renderLintCard`` in
``gateway/static/console.html``) now skips the per-item ``ROW_RENDERERS``
loop for C8/C9/C10 and instead renders one count + pointer line
(``queueCountLine``).

Following the pattern established by ``test_ui_console_lint_remediation.py``
(issue #363) and ``test_ui_console_lint_remediation_batch.py`` (issue #364),
these tests inspect the production ``gateway/static/console.html`` file's
text — no DOM, no fetch, no browser, no OPENAI_API_KEY (fully hermetic,
§6.3/§12.7). Visual rendering is verified manually; these structural
assertions are the committed test surface.

Note: the pre-existing ``ROW_RENDERERS.C8/C9/C10`` entries and the
``discardAction`` helper are intentionally left in place (dead code from
the Lint card's perspective) rather than deleted, because
``test_ui_console_lint_remediation.py`` pins their exact shape via regex
(e.g. ``test_c8_promotion_controls_stay_in_curation_queue_only``,
``test_c10_invalid_schema_gets_discard_button_in_lint_card``) and this slice
does not touch pre-existing test files. A follow-up slice can delete both
the now-unreachable renderer functions and the obsolete pinning tests
together.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors test_ui_console_lint_remediation_batch.py's helper)."""
    marker = f"function {name}("
    start = text.index(marker)
    depth = 0
    started = False
    i = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
            started = True
        elif text[i] == "}":
            depth -= 1
            if started and depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unterminated function body for {name}")


def _axis_loop_body(text: str) -> str:
    match = re.search(r"LINT_AXIS_ORDER\.forEach\(function\(axis\) \{.*?\n  \}\);", text, re.DOTALL)
    assert match is not None, "console.html must define the LINT_AXIS_ORDER.forEach render loop"
    return match.group(0)


# ---------------------------------------------------------------------------
# The queue-owned set exists and names exactly C8/C9/C10
# ---------------------------------------------------------------------------


def test_queue_owned_lifecycle_checks_set_is_exactly_c8_c9_c10():
    text = _console_text()
    match = re.search(r"var QUEUE_OWNED_LIFECYCLE_CHECKS\s*=\s*\{([^}]*)\};", text)
    assert match is not None, "console.html must define QUEUE_OWNED_LIFECYCLE_CHECKS (issue #438)"
    body = match.group(1)
    for code in ["C8", "C9", "C10"]:
        assert re.search(rf"\b{code}\s*:\s*true", body), (
            f"QUEUE_OWNED_LIFECYCLE_CHECKS missing {code}"
        )
    # No other check code should be marked queue-owned.
    for code in ["C1", "C2", "C3", "C4", "C5", "C6", "C11"]:
        assert not re.search(rf"\b{code}\s*:", body), (
            f"QUEUE_OWNED_LIFECYCLE_CHECKS unexpectedly includes {code}"
        )


# ---------------------------------------------------------------------------
# The axis loop skips per-item rows for queue-owned checks
# ---------------------------------------------------------------------------


def test_axis_loop_branches_on_queue_owned_checks_before_row_rendering():
    """The QUEUE_OWNED_LIFECYCLE_CHECKS guard must short-circuit (return)
    BEFORE the ROW_RENDERERS invocation, so C8/C9/C10 findings never reach
    the per-item row loop (issue #438 AC: 'each finding renders exactly
    once ... in the Curation Queue')."""
    text = _console_text()
    body = _axis_loop_body(text)
    guard_idx = body.index("if (QUEUE_OWNED_LIFECYCLE_CHECKS[code])")
    row_render_idx = body.index("axisBody.push(ROW_RENDERERS[code](i + 1, item))")
    assert guard_idx < row_render_idx, (
        "the queue-owned guard must appear before the ROW_RENDERERS loop so it can "
        "return early and skip per-item rows"
    )
    # The guard branch itself must return (skip to the next check code).
    guard_branch = body[guard_idx:row_render_idx]
    assert "return;" in guard_branch, (
        "the queue-owned branch must return before falling through to the "
        "per-item ROW_RENDERERS loop"
    )
    assert "queueCountLine(code, items.length, checkBatchBtnEl)" in guard_branch


def test_c10_discard_all_batch_button_still_flows_into_queue_count_line():
    """C10's existing 'Discard all' batch button (a bulk action, not a
    per-finding duplicate) must still be computed and passed through —
    this slice only removes per-item rows, not the check-level batch
    action already covered by test_ui_console_lint_remediation_batch.py."""
    text = _console_text()
    body = _axis_loop_body(text)
    checkBatchBtnEl_assign_idx = body.index("checkBatchBtnEl = discardAllAction(c10Slugs)")
    guard_idx = body.index("if (QUEUE_OWNED_LIFECYCLE_CHECKS[code])")
    assert checkBatchBtnEl_assign_idx < guard_idx, (
        "checkBatchBtnEl must be computed before the queue-owned guard so C10's "
        "Discard all button is still passed into queueCountLine"
    )


# ---------------------------------------------------------------------------
# queueCountLine: single line, count + bilingual pointer
# ---------------------------------------------------------------------------


def test_queue_count_line_renders_count_and_bilingual_pointer():
    text = _console_text()
    fn = _extract_function(text, "queueCountLine")
    assert 'code + " " + label + " (" + count + ") — " + pointer' in fn
    assert "LINT_CHROME[consoleLang].seeCurationQueue" in fn
    # batchBtnEl still rides along on the same one line (issue #438: "at
    # most a one-line count with a pointer").
    assert "batchBtnEl" in fn


def test_queue_count_line_pointer_chrome_strings_are_bilingual():
    text = _console_text()
    en_match = re.search(r"en:\s*\{(.*?)\n  \},\n  zh:", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},\n\};", text, re.DOTALL)
    assert en_match is not None and zh_match is not None, (
        "console.html must define the LINT_CHROME en/zh dictionaries"
    )
    en_pointer = re.search(r'seeCurationQueue:\s*"([^"]+)"', en_match.group(1))
    zh_pointer = re.search(r'seeCurationQueue:\s*"([^"]+)"', zh_match.group(1))
    assert en_pointer is not None, "LINT_CHROME.en must define seeCurationQueue (issue #438)"
    assert zh_pointer is not None, "LINT_CHROME.zh must define seeCurationQueue (issue #438)"
    assert en_pointer.group(1) != zh_pointer.group(1), (
        "the en/zh pointer strings must actually differ, not be a copy-paste"
    )
    assert "Curation Queue" in en_pointer.group(1)
    assert "Curation Queue" in zh_pointer.group(1), (
        "the Curation Queue proper noun itself stays English in both languages, "
        "matching renderCurationQueue's own (never localized) header text"
    )


# ---------------------------------------------------------------------------
# Curation Queue's own actions are unchanged (AC: "Promote / Discard / Edit
# unchanged")
# ---------------------------------------------------------------------------


def test_curation_queue_actions_are_unchanged():
    text = _console_text()
    fn = _extract_function(text, "renderCurationQueue")
    assert "promoteAllAction(" in fn, "C8 Promote action must still live in the Curation Queue"
    assert "buildDraftCard(f)" in fn, "C8 draft cards must still render in the Curation Queue"
    assert "buildRefileFlagCard(" in fn, "C9 Re-file action must still live in the Curation Queue"
    assert "buildFlagCard(" in fn, "C10 flag cards must still render in the Curation Queue"
    assert "doDiscard(slug, cardEl)" in fn, (
        "C10 Discard action must still live in the Curation Queue"
    )


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4) — re-checked for this slice's diff specifically
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
