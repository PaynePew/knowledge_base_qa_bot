"""Console Reader Feedback panel (issue #558).

Following the pattern in ``test_ui_console_budget_usage.py``: inspects the
production ``gateway/static/console.html`` file's text — no DOM, no fetch,
no browser, no OPENAI_API_KEY (fully hermetic, §6.3 / §12.7). Verifies:

  - The ``#feedback-panel-root`` markup exists AFTER the Curation Queue root
    (end of the step chain, outside the pipeline's action-arrow track).
  - LINT_CHROME carries the panel's bilingual chrome; the general
    ``test_lint_chrome_en_and_zh_key_sets_are_identical`` parity guard in
    ``test_ui_console_i18n_coverage.py`` already covers en/zh key parity for
    these new keys automatically.
  - ``buildFeedbackPanel`` fetches ``GET /feedback`` from a button click
    handler (one-shot, no polling — NOT a boot-time IIFE fetch like the
    budget banner).
  - ``renderFeedbackResult`` renders the bilingual empty state when there are
    no records, and a stats+table view otherwise, via textContent only.
  - ``applyConsoleLang`` re-renders the panel from ``lastFeedbackData``
    without a re-fetch.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling ``test_ui_console_*`` helper of the same name)."""
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


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


# ---------------------------------------------------------------------------
# Markup: panel root exists, placed at the END of the step chain
# ---------------------------------------------------------------------------


def test_feedback_panel_root_markup_present():
    text = _console_text()
    assert '<div id="feedback-panel-root">' in text


def test_feedback_panel_root_comes_after_curation_queue_root():
    text = _console_text()
    assert text.index('id="curation-queue-root"') < text.index('id="feedback-panel-root"'), (
        "the Reader Feedback panel must sit at the END of the console-main "
        "flow, after the Curation Queue — a monitoring surface outside the "
        "pipeline's action-arrow track (issue #558 AC)"
    )


# ---------------------------------------------------------------------------
# Bilingual chrome (key-set parity is covered generically by
# test_ui_console_i18n_coverage.py::test_lint_chrome_en_and_zh_key_sets_are_identical)
# ---------------------------------------------------------------------------


def test_feedback_chrome_keys_present_and_zh_is_real_chinese():
    text = _console_text()
    for key in (
        "feedbackPanelTitle",
        "feedbackPanelDesc",
        "feedbackLoadBtn",
        "feedbackLoading",
        "feedbackError",
        "feedbackEmpty",
        "feedbackUpLabel",
        "feedbackDownLabel",
        "feedbackTotalLabel",
        "feedbackColTime",
        "feedbackColReaction",
        "feedbackColQuery",
        "feedbackColComment",
        "feedbackColStack",
        "feedbackColGrounding",
    ):
        assert f"{key}:" in text, f"LINT_CHROME must define {key}"

    zh_match = re.search(r'feedbackEmpty:\s*"([^"]+)"', text.split("zh: {", 1)[1])
    assert zh_match is not None
    assert _has_cjk(zh_match.group(1)), "zh feedbackEmpty must contain real Chinese text"


def test_feedback_empty_state_matches_the_grilled_bilingual_copy():
    """Pins the exact zh empty-state string named in the issue AC."""
    text = _console_text()
    zh_block = text.split("zh: {", 1)[1]
    assert 'feedbackEmpty:       "尚無回饋 — Reader 答案卡上的 👍/👎 會出現在這裡"' in zh_block


# ---------------------------------------------------------------------------
# One-shot fetch on button click — not a boot-time IIFE (unlike the budget banner)
# ---------------------------------------------------------------------------


def test_feedback_panel_fetches_get_feedback_exactly_once_site():
    text = _console_text()
    assert text.count('fetch("/feedback")') == 1


def test_feedback_fetch_is_inside_a_click_handler_not_a_boot_iife():
    text = _console_text()
    fn = _extract_function(text, "buildFeedbackPanel")
    assert 'addEventListener("click"' in fn
    assert 'fetch("/feedback")' in fn


def test_feedback_panel_not_fetched_at_boot():
    """No polling / boot fetch — the console idiom is one-shot on click."""
    text = _console_text()
    # The budget banner's boot IIFE is `(function loadBudgetUsage() {`;
    # Reader Feedback has no equivalent boot-time IIFE calling GET /feedback.
    assert "(function loadFeedback" not in text


# ---------------------------------------------------------------------------
# renderFeedbackResult: empty state vs stats+table, textContent only
# ---------------------------------------------------------------------------


def test_render_feedback_result_shows_empty_state_for_zero_records():
    text = _console_text()
    fn = _extract_function(text, "renderFeedbackResult")
    assert "data.records.length === 0" in fn
    assert "feedback-empty" in fn


def test_render_feedback_result_renders_stats_and_table_for_nonempty():
    text = _console_text()
    fn = _extract_function(text, "renderFeedbackResult")
    assert "renderFeedbackStats(data.counts)" in fn
    assert '"feedback-table"' in fn


def test_render_feedback_row_uses_textcontent_only_and_all_columns():
    text = _console_text()
    fn = _extract_function(text, "renderFeedbackRow")
    assert ".innerHTML" not in fn
    for field in ("rec.ts", "rec.query", "rec.comment", "rec.stack", "rec.grounding"):
        assert field in fn


def test_render_feedback_row_truncates_query_but_not_comment():
    """AC: 'query (truncated), comment (full text, wrapped)' — only the
    query column is capped."""
    text = _console_text()
    fn = _extract_function(text, "renderFeedbackRow")
    assert "truncateFeedbackQuery(rec.query)" in fn
    assert "rec.comment || chrome.feedbackNoComment" in fn, (
        "the comment cell must render the FULL rec.comment text, never truncated"
    )


# ---------------------------------------------------------------------------
# Language toggle re-renders from cached data, no re-fetch
# ---------------------------------------------------------------------------


def test_apply_console_lang_rerenders_feedback_panel_without_refetch():
    text = _console_text()
    fn = _extract_function(text, "applyConsoleLang")
    assert "lastFeedbackData !== null" in fn
    assert "renderFeedbackResult(lastFeedbackData)" in fn


# ---------------------------------------------------------------------------
# No new innerHTML anywhere in the file (§12.4 regression guard, mirrors
# test_ui_console_i18n_coverage.py::test_console_no_inner_html_assignment_still_holds)
# ---------------------------------------------------------------------------


def test_console_still_has_no_inner_html_assignment():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
