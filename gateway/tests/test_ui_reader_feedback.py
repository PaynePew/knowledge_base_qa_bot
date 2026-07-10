"""Structural tests for the reader UI's Reader Feedback widget (issue #558).

The reader UI lives in ``gateway/static/index.html`` as a single vanilla
HTML/CSS/JS file (CODING_STANDARD §12.1). Following the established pattern
in ``test_ui_bilingual_starters.py``, these tests inspect the production UI
file's text — no DOM, no fetch, no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- Muted 👍/👎 row is built by ``renderFeedbackWidget`` and appended at the
  END of ``renderAnswerFoot`` — after the follow-up chips.
- Reaction posts to ``POST /feedback`` immediately (one click, no submit step).
- A comment append reuses the SAME ``answer_id``, minted once per answer card
  in ``onDone`` (not re-minted on every render).
- ``CHROME.en`` / ``CHROME.zh`` carry the widget's bilingual copy; zh values
  are real Chinese text.
- ``renderAnswerFoot`` unconditionally builds the widget (present on a
  Cannot Confirm card too — no gate on ``d.grounding.passed``); ``onError``
  never calls ``renderAnswerFoot``, so the widget is naturally absent from
  an error card.
- textContent-only (§12.4); no client-side re-implementation of grounding.
"""

from __future__ import annotations

import re
from pathlib import Path

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling console test helper of the same name)."""
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
# Widget exists, posts to POST /feedback, one click no submit step
# ---------------------------------------------------------------------------


def test_render_feedback_widget_function_exists():
    text = _ui_text()
    fn = _extract_function(text, "renderFeedbackWidget")
    assert "feedback-thumb" in fn
    assert '"aria-label"' in fn


def test_post_feedback_wrapper_posts_to_feedback_endpoint():
    text = _ui_text()
    fn = _extract_function(text, "postFeedback")
    assert 'fetch("/feedback"' in fn
    assert '"POST"' in fn or "'POST'" in fn


def test_thumb_click_posts_immediately_no_extra_submit_step():
    """Clicking a thumb calls submitRecord() synchronously inside reactTo,
    with no separate 'submit' control gating the reaction POST (AC4)."""
    text = _ui_text()
    fn = _extract_function(text, "renderFeedbackWidget")
    react_to = fn[fn.index("function reactTo(") : fn.index("function reactTo(") + 400]
    assert "submitRecord()" in react_to


def test_comment_send_reuses_reacted_value_and_appends_comment():
    text = _ui_text()
    fn = _extract_function(text, "renderFeedbackWidget")
    assert "submitRecord({ comment: comment })" in fn


# ---------------------------------------------------------------------------
# answer_id minted ONCE per answer card, in onDone — not re-minted per render
# ---------------------------------------------------------------------------


def test_answer_id_minted_once_in_ondone_not_on_every_render():
    text = _ui_text()
    on_done_fn = _extract_function(text, "onDone")
    assert "answerEl.feedbackCtx = {" in on_done_fn
    assert "answerId: mintAnswerId()" in on_done_fn

    # renderAnswerFoot / renderFeedbackWidget must NOT mint a fresh id — they
    # read the id already stashed on answerEl.feedbackCtx by onDone, so a
    # later re-render (language toggle) reuses the SAME answer_id.
    render_foot_fn = _extract_function(text, "renderAnswerFoot")
    feedback_widget_fn = _extract_function(text, "renderFeedbackWidget")
    assert "mintAnswerId()" not in render_foot_fn
    assert "mintAnswerId()" not in feedback_widget_fn


def test_render_feedback_widget_reads_answer_id_from_stashed_context():
    text = _ui_text()
    fn = _extract_function(text, "renderFeedbackWidget")
    assert "answerEl.feedbackCtx" in fn
    assert "ctx.answerId" in fn


# ---------------------------------------------------------------------------
# Widget order: end of renderAnswerFoot, after follow-up chips
# ---------------------------------------------------------------------------


def test_feedback_widget_appended_after_followups_in_render_answer_foot():
    text = _ui_text()
    fn = _extract_function(text, "renderAnswerFoot")
    followups_idx = fn.index('answerEl.append(fu)')
    feedback_idx = fn.index("renderFeedbackWidget(d)", fn.index("function renderAnswerFoot"))
    assert feedback_idx > followups_idx, (
        "the feedback widget must be appended AFTER the follow-up chips (AC: "
        "'appended at the END of renderAnswerFoot, after the follow-up chips')"
    )


def test_render_answer_foot_clears_prior_feedback_widget_on_rerender():
    """Idempotent re-render (language toggle) drops any prior .feedback copy,
    mirroring the existing .foot/.cnote/.followups cleanup."""
    text = _ui_text()
    fn = _extract_function(text, "renderAnswerFoot")
    assert '.querySelectorAll(".foot, .cnote, .followups, .feedback")' in fn


# ---------------------------------------------------------------------------
# Present on Cannot Confirm, absent on error (no pass-gating; onError never
# calls renderAnswerFoot)
# ---------------------------------------------------------------------------


def test_render_answer_foot_calls_renderFeedbackWidget_unconditionally():
    text = _ui_text()
    fn = _extract_function(text, "renderAnswerFoot")
    assert "var feedback = renderFeedbackWidget(d);" in fn
    assert "if (feedback) answerEl.append(feedback);" in fn


def test_ondone_calls_render_answer_foot_unconditionally():
    """onDone calls renderAnswerFoot(d) with no branch on d.grounding.passed,
    so the widget renders identically for a grounded answer and a Cannot
    Confirm one (AC4)."""
    text = _ui_text()
    fn = _extract_function(text, "onDone")
    assert re.search(r"^\s*renderAnswerFoot\(d\);\s*$", fn, re.MULTILINE) is not None


def test_onerror_never_calls_render_answer_foot():
    """onError builds its own .err block and never calls renderAnswerFoot, so
    the widget is naturally absent from an error card (AC4)."""
    text = _ui_text()
    fn = _extract_function(text, "onError")
    assert "renderAnswerFoot" not in fn
    assert "renderFeedbackWidget" not in fn


# ---------------------------------------------------------------------------
# Bilingual chrome
# ---------------------------------------------------------------------------


def test_chrome_defines_feedback_keys_in_both_languages():
    text = _ui_text()
    en_block = text.split("var CHROME = {", 1)[1].split("zh: {", 1)[0]
    zh_block = text.split("zh: {", 1)[1].split("};", 1)[0]
    keys = (
        "feedbackPrompt",
        "feedbackUpAria",
        "feedbackDownAria",
        "feedbackCommentPlaceholder",
        "feedbackSend",
        "feedbackSaving",
        "feedbackSaved",
        "feedbackFailed",
    )
    for key in keys:
        assert f"{key}:" in en_block, f"CHROME.en must define {key}"
        assert f"{key}:" in zh_block, f"CHROME.zh must define {key}"

    zh_prompt = re.search(r'feedbackPrompt:\s*"([^"]+)"', zh_block)
    assert zh_prompt is not None and _has_cjk(zh_prompt.group(1))


# ---------------------------------------------------------------------------
# §12.4: textContent only, no new EventSource
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_after_change():
    text = _ui_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text


def test_comment_input_has_500_char_maxlength_matching_server_cap():
    text = _ui_text()
    fn = _extract_function(text, "renderFeedbackWidget")
    assert 'maxlength: "500"' in fn
