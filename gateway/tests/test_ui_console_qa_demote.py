"""Structural tests for the Operator Console C10 Demote-to-draft UI
(issue #535, ADR-0037).

Following the pattern in ``test_ui_console_qa_refile.py``, these tests
inspect the production ``gateway/static/console.html`` file's text — no DOM,
no fetch, no browser, no OPENAI_API_KEY (fully hermetic, §6.3/§12.7).

Covers:

- The Curation Queue's C10 section renders ``buildDemoteFlagCard`` for a
  ``status: live`` schema-invalid finding instead of the old dead-end
  Discard button, and keeps the existing ``buildFlagCard``/``doDiscard``
  path for every non-live finding (unchanged AC).
- ``buildDemoteFlagCard`` posts to ``POST /wiki/qa/{slug}/demote``, marks
  the card in place on success (dims it, strikes the slug), self-heals on a
  404 (stale card), and guards against double-submit.
- The bilingual ``demoteCard*`` chrome strings exist and actually differ
  between ``en``/``zh``.
- The misleading "use re-ingest" copy is gone from the shared live-refusal
  message (``discardLiveRefused``), while the pinned "Cannot discard: page
  is live" substring (test_ui_console_stale_draft_404.py) is untouched.
- No new ``innerHTML`` assignment is introduced (§12.4).

Click -> demote -> card update is verified manually per §12.7.
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
    name)."""
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


def _c10_section_body(text: str) -> str:
    match = re.search(
        r"/\* .. C10: Invalid qa schema .*?\*/(.*?)queueSections\.push\(c10Section\);",
        text,
        re.DOTALL,
    )
    assert match is not None, "C10 Curation Queue block not found"
    return match.group(1)


# ---------------------------------------------------------------------------
# Curation Queue C10 section branches live vs non-live
# ---------------------------------------------------------------------------


def test_c10_section_branches_on_live_status():
    body = _c10_section_body(_console_text())
    assert 'items[0].status === "live"' in body, (
        "C10 must branch remediation on the finding's page-level status "
        "(ADR-0037: demote for live, discard for everything else)"
    )
    assert "buildDemoteFlagCard(" in body, (
        "a status:live C10 finding must render via buildDemoteFlagCard "
        "(issue #535 AC: 'shows 退回草稿 (demote), not the dead 拾棄')"
    )
    assert "buildFlagCard(" in body, (
        "a non-live C10 finding must still render via the existing buildFlagCard "
        "(AC: 'a non-live schema-invalid page is still discardable in one click')"
    )
    assert "doDiscard(slug, cardEl, statusMsgEl, null)" in body, (
        "the non-live path must still wire doDiscard exactly as before (unchanged AC)"
    )


def test_demote_card_posts_to_demote_endpoint():
    fn = _extract_function(_console_text(), "buildDemoteFlagCard")
    assert '"/wiki/qa/" + encodeURIComponent(slug) + "/demote"' in fn
    assert 'method: "POST"' in fn


def test_demote_card_success_marks_card_in_place():
    fn = _extract_function(_console_text(), "buildDemoteFlagCard")
    success_branch = re.search(r"if \(result\.ok\)\s*\{(.*?)\n(\s{10}\}|\s{8}\})", fn, re.DOTALL)
    assert success_branch is not None
    success_text = success_branch.group(0)
    assert "demoteCardSuccess" in success_text
    assert "cardEl.style.opacity" in success_text
    assert "flag-slug" in success_text
    assert "textDecoration" in success_text


def test_demote_card_self_heals_on_404():
    fn = _extract_function(_console_text(), "buildDemoteFlagCard")
    assert "resp.status === 404" in fn
    assert "handleStaleDraftGone(cardEl, statusMsgEl)" in fn


def test_demote_card_guards_against_double_submit():
    fn = _extract_function(_console_text(), "buildDemoteFlagCard")
    click_fn = re.search(r"demoteBtn\.addEventListener\(\"click\".*?\n  \}\);", fn, re.DOTALL)
    assert click_fn is not None
    assert "if (actionBusy) return;" in click_fn.group(0)


def test_demote_card_status_message_lives_in_full_width_row_not_actions():
    fn = _extract_function(_console_text(), "buildDemoteFlagCard")
    actions_line = re.search(r'el\("div", \{ class: "flag-actions" \}[^\n]*', fn)
    assert actions_line is not None, "buildDemoteFlagCard must render a .flag-actions container"
    assert "statusMsgEl" not in actions_line.group(0), (
        "the status message must not live inside .flag-actions (issue #496 convention)"
    )
    assert '"flag-status-row"' in fn


# ---------------------------------------------------------------------------
# Bilingual chrome
# ---------------------------------------------------------------------------


def test_demote_chrome_strings_exist_and_differ_between_languages():
    text = _console_text()
    en_match = re.search(r"en:\s*\{(.*?)\n  \},\n  zh:", text, re.DOTALL)
    zh_match = re.search(r"zh:\s*\{(.*?)\n  \},\n\};", text, re.DOTALL)
    assert en_match is not None and zh_match is not None

    for key in ("demoteCardLabel", "demoteCardDemoting", "demoteCardSuccess"):
        en_val = re.search(rf'{key}:\s*"([^"]+)"', en_match.group(1))
        zh_val = re.search(rf'{key}:\s*"([^"]+)"', zh_match.group(1))
        assert en_val is not None, f"LINT_CHROME.en must define {key}"
        assert zh_val is not None, f"LINT_CHROME.zh must define {key}"
        assert en_val.group(1) != zh_val.group(1), f"{key} must actually differ between en/zh"


# ---------------------------------------------------------------------------
# The misleading re-ingest copy is gone from the live-refusal message
# ---------------------------------------------------------------------------


def test_discard_live_refused_no_longer_suggests_reingest():
    text = _console_text()
    en_match = re.search(r'discardLiveRefused:\s*"([^"]+)"', text)
    assert en_match is not None
    assert "re-ingest" not in en_match.group(1).lower(), (
        "issue #535 / ADR-0037: re-ingest never regenerates a Filed Answer — "
        "the false remediation hint must be removed from the live-refusal message"
    )
    # Pinned substring (test_ui_console_stale_draft_404.py) must survive.
    assert "Cannot discard: page is live" in text


# ---------------------------------------------------------------------------
# No innerHTML / textContent-only discipline (§12.4)
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_in_demote_card():
    fn = _extract_function(_console_text(), "buildDemoteFlagCard")
    assert ".innerHTML =" not in fn and ".innerHTML=" not in fn
