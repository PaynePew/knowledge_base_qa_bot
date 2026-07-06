"""Structural tests for the Operator Console C9 Re-file UI (tier-B S4,
issue #380, ADR-0026 decision 1).

Following the pattern in ``test_ui_console_qa_edit.py`` / ``test_ui_console_
lint_remediation.py``, these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of the Re-file affordance:

- The Curation Queue's C9 section renders a real Re-file button wired to
  ``POST /wiki/qa/{slug}/refile`` (not the old read-only re-ingest hint).
- A successful refile marks the card in place (dims it, strikes the slug)
  and reports the observable consequence (old answer left the corpus,
  draft under Promotion Candidates) — gate legibility in operator language.
- A 422 (failed re-ground) renders an honest message, changes nothing
  visually beyond the status line, and re-enables the button.
- The shared taxonomy mirror (``LINT_CHECK_META``) reflects C9 as
  ``"authored"`` (matching ``markdown_kb/app/lint.py``'s
  ``_REMEDIATION_TAXONOMY``, the single source of truth).
- No new ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). Click -> refile -> card update is verified manually per §12.7.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _build_refile_flag_card_body() -> str:
    text = _console_text()
    match = re.search(
        r"function buildRefileFlagCard\(slug, subtitle\)\s*\{(.*?)\n\}\n", text, re.DOTALL
    )
    assert match is not None, "buildRefileFlagCard function not found in console.html"
    return match.group(1)


# ---------------------------------------------------------------------------
# Shared taxonomy mirror: C9 is authored (matches lint.py's single source of
# truth, ADR-0026 § Consequences)
# ---------------------------------------------------------------------------


def test_c9_taxonomy_mirror_is_authored():
    text = _console_text()
    match = re.search(r"var LINT_CHECK_META\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    assert match is not None
    body = match.group(1)
    c9_line = re.search(r"C9:\s*\{[^}]*\}", body)
    assert c9_line is not None, "LINT_CHECK_META must define C9"
    assert 'tier: "authored"' in c9_line.group(0), (
        f"C9 must mirror lint.py's authored tier (issue #380 / ADR-0026), got: {c9_line.group(0)}"
    )


# ---------------------------------------------------------------------------
# Curation Queue C9 section wires the real Re-file button
# ---------------------------------------------------------------------------


def test_curation_queue_c9_section_uses_refile_card():
    text = _console_text()
    c9_section = re.search(
        r"/\* .. C9: Stale filed answers .*?\*/(.*?)queueEl\.append\(c9Section\);",
        text,
        re.DOTALL,
    )
    assert c9_section is not None, "C9 Curation Queue block not found"
    body = c9_section.group(1)
    assert "buildRefileFlagCard(" in body, (
        "C9 stale-answer rows must render via buildRefileFlagCard (issue #380 AC "
        "'C9 findings gain a Re-file button')"
    )
    assert "buildFlagCard(" not in body, (
        "C9 must no longer use the old read-only buildFlagCard (Re-file replaces it)"
    )


def test_refile_button_present_and_posts_to_refile_endpoint():
    body = _build_refile_flag_card_body()
    assert '"edit-btn"' in body
    assert 'text: "Re-file"' in body, "buildRefileFlagCard must render a Re-file button"
    assert '"/wiki/qa/" + encodeURIComponent(slug) + "/refile"' in body
    assert 'method: "POST"' in body


# ---------------------------------------------------------------------------
# Success / failure rendering
# ---------------------------------------------------------------------------


def test_success_marks_card_and_reports_observable_consequence():
    """Gate legibility (ADR-0026 § Consequences): the success message states
    the observable consequence (old answer left the corpus; where to find
    the draft), not just a bare confirmation."""
    body = _build_refile_flag_card_body()
    assert "result.ok" in body
    success_branch = re.search(r"if \(result\.ok\)\s*\{(.*?)\n(\s{10}\}|\s{8}\})", body, re.DOTALL)
    assert success_branch is not None
    success_text = success_branch.group(0)
    assert "left the corpus" in success_text
    assert "Promotion Candidates" in success_text
    assert "cardEl.style.opacity" in success_text
    assert "flag-slug" in success_text


def test_422_renders_honest_reground_failure_and_reenables_button():
    body = _build_refile_flag_card_body()
    assert "result.status === 422" in body
    failure_branch = re.search(r"if \(result\.status === 422\)\s*\{(.*?)\n\s{8}\}", body, re.DOTALL)
    assert failure_branch is not None
    failure_text = failure_branch.group(0)
    assert "Cannot re-ground" in failure_text
    assert "nothing changed" in failure_text
    assert "refileBtn.disabled = false" in failure_text, (
        "a failed re-ground must re-enable the Re-file button — nothing was written"
    )


def test_refile_click_guards_against_double_submit():
    body = _build_refile_flag_card_body()
    click_fn = re.search(r"refileBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert click_fn is not None
    assert "if (actionBusy) return;" in click_fn.group(0)


# ---------------------------------------------------------------------------
# No innerHTML / textContent-only discipline (§12.4)
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_in_refile_card():
    body = _build_refile_flag_card_body()
    assert ".innerHTML =" not in body and ".innerHTML=" not in body


def test_no_inner_html_assignment_still_holds_console_wide():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text


# ---------------------------------------------------------------------------
# Full-width status row (issue #496): a long outcome message inside
# .flag-actions inflates that auto grid column's max-content, collapsing the
# 1fr body column to one word per line (prod repro: the C9 card's subtitle
# rendered as a single vertical word column after Re-file). The message must
# live on its own grid row spanning the whole card instead.
# ---------------------------------------------------------------------------


def _build_flag_card_body() -> str:
    text = _console_text()
    match = re.search(
        r"function buildFlagCard\(icon, slug, subtitle, showDiscard, onDiscard\)\s*\{(.*?)\n\}\n",
        text,
        re.DOTALL,
    )
    assert match is not None, "buildFlagCard function not found in console.html"
    return match.group(1)


def test_refile_status_message_lives_in_full_width_row_not_actions():
    body = _build_refile_flag_card_body()
    actions_line = re.search(r'el\("div", \{ class: "flag-actions" \}[^\n]*', body)
    assert actions_line is not None, "buildRefileFlagCard must render a .flag-actions container"
    assert "statusMsgEl" not in actions_line.group(0), (
        "the status message must not live inside .flag-actions — its length "
        "collapses the card's 1fr body column (issue #496)"
    )
    assert '"flag-status-row"' in body, (
        "buildRefileFlagCard must render the status message in the full-width "
        ".flag-status-row grid row (issue #496)"
    )


def test_flag_card_status_message_lives_in_full_width_row_not_actions():
    body = _build_flag_card_body()
    actions_line = re.search(r'el\("div", \{ class: "flag-actions" \}[^\n]*', body)
    assert actions_line is not None, "buildFlagCard must render a .flag-actions container"
    assert "statusMsgEl" not in actions_line.group(0), (
        "buildFlagCard's status message must not live inside .flag-actions (issue #496)"
    )
    assert '"flag-status-row"' in body


def test_flag_status_row_css_spans_grid_and_hides_when_empty():
    text = _console_text()
    assert ".flag-status-row { grid-column: 1 / -1;" in text, (
        ".flag-status-row must span the whole flag-card grid (issue #496)"
    )
    assert ".flag-status-row .card-status-msg:empty { display: none; }" in text, (
        "an empty status message must not reserve visible space (issue #496)"
    )


def test_c10_discard_threads_status_element_into_do_discard():
    """issue #505: the C10 wiring must pass the flag card's own statusMsgEl
    through to doDiscard — dropping it silently skips every outcome message
    ("Discarded." / the 409 live-page refusal / HTTP errors) and the row
    stays stuck on "Discarding…" forever."""
    text = _console_text()
    assert "doDiscard(slug, cardEl, statusMsgEl, null)" in text, (
        "C10's onDiscard must thread statusMsgEl into doDiscard (issue #505)"
    )
    body = _build_flag_card_body()
    assert "onDiscard(cardEl, statusMsgEl)" in body, (
        "buildFlagCard must hand its status element to the onDiscard callback (issue #505)"
    )
