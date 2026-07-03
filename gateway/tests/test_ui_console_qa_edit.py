"""Structural tests for the Operator Console edit-before-promote UI (tier-B S3,
issue #379, ADR-0026 decision 2).

Following the pattern in ``test_ui_console_legibility.py`` /
``test_ui_console_reconcile.py``, these tests inspect the production
``gateway/static/console.html`` file's text to assert the structural
invariants of the Edit affordance:

- The C8 draft card renders a real Edit button wired to ``PUT /wiki/qa/{slug}``
  (draft-only edit-before-promote).
- Save forwards the live textarea/input values (not a stale closure).
- A 422 grounding-re-check failure renders the per-item ``failures``
  list inline rather than a generic error.
- Card copy states what promoting *does* in operator language (no ADR
  reference required in this NEW copy — issue #379 AC), and the post-promote
  message closes the loop with an observable consequence.
- No new ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). Click → edit → save → re-render is verified manually per §12.7.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _build_draft_card_body() -> str:
    text = _console_text()
    match = re.search(r"function buildDraftCard\(f\)\s*\{(.*?)\n\}\n", text, re.DOTALL)
    assert match is not None, "buildDraftCard function not found in console.html"
    return match.group(1)


# ---------------------------------------------------------------------------
# Edit affordance exists and is wired to PUT /wiki/qa/{slug}
# ---------------------------------------------------------------------------


def test_edit_button_present_in_draft_card():
    body = _build_draft_card_body()
    assert '"edit-btn"' in body, "buildDraftCard must render an Edit button (issue #379 AC)"
    assert 'text: "Edit"' in body


def test_save_sends_put_to_qa_slug_endpoint():
    body = _build_draft_card_body()
    assert '"/wiki/qa/" + encodeURIComponent(slug)' in body
    assert 'method: "PUT"' in body, "Edit save must PUT (not POST) — draft-only edit, ADR-0026"


def test_save_forwards_live_input_values_not_stale_closure():
    """The client must send whatever is currently in the form fields, not a
    stale reference captured when the card was first built."""
    body = _build_draft_card_body()
    save_fn = re.search(r"saveBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert save_fn is not None
    save_body = save_fn.group(0)
    assert "questionInput.value" in save_body
    assert "bodyTextarea.value" in save_body


def test_edit_endpoint_is_put_not_post_and_matches_route():
    """PUT /qa/{slug} is a distinct route from POST /qa/{slug}/promote —
    confirm the Edit save does not accidentally hit the promote path."""
    body = _build_draft_card_body()
    save_fn = re.search(r"saveBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert save_fn is not None
    save_url_line = re.search(r'fetch\("/wiki/qa/"[^\n]*', save_fn.group(0))
    assert save_url_line is not None
    assert "/promote" not in save_url_line.group(0)


# ---------------------------------------------------------------------------
# Draft-only refusal / grounding-failure rendering
# ---------------------------------------------------------------------------


def test_422_renders_failure_list_inline():
    body = _build_draft_card_body()
    assert "result.status === 422" in body
    assert "detail.failures" in body, (
        "the 422 handler must render the LLM-free check's detail.failures list (ADR-0026)"
    )
    assert '"edit-validation-errors"' in body, (
        "422 rejection must render into a dedicated validation-errors element, not the generic status message"
    )


def test_save_success_updates_card_in_place():
    """A successful save must update the visible question/body text and exit
    edit mode — not just show a status message."""
    body = _build_draft_card_body()
    save_fn = re.search(r"saveBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert save_fn is not None
    save_body = save_fn.group(0)
    assert "questionEl.textContent = currentQuestion" in save_body
    assert "bodyPlaceholder.textContent" in save_body
    assert "closeEditForm()" in save_body


# ---------------------------------------------------------------------------
# Gate legibility (issue #379 AC): operator-language card copy + post-promote
# feedback that closes the loop
# ---------------------------------------------------------------------------


def test_card_states_what_promoting_does_in_operator_language():
    body = _build_draft_card_body()
    assert "draft-card-what-it-means" in body
    assert "the bot will answer similar questions with this approved answer" in body


def test_post_promote_message_closes_the_loop_with_observable_consequence():
    body = _build_draft_card_body()
    promote_fn = re.search(r"promoteBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert promote_fn is not None
    promote_body = promote_fn.group(0)
    assert "in chat" in promote_body
    assert "it now cites this page" in promote_body
    assert "currentQuestion" in promote_body, (
        "the post-promote message must include the actual question, not a generic phrase"
    )


# (The former test_existing_promote_line_copy_untouched guard is gone: it
# existed to protect the #347 ADR-0020 wording, which issue #379 AC3
# deliberately replaced with operator language. The updated pin lives in
# test_ui_console_legibility.py::test_console_promote_line_operator_copy.)


# ---------------------------------------------------------------------------
# No innerHTML / textContent-only discipline (§12.4)
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_in_draft_card():
    body = _build_draft_card_body()
    assert ".innerHTML =" not in body and ".innerHTML=" not in body
