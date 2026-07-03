"""Structural tests for the Operator Console C11 Confirmed orphan-delete UI
(tier-B S5, issue #381, ADR-0024/0025).

Following the pattern in ``test_ui_console_qa_refile.py`` / ``test_ui_console_
collision.py``, these tests inspect the production ``gateway/static/console.html``
file's text to assert the structural invariants of the delete affordance:

- The shared taxonomy mirror (``LINT_CHECK_META``) reflects C11 as
  ``"confirmed"`` (matching ``markdown_kb/app/lint.py``'s
  ``_REMEDIATION_TAXONOMY``, the single source of truth).
- The C11 row wires ``deleteOrphanAction(f.page_slug, f.full)`` — the client
  reads the server-computed full/partial eligibility off the finding, it
  never re-derives it (§12.5).
- A full orphan renders a real Delete button opening a Confirmed dialog; a
  partial orphan renders advisory text only, no button.
- The Confirmed dialog names the operation + target (the slug) before it
  runs, posts ``DELETE /wiki/pages/{slug}``, and on success closes itself
  and re-runs the shared re-lint + re-render path.
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the click -> confirm -> delete -> re-lint loop is
verified manually per §12.7 (visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _delete_orphan_action_body() -> str:
    text = _console_text()
    match = re.search(r"function deleteOrphanAction\(slug, full\)\s*\{(.*?)\n  \}", text, re.DOTALL)
    assert match is not None, "deleteOrphanAction function not found in console.html"
    return match.group(1)


def _open_delete_confirm_body() -> str:
    text = _console_text()
    match = re.search(
        r"function openDeleteOrphanConfirm\(slug, resultEl, siblingButtons\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert match is not None, "openDeleteOrphanConfirm function not found in console.html"
    return match.group(1)


# ---------------------------------------------------------------------------
# Shared taxonomy mirror: C11 is confirmed (matches lint.py's single source
# of truth, ADR-0025 § Consequences)
# ---------------------------------------------------------------------------


def test_c11_taxonomy_mirror_is_confirmed():
    text = _console_text()
    match = re.search(r"var LINT_CHECK_META\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    assert match is not None
    body = match.group(1)
    c11_line = re.search(r"C11:\s*\{[^}]*\}", body)
    assert c11_line is not None, "LINT_CHECK_META must define C11"
    assert 'tier: "confirmed"' in c11_line.group(0), (
        f"C11 must mirror lint.py's confirmed tier (issue #381 / ADR-0025), got: "
        f"{c11_line.group(0)}"
    )


# ---------------------------------------------------------------------------
# C11 row wires the real delete action, reading server-computed eligibility
# ---------------------------------------------------------------------------


def test_c11_row_wires_delete_orphan_action_with_full_flag():
    text = _console_text()
    c11_row = re.search(r"C11:\s*function\(i, f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert c11_row is not None, "C11 row renderer not found"
    assert "deleteOrphanAction(f.page_slug, f.full)" in c11_row.group(1), (
        "C11 rows must wire deleteOrphanAction reading the server-computed "
        "full/partial eligibility off the finding (issue #381 AC)"
    )


# ---------------------------------------------------------------------------
# Full orphan -> real Delete button; partial orphan -> advisory text only
# ---------------------------------------------------------------------------


def test_full_orphan_gets_delete_button():
    body = _delete_orphan_action_body()
    assert '"discard-btn"' in body
    assert "text: LINT_CHROME[consoleLang].deleteOrphan" in body
    assert "openDeleteOrphanConfirm(slug, resultEl, allButtons)" in body


def test_partial_orphan_renders_advisory_text_only():
    body = _delete_orphan_action_body()
    advisory_branch = re.search(r"if \(!full\)\s*\{(.*?)\n    \}", body, re.DOTALL)
    assert advisory_branch is not None
    branch_text = advisory_branch.group(0)
    assert "partialOrphanAdvisory" in branch_text
    assert "<button" not in branch_text and 'el("button"' not in branch_text, (
        "a partial orphan must never render a clickable delete affordance"
    )


def test_delete_click_guards_against_double_submit():
    body = _delete_orphan_action_body()
    click_fn = re.search(r"btn\.addEventListener\(\"click\".*?\n    \}\);", body, re.DOTALL)
    assert click_fn is not None
    assert "if (btn.disabled || remediationInFlight) return;" in click_fn.group(0)


# ---------------------------------------------------------------------------
# Confirmed dialog: names the operation + target, posts DELETE, closes +
# re-lints on success
# ---------------------------------------------------------------------------


def test_confirm_dialog_names_operation_and_target():
    body = _open_delete_confirm_body()
    assert "deleteConfirmTitle" in body
    assert "deleteConfirmBody" in body
    assert '.replace("{slug}", slug)' in body, (
        "the confirmation body must interpolate the concrete target slug "
        "(ADR-0024: 'names the irreversible operation and its target')"
    )


def test_confirm_posts_delete_to_pages_endpoint():
    text = _console_text()
    fn = re.search(r"function deleteOrphanRequest\(slug\)\s*\{(.*?)\n\}", text, re.DOTALL)
    assert fn is not None, "deleteOrphanRequest function not found"
    body = fn.group(1)
    assert '"/wiki/pages/" + encodeURIComponent(slug)' in body
    assert 'method: "DELETE"' in body


def test_confirm_success_closes_modal_and_relints():
    body = _open_delete_confirm_body()
    success_branch = re.search(r"\.then\(function\(\)\s*\{(.*?)\n      \}\)", body, re.DOTALL)
    assert success_branch is not None
    success_text = success_branch.group(0)
    assert "closeModal()" in success_text
    assert "finishBatchRemediation(" in success_text


def test_confirm_409_refusal_renders_honest_message_and_reenables():
    body = _open_delete_confirm_body()
    failure_branch = re.search(r"\.catch\(function\(err\)\s*\{(.*?)\n      \}\);", body, re.DOTALL)
    assert failure_branch is not None
    failure_text = failure_branch.group(0)
    assert "confirmBtn.disabled = false" in failure_text
    assert "cancelBtn.disabled = false" in failure_text
    assert "err.message" in failure_text


def test_confirm_click_guards_against_double_submit():
    body = _open_delete_confirm_body()
    click_fn = re.search(r"confirmBtn\.addEventListener\(\"click\".*?\n  \}\);", body, re.DOTALL)
    assert click_fn is not None
    assert "if (confirmBtn.disabled || remediationInFlight) return;" in click_fn.group(0)


# ---------------------------------------------------------------------------
# No innerHTML / textContent-only discipline (§12.4)
# ---------------------------------------------------------------------------


def test_no_inner_html_assignment_in_delete_orphan_code():
    action_body = _delete_orphan_action_body()
    confirm_body = _open_delete_confirm_body()
    for body in (action_body, confirm_body):
        assert ".innerHTML =" not in body and ".innerHTML=" not in body


def test_no_inner_html_assignment_still_holds_console_wide():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
