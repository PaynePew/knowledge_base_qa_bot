"""Structural tests for the Operator Console C5 Reconcile UI (tier-B S1, issue #376, ADR-0028).

Following the pattern in ``test_ui_console_lint_remediation.py``, these tests
inspect the production ``gateway/static/console.html`` file's text to assert
the structural invariants of the C5 Reconcile flow:

- The C5 row renders a real "Reconcile" button (not the disabled tier-b-btn
  every other Authored-tier check still renders — C4 unchanged; C1/C2 later
  gained their own real affordance in tier-B S7, issue #383, ADR-0027).
- Reconcile opens a generate -> side-by-side/editable-preview -> apply flow
  wired to the two new endpoints (``POST /wiki/pages/reconcile`` and
  ``POST /wiki/pages/reconcile/apply``), never recomputing the content hash
  client-side (CODING_STANDARD §12.5 — the hash values are forwarded
  verbatim from the generate response).
- 409 (stale draft) and 422 (grounding failure) responses render an honest,
  specific explanation rather than a generic error.
- A successful apply re-runs ``POST /wiki/lint?include_c5=false`` and
  re-renders both the Lint card and the Curation Queue (matching every
  other remediation's success path).
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the click -> preview -> apply -> re-lint loop is
verified manually per §12.7 (visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# C5 row renders a real Reconcile affordance, not the disabled tier-b-btn
# ---------------------------------------------------------------------------


def test_c5_row_wires_reconcile_action_not_tier_b():
    text = _console_text()
    c5_row = re.search(r"C5:\s*function\(i, f\)\s*\{(.*?)\},", text, re.DOTALL)
    assert c5_row is not None
    assert "reconcileAction(f.page_a, f.page_b)" in c5_row.group(1), (
        "C5 finding rows must wire the real Reconcile action (issue #376 AC)"
    )
    assert "tierBAffordance()" not in c5_row.group(1), (
        "C5 no longer renders the disabled tier-B placeholder — it ships a real affordance"
    )


# NOTE: the C1/C2 "still render disabled tier-B affordance" assertion that
# used to live here was removed in tier-B S7 (issue #383, ADR-0027) — C1/C2
# flip from Authored to Routed and gained a real "Fill via Import"
# navigation control, so the claim is no longer true. Coverage moved to
# ``test_ui_console_routed_coverage_fill.py``, mirroring how C4's own
# real-affordance coverage moved to ``test_ui_console_collision.py`` in
# tier-B S2 (see the module docstring above).


def test_reconcile_button_class_and_chrome_defined():
    text = _console_text()
    assert "reconcile-btn" in text
    assert '"reconcileApply"' not in text  # sanity: chrome keys are bare identifiers, not quoted
    assert re.search(r"reconcile:\s*\"Reconcile\"", text), (
        "LINT_CHROME.en.reconcile must be defined"
    )
    assert re.search(r"reconcileApply:\s*\"Apply\"", text)
    assert re.search(r"reconcileCancel:\s*\"Cancel\"", text)


# ---------------------------------------------------------------------------
# Endpoint wiring — the two new tier-B endpoints, hash forwarded verbatim
# ---------------------------------------------------------------------------


def test_generate_and_apply_endpoints_are_wired():
    text = _console_text()
    assert '"/wiki/pages/reconcile"' in text, "openReconcileModal must POST /wiki/pages/reconcile"
    assert '"/wiki/pages/reconcile/apply"' in text, "Apply must POST /wiki/pages/reconcile/apply"


def test_apply_forwards_hash_tokens_verbatim_never_recomputed():
    """CODING_STANDARD §12.5 — no business logic in the client: the client
    must forward the server-issued hash tokens unchanged, never compute its
    own hash of the (possibly edited) textarea content."""
    text = _console_text()
    apply_fn = re.search(r"applyBtn\.addEventListener\(\"click\".*?\n  \}\);", text, re.DOTALL)
    assert apply_fn is not None
    body = apply_fn.group(0)
    assert "hash_a: data.hash_a" in body
    assert "hash_b: data.hash_b" in body
    assert "sha256" not in body.lower(), "the client must never compute its own content hash"


def test_apply_sends_edited_textarea_content_not_the_original_draft():
    """The curator can edit the draft preview before applying — Apply must
    send the LIVE textarea values, not a stale closure over data.content_a/b."""
    text = _console_text()
    apply_fn = re.search(r"applyBtn\.addEventListener\(\"click\".*?\n  \}\);", text, re.DOTALL)
    assert apply_fn is not None
    body = apply_fn.group(0)
    assert "content_a: textareaA.value" in body
    assert "content_b: textareaB.value" in body


# ---------------------------------------------------------------------------
# Honest error rendering — 409 stale draft, 422 grounding failure
# ---------------------------------------------------------------------------


def _reconcile_preview_body() -> str:
    text = _console_text()
    preview_fn = re.search(r"function renderReconcilePreview\(.*?\n\}", text, re.DOTALL)
    assert preview_fn is not None
    return preview_fn.group(0)


def test_409_hash_mismatch_renders_specific_explanation():
    body = _reconcile_preview_body()
    assert "resp.status === 409" in body
    err_409 = re.search(r"resp\.status === 409\)\s*\{\s*throw new Error\(([\s\S]*?)\);", body)
    assert err_409 is not None
    message = err_409.group(1)
    assert "changed since" in message, "409 must explain WHY (a page changed since generate)"
    assert "re-open Reconcile" in message or "re-open reconcile" in message.lower()


def test_422_grounding_failure_lists_unsupported_claims():
    body = _reconcile_preview_body()
    assert "resp.status === 422" in body
    err_422 = re.search(r"resp\.status === 422\)[\s\S]*?unsupported_claims", body)
    assert err_422 is not None, "422 handling must read detail.unsupported_claims"
    assert "Grounding check failed" in body


# ---------------------------------------------------------------------------
# Success path — re-lints (fast path) and re-renders both the Lint card
# and the Curation Queue, matching runLintRemediation's established pattern
# ---------------------------------------------------------------------------


def test_apply_success_re_runs_fast_lint_and_rerenders():
    text = _console_text()
    preview_fn = re.search(r"function renderReconcilePreview\(.*?\n\}", text, re.DOTALL)
    assert preview_fn is not None
    body = preview_fn.group(0)
    assert '"/wiki/lint?include_c5=false"' in body, (
        "A successful apply must re-run POST /wiki/lint?include_c5=false (issue #376 AC)"
    )
    assert "renderLintCard(lintData, resultEl)" in body
    assert "renderCurationQueue(" in body
    assert "closeModal()" in body


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
