"""Structural tests for the Operator Console C5 conflict panel (issue #443).

Following the pattern in ``test_ui_console_reconcile.py`` (tier-B S1, issue
#376), these tests inspect the production ``gateway/static/console.html``
file's text to assert the structural invariants of issue #443:

- The C5 finding row surfaces ``page_a_claim`` / ``page_b_claim`` (both
  quoted, attributed to their own page), the full ``summary``, and
  ``suggested_action`` — no more ``f.summary.slice(0, 80)`` silent
  truncation, and no other finding fields dropped.
- The Reconcile modal renders the SAME conflict panel above the editable
  drafts, built from the C5 finding looked up client-side (the
  ``/wiki/pages/reconcile`` generate response carries no claim fields —
  ``schemas.py`` ``ReconcileGenerateResponse``), never a second server
  round-trip.
- Long claim/summary/suggested_action text collapses to a short preview
  with an explicit expand/collapse toggle — nothing is ever silently cut
  with no way to read the rest.
- The existing ``reconcileAction(f.page_a, f.page_b)`` call site inside
  ROW_RENDERERS.C5 is unchanged (test_ui_console_reconcile.py pins it) —
  the extra finding data reaches the modal via a page-pair lookup map,
  not a grown argument list.
- The new chrome strings are bilingual (issue #365 convention): only the
  static labels are translated; the dynamic claim/summary/suggested_action
  VALUES stay untouched English per-finding text.
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the actual expand-toggle click is verified manually
per §12.7 (visual rendering is out of scope for unit tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors test_ui_console_lint_bilingual.py's helper of the
    same name)."""
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


# ---------------------------------------------------------------------------
# C5 row: full evidence, no silent truncation, reconcileAction call unchanged
# ---------------------------------------------------------------------------


def test_c5_row_renders_conflict_panel_with_full_evidence():
    text = _console_text()
    c5_row = re.search(r"C5:\s*function\(i, f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert c5_row is not None
    body = c5_row.group(1)
    assert "c5ConflictPanel(f, reconcileAction(f.page_a, f.page_b))" in body, (
        "the C5 row must wire the conflict panel around the existing Reconcile "
        "action, without changing reconcileAction's own call signature "
        "(pinned by test_ui_console_reconcile.py)"
    )
    assert ".slice(0, 80)" not in body, (
        "the C5 row must no longer silently truncate the summary (issue #443 AC)"
    )


def test_reconcile_action_call_signature_unchanged():
    """Guards the invariant the design relies on: reconcileAction(f.page_a,
    f.page_b) — test_ui_console_reconcile.py pins this literal call site, so
    the finding data must reach the modal some other way (a lookup map)."""
    text = _console_text()
    c5_row = re.search(r"C5:\s*function\(i, f\)\s*\{(.*?)\n    \},", text, re.DOTALL)
    assert c5_row is not None
    assert "reconcileAction(f.page_a, f.page_b)" in c5_row.group(1)


def test_c5_conflict_panel_surfaces_all_four_ac_fields():
    text = _console_text()
    fn = _extract_function(text, "c5ConflictPanel")
    for field in ("f.page_a_claim", "f.page_b_claim", "f.summary", "f.suggested_action"):
        assert field in fn, f"c5ConflictPanel must render {field} (issue #443 AC)"
    # Claims are attributed to their own page slug, not a generic "Page A/B".
    assert "f.page_a +" in fn and "f.page_b +" in fn


# ---------------------------------------------------------------------------
# Reconcile modal: the SAME panel, built from a client-side finding lookup
# ---------------------------------------------------------------------------


def test_reconcile_action_looks_up_finding_and_forwards_it():
    text = _console_text()
    fn = _extract_function(text, "reconcileAction")
    assert "c5FindingByPair[pageA" in fn, (
        "reconcileAction must look up the full C5 finding by page-pair key"
    )
    assert re.search(r"openReconcileModal\(pageA, pageB, resultEl, allButtons, .*\)", fn), (
        "reconcileAction must forward the looked-up finding into openReconcileModal"
    )


def test_reconcile_modal_and_preview_thread_the_finding_through():
    text = _console_text()
    open_fn = _extract_function(text, "openReconcileModal")
    assert re.search(r"function openReconcileModal\([^)]*finding\)", text), (
        "openReconcileModal must accept the finding as an explicit parameter"
    )
    assert (
        "renderReconcilePreview(modal, data, resultEl, closeModal, siblingButtons, finding)"
        in open_fn
    )

    preview_fn = re.search(r"function renderReconcilePreview\(.*?\n\}", text, re.DOTALL)
    assert preview_fn is not None
    body = preview_fn.group(0)
    assert re.search(r"function renderReconcilePreview\([^)]*finding\)", body), (
        "renderReconcilePreview must accept the finding as an explicit parameter"
    )
    assert "c5ConflictPanel(finding)" in body, (
        "the Reconcile modal must render the SAME conflict panel above the "
        "editable drafts (issue #443 AC)"
    )


def test_reconcile_generate_response_has_no_claim_fields_client_never_assumes_it_does():
    """Documents WHY the finding must be threaded through client-side rather
    than read off the generate response: ReconcileGenerateResponse (schemas.py)
    carries no claim/summary/suggested_action fields."""
    schemas_path = Path(__file__).resolve().parents[2] / "markdown_kb" / "app" / "schemas.py"
    schemas_text = schemas_path.read_text(encoding="utf-8")
    m = re.search(
        r"class ReconcileGenerateResponse\(BaseModel\):(.*?)\n\n\n?class ", schemas_text, re.DOTALL
    )
    assert m is not None
    body = m.group(1)
    for field in ("page_a_claim", "page_b_claim", "suggested_action"):
        assert field not in body


# ---------------------------------------------------------------------------
# Collapsible long text — explicit toggle, never a silent truncation
# ---------------------------------------------------------------------------


def test_collapsible_span_never_discards_the_full_text():
    text = _console_text()
    fn = _extract_function(text, "collapsibleSpan")
    assert "C5_COLLAPSE_AT" in fn
    assert "text.slice(0, C5_COLLAPSE_AT)" in fn
    # The full (untruncated) `text` variable must still be reachable after
    # the short preview is built — i.e. the toggle can restore it in full.
    assert "expanded ? text : short" in fn, (
        "the expand toggle must be able to restore the FULL original text, "
        "not just re-show a longer-but-still-truncated slice"
    )


def test_expand_collapse_chrome_is_bilingual():
    text = _console_text()
    assert re.search(r"c5Expand:\s*\"Show more\"", text)
    assert re.search(r"c5Collapse:\s*\"Show less\"", text)
    assert "顯示更多" in text
    assert "顯示較少" in text


# ---------------------------------------------------------------------------
# Bilingual chrome — labels only, dynamic finding text stays untranslated
# ---------------------------------------------------------------------------


def test_c5_summary_and_suggested_action_labels_are_bilingual():
    text = _console_text()
    assert re.search(r"c5Summary:\s*\"Summary:\"", text)
    assert re.search(r"c5SuggestedAction:\s*\"Suggested action:\"", text)
    assert "摘要:" in text
    assert "建議動作:" in text


def test_dynamic_c5_finding_text_still_not_wrapped_in_lint_chrome():
    """Mirrors test_ui_console_lint_bilingual.py's
    test_dynamic_finding_text_is_not_wrapped_in_lint_chrome — the per-finding
    ROW_RENDERERS object literal must not reference LINT_CHROME directly;
    only helper functions defined outside it (like c5ConflictPanel) may."""
    text = _console_text()
    row_renderers = re.search(r"var ROW_RENDERERS = \{(.*?)\n  \};", text, re.DOTALL)
    assert row_renderers is not None
    assert "LINT_CHROME[consoleLang]" not in row_renderers.group(1)


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
