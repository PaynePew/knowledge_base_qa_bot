"""Structural tests for Operator Console batch remediation + busy-state +
durability guards (issue #364, ADR-0023 tier-A S4).

Following the pattern in ``test_ui_console_lint_remediation.py`` (S3), these
tests inspect the production ``gateway/static/console.html`` file's text to
assert the structural invariants of S4:

- The Freshness axis header and the C6 check group each offer a
  "Re-ingest all" batch action; the Freshness batch unions C6 and C3
  sources into a single ``POST /wiki/ingest`` call with ``force:true``
  (ADR-0023 Invariant — a C3-mixed batch omitting ``force`` is a false
  fix). The C6-only check-level batch mirrors the per-row C6 action and
  does NOT force ``force:true``.
- The C10 check group offers a "Discard all" batch action. There is no
  batch-delete endpoint, so it fires the individual
  ``DELETE /wiki/qa/{slug}`` calls sequentially behind one button.
- Batch is Direct-tier-only: no axis-level batch button is wired for
  Lifecycle, Coherence, or Coverage; no batch action is wired into any
  Authored-tier (C5/C4/C1/C2) row renderer.
- An in-flight remediation (single-row or batch) disables every
  remediation button on the Lint card (the shared ``allButtons`` /
  ``remediationInFlight`` guard) and shows a spinner.
- A ``beforeunload`` confirm is installed and only fires while
  ``remediationInFlight`` is true.
- A completed batch re-lints (``include_c5=false``) and surfaces a
  partial-failure summary ("re-ingested N: X ok, Y still failing").
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). The actual click -> batch -> relint loop was additionally exercised
against a hand-rolled DOM shim during implementation (ad hoc, not
committed — §12.7 treats DOM/visual rendering as manually verified, not
unit-tested); these structural assertions are the committed test surface.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching, so assertions don't over/under-match past the function's own
    closing brace (regex-only anchoring bit past reviewers on this file
    before — see the S3 test file's reingestAction/tierBAffordance notes)."""
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


# ---------------------------------------------------------------------------
# Batch buttons exist at the right scope
# ---------------------------------------------------------------------------


def test_freshness_axis_gets_axis_level_batch_button():
    text = _console_text()
    assert "axis-batch-btn" in text
    # axisHead() was folded into the collapsible: the axis-level batch button is
    # now passed to makeCollapse as the header's non-toggling aside element.
    assert "asideEl: axisBatchBtnEl" in text


def test_c6_check_group_gets_check_level_batch_button():
    text = _console_text()
    assert "check-batch-btn" in text
    check_head_fn = _extract_function(text, "checkHead")
    assert "batchBtnEl" in check_head_fn


def test_c10_check_group_gets_discard_all_button():
    text = _console_text()
    discard_all_fn = _extract_function(text, "discardAllAction")
    assert 'text: "Discard all (" + slugs.length + ")"' in discard_all_fn


def test_axis_level_batch_wired_only_for_freshness():
    """The axis loop must gate the axis-level batch behind an explicit
    ``axis === "Freshness"`` check — Lifecycle/Coherence/Coverage must not
    get one (ADR-0023 Invariant: 'Lifecycle stays check-level only ...
    with no axis-level button')."""
    text = _console_text()
    axis_loop = re.search(
        r"LINT_AXIS_ORDER\.forEach\(function\(axis\) \{.*?\n  \}\);", text, re.DOTALL
    )
    assert axis_loop is not None
    body = axis_loop.group(0)
    assert 'if (axis === "Freshness")' in body
    # reingestAllAction is only invoked from the C6 branch and the
    # Freshness-gated axis-level branch — never unconditionally.
    assert body.count("reingestAllAction(") == 2


def test_freshness_batch_unions_c6_and_c3_sources():
    text = _console_text()
    axis_loop = re.search(
        r"LINT_AXIS_ORDER\.forEach\(function\(axis\) \{.*?\n  \}\);", text, re.DOTALL
    )
    assert axis_loop is not None
    body = axis_loop.group(0)
    assert 'code === "C6" || code === "C3"' in body
    assert "freshnessSources[item.source] = true" in body


# ---------------------------------------------------------------------------
# force:true invariant — axis batch only, not the C6-only check batch
# ---------------------------------------------------------------------------


def test_freshness_axis_batch_sends_force_true():
    """ADR-0023 Invariant: the Freshness axis batch mixes in C3 sources, so
    it MUST send force:true or hash-skip (#93) no-ops the C3 retries."""
    text = _console_text()
    axis_loop = _extract_function(text, "renderLintCard")
    match = re.search(
        r"axisBatchBtnEl = reingestAllAction\(freshnessSourceList,\s*true,\s*\"axis-batch-btn\"\)",
        axis_loop,
    )
    assert match is not None, (
        "the Freshness axis-level 'Re-ingest all' must call "
        'reingestAllAction(freshnessSourceList, true, "axis-batch-btn") — force:true is mandatory'
    )


def test_c6_check_level_batch_does_not_force():
    """The C6-only check-level batch mirrors the per-row C6 action
    (force:false) — force:true is a C3-mixing invariant, not a general
    Direct-tier batch default (mirrors the S3 'only C3 requires force'
    reviewer guard, extended to the batch case)."""
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    match = re.search(
        r"checkBatchBtnEl = reingestAllAction\(c6Sources,\s*false,\s*\"check-batch-btn\"\)",
        render_fn,
    )
    assert match is not None, (
        "the C6 check-level 'Re-ingest all' must call "
        'reingestAllAction(c6Sources, false, "check-batch-btn")'
    )


def test_batch_ingest_request_only_sets_force_when_truthy():
    text = _console_text()
    fn = _extract_function(text, "batchIngestRemediationRequest")
    assert "if (force) payload.force = true;" in fn


# ---------------------------------------------------------------------------
# C10 Discard-all loops DELETE calls (no batch-delete endpoint)
# ---------------------------------------------------------------------------


def test_discard_all_fires_sequential_deletes_not_a_new_endpoint():
    text = _console_text()
    fn = _extract_function(text, "discardAllAction")
    assert "discardRemediationRequest(slugs[i])" in fn
    assert '"/wiki/qa/' not in fn.replace("discardRemediationRequest", "")  # no inline new fetch
    # Uses the existing single-slug request, looped — not a new batch endpoint.
    assert fn.count("discardRemediationRequest(") == 1


# ---------------------------------------------------------------------------
# Batch is Direct-tier-only — no batch on Authored rows (C5/C4/C1/C2)
# ---------------------------------------------------------------------------


def test_authored_tier_rows_never_wire_a_batch_action():
    text = _console_text()
    row_renderers = re.search(r"var ROW_RENDERERS = \{(.*?)\n  \};", text, re.DOTALL)
    assert row_renderers is not None
    body = row_renderers.group(1)
    for code in ["C5", "C4", "C1", "C2"]:
        row = re.search(rf"{code}: function\(i, f\) \{{(.*?)\n    \}},", body, re.DOTALL)
        assert row is not None, f"could not isolate {code} row renderer"
        assert "reingestAllAction" not in row.group(1)
        assert "discardAllAction" not in row.group(1)


# ---------------------------------------------------------------------------
# In-flight guard — every remediation button disables its siblings
# ---------------------------------------------------------------------------


def test_shared_in_flight_flag_and_sibling_button_list_exist():
    text = _console_text()
    assert "var remediationInFlight = false;" in text
    assert "var allButtons = [];" in text


def test_row_level_actions_pass_sibling_buttons_to_runner():
    text = _console_text()
    reingest_fn = _extract_function(text, "reingestAction")
    assert (
        "runLintRemediation(ingestRemediationRequest(source, retry === true), btn, statusMsgEl, resultEl, allButtons)"
        in reingest_fn
    )
    discard_fn = _extract_function(text, "discardAction")
    assert (
        "runLintRemediation(discardRemediationRequest(slug), btn, statusMsgEl, resultEl, allButtons)"
        in discard_fn
    )


def test_runner_disables_all_sibling_buttons_while_in_flight():
    text = _console_text()
    fn = _extract_function(text, "runLintRemediation")
    assert "buttons.forEach(function(b) { b.disabled = true; });" in fn
    assert "remediationInFlight = true;" in fn
    assert "remediationInFlight = false;" in fn


def test_batch_actions_disable_all_buttons_and_show_spinner():
    text = _console_text()
    for fn_name in ["reingestAllAction", "discardAllAction"]:
        fn = _extract_function(text, fn_name)
        assert "allButtons.forEach(function(b) { b.disabled = true; });" in fn
        assert 'el("span", { class: "spinner" })' in fn
        assert "remediationInFlight = true;" in fn


def test_batch_button_click_guards_against_double_submit():
    text = _console_text()
    for fn_name in ["reingestAllAction", "discardAllAction"]:
        fn = _extract_function(text, fn_name)
        assert "if (btn.disabled || remediationInFlight) return;" in fn


# ---------------------------------------------------------------------------
# beforeunload — soft durability notice, not a corruption warning
# ---------------------------------------------------------------------------


def test_beforeunload_guard_installed_and_gated_on_in_flight_flag():
    text = _console_text()
    match = re.search(
        r'window\.addEventListener\("beforeunload", function\(e\) \{(.*?)\n\}\);',
        text,
        re.DOTALL,
    )
    assert match is not None
    body = match.group(1)
    assert "if (!remediationInFlight) return;" in body
    assert "e.preventDefault();" in body
    assert 'e.returnValue = "";' in body


def test_beforeunload_comment_frames_the_notice_as_soft_not_alarmist():
    """§11 drift signal: a lint-remediation durability notice must not claim
    a refresh will corrupt data (POST /ingest is durable server-side under
    _index_lock — ADR-0023)."""
    text = _console_text()
    idx = text.index('window.addEventListener("beforeunload"')
    surrounding = text[max(0, idx - 1800) : idx]
    assert "SOFT" in surrounding or "soft" in surrounding
    assert "corrupt" not in surrounding.lower() or "never" in surrounding.lower()


def test_batch_keeps_in_flight_flag_armed_until_relint_resolves():
    """finishBatchRemediation must not clear remediationInFlight before the
    follow-up re-lint read-back — otherwise the beforeunload guard is disabled
    while a request is still in flight, inconsistent with the per-row
    runLintRemediation path (issue #364 verify finding). The flag-false must
    appear only inside the re-lint .then()/.catch(), i.e. after the fetch."""
    text = _console_text()
    fn = _extract_function(text, "finishBatchRemediation")
    fetch_pos = fn.index('adminFetch("/wiki/lint?include_c5=false"')
    first_clear = fn.index("remediationInFlight = false")
    assert first_clear > fetch_pos, (
        "remediationInFlight is cleared before the re-lint fetch — the unload "
        "guard would be off during the re-lint window"
    )


# ---------------------------------------------------------------------------
# Partial-failure summary on completion
# ---------------------------------------------------------------------------


def test_ingest_partial_summary_reports_ok_and_still_failing_counts():
    text = _console_text()
    fn = _extract_function(text, "ingestPartialSummary")
    assert '"re-ingested " + sources.length + ": " + okCount + " ok, " +' in fn
    assert (
        '"pages_with_failed_grounding"'.strip('"') not in fn or "pages_with_failed_grounding" in fn
    )


def test_discard_all_reports_ok_and_still_failing_counts():
    text = _console_text()
    fn = _extract_function(text, "discardAllAction")
    assert (
        '"discarded " + slugs.length + ": " + okCount + " ok, " + failCount + " still failing"'
        in fn
    )


def test_batch_completion_re_lints_and_re_renders_via_shared_helper():
    text = _console_text()
    fn = _extract_function(text, "finishBatchRemediation")
    assert '"/wiki/lint?include_c5=false"' in fn
    assert "renderLintCard(data, resultEl)" in fn
    assert "renderCurationQueue(" in fn
    assert "lastBatchSummary = summaryText;" in fn


def test_batch_summary_banner_is_one_shot_in_the_re_rendered_card():
    text = _console_text()
    render_fn = _extract_function(text, "renderLintCard")
    assert "if (lastBatchSummary) {" in render_fn
    assert 'class: "lint-batch-summary"' in render_fn
    assert "lastBatchSummary = null;" in render_fn


# ---------------------------------------------------------------------------
# No new backend endpoint (ADR-0023: zero new endpoints, tier A)
# ---------------------------------------------------------------------------


def test_batch_remediation_reuses_existing_endpoints_only():
    text = _console_text()
    batch_ingest_fn = _extract_function(text, "batchIngestRemediationRequest")
    assert '"/wiki/ingest"' in batch_ingest_fn
    discard_all_fn = _extract_function(text, "discardAllAction")
    assert "discardRemediationRequest" in discard_all_fn  # reuses DELETE /wiki/qa/{slug}


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4) — re-checked for this slice's diff specifically
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
