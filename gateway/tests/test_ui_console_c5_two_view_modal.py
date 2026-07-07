"""Structural tests for the C5 two-view Reconcile modal + fix-source batch
panel (issue #538, ADR-0036 decisions 2-5).

Following the pattern in ``test_ui_console_reconcile.py`` /
``test_ui_console_c3_routed_fix_source.py`` /
``test_ui_console_routed_coverage_fill.py``, these tests inspect the
production ``gateway/static/console.html`` file's text — no DOM, no fetch,
no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- The Reconcile modal header renders a real toggle between Wiki comparison
  (the pre-existing view) and Source comparison, and the default view is
  chosen from THIS generate's ``data.converged`` (source-rooted i.e. grounded-
  but-not-converged -> source, else -> wiki; ADR-0038 supersedes ADR-0036
  decision 2's ``grounding.passed`` routing), and Apply is gated on grounded
  AND converged — the system never auto-classifies which layer to fix, it only
  picks which evidence to show first and refuses to Apply a source-rooted pair.
- The Source comparison view renders both pages' own ``cited_sections_a``/
  ``cited_sections_b`` (issue #534's payload), each with a read-only
  ``/read/file`` view control and an edit entry point that accumulates the
  Source into the fix-source batch panel — never fetching anything itself.
- A citation that could not be resolved to content disables both controls
  (no false affordance).
- The fix-source batch panel accumulates MULTIPLE targets (an array,
  deduplicated), unlike C3's single-pending-slot banner, and its batch run
  drives exactly one force re-ingest and one deep-audit (ADR-0036
  decision 5), never a per-target stepper.
- A wiki-rooted Reconcile still edits -> applies exactly as before — the
  Apply button's request body and success path are untouched by the toggle.
- No ``innerHTML`` assignment is introduced (§12.4).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 /
§12.7). DOM rendering / the actual toggle-click / batch-upload loop is
verified manually per §12.7 (visual rendering is out of scope for unit
tests).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace
    matching (mirrors the sibling test files' helper of the same name)."""
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
# Header toggle: real Wiki/Source views, default chosen by grounding.passed
# ---------------------------------------------------------------------------


def test_view_toggle_chrome_defined_bilingually():
    text = _console_text()
    assert re.search(r'reconcileViewWiki:\s*"[^"]+"', text)
    assert re.search(r'reconcileViewSource:\s*"[^"]+"', text)
    assert "Wiki 比較" in text
    assert "Source 比較" in text


def test_default_view_is_chosen_by_convergence():
    """ADR-0038 (supersedes ADR-0036 decision 2's grounding-based routing):
    the default view follows CONVERGENCE, not grounding. A source-rooted pair
    (grounded but not converged) defaults to Source comparison; everything else
    defaults to Wiki. `data.converged` is read fail-safe (=== true)."""
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert "var converged = data.converged === true;" in fn
    assert 'var activeView = (passed && !converged) ? "source" : "wiki";' in fn


def test_apply_is_gated_on_grounding_and_convergence():
    """ADR-0038 Invariant: Apply is enabled iff grounded AND converged, so a
    source-rooted pair can never be Applied into a fresh contradiction."""
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert "var applyEnabled = passed && converged;" in fn
    assert "applyBtn.disabled = !applyEnabled;" in fn


def test_apply_handles_not_converged_422():
    """ADR-0038: an apply-time convergence refusal (a hand-edit that still
    contradicts) surfaces the source-rooted message, not a grounding one."""
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert 'detail.reason === "not_converged"' in fn


def test_apply_success_optimistically_clears_resolved_c5_pair():
    """issue #546: the fast relint excludes C5, so renderLintCard re-injects the
    preserved lastDeepC5Pairs. A successful reconcile apply must drop the
    resolved pair from that preserved list so the finding clears WITHOUT a manual
    deep re-audit — order-insensitive, and only on the 200 success path (a
    still-contradicting apply 422s, so there is no false clear)."""
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert "lastDeepC5Pairs = lastDeepC5Pairs.filter(" in fn
    assert "p.page_a === data.page_a && p.page_b === data.page_b" in fn
    assert "p.page_a === data.page_b && p.page_b === data.page_a" in fn


def test_both_toggle_buttons_always_rendered_and_wired():
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert 'class: "reconcile-view-toggle-btn"' in fn
    assert 'activeView = "wiki"; renderActiveView();' in fn
    assert 'activeView = "source"; renderActiveView();' in fn


def test_grounding_report_and_view_body_both_rendered_in_modal():
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert "viewToggleEl," in fn
    assert "groundingEl," in fn
    assert "viewBodyEl" in fn


# ---------------------------------------------------------------------------
# Source comparison view: both pages' cited sections, view link + edit entry
# ---------------------------------------------------------------------------


def test_source_view_renders_both_pages_cited_sections():
    text = _console_text()
    fn = _extract_function(text, "renderC5SourceComparisonView")
    assert "data.cited_sections_a" in fn
    assert "data.cited_sections_b" in fn
    assert "renderC5SourceCard(data.page_a" in fn
    assert "renderC5SourceCard(data.page_b" in fn


def test_source_view_shows_disagree_note_only_when_not_converged():
    """ADR-0038: the "Sources disagree" note keys on convergence, not grounding
    (grounding cannot see cross-page disagreement)."""
    text = _console_text()
    fn = _extract_function(text, "renderC5SourceComparisonView")
    assert "sources-disagree-note" in fn
    assert "var converged = data.converged === true;" in fn
    assert "if (!converged)" in fn
    assert "c5SourcesDisagreeNote" in fn


def test_source_card_offers_a_view_file_link_wired_to_read_file():
    """Mirrors C3's fix-source banner convention: reuse openFile() (GET
    /read/file), never a second fetch implementation, and never a
    client-built path guess — section.source_path is used verbatim."""
    fn = _extract_function(_console_text(), "renderC5SourceCard")
    assert "view-source-btn" in fn
    assert "openFile(section.source_path" in fn
    assert '"docs/" + ' not in fn


def test_source_card_edit_entry_point_never_fetches_itself():
    """ADR-0029/ADR-0036 Invariant mirrored: accumulating a fix-source
    target is a pure navigation — the click handler makes no request."""
    fn = _extract_function(_console_text(), "renderC5SourceCard")
    assert "fix-source-btn" in fn
    assert "addC5FixSourceTarget(section.source_path)" in fn
    assert "fetch(" not in fn


def test_source_card_disables_both_controls_when_unresolved():
    fn = _extract_function(_console_text(), "renderC5SourceCard")
    assert 'section.source_resolution === "resolved"' in fn
    assert "viewBtn.disabled = true;" in fn
    assert "editBtn.disabled = true;" in fn


# ---------------------------------------------------------------------------
# fix-source BATCH panel: multiple targets, one upload/re-ingest/deep-audit
# ---------------------------------------------------------------------------


def test_pending_targets_is_a_module_level_array_not_a_single_slot():
    text = _console_text()
    assert re.search(r"^var pendingC5FixSourceTargets = \[\];", text, re.MULTILINE), (
        "unlike C3's pendingFixSource (a single object), C5's batch panel "
        "must track an array of targets (ADR-0036 decision 5)"
    )


def test_add_target_dedupes_and_never_fetches():
    fn = _extract_function(_console_text(), "addC5FixSourceTarget")
    assert "pendingC5FixSourceTargets.some(" in fn, "must dedupe by sourcePath"
    assert "pendingC5FixSourceTargets.push(" in fn
    assert "fetch(" not in fn, "accumulating a target must never make a request"


def test_remove_target_filters_the_array():
    fn = _extract_function(_console_text(), "removeC5FixSourceTarget")
    assert "pendingC5FixSourceTargets.filter(" in fn
    assert "fetch(" not in fn


def test_panel_hidden_when_empty_and_visible_when_populated():
    fn = _extract_function(_console_text(), "renderC5FixSourcePanel")
    assert 'panel.classList.remove("visible");' in fn
    assert 'panel.classList.add("visible");' in fn
    assert "pendingC5FixSourceTargets.length === 0" in fn


def test_panel_css_hidden_by_default():
    text = _console_text()
    css_match = re.search(r"\.fix-source-batch-panel\s*\{([^}]*)\}", text, re.DOTALL)
    assert css_match is not None
    assert "display: none" in css_match.group(1)
    assert re.search(r"\.fix-source-batch-panel\.visible\s*\{([^}]*)\}", text)


def test_batch_run_uploads_sequentially_with_per_target_overwrite_relpath():
    """Each matched file uploads with its OWN target's resolved
    overwrite_relpath (ADR-0036 §6 / issue #533) — sequential, not
    Promise.all, mirroring runIngestBatches's own sequencing rationale."""
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert 'formData.append("overwrite_relpath", entry.target.sourcePath);' in fn
    assert "matched.reduce(function(chain, entry)" in fn
    assert "Promise.all(" not in fn


def test_batch_run_drives_exactly_one_force_reingest_and_one_deep_audit():
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert fn.count('fetch("/wiki/ingest"') == 1
    assert fn.count('fetch("/wiki/lint?include_c5=true"') == 1
    assert "force: true," in fn


def test_batch_run_reports_unmatched_files_without_uploading_them():
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert "unmatchedNames" in fn
    assert "c5FixSourceBatchMismatch" in fn


def test_batch_run_success_clears_targets_and_flags_a_genuine_deep_judgment():
    """Mirrors the Deep audit control's own one-shot signal (issue #489) —
    the fix-source batch's include_c5=true response is also a genuine
    judgment, not a fast relint's untouched empty default."""
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert "pendingDeepC5Audit = true;" in fn
    assert "pendingC5FixSourceTargets = [];" in fn
    assert "renderLintCard(lintData, lintResultEl)" in fn
    assert "renderCurationQueue(" in fn


def test_batch_run_checks_pending_coverage_and_fix_source_outcomes_first():
    """Same ordering convention as the other two include_c5 call sites."""
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    check_pos = fn.find("checkFixSourceOutcome(lintData)")
    render_pos = fn.find("renderLintCard(lintData, lintResultEl)")
    assert check_pos != -1 and render_pos != -1
    assert check_pos < render_pos


# ---------------------------------------------------------------------------
# Regression: wiki-rooted Reconcile still edits -> applies exactly as before
# ---------------------------------------------------------------------------


def test_apply_flow_unaffected_by_the_view_toggle():
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert '"/wiki/pages/reconcile/apply"' in fn
    assert "content_a: textareaA.value" in fn
    assert "content_b: textareaB.value" in fn
    assert "hash_a: data.hash_a" in fn
    assert "hash_b: data.hash_b" in fn


def test_wiki_view_still_contains_the_editable_columns():
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert 'var wikiViewEl = el("div", { class: "reconcile-columns" },' in fn
    assert "textareaA" in fn
    assert "textareaB" in fn


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
