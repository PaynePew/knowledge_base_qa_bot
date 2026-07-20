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
  ``/read/file`` view control and an edit entry point that opens an IN-MODAL
  Source editor (issue #632, ADR-0043 — supersedes ADR-0036 decision 4's
  "no in-app editor"): the whole file is fetched read-only via ``/read/file``
  and "Stage correction" stores the corrected text on the pending target —
  pure client state, never a write request.
- A citation that could not be resolved to content disables both controls
  (no false affordance).
- The fix-source batch panel accumulates MULTIPLE targets (an array,
  deduplicated), unlike C3's single-pending-slot banner, and EVERY byte-source
  (file-picker batch, in-modal staged corrections) converges on ONE shared
  pipeline that drives exactly one force re-ingest and one deep-audit
  (ADR-0036 decision 5, unchanged by ADR-0043), never a per-target stepper.
- The Wiki comparison view shows the sources-disagree note when the pair is
  not converged, so the disabled Apply is explained at point of use
  (ADR-0043 decision 5).
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


def test_source_card_edit_entry_point_opens_the_inline_editor():
    """ADR-0043 decision 1 (supersedes ADR-0036 decision 4): Fix this Source
    opens the in-modal editor instead of silently accumulating a target the
    curator cannot see behind the modal overlay."""
    fn = _extract_function(_console_text(), "renderC5SourceCard")
    assert "fix-source-btn" in fn
    assert "openC5SourceEditor(section" in fn
    assert "fetch(" not in fn, "the card itself still never fetches — the editor does"


def test_editor_fetches_the_whole_file_via_read_file_verbatim():
    """ADR-0043 decisions 1/6: the editor loads the WHOLE file (no client-side
    section splicing, §12.5) through the same GET /read/file convention as
    openFile, with section.source_path used verbatim — never a rebuilt path."""
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert '"/read/file?"' in fn
    assert "section.source_path" in fn
    assert '"docs/" + ' not in fn


def test_editor_stage_stores_content_and_never_writes():
    """ADR-0043 decision 2: Stage correction is pure client state — it routes
    through addC5FixSourceTarget(sourcePath, content); the editor makes no
    admin/mutating request itself (the only fetch is the read-only load)."""
    fn = _extract_function(_console_text(), "openC5SourceEditor")
    assert "addC5FixSourceTarget(section.source_path, textarea.value)" in fn
    assert "adminFetch(" not in fn
    assert '"/upload"' not in fn


def test_editor_chrome_defined_bilingually():
    text = _console_text()
    for key in (
        "c5EditSourceLoading",
        "c5EditSourceStage",
        "c5EditSourceStaged",
        "c5EditSourceCancel",
        "c5StagedBadge",
        "c5StagedBarPrefix",
        "c5UploadStaged",
    ):
        assert len(re.findall(rf'{key}:\s*"[^"]+"', text)) == 2, (
            f"chrome key {key} must be defined in BOTH language blocks"
        )


def test_source_card_renders_uncited_siblings_collapsed_with_toggle():
    """issue #635 (ADR-0044): sibling sections of the cited files (cited ===
    false) render collapsed to a toggle-able heading row tagged as not cited
    — the whole-file evidence the grounding report may quote is on screen,
    one click away, instead of invisible."""
    fn = _extract_function(_console_text(), "renderC5SourceCard")
    assert "section.cited !== false" in fn
    assert "uncited-section-toggle" in fn
    assert 'classList.toggle("collapsed")' in fn
    assert "c5UncitedTag" in fn


def test_uncited_sibling_chrome_defined_bilingually():
    text = _console_text()
    for key in ("c5UncitedTag", "c5GroundingUnionNote"):
        assert len(re.findall(rf'{key}:\s*"[^"]+"', text)) == 2, (
            f"chrome key {key} must be defined in BOTH language blocks"
        )


def test_collapsed_sibling_css_hides_content_and_actions():
    text = _console_text()
    assert ".reconcile-source-section.uncited.collapsed" in text


def test_grounding_report_names_its_whole_file_scope():
    """issue #635 (ADR-0044): when unsupported claims are listed, the modal
    states the check ran against the WHOLE Source files — a flagged claim may
    come from a section the page does not cite."""
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert "c5GroundingUnionNote" in fn


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
    assert "function addC5FixSourceTarget(sourcePath, content)" in fn, (
        "ADR-0043 decision 2: staging routes the corrected text through the same accumulator"
    )
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


def test_pipeline_uploads_sequentially_with_per_target_overwrite_relpath():
    """Each entry uploads with its OWN target's resolved overwrite_relpath
    (ADR-0036 §6 / issue #533) — sequential, not Promise.all, mirroring
    runIngestBatches's own sequencing rationale. The filename is the target's
    own (explicit third argument), so a staged Blob and a picked File upload
    identically (ADR-0043 decision 3)."""
    fn = _extract_function(_console_text(), "runC5FixSourcePipeline")
    assert 'formData.append("files", entry.file, entry.target.filename);' in fn
    assert 'formData.append("overwrite_relpath", entry.target.sourcePath);' in fn
    assert "entries.reduce(function(chain, entry)" in fn
    assert "Promise.all(" not in fn


def test_pipeline_drives_exactly_one_force_reingest_and_one_deep_audit():
    fn = _extract_function(_console_text(), "runC5FixSourcePipeline")
    assert fn.count('adminFetch("/wiki/ingest"') == 1
    assert fn.count('adminFetch("/wiki/lint?include_c5=true"') == 1
    assert "force: true," in fn


def test_every_byte_source_converges_on_the_one_pipeline():
    """ADR-0043 decision 3: the file-picker batch and the staged-correction
    runner both delegate to runC5FixSourcePipeline — neither owns a second
    upload/re-ingest/audit sequence."""
    text = _console_text()
    batch = _extract_function(text, "runC5FixSourceBatch")
    staged = _extract_function(text, "runC5FixSourceStaged")
    assert "runC5FixSourcePipeline(" in batch
    assert "runC5FixSourcePipeline(" in staged
    for fn in (batch, staged):
        assert 'adminFetch("/wiki/ingest"' not in fn
        assert 'adminFetch("/wiki/lint' not in fn
        assert 'adminFetch("/upload"' not in fn


def test_staged_runner_builds_blobs_from_staged_content_only():
    fn = _extract_function(_console_text(), "runC5FixSourceStaged")
    assert "t.content != null" in fn
    assert "new Blob([t.content]" in fn


def test_batch_run_reports_unmatched_files_without_uploading_them():
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert "unmatchedNames" in fn
    assert "c5FixSourceBatchMismatch" in fn


def test_pipeline_success_clears_targets_and_flags_a_genuine_deep_judgment():
    """Mirrors the Deep audit control's own one-shot signal (issue #489) —
    the fix-source pipeline's include_c5=true response is also a genuine
    judgment, not a fast relint's untouched empty default."""
    fn = _extract_function(_console_text(), "runC5FixSourcePipeline")
    assert "pendingDeepC5Audit = true;" in fn
    assert "pendingC5FixSourceTargets = [];" in fn
    assert "renderLintCard(lintData, lintResultEl)" in fn
    assert "renderCurationQueue(" in fn


def test_pipeline_checks_pending_coverage_and_fix_source_outcomes_first():
    """Same ordering convention as the other two include_c5 call sites."""
    fn = _extract_function(_console_text(), "runC5FixSourcePipeline")
    check_pos = fn.find("checkFixSourceOutcome(lintData)")
    render_pos = fn.find("renderLintCard(lintData, lintResultEl)")
    assert check_pos != -1 and render_pos != -1
    assert check_pos < render_pos


def test_staged_bar_rendered_into_source_view_and_wired_to_close_the_modal():
    """ADR-0043 decision 3: the in-modal staged-upload bar lives in the Source
    comparison view (where staging happens) and its success path closes the
    modal — the curator lands back on the re-rendered Lint card."""
    text = _console_text()
    preview = _extract_function(text, "renderReconcilePreview")
    assert "renderC5StagedBar(" in preview
    bar = _extract_function(text, "renderC5StagedBar")
    assert "runC5FixSourceStaged(" in bar
    assert "c5UploadStaged" in bar


def test_panel_marks_staged_targets_and_offers_direct_staged_upload():
    """ADR-0043 decision 4: the batch panel shows which targets carry an
    in-app correction and can upload the staged set without a file picker;
    the file-picker path survives as the local-editing escape hatch."""
    fn = _extract_function(_console_text(), "renderC5FixSourcePanel")
    assert "c5StagedBadge" in fn
    assert "runC5FixSourceStaged(" in fn
    assert "c5FixSourceBatchChooseFiles" in fn


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
    assert 'el("div", { class: "reconcile-columns" },' in fn
    assert "textareaA" in fn
    assert "textareaB" in fn


def test_wiki_view_explains_the_disabled_apply_when_not_converged():
    """ADR-0043 decision 5: the sources-disagree note renders in the Wiki
    comparison view too (not only the Source view), so a curator editing the
    draft sees WHY Apply is disabled instead of a dead button."""
    fn = _extract_function(_console_text(), "renderReconcilePreview")
    assert 'converged ? null : el("div", { class: "sources-disagree-note"' in fn


# ---------------------------------------------------------------------------
# No new innerHTML (§12.4)
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
