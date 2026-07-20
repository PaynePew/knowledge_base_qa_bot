"""Structural test for issue #544 — fix-source batch: duplicate-basename
pending targets across subdirs were unmatched.

Following the pattern in ``test_ui_console_c5_two_view_modal.py`` /
``test_ui_console_c3_routed_fix_source.py``, this test inspects the
production ``gateway/static/console.html`` file's text — no DOM, no fetch,
no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Bug (surfaced by the independent Verdict on PR #543, issue #538): the file-
to-target lookup in ``runC5FixSourceBatch`` matched a selected file against
``pendingC5FixSourceTargets`` by bare ``filename`` alone
(``t.filename === file.name``). When two pending targets shared a basename
across different ``docs/`` subdirs (e.g. ``docs/demo-zh/foo.md`` and
``docs/planted-zh/foo.md``), ``.filter(...)[0]`` always returned the SAME
first target for every same-named file in the batch — a second file meant
for the second target silently rebinds onto the first target's
``overwrite_relpath`` instead, and the second target's Source is never
touched.

Fix: track which pending target each match already claimed
(``claimedSourcePaths``, keyed by the target's own ``sourcePath`` — unique
per ``addC5FixSourceTarget``'s dedupe key) and exclude an already-claimed
target from the filter, so two same-basename files in one batch resolve to
two DIFFERENT targets, each keeping its own resolved origin relpath
(mirrors the destination-aware ``overwrite_relpath`` resolution from issue
#533, ADR-0036 §6).

Note (documented, not fixed here — out of scope for this client-only
slice): even with this fix, a REAL duplicate-basename batch still fails
server-side today, because ``markdown_kb/app/upload.py``'s
``_resolve_overwrite_target`` refuses ANY overwrite when the basename is
ambiguous anywhere under ``docs_dir``, independent of the caller-supplied
``overwrite_relpath`` (see ``test_overwrite_relpath_ambiguous_origin_refused``
in ``markdown_kb/tests/test_upload.py`` — an existing, accepted ADR-0036 §6
Guard). This test file only proves the CLIENT now resolves each file to its
own, distinct, correct target; the server-side ambiguity guard is a
separate, deeper follow-up.
"""

from __future__ import annotations

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
# Claim-tracking: a target matched by one file is excluded for the next
# ---------------------------------------------------------------------------


def test_matching_tracks_claimed_targets_per_batch_run():
    """A fresh claimed-target tracker is created for each batch run (not a
    module-level/persistent one — must not leak across separate batches)."""
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert "var claimedSourcePaths = {};" in fn


def test_filter_excludes_an_already_claimed_target():
    """The per-file target lookup must check BOTH filename equality AND
    that the target hasn't already been claimed by an earlier file in this
    same batch — filename equality alone reproduces issue #544."""
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert "t.filename === file.name && !claimedSourcePaths[t.sourcePath]" in fn


def test_a_match_marks_its_target_claimed_before_the_next_file_is_processed():
    """The claim is recorded (keyed by the target's own sourcePath — unique
    per addC5FixSourceTarget's dedupe key) inside the SAME per-file
    forEach callback that pushes the match, so the next file in the batch
    sees it excluded."""
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    foreach_start = fn.index("files.forEach(function(file) {")
    foreach_body = fn[foreach_start:]
    claim_pos = foreach_body.find("claimedSourcePaths[target.sourcePath] = true;")
    push_pos = foreach_body.find("matched.push(")
    assert claim_pos != -1, f"expected the claim write inside files.forEach: {foreach_body}"
    assert push_pos != -1
    assert claim_pos < push_pos, "the target must be claimed before it is pushed to matched"


# ---------------------------------------------------------------------------
# Regression: existing single-basename / no-match behavior is unchanged
# ---------------------------------------------------------------------------


def test_batch_run_still_reports_unmatched_files_without_uploading_them():
    """AC: 'existing single-basename batch behavior unchanged' — a file with
    no (remaining, unclaimed) matching target is still reported via
    unmatchedNames / c5FixSourceBatchMismatch, never silently uploaded."""
    fn = _extract_function(_console_text(), "runC5FixSourceBatch")
    assert "unmatchedNames" in fn
    assert "c5FixSourceBatchMismatch" in fn
    assert "unmatchedNames.push(file.name);" in fn


def test_batch_run_still_uploads_with_per_target_overwrite_relpath():
    """Regression: each matched entry still uploads with its OWN target's
    resolved overwrite_relpath (issue #533, ADR-0036 §6) — the claim-
    tracking fix must not disturb the upload wiring. Since issue #632
    (ADR-0043) the upload sequence lives in the shared
    ``runC5FixSourcePipeline``, which the batch run's matched entries feed."""
    text = _console_text()
    batch = _extract_function(text, "runC5FixSourceBatch")
    assert "runC5FixSourcePipeline(matched" in batch
    pipeline = _extract_function(text, "runC5FixSourcePipeline")
    assert 'formData.append("overwrite_relpath", entry.target.sourcePath);' in pipeline
    assert "entries.reduce(function(chain, entry)" in pipeline


def test_batch_run_still_drives_exactly_one_force_reingest_and_one_deep_audit():
    """Regression: still ONE force re-ingest + ONE deep-audit per batch
    (ADR-0036 decision 5, shared pipeline since ADR-0043) — unaffected by
    the matching-loop fix."""
    fn = _extract_function(_console_text(), "runC5FixSourcePipeline")
    assert fn.count('adminFetch("/wiki/ingest"') == 1
    assert fn.count('adminFetch("/wiki/lint?include_c5=true"') == 1
    assert "force: true," in fn


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found in console.html — §12.4 requires textContent only"
    )
