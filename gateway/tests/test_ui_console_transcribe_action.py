"""Structural tests for the Console's prominent Import failures + guarded
Transcribe action + real-time progress bar (issue #447).

Following the pattern in ``test_ui_console_file_viewer_error_feedback.py`` /
``test_ui_console_c3_routed_fix_source.py``, these tests inspect the
production ``gateway/static/console.html`` file's text — no DOM, no fetch,
no browser, no OPENAI_API_KEY (§6.3 / §12.7).

Covers:
- Import failures render FIRST (not buried after every success row) and show
  BOTH ``error_type`` and ``error_message`` (never one in place of the
  other), each failure row carrying the ``.rec-failed`` prominent treatment.
- A NoTextLayer failure — and only a NoTextLayer failure — offers a
  Transcribe action.
- The Transcribe action fetches the real page count (mechanical preflight,
  no model call) BEFORE opening the confirm dialog, which names that real
  page count and the configured page cap.
- Confirm submits the async batch+poll job (POST /wiki/transcribe/batch),
  never the synchronous POST /wiki/transcribe, because only the job status
  exposes per-page progress.
- The progress bar is driven by the job's real ``pages_done``/``pages_total``
  fields via recursive ``setTimeout`` polling (never ``setInterval``, never a
  client-guessed percentage).
- On success, the row turns into a normal success row in place and the new
  doc feeds ``lastImportedDocFilenames`` so Ingest picks it up.
- Bilingual chrome strings exist for both ``en``/``zh``.
- §12.4: no ``innerHTML`` introduced.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _function_body(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace-depth
    scanning (mirrors the sibling test files' helper of the same name)."""
    m = re.search(rf"function {name}\([^)]*\)\s*\{{", text)
    assert m is not None, f"console.html must define function {name}(...)"
    start = m.end() - 1  # index of the opening brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AssertionError(f"unbalanced braces scanning function {name}")


# ---------------------------------------------------------------------------
# Prominent failure rendering (AC1)
# ---------------------------------------------------------------------------


def test_failed_rows_render_before_success_rows():
    text = _console_text()
    body = _function_body(text, "renderImportCard")
    failed_call = body.index("failed.forEach(makeFailureRow)")
    imported_call = body.index("imported.forEach(makeSourceRow)")
    assert failed_call < imported_call, (
        "Import failures must render FIRST, not after every success row (issue #447 AC1)"
    )


def test_failure_row_carries_prominent_css_class():
    text = _console_text()
    body = _function_body(text, "makeFailureRow")
    assert '"rec rec-failed"' in body


def test_failure_row_shows_both_error_type_and_message_not_either_or():
    text = _console_text()
    body = _function_body(text, "makeFailureRow")
    assert "f.error_message || f.error_type" not in body, (
        "the old one-or-the-other rendering must be gone — both fields show now"
    )
    assert '"error-type"' in body
    assert "f.error_type" in body
    assert "f.error_message" in body


def test_prominent_failure_css_uses_fail_accent_and_background_tint():
    text = _console_text()
    css_match = re.search(r"\.rec-failed\s*\{([^}]*)\}", text, re.DOTALL)
    assert css_match is not None
    css_body = css_match.group(1)
    assert "background" in css_body
    assert "var(--fail)" in css_body


# ---------------------------------------------------------------------------
# Transcribe eligibility — NoTextLayer only (AC2)
# ---------------------------------------------------------------------------


def test_transcribe_action_offered_only_for_no_text_layer():
    text = _console_text()
    body = _function_body(text, "makeFailureRow")
    assert 'f.error_type === "NoTextLayer"' in body
    assert "transcribeAction(" in body


# ---------------------------------------------------------------------------
# Transcribe action: real page-count preflight BEFORE the confirm dialog
# ---------------------------------------------------------------------------


def test_transcribe_action_fetches_page_count_preflight():
    text = _console_text()
    body = _function_body(text, "transcribeAction")
    assert 'fetch("/wiki/transcribe/page-count' in body


def test_transcribe_action_opens_confirm_only_after_preflight_resolves():
    text = _console_text()
    body = _function_body(text, "transcribeAction")
    preflight_idx = body.index('fetch("/wiki/transcribe/page-count')
    confirm_idx = body.index("openTranscribeConfirm(")
    assert preflight_idx < confirm_idx, (
        "the confirm dialog must open only after the real page count is fetched, "
        "never before (no client-guessed page count)"
    )


def test_transcribe_action_never_calls_the_sync_transcribe_endpoint_directly():
    text = _console_text()
    body = _function_body(text, "transcribeAction")
    assert 'fetch("/wiki/transcribe"' not in body
    assert 'adminFetch("/wiki/transcribe"' not in body


# ---------------------------------------------------------------------------
# Confirm dialog: names the real page count + configured page cap
# ---------------------------------------------------------------------------


def test_confirm_dialog_names_real_page_count_and_max_pages():
    text = _console_text()
    body = _function_body(text, "openTranscribeConfirm")
    assert '.replace("{pageCount}", String(pageCount))' in body
    assert '.replace("{maxPages}", String(maxPages))' in body


def test_confirm_dialog_submits_the_async_batch_job_not_the_sync_endpoint():
    """Only the batch+poll job (issue #459) exposes per-page progress — the
    synchronous POST /wiki/transcribe has no progress signal at all."""
    text = _console_text()
    body = _function_body(text, "openTranscribeConfirm")
    assert 'adminFetch("/wiki/transcribe/batch"' in body
    assert "JSON.stringify({ sources: [source] })" in body


def test_confirm_dialog_reuses_existing_modal_chrome():
    text = _console_text()
    body = _function_body(text, "openTranscribeConfirm")
    assert '"reconcile-overlay"' in body
    assert '"reconcile-modal"' in body
    assert '"reconcile-actions"' in body


# ---------------------------------------------------------------------------
# Real-time progress bar (owner-directive AC addition)
# ---------------------------------------------------------------------------


def test_progress_bar_driven_by_real_server_reported_counts():
    """The progress callback must read the job's own pages_done/pages_total
    (server truth), never derive or guess a number client-side."""
    text = _console_text()
    body = _function_body(text, "pollTranscribeJob")
    assert "job.pages_done" in body
    assert "job.pages_total" in body
    assert "onProgress(job.pages_done, job.pages_total)" in body


def test_poll_uses_recursive_settimeout_not_setinterval():
    # The tick loop lives in the generic pollJobStatus since issue #497;
    # pollTranscribeJob delegates to it.
    text = _console_text()
    body = _function_body(text, "pollJobStatus")
    assert "setTimeout(tick" in body
    assert "setInterval(" not in body


def test_confirm_dialog_progress_track_starts_hidden_until_submit():
    text = _console_text()
    body = _function_body(text, "openTranscribeConfirm")
    assert 'progressTrackEl.style.display = "none"' in body
    assert 'progressTrackEl.style.display = "block"' in body


def test_progress_fill_width_derives_from_real_pages_done_ratio():
    text = _console_text()
    body = _function_body(text, "openTranscribeConfirm")
    set_progress = body[
        body.index("function setProgress") : body.index("confirmBtn.addEventListener")
    ]
    assert "pagesDone / denominator" in set_progress
    assert "progressFillEl.style.width" in set_progress


# ---------------------------------------------------------------------------
# Success path: row mutates in place + feeds the Ingest step
# ---------------------------------------------------------------------------


def test_transcribe_success_updates_last_imported_doc_filenames():
    text = _console_text()
    body = _function_body(text, "makeFailureRow")
    assert "lastImportedDocFilenames.push(docsName)" in body


def test_transcribe_success_marks_origin_via_success_copy():
    text = _console_text()
    body = _function_body(text, "makeFailureRow")
    assert "LINT_CHROME[consoleLang].transcribeSuccess" in body


# ---------------------------------------------------------------------------
# Bilingual chrome strings (AC5)
# ---------------------------------------------------------------------------

_TRANSCRIBE_CHROME_KEYS = [
    "transcribe:",
    "transcribeChecking:",
    "transcribeConfirmTitle:",
    "transcribeConfirmBody:",
    "transcribeConfirm:",
    "transcribeCancel:",
    "transcribeWorking:",
    "transcribeProgress:",
    "transcribeSuccess:",
    "transcribeClose:",
]


def test_transcribe_chrome_keys_present_in_both_languages():
    text = _console_text()
    en_block = re.search(r"var LINT_CHROME = \{\s*en: \{(.*?)\n  \},\n  zh: \{", text, re.DOTALL)
    zh_block = re.search(r"zh: \{(.*?)\n  \},\n\};", text, re.DOTALL)
    assert en_block is not None and zh_block is not None

    for key in _TRANSCRIBE_CHROME_KEYS:
        assert key in en_block.group(1), f"missing en LINT_CHROME key {key}"
        assert key in zh_block.group(1), f"missing zh LINT_CHROME key {key}"


def test_transcribe_zh_confirm_body_is_real_cjk_not_english_reuse():
    text = _console_text()
    zh_block = re.search(r"zh: \{(.*?)\n  \},\n\};", text, re.DOTALL)
    assert zh_block is not None
    zh_match = re.search(r'transcribeConfirmBody:\s*"([^"]+)"', zh_block.group(1))
    assert zh_match is not None
    assert any("一" <= ch <= "鿿" for ch in zh_match.group(1)), (
        "zh transcribeConfirmBody must contain actual CJK text"
    )


# ---------------------------------------------------------------------------
# §12.4: still textContent/safe-DOM-construction only
# ---------------------------------------------------------------------------


def test_console_no_inner_html_assignment_still_holds():
    text = _console_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text
