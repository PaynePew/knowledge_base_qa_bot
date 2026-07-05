"""Structural tests for the Operator Console's async Import step (issue #497).

Following the sibling ``test_ui_console_*`` pattern, these inspect
``gateway/static/console.html``'s text to assert:

- The Import step submits POST /wiki/import/jobs and polls (via the shared
  pollJobStatus) instead of one synchronous POST /wiki/import — the sync
  request auto-transcribes scans inline (ADR-0032), long enough to blow the
  edge proxy's window and surface as a bare "HTTP 502:".
- Progress rendering reads the job's own server-owned counts
  (pages_done/pages_total, files_done/files_total) — §12.8 bans
  client-GUESSED percentages, not real server-reported counts.
- httpError composes an honest message when an error response has an EMPTY
  body (the edge-502 case), instead of a bare "HTTP 502:".
- No new innerHTML (§12.4).

No DOM, no fetch, no browser — fully hermetic (§12.7); the live click-path
is verified against prod per the batch's real-artifact verification pass.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _function_body(text: str, name: str) -> str:
    """Extract a top-level ``function <name>(...) { ... }`` body by brace-depth
    scanning — robust to nested braces inside the body (mirrors the sibling
    test files' helper)."""
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
                return text[start + 1 : i]
    raise AssertionError(f"unbalanced braces scanning function {name}")


def _import_step_run_body() -> str:
    text = _console_text()
    m = re.search(r'id: "import",(.*?)\n  \},', text, re.DOTALL)
    assert m is not None, "STEP_DEFS must define the import step"
    return m.group(1)


# ---------------------------------------------------------------------------
# Submit + poll instead of one synchronous request
# ---------------------------------------------------------------------------


def test_import_step_submits_the_async_job_not_the_sync_endpoint():
    body = _import_step_run_body()
    assert 'fetch("/wiki/import/jobs", { method: "POST" })' in body, (
        "the Import step must submit the async job (issue #497)"
    )
    assert 'fetch("/wiki/import", { method: "POST" })' not in body, (
        "the synchronous /wiki/import call is exactly the request the edge "
        "502s on a real scan — the step must not use it (issue #497)"
    )


def test_import_step_polls_through_the_shared_poller():
    body = _import_step_run_body()
    assert "pollJobStatus(" in body
    assert '"/wiki/import/jobs/" + encodeURIComponent(' in body


def test_import_step_renders_failed_job_and_poll_errors_honestly():
    body = _import_step_run_body()
    assert 'job.status === "failed"' in body
    assert 'renderErrorCard("Import"' in body


def test_import_step_still_feeds_ingest_with_imported_filenames():
    """The lastImportedDocFilenames capture (§12.8 single-batch rule) must
    survive the async rewrite — it now reads the job's terminal result."""
    body = _import_step_run_body()
    assert "lastImportedDocFilenames" in body
    assert "imported_sources" in body and "skipped_sources" in body


# ---------------------------------------------------------------------------
# Progress: real server-owned counts only
# ---------------------------------------------------------------------------


def test_progress_card_reads_server_owned_counts():
    text = _console_text()
    body = _function_body(text, "makeImportProgressCard")
    assert "job.pages_done" in body and "job.pages_total" in body
    assert "job.files_done" in body and "job.files_total" in body
    assert "transcribe-progress-track" in body, (
        "the determinate bar must reuse the established progress chrome"
    )


def test_progress_bar_hidden_until_real_counts_arrive():
    """No fake motion: the track only appears once pages_total > 0 (a run
    with no scans keeps the plain busy spinner)."""
    text = _console_text()
    body = _function_body(text, "makeImportProgressCard")
    assert 'trackEl.style.display = "none"' in body
    assert "job.pages_total > 0" in body


# ---------------------------------------------------------------------------
# Honest empty-body errors (the edge-502 case)
# ---------------------------------------------------------------------------


def test_http_error_helper_handles_empty_bodies():
    text = _console_text()
    body = _function_body(text, "httpError")
    assert "bodyText.trim()" in body, "a non-empty body must still be shown"
    assert "502" in body and "504" in body, (
        "the gateway-timeout family must get an explanatory fallback, not a "
        "bare 'HTTP 502:' (issue #497)"
    )


def test_poller_and_import_submit_use_http_error():
    text = _console_text()
    assert "throw httpError(resp, t)" in _function_body(text, "pollJobStatus")
    assert "throw httpError(resp, t)" in _import_step_run_body()


# ---------------------------------------------------------------------------
# §12.4 discipline holds
# ---------------------------------------------------------------------------


def test_no_inner_html_in_new_import_code():
    text = _console_text()
    assert ".innerHTML" not in _function_body(text, "makeImportProgressCard")
    assert ".innerHTML" not in _function_body(text, "pollJobStatus")
    assert ".innerHTML" not in _import_step_run_body()
