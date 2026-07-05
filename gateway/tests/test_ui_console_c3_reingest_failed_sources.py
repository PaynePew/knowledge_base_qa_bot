"""Structural tests for per-row Re-ingest failure surfacing (issue #475).

``POST /wiki/ingest`` answers HTTP 200 even when the Source failed — failures
ride in the batch-shaped body's ``failed_sources`` (routes.py). Before #475
the per-row ``ingestRemediationRequest`` only checked ``resp.ok``, so a C3
"Re-ingest (retry)" that dead-ended in ``failed_sources`` rendered as success
while the finding silently stayed. These tests pin the structural fix: the
single-source wrapper must inspect ``failed_sources`` and reject, which
routes the failure through ``runLintRemediation``'s existing err-status
per-row rendering.

Following the ``test_ui_console_lint_remediation.py`` pattern: text-level
assertions against the production console.html — no DOM, no fetch, no
browser. The real click -> fail -> per-row-error loop is proven in a driven
browser against a live gateway (issue #475 AC).
"""

from __future__ import annotations

import re
from pathlib import Path

_CONSOLE_HTML = Path(__file__).resolve().parents[2] / "gateway" / "static" / "console.html"


def _console_text() -> str:
    return _CONSOLE_HTML.read_text(encoding="utf-8")


def _ingest_remediation_request_body() -> str:
    """Return the source text of the ingestRemediationRequest function."""
    text = _console_text()
    match = re.search(
        r"function ingestRemediationRequest\(source, force\) \{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert match is not None, "console.html must define ingestRemediationRequest"
    return match.group(1)


def test_single_source_reingest_checks_failed_sources():
    """The per-row wrapper inspects the 200 body's failed_sources (issue #475)."""
    body = _ingest_remediation_request_body()
    assert "failed_sources" in body, (
        "ingestRemediationRequest must inspect failed_sources — a 200 with a "
        "failed_sources entry is a FAILED re-ingest, not a success (issue #475)"
    )


def test_single_source_reingest_rejects_on_failed_sources():
    """A failed_sources entry rejects (throw) so runLintRemediation's .catch
    renders the per-row err status instead of a silent success."""
    body = _ingest_remediation_request_body()
    match = re.search(r"if \(data && data\.failed_sources.*?\{(.*?)\}", body, re.DOTALL)
    assert match is not None, (
        "ingestRemediationRequest must gate on data.failed_sources being non-empty"
    )
    assert "throw new Error" in match.group(1), (
        "the failed_sources branch must throw — rejection is what routes the "
        "failure into runLintRemediation's err-status per-row rendering"
    )


def test_batch_reingest_summary_path_unchanged():
    """The batch path keeps its own partial-failure summary (ingestPartialSummary)
    — #475 only changes the per-row single-source wrapper."""
    text = _console_text()
    assert "function ingestPartialSummary(sources, data)" in text
    match = re.search(
        r"function batchIngestRemediationRequest\(sources, force\) \{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert match is not None, "console.html must define batchIngestRemediationRequest"
    assert "failed_sources" not in match.group(1), (
        "batch wrapper must NOT throw on partial failure — its N-source result "
        "renders via ingestPartialSummary ('N ok, M still failing'), issue #364"
    )
