"""Fast hermetic unit tests for ops/loadtest/transcribe_load.py's pure PDF
assembly helper, plus summarize.py's Transcribe column (issue #627). No
server, no network. ``_build_multipage_pdf`` reads the already-committed
single-page scanned fixture and asserts the in-memory multi-page PDF it
builds has the right page count. The summarize.py tests live here (a new
file for this slice) rather than appended to test_summarize.py, so this
slice's tests never touch a file another slice (#600) already delivered.
The network-driving ``run_transcribe_load`` itself is exercised manually via
``harness.py run S5_...``, same as ``import_load.py``/``chat_load.py`` (no
server means no unit test for those either — see this package's tests/
directory for the established pattern).
"""

from __future__ import annotations

import pypdfium2 as pdfium

from ops.loadtest.summarize import to_markdown_table
from ops.loadtest.transcribe_load import _build_multipage_pdf


def test_build_multipage_pdf_has_requested_page_count():
    data = _build_multipage_pdf(16)
    pdf = pdfium.PdfDocument(data)
    try:
        assert len(pdf) == 16
    finally:
        pdf.close()


def test_build_multipage_pdf_single_page():
    data = _build_multipage_pdf(1)
    pdf = pdfium.PdfDocument(data)
    try:
        assert len(pdf) == 1
    finally:
        pdf.close()


def _summarize_result(**overrides):
    base = {
        "scenario_id": "S5_transcribe_c16",
        "description": "test transcribe scenario",
        "wall_clock_sec": 5.0,
        "memory": {
            "peak_rss_polled_mb": 180.0,
            "peak_wset_os_mb": 185.0,
            "sample_count": 20,
        },
        "chat_load": None,
        "import_load": None,
        "transcribe_load": None,
    }
    base.update(overrides)
    return base


def test_to_markdown_table_renders_transcribe_status_when_present():
    r = _summarize_result(
        transcribe_load={
            "status": "completed",
            "files_total": 1,
            "files_done": 1,
            "pages_total": 16,
            "pages_done": 16,
            "wall_clock_sec": 5.0,
        }
    )
    table = to_markdown_table([r])
    assert "completed (pages 16/16)" in table


def test_to_markdown_table_transcribe_status_dash_when_absent():
    table = to_markdown_table([_summarize_result()])
    lines = table.strip().splitlines()
    assert lines[-1].rstrip().endswith("| - |")
