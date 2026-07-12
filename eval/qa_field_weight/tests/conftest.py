"""Isolation fixture for the qa_field_weight calibration eval suite.

``calibrate.py`` builds its synthetic corpus by assigning directly to
``markdown_kb.app.indexer.sections`` (no file I/O, no ``build_index()`` call),
so the only isolation needed is restoring that module-level state after each
test — mirrors the snapshot/restore convention used by
``markdown_kb/tests/test_indexer_qa_question_weight.py`` (CODING_STANDARD
§6.5).
"""

from __future__ import annotations

import pytest

import markdown_kb.app.indexer as mk_indexer


@pytest.fixture(autouse=True)
def _isolate_indexer_module_state():
    sections_snapshot = list(mk_indexer.sections)
    doc_freq_snapshot = mk_indexer.doc_freq.copy()
    avg_doc_len_snapshot = mk_indexer.avg_doc_len
    files_indexed_snapshot = mk_indexer.files_indexed
    yield
    mk_indexer.sections = sections_snapshot
    mk_indexer.doc_freq = doc_freq_snapshot
    mk_indexer.avg_doc_len = avg_doc_len_snapshot
    mk_indexer.files_indexed = files_indexed_snapshot
