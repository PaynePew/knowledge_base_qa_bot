"""Production-isolation fixture for the contaminated-session eval suite.

Mirrors ``eval.negative_case.tests.conftest``: every test that builds an
index gets markdown_kb's ``INDEX_PATH`` / ``WIKI_DIR`` / ``LOG_PATH`` pointed
at ``tmp_path``, and ``SOURCE_DIRS`` is snapshotted + restored, so no test can
pollute production ``.kb/`` / ``wiki/`` (CODING_STANDARD §6.5). No test in
this suite calls the real ``rewrite_query`` LLM, so no API key or LLM stub is
needed here.
"""

from __future__ import annotations

import pytest

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger


@pytest.fixture(autouse=True)
def _isolate_markdown_kb_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(mk_indexer, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(mk_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")
    source_dirs_snapshot = mk_indexer.SOURCE_DIRS
    yield
    mk_indexer.SOURCE_DIRS = source_dirs_snapshot
    mk_indexer.sections.clear()
