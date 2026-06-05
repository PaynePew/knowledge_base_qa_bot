"""Shared isolation fixtures for the kb_cli test suite.

Mirrors the pattern in ``kb_mcp/tests/conftest.py``: an autouse, per-test
safety net that keeps every kb_cli test off the real on-disk index / log files
and snapshot-restores the in-process module globals the retrieval stacks
mutate.

Why this conftest exists: kb_cli imports ``markdown_kb`` deep modules (indexer,
retrieval) which mutate module-level globals (``sections``, ``doc_freq``, etc.)
and append to the real ``wiki/log.md``.  Without isolation, kb_cli tests leak
state into the root suite, causing collection-order-sensitive failures in other
test packages (see kb_mcp conftest docstring for the exact failure mode).

``_isolate_module_state`` (autouse, per-test):
  1. Redirects ``markdown_kb.app.indexer.INDEX_PATH`` and
     ``kb_mcp.freshness.INDEX_PATH`` to ``tmp_path`` so no test reads/writes
     the real ``.kb/index.json``.
  2. Redirects ``markdown_kb.app.indexer.WIKI_DIR`` and
     ``markdown_kb.app.logger.LOG_PATH`` to ``tmp_path``.
  3. Snapshots and restores every indexer module-level global that
     ``load_index_json`` / ``build_index`` mutate (``sections``, ``doc_freq``,
     ``avg_doc_len``, ``files_indexed``, ``last_wiki_index_outcome``).
  4. Resets ``kb_mcp.freshness._last_mtime`` to ``None`` so the mtime cache
     is always in the cold-start state at the start of each test.
  5. Resets the ``vector_rag.app.indexer`` index globals (``vectorstore``,
     ``files_indexed``, ``chunks_indexed``) on teardown.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_module_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Autouse safety net for every kb_cli test — see module docstring."""
    import kb_mcp.freshness as freshness_mod
    import markdown_kb.app.indexer as current_indexer
    import markdown_kb.app.logger as current_logger
    import vector_rag.app.indexer as rag_indexer

    # --- 1 & 2: redirect every real on-disk path to tmp ---
    tmp_index = tmp_path / ".kb" / "index.json"
    monkeypatch.setattr(current_indexer, "INDEX_PATH", tmp_index)
    monkeypatch.setattr(freshness_mod, "INDEX_PATH", tmp_index)
    monkeypatch.setattr(current_indexer, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(current_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")

    # --- 3: snapshot indexer globals ---
    saved_sections = list(current_indexer.sections)
    saved_doc_freq = Counter(current_indexer.doc_freq)
    saved_avg_doc_len = current_indexer.avg_doc_len
    saved_files_indexed = current_indexer.files_indexed
    saved_last_wiki_index_outcome = current_indexer.last_wiki_index_outcome

    # --- 4: reset freshness mtime cache to the cold-start state ---
    monkeypatch.setattr(freshness_mod, "_last_mtime", None)

    yield

    # --- 3 (teardown): restore indexer globals ---
    current_indexer.sections.clear()
    current_indexer.sections.extend(saved_sections)
    current_indexer.doc_freq = saved_doc_freq
    current_indexer.avg_doc_len = saved_avg_doc_len
    current_indexer.files_indexed = saved_files_indexed
    current_indexer.last_wiki_index_outcome = saved_last_wiki_index_outcome

    # --- 5 (teardown): reset vector_rag index globals (mirrors vr conftest) ---
    rag_indexer.vectorstore = None
    rag_indexer.files_indexed = 0
    rag_indexer.chunks_indexed = 0
