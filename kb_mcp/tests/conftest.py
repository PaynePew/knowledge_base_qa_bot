"""Shared isolation fixtures for the kb_mcp test suite.

Mirrors the pattern in ``markdown_kb/tests/conftest.py`` and
``vector_rag/tests/conftest.py``: an autouse, per-test safety net that keeps
every kb_mcp test off the real on-disk index / log files and snapshot-restores
the in-process module globals the retrieval stacks mutate.

Why this conftest exists (issue #200): the kb_mcp slice added the first tests
that import ``markdown_kb.app.indexer`` and ``kb_mcp.freshness`` into the root
suite.  Because ``kb_search_v1`` calls ``reload_if_stale()`` with no argument,
the default target is ``kb_mcp.freshness.INDEX_PATH`` which resolves (parents[3])
to the *same* repo-root ``.kb/index.json`` as ``markdown_kb.app.indexer``'s
``INDEX_PATH`` — and ``load_index_json`` mutates the indexer's module-level
globals (``sections`` / ``doc_freq`` / ``avg_doc_len`` / ``files_indexed``) plus
appends to the real ``wiki/log.md`` via ``log_event('index_loaded', ...)``.

Without this isolation, a collection order that places kb_mcp tests first leaks
those populated globals into the gateway test
``test_wiki_stream_cannot_confirm_does_not_file`` (and others that rely on a
particular indexer state): retrieval over the polluted globals raises, the
gateway's last-resort ``except Exception`` swallows it, and the SSE stream
emits zero ``done`` events — ``done_events == []``.  The leak surface is
*latent* under the current test bodies (they patch ``reload_if_stale`` and the
``search`` leaves), so this fixture is the safety net that holds the suite green
if a future kb_mcp test exercises the real reload / build path.

``_isolate_module_state`` (autouse, per-test):
  1. Redirects ``markdown_kb.app.indexer.INDEX_PATH`` **and**
     ``kb_mcp.freshness.INDEX_PATH`` to ``tmp_path`` so kb_mcp tests never read
     or write the real ``.kb/index.json`` (both names resolve to the same file,
     and ``reload_if_stale()`` uses the freshness one as its default argument).
  2. Redirects ``markdown_kb.app.indexer.WIKI_DIR`` and
     ``markdown_kb.app.logger.LOG_PATH`` to ``tmp_path`` (defensive parity with
     the markdown_kb / vector_rag conftests): ``load_index_json`` unconditionally
     calls ``log_event('index_loaded', ...)`` which appends to the real
     ``wiki/log.md`` on any real reload.
  3. Snapshots and restores every indexer module-level global that
     ``load_index_json`` / ``build_index`` mutate (``sections``, ``doc_freq``,
     ``avg_doc_len``, ``files_indexed``, ``last_wiki_index_outcome``) around each
     test.  List/Counter globals are restored in place (clear + extend / rebind)
     so the module identity callers captured stays valid.
  4. Resets ``kb_mcp.freshness._last_mtime`` to ``None`` so the mtime cache is
     always in the cold-start state at the start of each test.
  5. Resets the ``vector_rag.app.indexer`` index globals (``vectorstore``,
     ``files_indexed``, ``chunks_indexed``) on teardown — the rag stack path
     (``server.py``) imports ``vector_rag.app.indexer.search``; today's tests
     patch that leaf, but a real rag-path test would mutate these globals.
     Mirrors the ``vector_rag/tests/conftest.py`` teardown.
  6. Redirects ``kb_mcp.hot_cache.HOT_PATH`` to ``tmp_path / "wiki" / "hot.md"``
     so hot-cache tools never read/write the real ``wiki/hot.md`` (issue #202).
     The ``kb_read_hot_v1`` and ``kb_save_hot_v1`` tools look up HOT_PATH at
     call time; per-test monkeypatching is sufficient.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_module_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Autouse safety net for every kb_mcp test — see module docstring (issues #200, #202)."""
    import markdown_kb.app.indexer as current_indexer
    import markdown_kb.app.logger as current_logger
    import vector_rag.app.indexer as rag_indexer

    import kb_mcp.freshness as freshness_mod
    import kb_mcp.hot_cache as hot_cache_mod

    # --- 1 & 2: redirect every real on-disk path to tmp ---
    # Both INDEX_PATH names resolve to the same repo-root .kb/index.json; redirect
    # both because reload_if_stale() defaults to the freshness module's name while
    # load_index_json() reads the indexer module's name.
    tmp_index = tmp_path / ".kb" / "index.json"
    monkeypatch.setattr(current_indexer, "INDEX_PATH", tmp_index)
    monkeypatch.setattr(freshness_mod, "INDEX_PATH", tmp_index)
    # load_index_json -> log_event appends to the real wiki/log.md on any real
    # reload; build_index touches WIKI_DIR. Redirect both for parity with the
    # markdown_kb / vector_rag conftests.
    monkeypatch.setattr(current_indexer, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(current_logger, "LOG_PATH", tmp_path / "wiki" / "log.md")

    # --- 6: redirect hot-cache path so no test touches the real wiki/hot.md ---
    # kb_read_hot_v1 / kb_save_hot_v1 look up HOT_PATH at call time (not at
    # import time), so monkeypatching the module-level name is sufficient.
    monkeypatch.setattr(hot_cache_mod, "HOT_PATH", tmp_path / "wiki" / "hot.md")

    # --- 3: snapshot indexer globals ---
    # sections / doc_freq are mutable containers callers may hold by identity, so
    # restore them in place (clear + extend / rebind) rather than rebinding to a
    # fresh object only on teardown.
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
