"""Public warmup seam ``ensure_indexes_loaded`` — registration + idempotency (#326).

Pins the CODING_STANDARD §2.4 escalation: ``_ensure_indexes_loaded`` (private)
promoted to ``ensure_indexes_loaded`` (public) so ``kb_mcp`` and any future
caller can cold-start the Hybrid stack without a cross-package private import.

Three assertions:
  1. ``ensure_indexes_loaded`` is listed in ``query.__all__`` (registered public API).
  2. ``ensure_indexes_loaded`` is callable (the symbol exists and is a function).
  3. Calling it twice with both arms already warm is a no-op — no exception, no
     reload attempt on either arm (idempotency guarantee for MCP / Gateway callers
     that may invoke it defensively on every request).

The conftest autouse ``_redirect_paths_to_tmp`` fixture handles path redirection
and global teardown; this file adds no extra fixtures.
"""

from __future__ import annotations

import hybrid_kb.app.dense_index as dense_index
import hybrid_kb.app.query as query_module
import markdown_kb.app.indexer as bm25_indexer


# ---------------------------------------------------------------------------
# Registration assertions (these are pure symbol checks — no I/O)
# ---------------------------------------------------------------------------


def test_ensure_indexes_loaded_in_all():
    """``ensure_indexes_loaded`` must appear in ``query.__all__`` (public API)."""
    assert "ensure_indexes_loaded" in query_module.__all__


def test_ensure_indexes_loaded_is_callable():
    """``ensure_indexes_loaded`` must be callable (exists, is a function)."""
    assert callable(query_module.ensure_indexes_loaded)


# ---------------------------------------------------------------------------
# Idempotency: warm arms → no reload on either call
# ---------------------------------------------------------------------------


def test_ensure_indexes_loaded_idempotent_when_warm(monkeypatch):
    """Both calls are no-ops when both arms are already warm.

    Pre-warms both arms with lightweight sentinel values:
      * BM25 sections replaced with a non-empty list (triggers the ``if not
        bm25_indexer.sections`` guard to short-circuit).
      * ``dense_index.vectorstore`` set to a non-None sentinel (triggers the
        ``if dense_index.vectorstore is None`` guard to short-circuit).

    Neither ``load_index_json`` nor ``load_dense_index`` must be called on
    either the first or the second invocation.
    """
    load_json_calls: list[int] = []
    load_dense_calls: list[int] = []

    # Replace ``sections`` with a non-empty list so the guard sees a warm BM25 arm.
    monkeypatch.setattr(bm25_indexer, "sections", ["_warm_sentinel"])
    # Set vectorstore to a non-None sentinel so the guard sees a warm dense arm.
    monkeypatch.setattr(dense_index, "vectorstore", object())

    # Intercept any load attempt — both should be completely bypassed.
    monkeypatch.setattr(
        bm25_indexer, "load_index_json", lambda: load_json_calls.append(1)
    )
    monkeypatch.setattr(
        dense_index, "load_dense_index", lambda: load_dense_calls.append(1)
    )

    # First call — both arms warm; must be a no-op.
    query_module.ensure_indexes_loaded()
    assert load_json_calls == [], (
        "load_index_json must be skipped when BM25 arm is warm"
    )
    assert load_dense_calls == [], (
        "load_dense_index must be skipped when dense arm is warm"
    )

    # Second call — still warm; still no-op.
    query_module.ensure_indexes_loaded()
    assert load_json_calls == [], "load_index_json must be skipped on second call too"
    assert load_dense_calls == [], "load_dense_index must be skipped on second call too"
