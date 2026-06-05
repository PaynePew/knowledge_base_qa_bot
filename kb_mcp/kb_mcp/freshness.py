"""Reusable index-freshness helper shared by MCP server and CLI REPL.

``reload_if_stale()`` reloads ``.kb/index.json`` into the markdown_kb indexer's
in-process module global only when the file's mtime has changed since the last
load.  This is the shared freshness mechanism described in ADR-0016 and PRD #198
(§Implementation Decisions — "Index freshness (shared deep module)").

Because ``indexer.search`` reads the module-level ``sections`` list (not a per-call
argument), a long-lived server process must reload the list when the operator
rebuilds the index on disk.  mtime-gating avoids unnecessary reloads.
"""

from __future__ import annotations

from pathlib import Path

# Module-level mtime cache — tracks when we last successfully loaded the index.
# ``None`` means never loaded (cold start).
_last_mtime: float | None = None

# Default index path — resolved at import time so callers see the same singleton.
# Tests monkeypatch this name after importing the module.
INDEX_PATH: Path = Path(__file__).resolve().parents[3] / ".kb" / "index.json"


def reload_if_stale(index_path: Path | None = None) -> bool:
    """Reload ``.kb/index.json`` when its mtime has changed since the last load.

    Args:
        index_path: Path to the index JSON file.  Defaults to the module-level
            ``INDEX_PATH`` constant (which points at the repo-root ``.kb/index.json``).

    Returns:
        ``True`` when a reload was performed, ``False`` when the index is already
        up to date or the file does not exist.

    Side-effects:
        Calls ``markdown_kb.app.indexer.load_index_json(index_path)`` which
        populates the module-level ``sections`` / ``doc_freq`` globals used by
        ``indexer.search``.  Also updates the module-level ``_last_mtime`` cache.
    """
    global _last_mtime

    if index_path is None:
        index_path = INDEX_PATH

    if not index_path.exists():
        return False

    current_mtime = index_path.stat().st_mtime
    if _last_mtime is not None and current_mtime == _last_mtime:
        return False

    # mtime changed (or cold start) — reload.
    from markdown_kb.app.indexer import load_index_json

    load_index_json(index_path)
    _last_mtime = current_mtime
    return True
