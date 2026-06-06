"""Hot Cache deep module — working-memory persistence for the MCP agent.

Phase 12 Slice 2 (ADR-0016, PRD #198).

Provides ``read_hot()`` and ``save_hot()`` as the only entry points.  The MCP
tools ``kb_read_hot_v1`` and ``kb_save_hot_v1`` in ``server.py`` delegate here.

``wiki/hot.md`` is git-ignored (per-user runtime cache).  A missing file is a
valid state (first session) and returns an empty string, not an error.

``save_hot`` writes atomically via ``write_text_atomic`` from
``markdown_kb.app.atomic`` (tmp-file + ``os.replace`` with a bounded retry for
transient Windows AV/Search file locks).  ``kb_mcp → markdown-kb`` is a declared
dependency; the prior ``kb_mcp → eval`` direction was undeclared (issue #211).
"""

from __future__ import annotations

from pathlib import Path

# Import the atomic module under ``_loader`` so tests can monkeypatch
# ``hc._loader.os.replace`` — the established seam for the replace spy in
# test_hot_cache.py.  ``markdown_kb.app.atomic`` exposes ``os`` and ``time`` as
# module-level attributes, identical to the old ``eval.paraphrase_comparison.loader``
# seam.  The noqa comment on the import line suppresses F401; this is intentional.
import markdown_kb.app.atomic as _loader  # noqa: F401
from markdown_kb.app.atomic import write_text_atomic

# Default hot-cache path — resolved at import time so callers see the same
# singleton.  Tests monkeypatch this name after importing the module.
HOT_PATH: Path = Path(__file__).resolve().parents[3] / "wiki" / "hot.md"


def read_hot(*, hot_path: Path | None = None) -> str:
    """Return the contents of the hot cache file, or '' when absent.

    Args:
        hot_path: Path to ``hot.md``.  Defaults to the module-level ``HOT_PATH``
            constant (repo-root ``wiki/hot.md``).

    Returns:
        The file's text contents (UTF-8), or an empty string when the file does
        not exist.  An absent file is a normal first-session state — it is not
        an error.
    """
    path = hot_path if hot_path is not None else HOT_PATH
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_hot(summary: str, *, hot_path: Path | None = None) -> None:
    """Persist ``summary`` to the hot cache file atomically.

    Writes via ``write_text_atomic`` (tmp-file + ``os.replace``), which is the
    Windows-safe atomic-write pattern from CODING_STANDARD §2.6.  Parent
    directories are created automatically.

    Args:
        summary: The working-memory summary string to persist.
        hot_path: Path to ``hot.md``.  Defaults to ``HOT_PATH``.
    """
    path = hot_path if hot_path is not None else HOT_PATH
    write_text_atomic(path, summary)
