"""Shallow module per Ousterhout. Public surface: ``DOCS_DIR``, ``WIKI_DIR``, ``INDEX_PATH``.

Canonical filesystem locations for the markdown_kb app.

Centralises the three path constants previously declared in
:mod:`markdown_kb.app.indexer`. Both ``indexer.py`` (BM25 indexing /
``/index`` route) and ``ingest.py`` (Source -> wiki/ synthesis,
``/ingest`` route) reference the same canonical Sources directory, so
having ``ingest`` import the constant from ``indexer`` created a
one-way coupling that the Slice #29 implementer flagged.

The constants are intentionally module-level Path objects rather than
helper functions so that:

  * Existing test ``monkeypatch.setattr(<module>, "DOCS_DIR", ...)`` calls
    continue to work — ``from ._paths import DOCS_DIR`` rebinds the name
    inside the importing module's namespace, and the monkeypatch patches
    that local binding (the same machinery as before).
  * The ``is``-based default-sentinel check in ``indexer.build_index``
    (``docs_dir is not DOCS_DIR``) keeps its identity semantics — both
    callers see the same Path instance.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DOCS_DIR = _REPO_ROOT / "docs"
WIKI_DIR = _REPO_ROOT / "wiki"
INDEX_PATH = _REPO_ROOT / ".kb" / "index.json"
