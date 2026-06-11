"""Repo-root conftest: a session-scoped safety net for committed invariant files.

#204 made ``.kb/index.json`` byte-stable across the normal suite, but that relied
on every test — across every package — remembering to redirect ``INDEX_PATH`` to
tmp. The live suite (``pytest -m live``) exercises real build/ingest paths and has
leaked the committed ``.kb/index.json`` back to disk: the markdown_kb autouse
redirect is package-local, and ``indexer.SOURCE_DIRS`` is frozen at import time
(the autouse patches ``WIKI_DIR`` but not the already-computed ``SOURCE_DIRS``), so
a ``build_index()`` whose ``INDEX_PATH`` isolation slips writes the real file.

Rather than chase each leaking test, this session guard enforces the invariant
centrally and package-agnostically: snapshot each tracked invariant file before the
session, and restore it afterwards if a test mutated it — emitting a warning so the
leak is visible rather than silent. It changes no test's behaviour; it only protects
the committed bytes (#204).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import conftest_support as _guard

# Git-tracked invariant files that no test run may leave mutated (see #204).
_PROTECTED: list[Path] = [Path(__file__).parent / ".kb" / "index.json"]


@pytest.fixture(scope="session", autouse=True)
def _protect_committed_invariants():
    """Snapshot tracked invariant files; restore any a test mutated, with a warning."""
    snapshots = {p: _guard.read_bytes_or_none(p) for p in _PROTECTED}
    yield
    for path, original in snapshots.items():
        if _guard.restore_if_changed(path, original):
            warnings.warn(
                f"A test mutated the committed invariant file {path} during this "
                f"session; it has been restored to its committed bytes. Some test is "
                f"writing a real path instead of tmp — likely a live test whose "
                f"INDEX_PATH / SOURCE_DIRS isolation leaked (see #204).",
                stacklevel=1,
            )
