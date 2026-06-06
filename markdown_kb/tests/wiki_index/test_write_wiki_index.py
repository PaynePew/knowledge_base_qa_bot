"""Tests for wiki_index.write_wiki_index — filesystem wrapper.

Covers:
  - Happy path: file written, tuple (True, path, None) returned.
  - Directory absent: wiki_dir is created automatically.
  - Permission denied: returns failure tuple, no exception escapes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import app.indexer as _indexer
from app.wiki_index import project_wiki_index, write_wiki_index

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path(tmp_path, indexed_corpus):
    """write_wiki_index writes index.md and returns (True, path, None)."""
    wiki_dir = tmp_path / "output_wiki"
    wiki_dir.mkdir()

    written, path, error = write_wiki_index(_indexer.sections, wiki_dir=wiki_dir)

    assert written is True
    assert error is None
    assert path == (wiki_dir / "index.md").resolve()
    assert (wiki_dir / "index.md").exists()
    # No lingering .tmp file
    assert not list(wiki_dir.glob("*.tmp"))
    # Content matches projection
    assert (wiki_dir / "index.md").read_text(encoding="utf-8") == project_wiki_index(
        _indexer.sections
    )


# ---------------------------------------------------------------------------
# Directory absent — should be created
# ---------------------------------------------------------------------------


def test_directory_absent(tmp_path, indexed_corpus):
    """write_wiki_index creates wiki_dir if it does not exist, then writes."""
    wiki_dir = tmp_path / "deep" / "wiki"
    assert not wiki_dir.exists()

    written, path, error = write_wiki_index(_indexer.sections, wiki_dir=wiki_dir)

    assert written is True
    assert error is None
    assert wiki_dir.exists()
    assert (wiki_dir / "index.md").exists()


# ---------------------------------------------------------------------------
# Permission denied — must not raise; returns failure tuple
# ---------------------------------------------------------------------------


def test_permission_denied(tmp_path, indexed_corpus):
    """write_wiki_index returns (False, None, '<Class>: <msg>') on filesystem error."""
    wiki_dir = tmp_path / "perm_wiki"
    wiki_dir.mkdir()

    # write_wiki_index now delegates to write_text_atomic in app.atomic, so the
    # os.replace seam has moved there (not app.wiki_index.os.replace any more).
    with (
        patch("app.atomic.os.replace", side_effect=PermissionError("Permission denied")),
        patch("app.atomic.time.sleep"),
    ):
        written, path, error = write_wiki_index(_indexer.sections, wiki_dir=wiki_dir)

    assert written is False
    assert path is None
    assert error is not None
    assert "PermissionError" in error
    assert "Permission denied" in error
    # index.md must not exist (write failed before rename)
    assert not (wiki_dir / "index.md").exists()
