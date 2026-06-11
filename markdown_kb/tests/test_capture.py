"""Hermetic tests for the ``capture`` deep module.

Tests exercise the public entry point ``capture_source(filename, content)``
directly — no interface, no LLM.  Mirrors the upload.py / importer.py test
pattern: validate behaviour at the boundary (bytes written, frontmatter
stamped, rejection raised, Import not invoked).

Isolation
---------
The autouse ``_redirect_paths_to_tmp`` fixture in conftest.py redirects
``INDEX_PATH`` and ``LOG_PATH`` to tmp.  Capture tests also redirect
``app.capture.DOCS_DIR`` to ``tmp_path`` so no write ever lands in the
real ``docs/``.

AC reference (issue #230)
--------------------------
AC-1  valid filename + content → docs/ Source with mandatory provenance frontmatter
AC-2  unsafe filename (traversal / separators) is rejected; nothing written outside docs/
AC-3  Import is NOT invoked (no format conversion)
AC-4  (MCP tool — tested in test_kb_capture.py)
AC-5  hermetic tests cover deep module
AC-6  .kb/index.json / wiki/ byte-stability respected; tests write only to tmp
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# AC-1: valid filename + content → docs/ Source with mandatory provenance
# ---------------------------------------------------------------------------


def test_capture_source_writes_file_to_docs(tmp_path, monkeypatch):
    """capture_source writes the file into docs_dir."""
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    capture_mod.capture_source("my_note.md", "# My Note\n\nSome content.\n", docs_dir=docs_dir)

    assert (docs_dir / "my_note.md").exists(), "File was not written to docs_dir"


def test_capture_source_content_in_file(tmp_path, monkeypatch):
    """capture_source writes the supplied content into the file."""
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    content = "# Hello\n\nThis is my captured note.\n"
    capture_mod.capture_source("captured.md", content, docs_dir=docs_dir)

    text = (docs_dir / "captured.md").read_text(encoding="utf-8")
    assert "This is my captured note." in text


def test_capture_source_stamps_origin_frontmatter(tmp_path, monkeypatch):
    """capture_source stamps ``origin: mcp-conversation`` in YAML frontmatter."""
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    capture_mod.capture_source("note.md", "# Note\n\nBody.\n", docs_dir=docs_dir)

    text = (docs_dir / "note.md").read_text(encoding="utf-8")
    assert "origin: mcp-conversation" in text, (
        f"Mandatory 'origin: mcp-conversation' not found in:\n{text}"
    )


def test_capture_source_stamps_authored_by_frontmatter(tmp_path, monkeypatch):
    """capture_source stamps ``authored_by: agent`` in YAML frontmatter."""
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    capture_mod.capture_source("note.md", "# Note\n\nBody.\n", docs_dir=docs_dir)

    text = (docs_dir / "note.md").read_text(encoding="utf-8")
    assert "authored_by: agent" in text, (
        f"Mandatory 'authored_by: agent' not found in:\n{text}"
    )


def test_capture_source_stamps_created_at_frontmatter(tmp_path, monkeypatch):
    """capture_source stamps a ``created_at`` ISO-8601 timestamp in YAML frontmatter."""
    import re

    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    capture_mod.capture_source("note.md", "# Note\n\nBody.\n", docs_dir=docs_dir)

    text = (docs_dir / "note.md").read_text(encoding="utf-8")
    # created_at: 2026-06-11T...  (ISO-8601 prefix sufficient)
    assert re.search(r"created_at: \d{4}-\d{2}-\d{2}T", text), (
        f"Mandatory 'created_at' ISO-8601 timestamp not found in:\n{text}"
    )


def test_capture_source_frontmatter_is_valid_yaml_block(tmp_path, monkeypatch):
    """The written file starts with a valid YAML front-matter block (--- ... ---)."""
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    capture_mod.capture_source("note.md", "# Note\n\nBody.\n", docs_dir=docs_dir)

    text = (docs_dir / "note.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"File does not start with YAML front-matter '---': {text[:80]!r}"
    # The closing delimiter must also appear
    assert "\n---\n" in text, f"No closing '---' delimiter found in:\n{text}"


def test_capture_source_write_is_atomic(tmp_path, monkeypatch):
    """capture_source uses the atomic write helper — no partial files on crash.

    We verify indirectly: if a .tmp file is left behind after a successful
    write, the atomic helper contract was broken.  On a successful call, no
    *.tmp files should remain in docs_dir.
    """
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    capture_mod.capture_source("note.md", "# Note\n\nBody.\n", docs_dir=docs_dir)

    tmp_files = list(docs_dir.glob("*.tmp"))
    assert tmp_files == [], f"Leftover .tmp files after write: {tmp_files}"


# ---------------------------------------------------------------------------
# AC-2: unsafe filenames are rejected; nothing written outside docs/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_filename",
    [
        "../evil.md",
        "../../etc/passwd",
        "sub/dir/note.md",
        "sub\\dir\\note.md",
        "/absolute/path.md",
        "note\x00.md",
        "",
        "   ",
    ],
    ids=[
        "traversal-dotdot",
        "traversal-deep",
        "forward-slash",
        "backslash",
        "absolute-path",
        "null-byte",
        "empty",
        "whitespace-only",
    ],
)
def test_capture_source_rejects_unsafe_filename(tmp_path, monkeypatch, bad_filename):
    """capture_source raises ValueError for unsafe filenames; nothing is written."""
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    with pytest.raises(ValueError, match=r"(?i)(filename|path|unsafe|traversal|separator|empty|absolute|control)"):
        capture_mod.capture_source(bad_filename, "# Content\n", docs_dir=docs_dir)

    # Nothing should have been written anywhere under tmp_path
    all_written = list(tmp_path.rglob("*.md"))
    assert all_written == [], f"Files were written despite unsafe filename: {all_written}"


# ---------------------------------------------------------------------------
# AC-3: Import is NOT invoked
# ---------------------------------------------------------------------------


def test_capture_source_does_not_call_import(tmp_path, monkeypatch):
    """capture_source does not call any importer function (content is already Markdown)."""
    import app.capture as capture_mod

    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(capture_mod, "DOCS_DIR", docs_dir)

    import_called = []

    # Patch the importer at the module level to detect any call
    try:
        import app.importer as importer_mod

        original_import_sources = importer_mod.import_sources

        def _spy_import_sources(*args, **kwargs):
            import_called.append(("import_sources", args, kwargs))
            return original_import_sources(*args, **kwargs)

        monkeypatch.setattr(importer_mod, "import_sources", _spy_import_sources)
    except (ImportError, AttributeError):
        pass  # importer may not exist or may have different structure

    capture_mod.capture_source("note.md", "# Note\n\nBody.\n", docs_dir=docs_dir)

    assert import_called == [], (
        f"capture_source must NOT call the importer; called with: {import_called}"
    )
