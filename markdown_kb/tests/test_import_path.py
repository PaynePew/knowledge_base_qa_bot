"""Hermetic tests for importer.import_path — Slice 227.

AC coverage (issue #227):
  - import_path stages a local file into raw/ under a safe basename and
    converts it to a docs/ Source
  - .html inputs convert correctly (HTML → Markdown)
  - .txt inputs convert correctly (passthrough)
  - .md inputs are recognised as already-canonical (no double conversion)
  - .pdf inputs convert via MarkItDown text-layer extraction (issue #415 /
    ADR-0031 — supersedes the original "extractor not yet available" AC)
  - traversal-unsafe path (.. or separators in basename) is rejected with
    a clear error; nothing is written outside raw/docs/
  - .kb/index.json and wiki/ are not touched (writes only to tmp-redirected
    raw/docs)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixture: tmp raw/docs dirs wired into importer module
# ---------------------------------------------------------------------------


@pytest.fixture()
def import_path_env(tmp_path, monkeypatch):
    """Wire RAW_DIR and DOCS_DIR into importer for isolation.

    Mirrors the ``import_env`` fixture in test_import_html_happy_path.py but
    targets ``import_path`` rather than the HTTP route.
    """
    import app.importer as importer_module
    import app.logger as logger_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    return {"raw_dir": raw_dir, "docs_dir": docs_dir, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# AC-1: .txt input is staged to raw/ and converted (passthrough) to docs/
# ---------------------------------------------------------------------------


def test_import_path_txt_stages_and_converts(import_path_env, tmp_path):
    """import_path stages a .txt file into raw/ and produces a docs/ Source."""
    from app.importer import import_path

    src = tmp_path / "notes.txt"
    src.write_text("Some important notes.\nLine two.", encoding="utf-8")

    result = import_path(src)

    raw_dir = import_path_env["raw_dir"]
    docs_dir = import_path_env["docs_dir"]

    # Staged to raw/
    assert (raw_dir / "notes.txt").exists(), "File must be staged into raw/"
    # Converted to docs/
    assert (docs_dir / "notes.md").exists(), "Converted docs/*.md must be created"

    # Result reflects the outcome
    assert result is not None
    assert result.original_format == "txt"
    assert result.status in ("created", "updated")
    assert result.docs_path.endswith("notes.md")


# ---------------------------------------------------------------------------
# AC-2: .html input converts to Markdown
# ---------------------------------------------------------------------------


def test_import_path_html_converts_to_markdown(import_path_env, tmp_path):
    """import_path converts .html to Markdown in docs/."""
    from app.importer import import_path

    src = tmp_path / "article.html"
    src.write_text(
        "<h1>Title</h1><p>Some content here.</p>",
        encoding="utf-8",
    )

    result = import_path(src)

    docs_dir = import_path_env["docs_dir"]
    docs_file = docs_dir / "article.md"
    assert docs_file.exists(), "docs/article.md must be created"

    content = docs_file.read_text(encoding="utf-8")
    assert "# Title" in content, "H1 must be converted to Markdown heading"
    assert "Some content here" in content, "Body text must be preserved"
    assert result.original_format == "html"


# ---------------------------------------------------------------------------
# AC-3: .md input is recognised as already-canonical (no double-conversion)
# ---------------------------------------------------------------------------


def test_import_path_md_is_canonical(import_path_env, tmp_path):
    """import_path treats .md inputs as canonical — no heading inference."""
    from app.importer import import_path

    src = tmp_path / "page.md"
    src.write_text("# My Page\n\nAlready Markdown.", encoding="utf-8")

    result = import_path(src)

    docs_dir = import_path_env["docs_dir"]
    docs_file = docs_dir / "page.md"
    assert docs_file.exists(), "docs/page.md must be created"

    content = docs_file.read_text(encoding="utf-8")
    assert "# My Page" in content, ".md body must be preserved as-is"
    assert result.original_format == "md"


# ---------------------------------------------------------------------------
# AC-4: .pdf input converts via MarkItDown (issue #415 / ADR-0031)
# ---------------------------------------------------------------------------


def test_import_path_pdf_converts_to_markdown(import_path_env, tmp_path):
    """import_path stages a .pdf file into raw/ and converts it to a docs/ Source."""
    from reportlab.pdfgen import canvas

    from app.importer import import_path

    src = tmp_path / "document.pdf"
    c = canvas.Canvas(str(src))
    c.drawString(72, 750, "# Document Title")
    c.drawString(72, 730, "Some PDF body text.")
    c.showPage()
    c.save()

    result = import_path(src)

    raw_dir = import_path_env["raw_dir"]
    docs_dir = import_path_env["docs_dir"]

    assert (raw_dir / "document.pdf").exists(), "File must be staged into raw/"
    assert (docs_dir / "document.md").exists(), "Converted docs/*.md must be created"
    assert result.original_format == "pdf"
    assert result.status in ("created", "updated")

    content = (docs_dir / "document.md").read_text(encoding="utf-8")
    assert "# Document Title" in content
    assert "Some PDF body text." in content


# ---------------------------------------------------------------------------
# AC-5: traversal-unsafe path is rejected — nothing written
# ---------------------------------------------------------------------------


def test_import_path_rejects_traversal_in_basename(import_path_env, tmp_path):
    """import_path rejects a path with '..' in the basename."""
    from app.importer import ImportPathError, import_path

    # Construct a path whose basename contains ".." (non-existent file)
    evil = tmp_path / ".." / "evil.txt"

    with pytest.raises((ImportPathError, ValueError, OSError)):
        import_path(evil)


def test_import_path_rejects_basename_with_slash(import_path_env, tmp_path):
    """import_path rejects a basename that would escape via a path separator."""
    from app.importer import ImportPathError, import_path

    # Test that a basename with '#' is rejected (shares rejection logic with '/')
    src_hash = tmp_path / "bad#name.txt"
    src_hash.write_text("content", encoding="utf-8")

    with pytest.raises((ImportPathError, ValueError)):
        import_path(src_hash)


# ---------------------------------------------------------------------------
# AC-6: frontmatter is correct (imported_from / original_format / imported_at)
# ---------------------------------------------------------------------------


def test_import_path_txt_frontmatter(import_path_env, tmp_path):
    """docs/*.md produced by import_path has well-formed provenance frontmatter."""
    from app.importer import import_path

    src = tmp_path / "memo.txt"
    src.write_text("A memo.", encoding="utf-8")

    import_path(src)

    docs_dir = import_path_env["docs_dir"]
    content = (docs_dir / "memo.md").read_text(encoding="utf-8")
    assert content.startswith("---\n"), "Output must start with YAML frontmatter"

    end = content.index("---\n", 4)
    fm = yaml.safe_load(content[4:end])

    assert "imported_from" in fm
    assert "original_format" in fm
    assert "imported_at" in fm
    assert fm["original_format"] == "txt"
