"""Integration tests for POST /import — Slice 7-1 HTML happy path.

AC coverage (issue #90 — Slice 7-1):
  - POST /import (batch mode) with .html in raw/ writes docs/<basename>.md
  - Output frontmatter has imported_from, original_format, imported_at, content_sha256
  - HTML conversion: headings, paragraphs, lists, code, links preserved
  - HTML conversion: script, style, iframe, form, comments stripped
  - POST /import with source=<filename> processes single specified file
  - Nested raw/sub/foo.html discovered by batch glob, output flattens to docs/foo.md
  - Atomic write: no .tmp file lingers after success
  - FileNotFoundError populates failed_sources with correct error_type
  - POST /import always returns HTTP 200

Extended in Slice 7-3 (issue #92):
  - content_sha256 is present in frontmatter on first write
  - status='created' on first write
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


# ---------------------------------------------------------------------------
# App client fixture with tmp dirs wired
# ---------------------------------------------------------------------------


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir and docs_dir into importer.py for isolation."""
    import app.importer as importer_module
    import app.logger as logger_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    from app.main import app

    client = TestClient(app)
    return {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir}


# ---------------------------------------------------------------------------
# Batch-mode HTML happy path
# ---------------------------------------------------------------------------


def test_import_html_batch_creates_docs_file(import_env):
    """POST /import (no body) processes .html from raw/ and writes docs/<basename>.md."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    # Place a fixture HTML file in raw/
    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    assert data["failed_sources"] == []
    assert data["skipped_sources"] == []

    result = data["imported_sources"][0]
    assert result["original_format"] == "html"
    assert result["status"] == "created", "First import must have status='created'"
    assert result["content_sha256"] != "", (
        "content_sha256 must be populated on first import (slice 7-3)"
    )
    assert result["raw_path"].endswith("clean_article.html")
    assert result["docs_path"].endswith("clean_article.md")

    # File exists on disk
    docs_file = docs_dir / "clean_article.md"
    assert docs_file.exists(), "docs/clean_article.md must be created"


def test_import_html_frontmatter(import_env):
    """Output docs/*.md has imported_from, original_format, imported_at frontmatter."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    content = (docs_dir / "clean_article.md").read_text(encoding="utf-8")
    assert content.startswith("---\n"), "File must start with YAML frontmatter"

    # Extract frontmatter block
    end = content.index("---\n", 4)
    fm = yaml.safe_load(content[4:end])

    assert "imported_from" in fm, "imported_from must be in frontmatter"
    assert "original_format" in fm, "original_format must be in frontmatter"
    assert "imported_at" in fm, "imported_at must be in frontmatter"
    assert fm["original_format"] == "html"
    assert "clean_article.html" in fm["imported_from"]
    # content_sha256 added in slice 7-3
    assert "content_sha256" in fm, "content_sha256 must be in frontmatter (slice 7-3)"
    assert fm["content_sha256"], "content_sha256 must not be empty"

    # imported_at is ISO-8601 UTC
    ts = fm["imported_at"]
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*Z", ts), (
        f"imported_at must be ISO-8601 UTC, got: {ts}"
    )


def test_import_html_body_preserves_structure(import_env):
    """HTML headings, paragraphs, lists, code, links are preserved in output Markdown."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    body = (docs_dir / "clean_article.md").read_text(encoding="utf-8")
    # Strip frontmatter
    body_after_fm = body.split("---\n", 2)[2]

    # Headings preserved
    assert "# Clean Article Title" in body_after_fm, "H1 must be preserved"
    assert "## Section One" in body_after_fm, "H2 must be preserved"
    assert "## Section Two" in body_after_fm, "H2 must be preserved"

    # Formatting preserved
    assert "**bold text**" in body_after_fm or "bold text" in body_after_fm
    assert "example site" in body_after_fm, "Link text must be preserved"
    assert "https://example.com" in body_after_fm, "Link URL must be preserved"

    # List items preserved
    assert "List item one" in body_after_fm
    assert "List item two" in body_after_fm

    # Code preserved
    assert "code snippet" in body_after_fm
    assert "function example" in body_after_fm

    # Blockquote preserved
    assert "notable quotation" in body_after_fm


def test_import_html_strips_unwanted_elements(import_env):
    """HTML script, style, iframe, form, comments are stripped from output."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    body = (docs_dir / "clean_article.md").read_text(encoding="utf-8")

    assert "console.log" not in body, "script content must be stripped"
    assert "font-family" not in body, "style content must be stripped"
    assert "ads.example.com" not in body, "iframe src must be stripped"
    assert "<form" not in body, "form tags must not appear in output"
    assert "<input" not in body, "input tags must not appear in output"
    assert "<!-- " not in body, "HTML comments must be stripped"
    assert "Please enable JavaScript" not in body, "noscript must be stripped"


# ---------------------------------------------------------------------------
# Single-mode
# ---------------------------------------------------------------------------


def test_import_html_single_mode(import_env):
    """POST /import with source=<filename> processes one specific file."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    # Place two files, request only one
    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())
    (raw_dir / "other.html").write_text("<h1>Other</h1><p>Other content.</p>", encoding="utf-8")

    resp = client.post("/import", json={"source": "clean_article.html"})
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["docs_path"].endswith("clean_article.md")

    # other.html should NOT have been processed
    assert not (docs_dir / "other.md").exists()


# ---------------------------------------------------------------------------
# Nested glob + flatten
# ---------------------------------------------------------------------------


def test_import_html_nested_raw_flattens(import_env):
    """Nested raw/sub/foo.html discovered by batch glob, output is docs/foo.md."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    nested = raw_dir / "2024" / "q3"
    nested.mkdir(parents=True)
    (nested / "nested_article.html").write_text(
        "<h1>Nested</h1><p>Nested content.</p>", encoding="utf-8"
    )

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    # Output must be flattened to docs/nested_article.md (no subdirectory)
    result = data["imported_sources"][0]
    assert result["docs_path"].endswith("nested_article.md")
    assert (docs_dir / "nested_article.md").exists()


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_import_html_no_tmp_file_lingers(import_env):
    """No .tmp file remains in docs/ after a successful import."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "clean_article.html"
    (raw_dir / "clean_article.html").write_bytes(src.read_bytes())

    client.post("/import")

    tmp_files = list(docs_dir.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files in docs/: {tmp_files}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_import_single_mode_missing_file_returns_200(import_env):
    """POST /import with non-existent source returns 200 with failed_sources populated."""
    client = import_env["client"]

    resp = client.post("/import", json={"source": "nonexistent.html"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["imported_sources"] == []
    assert len(data["failed_sources"]) == 1

    failure = data["failed_sources"][0]
    assert failure["error_type"] == "FileNotFoundError"
    assert "nonexistent.html" in failure["raw_path"]
    assert len(failure["error_message"]) <= 200
