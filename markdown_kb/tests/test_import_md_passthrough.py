"""Tests for POST /import — .md passthrough (issue #141).

AC coverage (issue #141 — Phase 8.5 S4):
  - POST /import on .md writes docs/<basename>.md with content preserved verbatim
  - .md output carries correct frontmatter (original_format=md, imported_from, imported_at,
    content_sha256)
  - .md imports pass through collision detection identically to other formats
  - .md imports pass through filename validation identically to other formats
  - batch mode picks up .md files via glob
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


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

    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    return {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir}


def test_import_md_creates_docs_file(import_env):
    """POST /import on .md creates docs/<basename>.md."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.md"
    (raw_dir / "simple.md").write_bytes(src.read_bytes())

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    result = data["imported_sources"][0]
    assert result["original_format"] == "md"
    assert result["status"] == "created"
    assert result["docs_path"].endswith("simple.md")

    assert (docs_dir / "simple.md").exists()


def test_import_md_frontmatter(import_env):
    """.md output carries correct frontmatter fields."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.md"
    (raw_dir / "simple.md").write_bytes(src.read_bytes())

    client.post("/import")

    content = (docs_dir / "simple.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    end = content.index("---\n", 4)
    fm = yaml.safe_load(content[4:end])

    assert fm["original_format"] == "md"
    assert "simple.md" in fm["imported_from"]
    assert "imported_at" in fm
    assert "content_sha256" in fm, "content_sha256 must be present in frontmatter"
    assert fm["content_sha256"], "content_sha256 must not be empty"


def test_import_md_body_passthrough(import_env):
    """.md content is preserved verbatim in the output body."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.md"
    (raw_dir / "simple.md").write_bytes(src.read_bytes())

    client.post("/import")

    content = (docs_dir / "simple.md").read_text(encoding="utf-8")
    # Strip frontmatter
    body = content.split("---\n", 2)[2]

    # Original Markdown content must be preserved verbatim
    assert "# Simple Markdown Source" in body
    assert "## Section One" in body
    assert "Content under section one." in body


def test_import_md_single_mode(import_env):
    """POST /import?source=simple.md works in single-file mode."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.md"
    (raw_dir / "simple.md").write_bytes(src.read_bytes())

    resp = client.post("/import", params={"source": "simple.md"})
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["original_format"] == "md"

    assert (docs_dir / "simple.md").exists()


def test_import_md_batch_glob_includes_md(import_env):
    """Batch mode (no source filter) picks up .md files alongside .txt and .html."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    src_md = FIXTURES / "simple.md"
    src_txt = FIXTURES / "simple.txt"
    (raw_dir / "page.md").write_bytes(src_md.read_bytes())
    (raw_dir / "note.txt").write_bytes(src_txt.read_bytes())

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    # Both files should be imported
    imported_names = {
        r["docs_path"].split("/")[-1].split("\\")[-1] for r in data["imported_sources"]
    }
    assert "page.md" in imported_names
    assert "note.md" in imported_names


def test_import_md_collision_detection(import_env):
    """Hand-authored collision detection applies to .md imports identically to other formats."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.md"
    (raw_dir / "simple.md").write_bytes(src.read_bytes())

    # Place a hand-authored docs file (no imported_from) at the target path
    hand_authored = docs_dir / "simple.md"
    hand_authored.write_text("# Hand-authored\n\nThis has no frontmatter.\n", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 0
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "HandAuthoredCollision"


def test_import_md_filename_validation(import_env):
    """Filename validation rejects .md files with invalid chars identically to other formats."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    src = FIXTURES / "simple.md"
    (raw_dir / "bad#name.md").write_bytes(src.read_bytes())

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 0
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "InvalidFilename"
