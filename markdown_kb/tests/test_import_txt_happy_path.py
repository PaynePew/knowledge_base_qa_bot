"""Integration tests for POST /import — Slice 7-1 TXT happy path.

AC coverage (issue #90 — Slice 7-1):
  - POST /import on .txt writes docs/<basename>.md with no heading inference
  - .txt output parsed via indexer.parse_markdown yields single Section with id=<basename>.md
  - .txt frontmatter: original_format=txt, imported_from, imported_at present
  - .txt passthrough: content preserved verbatim in body
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_import_txt_creates_docs_file(import_env):
    """POST /import on .txt creates docs/<basename>.md."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.txt"
    (raw_dir / "simple.txt").write_bytes(src.read_bytes())

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    result = data["imported_sources"][0]
    assert result["original_format"] == "txt"
    assert result["status"] == "created"
    assert result["docs_path"].endswith("simple.md")

    assert (docs_dir / "simple.md").exists()


def test_import_txt_frontmatter(import_env):
    """TXT output carries correct frontmatter fields."""
    import yaml

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.txt"
    (raw_dir / "simple.txt").write_bytes(src.read_bytes())

    client.post("/import")

    content = (docs_dir / "simple.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    end = content.index("---\n", 4)
    fm = yaml.safe_load(content[4:end])

    assert fm["original_format"] == "txt"
    assert "simple.txt" in fm["imported_from"]
    assert "imported_at" in fm
    assert "content_sha256" not in fm


def test_import_txt_body_passthrough(import_env):
    """TXT content is preserved verbatim in the output body."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.txt"
    (raw_dir / "simple.txt").write_bytes(src.read_bytes())

    client.post("/import")

    content = (docs_dir / "simple.md").read_text(encoding="utf-8")
    # Strip frontmatter
    body = content.split("---\n", 2)[2]

    # Original text content must be present
    assert "Plain text content with no headings" in body


def test_import_txt_single_section_degraded(import_env):
    """TXT output parsed via parse_markdown yields a single Section with id=<basename>.md."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    src = FIXTURES / "simple.txt"
    (raw_dir / "simple.txt").write_bytes(src.read_bytes())

    client.post("/import")

    from app.indexer import parse_markdown

    docs_file = docs_dir / "simple.md"
    sections = parse_markdown(docs_file)

    assert len(sections) == 1, f"Expected 1 section (degraded), got {len(sections)}"
    # Per CONTEXT.md degraded-Section rule: id = bare filename
    assert sections[0].id == "simple.md", (
        f"Degraded section id must be 'simple.md', got '{sections[0].id}'"
    )
