"""Integration tests for POST /import — PDF happy path (issue #415 / ADR-0031).

AC coverage (PRD #414 / issue #415):
  - Batch and single-source Import convert a digital-native .pdf to a docs/
    Source with standard provenance frontmatter, including
    original_format=pdf and content_sha256.
  - The English fixture converts to Markdown whose headings index into >=2
    Sections and whose table renders as a Markdown table.
  - The CJK fixture converts with Chinese text intact (verbatim substrings).
  - Re-importing an unchanged PDF hash-skips; a byte-modified PDF overwrites
    with status='updated'.

Fixtures: markdown_kb/tests/fixtures/raw_import/sample_english.pdf and
sample_cjk.pdf, regenerable via
markdown_kb/tests/fixtures/generate_pdf_fixtures.py (dev-only, reportlab).
"""

from __future__ import annotations

import shutil
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


# ---------------------------------------------------------------------------
# English fixture: batch import, frontmatter, headings -> Sections, table
# ---------------------------------------------------------------------------


def test_import_pdf_batch_creates_docs_file(import_env):
    """Batch POST /import converts sample_english.pdf to docs/sample_english.md."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "sample_english.pdf", raw_dir / "sample_english.pdf")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    result = data["imported_sources"][0]
    assert result["original_format"] == "pdf"
    assert result["status"] == "created"
    assert result["docs_path"].endswith("sample_english.md")
    assert (docs_dir / "sample_english.md").exists()


def test_import_pdf_single_mode(import_env):
    """Single-source POST /import (source filter) converts one named .pdf."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    shutil.copy(FIXTURES / "sample_english.pdf", raw_dir / "sample_english.pdf")

    resp = client.post("/import", json={"source": "sample_english.pdf"})
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["original_format"] == "pdf"


def test_import_pdf_frontmatter(import_env):
    """PDF output carries standard provenance frontmatter incl. original_format=pdf."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "sample_english.pdf", raw_dir / "sample_english.pdf")
    client.post("/import")

    content = (docs_dir / "sample_english.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    end = content.index("---\n", 4)
    fm = yaml.safe_load(content[4:end])

    assert fm["original_format"] == "pdf"
    assert "sample_english.pdf" in fm["imported_from"]
    assert "imported_at" in fm
    assert fm["content_sha256"], "content_sha256 must not be empty"


def test_import_pdf_headings_index_into_multiple_sections(import_env):
    """English fixture's literal '#'/'##' text lines index into >=2 Sections."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "sample_english.pdf", raw_dir / "sample_english.pdf")
    client.post("/import")

    from app.indexer import parse_markdown

    sections = parse_markdown(docs_dir / "sample_english.md")
    assert len(sections) >= 2, f"Expected >=2 Sections, got {len(sections)}: {sections}"
    headings = {s.heading for s in sections}
    assert "Getting Started" in headings
    assert "Installation" in headings


def test_import_pdf_table_renders_as_markdown_table(import_env):
    """The English fixture's table converts to a pipe-delimited Markdown table."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "sample_english.pdf", raw_dir / "sample_english.pdf")
    client.post("/import")

    content = (docs_dir / "sample_english.md").read_text(encoding="utf-8")
    assert "| Plan" in content, f"Expected a Markdown table header row, got:\n{content}"
    assert "| Basic" in content
    assert "$9" in content


# ---------------------------------------------------------------------------
# CJK fixture: Traditional Chinese text intact
# ---------------------------------------------------------------------------


def test_import_pdf_cjk_text_intact(import_env):
    """The CJK fixture converts with Chinese text intact (verbatim substrings)."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "sample_cjk.pdf", raw_dir / "sample_cjk.pdf")
    resp = client.post("/import")
    assert resp.status_code == 200
    assert resp.json()["imported_sources"][0]["original_format"] == "pdf"

    content = (docs_dir / "sample_cjk.md").read_text(encoding="utf-8")
    assert "退款政策" in content, f"Expected CJK heading text intact, got:\n{content}"
    assert "常見問題" in content, f"Expected CJK heading text intact, got:\n{content}"
    assert "如果您在購買後七天內申請退款" in content, (
        f"Expected CJK body text intact, got:\n{content}"
    )


# ---------------------------------------------------------------------------
# Hash-skip / hash-drift parity with the other formats
# ---------------------------------------------------------------------------


def test_import_pdf_reimport_unchanged_hash_skips(import_env):
    """Re-importing an unchanged PDF hash-skips (status='skipped', no disk write)."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "sample_english.pdf", raw_dir / "sample_english.pdf")

    resp1 = client.post("/import")
    assert len(resp1.json()["imported_sources"]) == 1

    docs_file = docs_dir / "sample_english.md"
    mtime_after_first = docs_file.stat().st_mtime

    resp2 = client.post("/import")
    data2 = resp2.json()
    assert data2["imported_sources"] == []
    assert len(data2["skipped_sources"]) == 1
    assert data2["skipped_sources"][0]["status"] == "skipped"
    assert docs_file.stat().st_mtime == mtime_after_first, (
        "Docs file must not be rewritten on hash-match skip"
    )


def test_import_pdf_byte_modified_overwrites_as_updated(import_env):
    """A byte-modified PDF re-import overwrites the previous conversion (status='updated')."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    raw_path = raw_dir / "sample_english.pdf"
    shutil.copy(FIXTURES / "sample_english.pdf", raw_path)

    client.post("/import")  # first import: created

    # Byte-modify (append trailing bytes — still resolves as the same PDF
    # object stream for our purposes; only the hash needs to change).
    raw_path.write_bytes(raw_path.read_bytes() + b"%stray-comment-bytes")

    resp = client.post("/import")
    data = resp.json()
    assert data["skipped_sources"] == []
    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["status"] == "updated"
