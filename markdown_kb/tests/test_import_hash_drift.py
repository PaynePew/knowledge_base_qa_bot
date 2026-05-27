"""Integration tests for POST /import — Slice 7-3 hash drift (re-import on changed file).

AC coverage (issue #92 — Slice 7-3):
  - Re-import of modified raw file (bytes changed): imported_sources entry, status='updated'
  - Docs file is overwritten with new content
  - content_sha256 in docs frontmatter reflects new hash after update
  - ImportSourceResult.status='updated' for drift case
  - No skipped_sources entry on drift (hash-mismatch means reprocess)
  - Wiki Log emits import_source event (not import_skipped) on drift
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml  # noqa: F401 — used by yaml.safe_load in inline assertions

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log_path for isolation."""
    import app.importer as importer_module
    import app.logger as logger_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    log_path = tmp_path / "wiki" / "log.md"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    from app.main import app

    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    return {
        "client": client,
        "raw_dir": raw_dir,
        "docs_dir": docs_dir,
        "log_path": log_path,
    }


# ---------------------------------------------------------------------------
# Drift: modified raw file → status='updated' in imported_sources
# ---------------------------------------------------------------------------


def test_reimport_modified_file_goes_to_imported_sources_with_status_updated(import_env):
    """Re-import of a modified raw file produces imported_sources entry with status='updated'."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    raw_file = raw_dir / "article.html"
    raw_file.write_text("<h1>Original</h1><p>Original content.</p>", encoding="utf-8")

    # First import — creates
    resp1 = client.post("/import")
    assert resp1.json()["imported_sources"][0]["status"] == "created"

    # Modify the raw file
    raw_file.write_text("<h1>Updated</h1><p>New content after modification.</p>", encoding="utf-8")

    # Second import — drift, should update
    resp2 = client.post("/import")
    assert resp2.status_code == 200
    data = resp2.json()

    assert data["skipped_sources"] == [], "Modified file must NOT appear in skipped_sources"
    assert len(data["imported_sources"]) == 1, "Modified file must appear in imported_sources"
    result = data["imported_sources"][0]
    assert result["status"] == "updated", (
        f"Re-import of modified file must have status='updated', got: {result['status']}"
    )
    assert result["raw_path"].endswith("article.html")
    assert result["docs_path"].endswith("article.md")


def test_reimport_modified_file_overwrites_docs_content(import_env):
    """Re-import of modified raw file overwrites docs/<basename>.md with new content."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    raw_file = raw_dir / "article.html"
    raw_file.write_text("<h1>Original</h1><p>Original content.</p>", encoding="utf-8")

    client.post("/import")

    docs_file = docs_dir / "article.md"
    content_after_first = docs_file.read_text(encoding="utf-8")
    assert "Original content" in content_after_first

    # Modify the raw file
    raw_file.write_text("<h1>Refreshed</h1><p>Refreshed content after drift.</p>", encoding="utf-8")

    client.post("/import")

    content_after_second = docs_file.read_text(encoding="utf-8")
    assert "Refreshed content after drift" in content_after_second, (
        "Docs file must be overwritten with new content on drift"
    )
    assert "Original content" not in content_after_second, (
        "Old content must not remain after drift overwrite"
    )


def test_reimport_modified_file_updates_content_sha256_in_frontmatter(import_env):
    """After drift re-import, docs frontmatter content_sha256 reflects the new raw bytes."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    original_bytes = b"<h1>Original</h1><p>Original content.</p>"
    raw_file = raw_dir / "article.html"
    raw_file.write_bytes(original_bytes)

    client.post("/import")

    docs_file = docs_dir / "article.md"
    content_v1 = docs_file.read_text(encoding="utf-8")
    end_v1 = content_v1.index("---\n", 4)
    fm_v1 = yaml.safe_load(content_v1[4:end_v1])
    sha_v1 = fm_v1["content_sha256"]
    assert sha_v1 == hashlib.sha256(original_bytes).hexdigest()

    # Modify the raw file
    updated_bytes = b"<h1>Updated</h1><p>Updated content after drift.</p>"
    raw_file.write_bytes(updated_bytes)

    client.post("/import")

    content_v2 = docs_file.read_text(encoding="utf-8")
    end_v2 = content_v2.index("---\n", 4)
    fm_v2 = yaml.safe_load(content_v2[4:end_v2])
    sha_v2 = fm_v2["content_sha256"]

    expected_v2 = hashlib.sha256(updated_bytes).hexdigest()
    assert sha_v2 == expected_v2, (
        f"content_sha256 in frontmatter must be updated to reflect new raw bytes: "
        f"expected {expected_v2}, got {sha_v2}"
    )
    assert sha_v2 != sha_v1, "SHA-256 must change after file modification"


def test_reimport_modified_file_emits_import_source_not_import_skipped(import_env):
    """Drift re-import emits import_source event, NOT import_skipped."""
    import re

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    log_path = import_env["log_path"]

    raw_file = raw_dir / "article.html"
    raw_file.write_text("<h1>Original</h1><p>Content.</p>", encoding="utf-8")

    client.post("/import")  # first import

    # Clear log between imports by re-initializing path reference — we just check new events
    # We'll check the combined log and verify no import_skipped for the second call
    raw_file.write_text("<h1>Drifted</h1><p>Drifted content.</p>", encoding="utf-8")

    client.post("/import")  # drift import

    log_re = re.compile(r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] (\S+) \| (.+)$")
    lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    events = [(m.group(2), m.group(3)) for line in lines if (m := log_re.match(line))]

    skipped_events = [e for e in events if e[0] == "import_skipped"]
    source_events = [e for e in events if e[0] == "import_source"]

    assert skipped_events == [], (
        f"import_skipped must NOT be emitted on drift re-import, got: {skipped_events}"
    )
    # Should have 2 import_source events total (one per import call)
    assert len(source_events) >= 2, (
        f"Drift import must emit import_source event; found: {source_events}"
    )


def test_reimport_modified_file_response_content_sha256_updated(import_env):
    """Drift re-import response ImportSourceResult.content_sha256 reflects the new hash."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    original_bytes = b"<h1>Original</h1><p>Content.</p>"
    updated_bytes = b"<h1>Changed</h1><p>Different content entirely.</p>"

    raw_file = raw_dir / "article.html"
    raw_file.write_bytes(original_bytes)

    resp1 = client.post("/import")
    sha_v1 = resp1.json()["imported_sources"][0]["content_sha256"]
    assert sha_v1 == hashlib.sha256(original_bytes).hexdigest()

    raw_file.write_bytes(updated_bytes)
    resp2 = client.post("/import")
    sha_v2 = resp2.json()["imported_sources"][0]["content_sha256"]

    expected_v2 = hashlib.sha256(updated_bytes).hexdigest()
    assert sha_v2 == expected_v2, (
        f"Response content_sha256 must reflect updated bytes: expected {expected_v2}, got {sha_v2}"
    )
    assert sha_v2 != sha_v1, "SHA-256 in response must change when raw bytes change"


# ---------------------------------------------------------------------------
# Three-way matrix: created / updated / skipped all exercised in one batch
# ---------------------------------------------------------------------------


def test_all_three_statuses_exercised_in_one_session(import_env):
    """ImportSourceResult.status enum — 'created', 'updated', 'skipped' all exercised."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    # Place two files
    (raw_dir / "unchanged.html").write_text(
        "<h1>Unchanged</h1><p>Will not change.</p>", encoding="utf-8"
    )
    (raw_dir / "drifted.html").write_text(
        "<h1>Original</h1><p>Will be modified.</p>", encoding="utf-8"
    )

    # First import: both are 'created'
    resp1 = client.post("/import")
    statuses1 = {r["status"] for r in resp1.json()["imported_sources"]}
    assert statuses1 == {"created"}, f"All first-import statuses must be 'created': {statuses1}"

    # Modify drifted.html
    (raw_dir / "drifted.html").write_text(
        "<h1>Drifted</h1><p>Content has changed.</p>", encoding="utf-8"
    )

    # Add a brand new file
    (raw_dir / "fresh.html").write_text("<h1>Fresh</h1><p>Never seen before.</p>", encoding="utf-8")

    # Second import: unchanged='skipped', drifted='updated', fresh='created'
    resp2 = client.post("/import")
    data2 = resp2.json()

    imported = {r["status"] for r in data2["imported_sources"]}
    skipped = {r["status"] for r in data2["skipped_sources"]}

    assert "created" in imported, f"fresh.html must produce 'created' status: {data2}"
    assert "updated" in imported, f"drifted.html must produce 'updated' status: {data2}"
    assert skipped == {"skipped"}, (
        f"unchanged.html must produce 'skipped' status: {data2['skipped_sources']}"
    )
