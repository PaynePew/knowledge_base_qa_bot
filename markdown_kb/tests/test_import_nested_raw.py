"""Tests for nested raw/ glob with FilenameCollision — Slice 7-2.

AC coverage (issue #91 — Slice 7-2):
  - Nested raw/2024/policy.html + raw/2025/policy.html → both target docs/policy.md
  - First one succeeds, second fails as FilenameCollision
  - HTTP 200 returned regardless
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log path into importer.py for isolation."""
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
    return {
        "client": client,
        "raw_dir": raw_dir,
        "docs_dir": docs_dir,
    }


def test_nested_same_basename_collision(import_env):
    """raw/2024/policy.html + raw/2025/policy.html → second fails as FilenameCollision."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    # Create nested directories
    (raw_dir / "2024").mkdir()
    (raw_dir / "2025").mkdir()

    (raw_dir / "2024" / "policy.html").write_text(
        "<h1>Policy 2024</h1><p>2024 content.</p>", encoding="utf-8"
    )
    (raw_dir / "2025" / "policy.html").write_text(
        "<h1>Policy 2025</h1><p>2025 content.</p>", encoding="utf-8"
    )

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    # Exactly one succeeds, one fails
    assert len(data["imported_sources"]) == 1, (
        f"Expected 1 imported, got: {data['imported_sources']}"
    )
    assert len(data["failed_sources"]) == 1, f"Expected 1 failed, got: {data['failed_sources']}"
    assert data["failed_sources"][0]["error_type"] == "FilenameCollision", (
        f"Expected FilenameCollision, got: {data['failed_sources'][0]['error_type']}"
    )

    # docs/policy.md exists (from the winner)
    assert (docs_dir / "policy.md").exists()


def test_nested_no_collision_different_basenames(import_env):
    """Nested files with different basenames succeed independently."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "2024").mkdir()
    (raw_dir / "2025").mkdir()

    (raw_dir / "2024" / "policy2024.html").write_text(
        "<h1>Policy 2024</h1><p>Content.</p>", encoding="utf-8"
    )
    (raw_dir / "2025" / "policy2025.html").write_text(
        "<h1>Policy 2025</h1><p>Content.</p>", encoding="utf-8"
    )

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 2
    assert data["failed_sources"] == []


def test_nested_three_level_collision(import_env):
    """Three levels of nesting, same basename — first wins."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "a" / "b").mkdir(parents=True)
    (raw_dir / "c").mkdir()

    (raw_dir / "a" / "b" / "faq.html").write_text("<h1>FAQ</h1><p>Level AB.</p>", encoding="utf-8")
    (raw_dir / "c" / "faq.html").write_text("<h1>FAQ</h1><p>Level C.</p>", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "FilenameCollision"
