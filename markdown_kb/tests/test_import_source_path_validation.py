"""Tests for single-mode source path validation — Slice 7-2.

AC coverage (issue #91 — Slice 7-2):
  - InvalidSourcePath raised for absolute paths (/etc/foo.html, C:\\...)
  - InvalidSourcePath raised for '..' path traversal
  - InvalidSourcePath raised for 'raw/' prefix (require 'foo.html' not 'raw/foo.html')
  - Valid relative source paths are accepted
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


def _assert_invalid_source_path(data: dict) -> None:
    """Assert exactly one InvalidSourcePath failure in the response."""
    assert data["imported_sources"] == []
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "InvalidSourcePath", (
        f"Expected InvalidSourcePath, got: {data['failed_sources'][0]['error_type']}"
    )
    assert len(data["failed_sources"][0]["error_message"]) <= 200


# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------


def test_rejects_unix_absolute_path(import_env):
    """Unix absolute path /etc/passwd → InvalidSourcePath."""
    client = import_env["client"]
    resp = client.post("/import", json={"source": "/etc/passwd"})
    assert resp.status_code == 200
    _assert_invalid_source_path(resp.json())


def test_rejects_unix_absolute_path_etc_foo(import_env):
    """Unix absolute path /etc/foo.html → InvalidSourcePath."""
    client = import_env["client"]
    resp = client.post("/import", json={"source": "/etc/foo.html"})
    assert resp.status_code == 200
    _assert_invalid_source_path(resp.json())


def test_rejects_windows_absolute_path(import_env):
    """Windows-style absolute path (C:\\...) is rejected."""
    import app.importer as importer_module

    failure = importer_module._validate_source_path("C:\\Users\\secret.html")
    assert failure is not None
    assert failure.error_type == "InvalidSourcePath"


# ---------------------------------------------------------------------------
# Path traversal (..)
# ---------------------------------------------------------------------------


def test_rejects_double_dot_traversal(import_env):
    """Path with '..' traversal → InvalidSourcePath."""
    client = import_env["client"]
    resp = client.post("/import", json={"source": "../secret.html"})
    assert resp.status_code == 200
    _assert_invalid_source_path(resp.json())


def test_rejects_nested_double_dot_traversal(import_env):
    """Nested '..' traversal → InvalidSourcePath."""
    import app.importer as importer_module

    failure = importer_module._validate_source_path("sub/../../etc/passwd")
    assert failure is not None
    assert failure.error_type == "InvalidSourcePath"


def test_rejects_double_dot_in_middle(import_env):
    """'..' in the middle of path → InvalidSourcePath."""
    import app.importer as importer_module

    failure = importer_module._validate_source_path("sub/../../../etc/foo.html")
    assert failure is not None
    assert failure.error_type == "InvalidSourcePath"


# ---------------------------------------------------------------------------
# raw/ prefix
# ---------------------------------------------------------------------------


def test_rejects_raw_prefix(import_env):
    """Source 'raw/foo.html' with 'raw/' prefix → InvalidSourcePath."""
    client = import_env["client"]
    resp = client.post("/import", json={"source": "raw/foo.html"})
    assert resp.status_code == 200
    _assert_invalid_source_path(resp.json())


def test_rejects_raw_prefix_case_insensitive(import_env):
    """'RAW/foo.html' prefix is also rejected."""
    import app.importer as importer_module

    failure = importer_module._validate_source_path("RAW/foo.html")
    assert failure is not None
    assert failure.error_type == "InvalidSourcePath"


# ---------------------------------------------------------------------------
# Valid paths accepted
# ---------------------------------------------------------------------------


def test_accepts_simple_filename(import_env):
    """'foo.html' is a valid single-mode source path."""
    import app.importer as importer_module

    failure = importer_module._validate_source_path("foo.html")
    assert failure is None, f"Expected None for simple filename, got {failure}"


def test_accepts_subdirectory_path(import_env):
    """'2024/report.html' (nested under raw/) is a valid path."""
    import app.importer as importer_module

    failure = importer_module._validate_source_path("2024/report.html")
    assert failure is None, f"Expected None for subdirectory path, got {failure}"


def test_accepts_nested_subdirectory(import_env):
    """'2024/q3/policy.html' (nested) is a valid path."""
    import app.importer as importer_module

    failure = importer_module._validate_source_path("2024/q3/policy.html")
    assert failure is None, f"Expected None for nested path, got {failure}"
