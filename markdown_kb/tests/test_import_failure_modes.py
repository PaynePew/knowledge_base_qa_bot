"""Integration tests for POST /import — Slice 7-2 full 12 failure modes.

AC coverage (issue #91 — Slice 7-2):
  - All 12 typed error_type values reachable from corresponding raw input cases
  - One test per error_type per AC requirement
  - failed_sources entries carry typed error_type (string) and truncated error_message (<=200)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"

LOG_LINE_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] (\S+) \| (.+)$")


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
        "log_path": tmp_path / "wiki" / "log.md",
    }


def _assert_failure(data: dict, expected_error_type: str) -> dict:
    """Assert a single failure entry with the given error_type and return it."""
    assert data["imported_sources"] == [], "No sources should be imported on failure"
    assert len(data["failed_sources"]) == 1, (
        f"Expected 1 failed_sources entry, got: {data['failed_sources']}"
    )
    failure = data["failed_sources"][0]
    assert failure["error_type"] == expected_error_type, (
        f"Expected error_type={expected_error_type!r}, got {failure['error_type']!r}"
    )
    assert len(failure["error_message"]) <= 200, (
        f"error_message must be <=200 chars, got {len(failure['error_message'])}"
    )
    return failure


# ---------------------------------------------------------------------------
# 1. HandAuthoredCollision
# ---------------------------------------------------------------------------


def test_failure_hand_authored_collision(import_env):
    """Docs target exists without imported_from frontmatter → HandAuthoredCollision."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    # Create raw source
    (raw_dir / "policy.html").write_text("<h1>Policy</h1><p>Content.</p>", encoding="utf-8")
    # Create docs file WITHOUT imported_from (hand-authored)
    (docs_dir / "policy.md").write_text("# Policy\n\nHand-authored content.\n", encoding="utf-8")

    resp = client.post("/import", json={"source": "policy.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "HandAuthoredCollision")


# ---------------------------------------------------------------------------
# 2. UnicodeDecodeError
# ---------------------------------------------------------------------------


def test_failure_unicode_decode_error(import_env):
    """Non-UTF-8 raw file → UnicodeDecodeError."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    # Write Big5 encoded content (not valid UTF-8)
    (raw_dir / "big5.html").write_bytes(b"\xa4\xa4\xa4\xe5")  # Big5 bytes

    resp = client.post("/import", json={"source": "big5.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "UnicodeDecodeError")


# ---------------------------------------------------------------------------
# 3. EmptySource
# ---------------------------------------------------------------------------


def test_failure_empty_source(import_env):
    """0-byte raw file → EmptySource (not silent skip)."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "empty.html").write_bytes(b"")

    resp = client.post("/import", json={"source": "empty.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "EmptySource")


# ---------------------------------------------------------------------------
# 4. OversizedSource
# ---------------------------------------------------------------------------


def test_failure_oversized_source(import_env, monkeypatch):
    """Raw file > 10 MB → OversizedSource before markdownify invocation."""
    import app.importer as importer_module

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    # Write a file that reports > 10 MB without actually allocating 10 MB
    html_content = "<h1>Large</h1>" + "x" * 100

    (raw_dir / "large.html").write_text(html_content, encoding="utf-8")

    # Monkeypatch _MAX_SOURCE_BYTES to a tiny value to trigger OversizedSource
    monkeypatch.setattr(importer_module, "_MAX_SOURCE_BYTES", 10)

    resp = client.post("/import", json={"source": "large.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "OversizedSource")


# ---------------------------------------------------------------------------
# 5. UnsupportedExtension (single mode)
# ---------------------------------------------------------------------------


def test_failure_unsupported_extension_single_mode(import_env):
    """Single-mode source with unsupported extension (.pdf) → UnsupportedExtension."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "document.pdf").write_bytes(b"%PDF-1.4 fake content")

    resp = client.post("/import", json={"source": "document.pdf"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "UnsupportedExtension")


# ---------------------------------------------------------------------------
# 6. FileNotFoundError
# ---------------------------------------------------------------------------


def test_failure_file_not_found(import_env):
    """Single-mode source not in raw/ → FileNotFoundError."""
    client = import_env["client"]

    resp = client.post("/import", json={"source": "nonexistent.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "FileNotFoundError")


# ---------------------------------------------------------------------------
# 7. MarkdownifyError
# ---------------------------------------------------------------------------


def test_failure_markdownify_error(import_env, monkeypatch):
    """markdownify raises an exception → MarkdownifyError (not re-raised)."""
    import app.importer as importer_module

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "broken.html").write_text("<h1>Broken</h1>", encoding="utf-8")

    # Patch _convert_to_markdown to raise
    def raise_markdownify(*args, **kwargs):
        raise RuntimeError("markdownify internal error")

    monkeypatch.setattr(importer_module, "_convert_to_markdown", raise_markdownify)

    resp = client.post("/import", json={"source": "broken.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "MarkdownifyError")


# ---------------------------------------------------------------------------
# 8. IOError
# ---------------------------------------------------------------------------


def test_failure_io_error(import_env, monkeypatch):
    """Atomic-write os.replace failure → IOError."""
    import app.importer as importer_module

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "ioerr.html").write_text("<h1>IO</h1><p>Content.</p>", encoding="utf-8")

    def raise_io(*args, **kwargs):
        raise OSError("Simulated disk full")

    monkeypatch.setattr(importer_module, "_atomic_write", raise_io)

    resp = client.post("/import", json={"source": "ioerr.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "IOError")


# ---------------------------------------------------------------------------
# 9. InvalidFilename
# ---------------------------------------------------------------------------


def test_failure_invalid_filename(import_env):
    """Basename containing '#' → InvalidFilename (single mode via source path check)."""
    client = import_env["client"]

    resp = client.post("/import", json={"source": "bad#name.html"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "InvalidFilename")


# ---------------------------------------------------------------------------
# 10. InvalidSourcePath
# ---------------------------------------------------------------------------


def test_failure_invalid_source_path(import_env):
    """Single-mode source with absolute path → InvalidSourcePath."""
    client = import_env["client"]

    resp = client.post("/import", json={"source": "/etc/passwd"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "InvalidSourcePath")


# ---------------------------------------------------------------------------
# 11. FilenameCollision
# ---------------------------------------------------------------------------


def test_failure_filename_collision(import_env):
    """Two batch files map to same docs/<stem>.md → second fails as FilenameCollision."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    # Both produce docs/article.md
    (raw_dir / "article.html").write_text("<h1>Article HTML</h1><p>Content.</p>", encoding="utf-8")
    (raw_dir / "article.txt").write_text("Article TXT content.", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    # One succeeds, one fails
    assert len(data["imported_sources"]) == 1
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "FilenameCollision"


# ---------------------------------------------------------------------------
# 12. UnsupportedExtension silently skipped in batch mode
# ---------------------------------------------------------------------------


def test_unsupported_extension_batch_mode_silent_skip(import_env):
    """Batch mode silently skips .pdf files (no failed_sources entry)."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "document.pdf").write_bytes(b"%PDF-1.4 fake content")
    (raw_dir / "good.html").write_text("<h1>Good</h1><p>Content.</p>", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1, "Only the .html should be imported"
    assert data["failed_sources"] == [], "Batch mode must silently skip unsupported extensions"


# ---------------------------------------------------------------------------
# Verify error_message truncation to 200 chars
# ---------------------------------------------------------------------------


def test_failure_error_message_truncation(import_env, monkeypatch):
    """error_message is truncated to <=200 chars regardless of exception length."""
    import app.importer as importer_module

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "trunctest.html").write_text("<h1>Test</h1>", encoding="utf-8")

    long_message = "X" * 500

    def raise_long(*args, **kwargs):
        raise RuntimeError(long_message)

    monkeypatch.setattr(importer_module, "_convert_to_markdown", raise_long)

    resp = client.post("/import", json={"source": "trunctest.html"})
    assert resp.status_code == 200

    failure = resp.json()["failed_sources"][0]
    assert len(failure["error_message"]) <= 200, (
        f"error_message must be <=200 chars, got {len(failure['error_message'])}"
    )
