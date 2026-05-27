"""Tests for filename validation — Slice 7-2.

AC coverage (issue #91 — Slice 7-2):
  - InvalidFilename raised for basename containing '#'
  - InvalidFilename raised for basename containing '/'
  - InvalidFilename raised for basename containing '\\'
  - InvalidFilename raised for basename containing ':'
  - InvalidFilename raised for basename containing control chars (\\x00-\\x1f)
  - InvalidFilename raised for basename containing bidi control chars
    U+202A-E, U+2066-9 (CVE-2021-42574 Trojan Source)
  - Valid filenames pass through without error
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


def _assert_invalid_filename(data: dict) -> None:
    """Assert that the response contains exactly one InvalidFilename failure."""
    assert data["imported_sources"] == []
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "InvalidFilename", (
        f"Expected InvalidFilename, got: {data['failed_sources'][0]['error_type']}"
    )
    assert len(data["failed_sources"][0]["error_message"]) <= 200


# ---------------------------------------------------------------------------
# Hash character (#)
# ---------------------------------------------------------------------------


def test_filename_rejects_hash(import_env):
    """Basename containing '#' is rejected with InvalidFilename."""
    client = import_env["client"]
    resp = client.post("/import", json={"source": "bad#name.html"})
    assert resp.status_code == 200
    _assert_invalid_filename(resp.json())


# ---------------------------------------------------------------------------
# Forward slash (/)
# ---------------------------------------------------------------------------


def test_filename_rejects_forward_slash(import_env):
    """Basename containing '/' is rejected with InvalidFilename."""
    import app.importer as importer_module

    # _validate_filename directly since '/' in path would be interpreted as path sep
    failure = importer_module._validate_filename("bad/name.html", "raw/bad/name.html")
    assert failure is not None
    assert failure.error_type == "InvalidFilename"
    assert "/" in failure.error_message or "InvalidFilename" in failure.error_type


# ---------------------------------------------------------------------------
# Backslash (\)
# ---------------------------------------------------------------------------


def test_filename_rejects_backslash(import_env):
    """Basename containing '\\' is rejected with InvalidFilename."""
    import app.importer as importer_module

    failure = importer_module._validate_filename("bad\\name.html", "raw/bad\\name.html")
    assert failure is not None
    assert failure.error_type == "InvalidFilename"


# ---------------------------------------------------------------------------
# Colon (:)
# ---------------------------------------------------------------------------


def test_filename_rejects_colon(import_env):
    """Basename containing ':' is rejected with InvalidFilename."""
    client = import_env["client"]
    resp = client.post("/import", json={"source": "bad:name.html"})
    assert resp.status_code == 200
    _assert_invalid_filename(resp.json())


# ---------------------------------------------------------------------------
# Control characters (\x00-\x1f)
# ---------------------------------------------------------------------------


def test_filename_rejects_control_chars(import_env):
    """Basename containing control chars (\\x00-\\x1f) is rejected with InvalidFilename."""
    import app.importer as importer_module

    # Test a few control chars
    for ctrl in ["\x00", "\x01", "\x1f", "\t", "\n"]:
        failure = importer_module._validate_filename(
            f"bad{ctrl}name.html", f"raw/bad{ctrl}name.html"
        )
        assert failure is not None, f"Expected failure for control char {repr(ctrl)}"
        assert failure.error_type == "InvalidFilename", (
            f"Expected InvalidFilename for {repr(ctrl)}, got {failure.error_type}"
        )


# ---------------------------------------------------------------------------
# Bidi control chars (CVE-2021-42574 Trojan Source)
# ---------------------------------------------------------------------------


def test_filename_rejects_bidi_lre(import_env):
    """Basename containing LRE (U+202A) is rejected with InvalidFilename."""
    import app.importer as importer_module

    # U+202A LEFT-TO-RIGHT EMBEDDING
    failure = importer_module._validate_filename("bad‪name.html", "raw/bad‪name.html")
    assert failure is not None
    assert failure.error_type == "InvalidFilename"
    assert "202A" in failure.error_message or "bidi" in failure.error_message.lower()


def test_filename_rejects_bidi_rle(import_env):
    """Basename containing RLE (U+202B) is rejected with InvalidFilename."""
    import app.importer as importer_module

    failure = importer_module._validate_filename("bad‫name.html", "raw/bad‫name.html")
    assert failure is not None
    assert failure.error_type == "InvalidFilename"


def test_filename_rejects_bidi_rlo(import_env):
    """Basename containing RLO (U+202E) is rejected with InvalidFilename."""
    import app.importer as importer_module

    failure = importer_module._validate_filename("bad‮name.html", "raw/bad‮name.html")
    assert failure is not None
    assert failure.error_type == "InvalidFilename"


def test_filename_rejects_bidi_lri(import_env):
    """Basename containing LRI (U+2066) is rejected with InvalidFilename."""
    import app.importer as importer_module

    failure = importer_module._validate_filename("bad⁦name.html", "raw/bad⁦name.html")
    assert failure is not None
    assert failure.error_type == "InvalidFilename"


def test_filename_rejects_bidi_pdi(import_env):
    """Basename containing PDI (U+2069) is rejected with InvalidFilename."""
    import app.importer as importer_module

    failure = importer_module._validate_filename("bad⁩name.html", "raw/bad⁩name.html")
    assert failure is not None
    assert failure.error_type == "InvalidFilename"


# ---------------------------------------------------------------------------
# Valid filenames pass through
# ---------------------------------------------------------------------------


def test_filename_allows_valid_names(import_env):
    """Normal filenames without rejected chars pass validation without error."""
    import app.importer as importer_module

    valid_names = [
        "policy.html",
        "faq-2024.txt",
        "article_v2.html",
        "2024-report.html",
        "hello world.html",  # space is allowed
        "café.html",  # accented chars allowed
        "日本語.html",  # Unicode letters allowed
    ]
    for name in valid_names:
        failure = importer_module._validate_filename(name, f"raw/{name}")
        assert failure is None, f"Expected None for valid filename {repr(name)}, got {failure}"
