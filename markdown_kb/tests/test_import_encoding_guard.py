"""Encoding guard tests — Phase 16 Slice 3 (issue #167).

AC: a non-UTF-8 (Big5/cp950) raw file fed to Import fails loud with a per-file
``UnicodeDecodeError``; other files in the same batch are unaffected.

This documents the already-correct fail-loud behaviour as a regression guard.
Auto-detection / transcoding is explicitly out of scope — the test asserts that
non-UTF-8 bytes are REJECTED with a clean error, not silently decoded.

Big5/cp950 bytes are generated in-test via ``"退款政策".encode("big5")`` —
no machine-specific paths, no pre-baked fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest


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


# ---------------------------------------------------------------------------
# Batch mode: Big5 file fails loud; sibling files proceed normally
# ---------------------------------------------------------------------------


def test_batch_big5_file_fails_with_unicode_decode_error(import_env):
    """A Big5-encoded .txt file in a batch fails with UnicodeDecodeError.

    ``"退款政策".encode("big5")`` produces bytes that are not valid UTF-8
    (e.g. 0xB0, 0xCB, 0xA4, 0x7A — all high-byte pairs in Big5).
    The importer must surface this as a typed UnicodeDecodeError, not as
    silent mojibake or an ImportSourceResult.
    """
    raw_dir = import_env["raw_dir"]

    big5_bytes = "退款政策".encode("big5")
    (raw_dir / "big5_content.txt").write_bytes(big5_bytes)

    client = import_env["client"]
    resp = client.post("/import", json={"source": "big5_content.txt"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["imported_sources"] == [], (
        "A Big5 file must not appear in imported_sources — it must be rejected"
    )
    assert len(data["failed_sources"]) == 1
    failure = data["failed_sources"][0]
    assert failure["error_type"] == "UnicodeDecodeError", (
        f"Expected UnicodeDecodeError, got {failure['error_type']!r}"
    )
    assert len(failure["error_message"]) <= 200


def test_batch_big5_failure_does_not_affect_sibling_files(import_env):
    """Batch: one Big5 file fails; sibling UTF-8 files are imported normally.

    This is the core regression guard — continue-on-error semantics must hold
    for encoding failures exactly as they do for EmptySource / OversizedSource.
    """
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    # Good UTF-8 siblings
    (raw_dir / "before.txt").write_text("Good content before the bad file.", encoding="utf-8")
    (raw_dir / "after.html").write_text("<h1>After</h1><p>Still good.</p>", encoding="utf-8")

    # Bad Big5 file in the middle
    big5_bytes = "退款政策".encode("big5")
    (raw_dir / "bad_encoding.txt").write_bytes(big5_bytes)

    client = import_env["client"]
    resp = client.post("/import")  # batch mode
    assert resp.status_code == 200

    data = resp.json()
    # Both UTF-8 siblings must succeed
    assert len(data["imported_sources"]) == 2, (
        f"Expected 2 imported sources (the two UTF-8 siblings), "
        f"got {len(data['imported_sources'])}: {data['imported_sources']}"
    )
    # Exactly one failure for the Big5 file
    assert len(data["failed_sources"]) == 1, (
        f"Expected exactly 1 failed source, got {len(data['failed_sources'])}: "
        f"{data['failed_sources']}"
    )
    failure = data["failed_sources"][0]
    assert failure["error_type"] == "UnicodeDecodeError"
    assert "bad_encoding.txt" in failure["raw_path"]

    # The successfully imported docs files must exist on disk
    assert (docs_dir / "before.md").exists(), "before.md must exist after successful import"
    assert (docs_dir / "after.md").exists(), "after.md must exist after successful import"


def test_batch_cp950_bytes_also_fail_loud(import_env):
    """cp950 (Windows Big5 superset) bytes also raise UnicodeDecodeError.

    cp950 is the Windows code page closest to Big5; some Windows-authored
    Chinese files arrive in this encoding. The importer must reject them
    with the same typed error — no silent transcoding.
    """
    raw_dir = import_env["raw_dir"]

    # cp950 bytes for a Traditional Chinese phrase not representable in UTF-8 as-is
    cp950_bytes = "退款政策".encode("cp950")
    (raw_dir / "cp950_content.html").write_bytes(cp950_bytes)

    client = import_env["client"]
    resp = client.post("/import", json={"source": "cp950_content.html"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["imported_sources"] == []
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "UnicodeDecodeError"
