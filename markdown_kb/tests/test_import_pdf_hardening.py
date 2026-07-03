"""Integration tests for POST /import — PDF failure-mode hardening (issue #416 / ADR-0031).

AC coverage (issue #416 — Slice 2):
  - Encrypted fixture fails typed ``EncryptedPdf`` on every seam (batch,
    single, path-import); message names encryption and the next action.
  - Image-only fixture fails typed ``NoTextLayer`` with OCR-elsewhere
    guidance (complements test_import_pdf_failure_modes.py's inline-blank-PDF
    case with a committed, image-bearing fixture).
  - Corrupt/truncated-bytes fixture fails typed ``PdfExtractionError`` with a
    <=200-char message and no stack trace.
  - A mixed batch (1 good + all 3 failing fixtures) converts the good file,
    records three typed failures, and completes with correct counts.

Fixtures: markdown_kb/tests/fixtures/raw_import/{image_only,encrypted,corrupt}.pdf,
regenerable via markdown_kb/tests/fixtures/generate_pdf_fixtures.py (dev-only,
reportlab + Pillow).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

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


def _parse_log(log_path: Path) -> list[tuple[str, str]]:
    """Return list of (kind, summary) tuples from the log file."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    result = []
    for line in lines:
        m = LOG_LINE_RE.match(line)
        if m:
            result.append((m.group(2), m.group(3)))
    return result


# ---------------------------------------------------------------------------
# EncryptedPdf — password-protected PDF, every seam
# ---------------------------------------------------------------------------


def test_encrypted_pdf_fails_typed_batch(import_env):
    """A password-protected PDF fails typed EncryptedPdf in batch mode."""
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]
    client = import_env["client"]

    shutil.copy(FIXTURES / "encrypted.pdf", raw_dir / "encrypted.pdf")

    resp = client.post("/import")
    assert resp.status_code == 200
    _assert_failure(resp.json(), "EncryptedPdf")
    assert not (docs_dir / "encrypted.md").exists(), (
        "No docs/ Source must be written on EncryptedPdf"
    )


def test_encrypted_pdf_fails_typed_single_mode(import_env):
    """A password-protected PDF fails typed EncryptedPdf in single-source mode."""
    raw_dir = import_env["raw_dir"]
    client = import_env["client"]

    shutil.copy(FIXTURES / "encrypted.pdf", raw_dir / "encrypted.pdf")

    resp = client.post("/import", json={"source": "encrypted.pdf"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "EncryptedPdf")


def test_encrypted_pdf_message_names_encryption_and_next_action(import_env):
    """EncryptedPdf's error_message names the cause and the concrete next action."""
    raw_dir = import_env["raw_dir"]
    client = import_env["client"]

    shutil.copy(FIXTURES / "encrypted.pdf", raw_dir / "encrypted.pdf")

    resp = client.post("/import", json={"source": "encrypted.pdf"})
    failure = _assert_failure(resp.json(), "EncryptedPdf")

    message_lower = failure["error_message"].lower()
    assert "encrypt" in message_lower or "password" in message_lower, (
        f"EncryptedPdf message must name encryption as the cause, got: {failure}"
    )
    assert "decrypt" in message_lower, (
        f"EncryptedPdf message must direct the curator to supply a decrypted copy, got: {failure}"
    )


def test_encrypted_pdf_via_import_path(import_env, tmp_path):
    """import_path (the CLI/MCP shared seam) raises ImportPathError typed EncryptedPdf."""
    from app.importer import ImportPathError, import_path

    src = tmp_path / "encrypted.pdf"
    src.write_bytes((FIXTURES / "encrypted.pdf").read_bytes())

    with pytest.raises(ImportPathError) as exc_info:
        import_path(src)

    assert exc_info.value.error_type == "EncryptedPdf"
    assert len(exc_info.value.message) <= 200


# ---------------------------------------------------------------------------
# NoTextLayer — committed image-only fixture (scanned-PDF proxy)
# ---------------------------------------------------------------------------


def test_image_only_fixture_fails_typed_no_text_layer(import_env):
    """The committed image-only fixture (no drawn text) fails typed NoTextLayer."""
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]
    client = import_env["client"]

    shutil.copy(FIXTURES / "image_only.pdf", raw_dir / "image_only.pdf")

    resp = client.post("/import", json={"source": "image_only.pdf"})
    assert resp.status_code == 200
    failure = _assert_failure(resp.json(), "NoTextLayer")

    assert "ocr" in failure["error_message"].lower(), (
        f"NoTextLayer message must guide the curator to OCR externally, got: {failure}"
    )
    assert not (docs_dir / "image_only.md").exists(), (
        "No docs/ Source must be written on NoTextLayer"
    )


# ---------------------------------------------------------------------------
# PdfExtractionError — committed corrupt/truncated fixture
# ---------------------------------------------------------------------------


def test_corrupt_fixture_fails_typed_pdf_extraction_error(import_env):
    """The committed truncated-PDF fixture fails typed PdfExtractionError, no stack trace."""
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]
    client = import_env["client"]

    shutil.copy(FIXTURES / "corrupt.pdf", raw_dir / "corrupt.pdf")

    resp = client.post("/import", json={"source": "corrupt.pdf"})
    assert resp.status_code == 200
    failure = _assert_failure(resp.json(), "PdfExtractionError")

    assert "Traceback" not in failure["error_message"], (
        f"error_message must not leak a stack trace, got: {failure}"
    )
    assert not (docs_dir / "corrupt.md").exists(), (
        "No docs/ Source must be written on PdfExtractionError"
    )


# ---------------------------------------------------------------------------
# Mixed batch: 1 good + all 3 failing fixtures — continue-on-error proof
# ---------------------------------------------------------------------------


def test_mixed_batch_one_good_three_failing_completes(import_env):
    """1 good PDF + image_only/encrypted/corrupt fixtures: batch converts the good one,
    records three typed failures, and completes with correct counts."""
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]
    log_path = import_env["log_path"]
    client = import_env["client"]

    shutil.copy(FIXTURES / "sample_english.pdf", raw_dir / "sample_english.pdf")
    shutil.copy(FIXTURES / "image_only.pdf", raw_dir / "image_only.pdf")
    shutil.copy(FIXTURES / "encrypted.pdf", raw_dir / "encrypted.pdf")
    shutil.copy(FIXTURES / "corrupt.pdf", raw_dir / "corrupt.pdf")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1, "Only sample_english.pdf should import"
    assert data["imported_sources"][0]["original_format"] == "pdf"
    assert (docs_dir / "sample_english.md").exists()

    assert len(data["failed_sources"]) == 3
    failed_types = {f["error_type"] for f in data["failed_sources"]}
    assert failed_types == {"NoTextLayer", "EncryptedPdf", "PdfExtractionError"}
    for failure in data["failed_sources"]:
        assert len(failure["error_message"]) <= 200

    # Wiki Log event sequence: started -> 1 import_source + 3 import_error (any order) -> completed
    events = _parse_log(log_path)
    import_kinds = [k for k, _ in events if k.startswith("import_")]

    assert import_kinds[0] == "import_batch_started"
    assert import_kinds[-1] == "import_batch_completed"
    assert import_kinds.count("import_source") == 1
    assert import_kinds.count("import_error") == 3

    completed_summary = next(s for k, s in events if k == "import_batch_completed")
    assert "imported=1" in completed_summary
    assert "failed=3" in completed_summary
