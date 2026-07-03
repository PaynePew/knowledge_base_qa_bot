"""Integration tests for POST /import — PDF-specific failure modes (issue #415 / ADR-0031).

AC coverage:
  - An empty-extraction PDF (no text layer — scanned/image-only) fails typed
    ``NoTextLayer`` with curator OCR guidance.
  - An extractor crash fails typed ``PdfExtractionError`` with a truncated
    (<=200 char) message.
  - Neither failure aborts a batch (continue-on-error, matching the existing
    12 failure modes in test_import_failure_modes.py).

``EncryptedPdf`` is explicitly deferred to Slice 2 (issue #415 scope) and is
not tested here.
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
    return {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir}


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
# NoTextLayer — empty/whitespace extraction (scanned/image-only PDF proxy)
# ---------------------------------------------------------------------------


def test_no_text_layer_fails_typed(import_env):
    """A blank (no text layer) PDF fails typed NoTextLayer, not a silent empty Source."""
    from reportlab.pdfgen import canvas

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    blank_pdf = raw_dir / "scanned.pdf"
    c = canvas.Canvas(str(blank_pdf))
    c.showPage()
    c.save()

    resp = client.post("/import", json={"source": "scanned.pdf"})
    assert resp.status_code == 200
    failure = _assert_failure(resp.json(), "NoTextLayer")

    assert "ocr" in failure["error_message"].lower(), (
        f"NoTextLayer message must guide the curator to OCR externally, got: {failure}"
    )
    assert not (docs_dir / "scanned.md").exists(), "No docs/ Source must be written on NoTextLayer"


# ---------------------------------------------------------------------------
# PdfExtractionError — extractor internal exception (corrupt PDF)
# ---------------------------------------------------------------------------


def test_pdf_extraction_error_fails_typed(import_env):
    """A corrupt PDF (valid magic header, malformed body) fails typed PdfExtractionError."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    corrupt = raw_dir / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\ncorrupt garbage, not a real xref table" + bytes(range(200)))

    resp = client.post("/import", json={"source": "corrupt.pdf"})
    assert resp.status_code == 200
    _assert_failure(resp.json(), "PdfExtractionError")

    assert not (docs_dir / "corrupt.md").exists(), (
        "No docs/ Source must be written on PdfExtractionError"
    )


def test_pdf_extraction_error_message_truncated(import_env, monkeypatch):
    """error_message stays <=200 chars even when the underlying exception is long."""
    import app.importer as importer_module

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "trunctest.pdf").write_bytes(b"%PDF-1.4 fake")

    long_message = "X" * 500

    def raise_long(*args, **kwargs):
        raise RuntimeError(long_message)

    monkeypatch.setattr(importer_module, "_convert_pdf_to_markdown", raise_long)

    resp = client.post("/import", json={"source": "trunctest.pdf"})
    assert resp.status_code == 200
    failure = _assert_failure(resp.json(), "PdfExtractionError")
    assert failure["error_message"] == "X" * 200


# ---------------------------------------------------------------------------
# Continue-on-error: neither PDF failure mode aborts a batch
# ---------------------------------------------------------------------------


def test_pdf_failures_do_not_abort_batch(import_env):
    """A NoTextLayer PDF and a PdfExtractionError PDF alongside a good file: batch continues."""
    from reportlab.pdfgen import canvas

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    blank_pdf = raw_dir / "blank.pdf"
    c = canvas.Canvas(str(blank_pdf))
    c.showPage()
    c.save()

    (raw_dir / "corrupt.pdf").write_bytes(b"%PDF-1.4\ncorrupt" + bytes(range(200)))
    (raw_dir / "good.html").write_text("<h1>Good</h1><p>Content.</p>", encoding="utf-8")

    resp = client.post("/import")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["imported_sources"]) == 1, "The good .html must still import"
    assert data["imported_sources"][0]["original_format"] == "html"

    failed_types = {f["error_type"] for f in data["failed_sources"]}
    assert failed_types == {"NoTextLayer", "PdfExtractionError"}
