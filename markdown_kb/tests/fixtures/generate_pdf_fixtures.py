#!/usr/bin/env python3
"""Regenerate the committed PDF Import fixtures (issue #415/#416 / ADR-0031).

Dev-only tooling (CODING_STANDARD §7.2) — requires ``reportlab`` (a dev
dependency of the ``markdown-kb`` member only; never imported by app code).
Not itself a test: pytest does not collect this file (no ``test_`` prefix).

Usage:
    uv run --package markdown-kb python markdown_kb/tests/fixtures/generate_pdf_fixtures.py

Regenerates five KB-scale PDFs under ``raw_import/``:

- ``sample_english.pdf`` — an H1/H2 heading hierarchy plus a borderless table,
  so both the MarkItDown extractor's table heuristic and its plain-text
  passthrough of heading lines fire.
- ``sample_cjk.pdf`` — Traditional Chinese body text drawn via a registered
  CID font (STSong-Light), proving the text layer round-trips CJK losslessly.
- ``image_only.pdf`` — issue #416 (Slice 2): a single embedded raster image
  and no drawn text, proxying a scanned/image-only PDF. MarkItDown's
  extraction yields an empty string here, which the importer maps to the
  typed ``NoTextLayer`` failure.
- ``encrypted.pdf`` — issue #416 (Slice 2): password-protected via
  ``reportlab``'s ``StandardEncryption``. MarkItDown's PDF converter (via
  pdfminer.six) raises ``PDFPasswordIncorrect``/``PDFEncryptionError`` on
  open, which the importer maps to the typed ``EncryptedPdf`` failure.
- ``corrupt.pdf`` — issue #416 (Slice 2): a valid small PDF truncated to half
  its byte length, cutting off the xref table/trailer. MarkItDown raises a
  parse exception (e.g. ``PSEOF``), which the importer maps to the typed
  ``PdfExtractionError`` failure.

Headings are literal ``#`` / ``##`` characters drawn as ordinary page text.
MarkItDown's PDF converter does no font-size-based heading inference (plain
text-layer extraction only — pdfplumber + pdfminer.six, ADR-0031); the ``#``
markers must already be present in the extracted text for
``indexer.parse_markdown``'s ``HEADING_RE`` to recognise them. This mirrors
the existing ``.txt``/``.md`` passthrough precedent: Import performs no
heading inference of its own.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

_FIXTURES_DIR = Path(__file__).resolve().parent / "raw_import"

# Line-by-line body content for the English fixture. Blank strings advance the
# cursor without drawing (visual paragraph spacing only — HEADING_RE matches
# per-line, so blank lines have no effect on Section splitting).
_ENGLISH_LINES = [
    "# Getting Started",
    "",
    "This is the introduction paragraph explaining the product.",
    "It has more than one line of body text under the H1.",
    "",
    "## Installation",
    "",
    "Follow these steps to install the product on your machine.",
    "",
    "## Pricing Table",
    "",
]

_ENGLISH_TABLE = [
    ["Plan", "Price", "Seats"],
    ["Basic", "$9", "1"],
    ["Team", "$29", "5"],
    ["Enterprise", "$99", "50"],
]
_ENGLISH_TABLE_COLUMN_X = [72, 200, 300]

_CJK_LINES = [
    "# 退款政策",
    "如果您在購買後七天內申請退款，我們將全額退還費用。",
    "退款將於五個工作天內處理完成。",
    "## 常見問題",
    "如有任何問題，請聯絡客服人員協助處理。",
]


def _write_english_fixture(path: Path) -> None:
    """Write the English H1/H2 + borderless-table fixture."""
    c = canvas.Canvas(str(path))
    y = 750
    for text in _ENGLISH_LINES:
        if text:
            c.setFont("Helvetica", 11)
            c.drawString(72, y, text)
        y -= 16
    for row in _ENGLISH_TABLE:
        c.setFont("Helvetica", 10)
        for x, cell in zip(_ENGLISH_TABLE_COLUMN_X, row, strict=True):
            c.drawString(x, y, cell)
        y -= 16
    c.showPage()
    c.save()


def _write_cjk_fixture(path: Path) -> None:
    """Write the Traditional Chinese fixture via a registered CID font."""
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    c = canvas.Canvas(str(path))
    y = 750
    for text in _CJK_LINES:
        c.setFont("STSong-Light", 12)
        c.drawString(72, y, text)
        y -= 20
    c.showPage()
    c.save()


def _write_image_only_fixture(path: Path) -> None:
    """Write a scanned-PDF proxy: one embedded raster image, no drawn text.

    MarkItDown's PDF converter extracts no text from this page (nothing but
    an image XObject), so the importer's NoTextLayer detector fires — the
    same contract as a real scanned/image-only PDF.
    """
    image = Image.new("RGB", (400, 200), color=(230, 230, 230))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    c = canvas.Canvas(str(path))
    c.drawImage(ImageReader(buf), 72, 600, width=400, height=200)
    c.showPage()
    c.save()


def _write_encrypted_fixture(path: Path) -> None:
    """Write a password-protected PDF (RC4/AES via reportlab's StandardEncryption).

    MarkItDown's PDF converter (pdfminer.six underneath) cannot open this
    without the password, so it raises PDFPasswordIncorrect/PDFEncryptionError —
    the importer maps that specifically to the typed EncryptedPdf failure.
    """
    encryption = StandardEncryption("kb-user-pw", ownerPassword="kb-owner-pw", canPrint=1)
    c = canvas.Canvas(str(path), encrypt=encryption)
    c.drawString(72, 750, "This content is password-protected.")
    c.showPage()
    c.save()


def _write_corrupt_fixture(path: Path) -> None:
    """Write a corrupt/truncated PDF: a valid small PDF cut to half its bytes.

    Removing the tail (xref table + trailer) leaves a structurally invalid
    PDF that MarkItDown's extractors fail to parse (e.g. pdfminer's PSEOF),
    which the importer maps to the generic typed PdfExtractionError failure.
    """
    c = canvas.Canvas(str(path))
    c.drawString(72, 750, "This PDF will be truncated after generation.")
    c.showPage()
    c.save()

    full_bytes = path.read_bytes()
    path.write_bytes(full_bytes[: len(full_bytes) // 2])


def main() -> None:
    """Regenerate all five committed PDF fixtures in place."""
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "english": _FIXTURES_DIR / "sample_english.pdf",
        "cjk": _FIXTURES_DIR / "sample_cjk.pdf",
        "image_only": _FIXTURES_DIR / "image_only.pdf",
        "encrypted": _FIXTURES_DIR / "encrypted.pdf",
        "corrupt": _FIXTURES_DIR / "corrupt.pdf",
    }

    _write_english_fixture(paths["english"])
    _write_cjk_fixture(paths["cjk"])
    _write_image_only_fixture(paths["image_only"])
    _write_encrypted_fixture(paths["encrypted"])
    _write_corrupt_fixture(paths["corrupt"])

    for p in paths.values():
        print(f"Wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
