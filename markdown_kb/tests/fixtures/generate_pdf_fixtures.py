#!/usr/bin/env python3
"""Regenerate the committed PDF Import fixtures (issue #415 / ADR-0031).

Dev-only tooling (CODING_STANDARD §7.2) — requires ``reportlab`` (a dev
dependency of the ``markdown-kb`` member only; never imported by app code).
Not itself a test: pytest does not collect this file (no ``test_`` prefix).

Usage:
    uv run --package markdown-kb python markdown_kb/tests/fixtures/generate_pdf_fixtures.py

Regenerates two KB-scale, digital-native PDFs under ``raw_import/``:

- ``sample_english.pdf`` — an H1/H2 heading hierarchy plus a borderless table,
  so both the MarkItDown extractor's table heuristic and its plain-text
  passthrough of heading lines fire.
- ``sample_cjk.pdf`` — Traditional Chinese body text drawn via a registered
  CID font (STSong-Light), proving the text layer round-trips CJK losslessly.

Headings are literal ``#`` / ``##`` characters drawn as ordinary page text.
MarkItDown's PDF converter does no font-size-based heading inference (plain
text-layer extraction only — pdfplumber + pdfminer.six, ADR-0031); the ``#``
markers must already be present in the extracted text for
``indexer.parse_markdown``'s ``HEADING_RE`` to recognise them. This mirrors
the existing ``.txt``/``.md`` passthrough precedent: Import performs no
heading inference of its own.
"""

from __future__ import annotations

from pathlib import Path

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


def main() -> None:
    """Regenerate both committed PDF fixtures in place."""
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    english_path = _FIXTURES_DIR / "sample_english.pdf"
    cjk_path = _FIXTURES_DIR / "sample_cjk.pdf"

    _write_english_fixture(english_path)
    _write_cjk_fixture(cjk_path)

    print(f"Wrote {english_path} ({english_path.stat().st_size} bytes)")
    print(f"Wrote {cjk_path} ({cjk_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
