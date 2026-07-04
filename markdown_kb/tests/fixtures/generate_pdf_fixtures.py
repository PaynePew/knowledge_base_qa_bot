#!/usr/bin/env python3
"""Regenerate the committed PDF Import fixtures (issue #415/#416/#425 / ADR-0031).

Dev-only tooling (CODING_STANDARD §7.2) — requires ``reportlab`` (a dev
dependency of the ``markdown-kb`` member only; never imported by app code).
Not itself a test: pytest does not collect this file (no ``test_`` prefix).

Usage:
    uv run --package markdown-kb python markdown_kb/tests/fixtures/generate_pdf_fixtures.py

Regenerates six KB-scale PDFs under ``raw_import/``:

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
- ``kangxi_contamination.pdf`` — issue #425: plants real Kangxi-radical
  codepoints (U+2F6C, U+2F00, U+2F64, U+2F83) in the extracted text, proving
  the importer's post-processing step corrects them. Built by hand
  (``_write_kangxi_contamination_fixture``, no ``reportlab``): a real PDF
  generated via ``reportlab``'s ``UnicodeCIDFont`` cannot plant these
  codepoints at all — its Unicode-to-CID lookup for the CJK CID font already
  collapses Kangxi-radical input to the base ideograph before the byte
  stream is ever written (verified empirically), which is the opposite
  direction of the real-world font-subsetting bug this fixture exists to
  simulate. Instead this fixture hand-assembles a minimal single-page PDF
  with a simple (non-CID) font whose ``/ToUnicode`` CMap is fully
  hand-specified, so pdfminer's text extraction returns exactly the planted
  codepoints regardless of what the (irrelevant, never-rendered) glyphs are.

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


# ---------------------------------------------------------------------------
# kangxi_contamination.pdf (issue #425) — hand-assembled, no reportlab
# ---------------------------------------------------------------------------
# Contaminated heading + body line. Each contains real Kangxi-radical
# codepoints planted amongst ordinary CJK text:
#   ⽬ U+2F6C KANGXI RADICAL EYE       (corrected form: 目 U+76EE)
#   ⼀ U+2F00 KANGXI RADICAL ONE       (corrected form: 一 U+4E00)
#   ⽤ U+2F64 KANGXI RADICAL USE       (corrected form: 用 U+7528)
#   ⾃ U+2F83 KANGXI RADICAL SELF      (corrected form: 自 U+81EA)
_KANGXI_HEADING_LINE = "# 題⽬"
_KANGXI_BODY_LINE = "如果您需要⼀些⽤法或⾃我說明，請參閱本文件。"


def _assign_byte_codes(*lines: str) -> tuple[dict[str, int], list[bytes]]:
    """Assign each distinct character across ``lines`` a unique byte 1..255.

    Returns the char->code map (feeds the ToUnicode CMap) and, per line, the
    bytes encoding that line under the assignment (feeds the content stream).
    """
    char_to_code: dict[str, int] = {}
    next_code = 1
    encoded_lines: list[bytes] = []
    for line in lines:
        codes = bytearray()
        for ch in line:
            if ch not in char_to_code:
                char_to_code[ch] = next_code
                next_code += 1
            codes.append(char_to_code[ch])
        encoded_lines.append(bytes(codes))
    return char_to_code, encoded_lines


def _build_kangxi_pdf_bytes(*lines: str) -> bytes:
    """Hand-assemble a minimal single-page PDF planting ``lines`` verbatim.

    A real ``reportlab``-generated CJK PDF cannot plant Kangxi-radical
    codepoints (see the module docstring): its ``UnicodeCIDFont`` support
    silently collapses them to the base ideograph during Unicode-to-CID
    encoding, before any bytes reach the page. This function instead builds
    the PDF objects directly: a simple (non-CID) ``/Type1`` font whose
    ``/ToUnicode`` CMap is fully hand-specified, mapping one single-byte code
    per distinct character to its exact intended Unicode codepoint. pdfminer
    (via MarkItDown) trusts ``/ToUnicode`` for text extraction regardless of
    what the font's actual glyphs look like, so this reliably plants exactly
    the codepoints in ``lines`` — nothing else needs to be "real" (no glyph
    outlines are ever inspected).
    """
    char_to_code, encoded_lines = _assign_byte_codes(*lines)

    bfchar_entries = "\n".join(f"<{code:02X}> <{ord(ch):04X}>" for ch, code in char_to_code.items())
    cmap_body = (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "1 begincodespacerange\n"
        "<00> <FF>\n"
        "endcodespacerange\n"
        f"{len(char_to_code)} beginbfchar\n"
        f"{bfchar_entries}\n"
        "endbfchar\n"
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end\n"
    ).encode("ascii")

    text_ops = [f"72 {750 - 20 * i} Td\n<{enc.hex()}> Tj\n" for i, enc in enumerate(encoded_lines)]
    content_stream = ("BT\n/F1 12 Tf\n" + "".join(text_ops) + "ET\n").encode("ascii")

    n_codes = len(char_to_code)
    widths = " ".join("600" for _ in range(n_codes))
    font_dict = (
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        "/Encoding /WinAnsiEncoding /ToUnicode 6 0 R "
        f"/FirstChar 1 /LastChar {n_codes} /Widths [{widths}] >>"
    ).encode("ascii")

    def obj(n: int, body: bytes) -> bytes:
        return f"{n} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    objects = [
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
            b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        ),
        obj(4, font_dict),
        obj(
            5,
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
            + content_stream
            + b"endstream",
        ),
        obj(
            6,
            f"<< /Length {len(cmap_body)} >>\nstream\n".encode("ascii") + cmap_body + b"endstream",
        ),
    ]

    buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for o in objects:
        offsets.append(len(buf))
        buf.extend(o)

    xref_offset = len(buf)
    n_objs = len(objects) + 1
    buf.extend(f"xref\n0 {n_objs}\n".encode("ascii"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(
        f"trailer\n<< /Size {n_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode(
            "ascii"
        )
    )
    return bytes(buf)


def _write_kangxi_contamination_fixture(path: Path) -> None:
    """Write the hand-assembled PDF planting Kangxi-radical codepoints."""
    path.write_bytes(_build_kangxi_pdf_bytes(_KANGXI_HEADING_LINE, _KANGXI_BODY_LINE))


def main() -> None:
    """Regenerate all six committed PDF fixtures in place."""
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "english": _FIXTURES_DIR / "sample_english.pdf",
        "cjk": _FIXTURES_DIR / "sample_cjk.pdf",
        "image_only": _FIXTURES_DIR / "image_only.pdf",
        "encrypted": _FIXTURES_DIR / "encrypted.pdf",
        "corrupt": _FIXTURES_DIR / "corrupt.pdf",
        "kangxi_contamination": _FIXTURES_DIR / "kangxi_contamination.pdf",
    }

    _write_english_fixture(paths["english"])
    _write_cjk_fixture(paths["cjk"])
    _write_image_only_fixture(paths["image_only"])
    _write_encrypted_fixture(paths["encrypted"])
    _write_corrupt_fixture(paths["corrupt"])
    _write_kangxi_contamination_fixture(paths["kangxi_contamination"])

    for p in paths.values():
        print(f"Wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
