"""Deep module per Ousterhout. Public surface: normalize_kangxi_radicals, KANGXI_RADICAL_MAP.

Kangxi-radical codepoint normalization (issue #425 / PRD #424, amending
ADR-0031). Real-artifact evidence (issue #419) showed that designed PDFs with
subsetted fonts sometimes emit CJK ideographs as their visually-identical
Kangxi-radical codepoints -- e.g. U+2F6C "⽬" instead of U+76EE "目" -- which
the CJK bigram tokenizer (ADR-0014) treats as a literally different
character, so retrieval silently misses those tokens.

This module provides a deterministic fix scoped to exactly two contiguous
Unicode blocks -- CJK Radicals Supplement (U+2E80-2EFF) and Kangxi Radicals
(U+2F00-2FDF) -- mapped to their corresponding CJK Unified Ideographs. A
blanket ``unicodedata.normalize("NFKC", text)`` pass was considered and
rejected: it would also rewrite fullwidth forms and compatibility ligatures
elsewhere in the text, which must not change. The two blocks above are the
entirety of the scope; every other character, including ordinary CJK
ideographs, is guaranteed untouched.

Applied by the mechanical PDF Import converter
(``importer._convert_pdf_to_markdown``) on its extracted body, and intended
for reuse by the future Transcribe converter (PRD #424) as defense in depth
against the same contamination class surfacing in model output.
"""

from __future__ import annotations

import unicodedata

# ---------------------------------------------------------------------------
# Scoped Unicode block ranges (inclusive)
# ---------------------------------------------------------------------------
# CJK Radicals Supplement (U+2E80-2EFF) and Kangxi Radicals (U+2F00-2FDF) are
# adjacent Unicode blocks (0x2EFF + 1 == 0x2F00), so one contiguous range
# below covers exactly their union with nothing else.
_CJK_RADICALS_SUPPLEMENT_START = 0x2E80
_KANGXI_RADICALS_END = 0x2FDF


def _build_radical_map() -> dict[str, str]:
    """Build the Kangxi-radical -> CJK Unified Ideograph lookup table.

    Computed once from Unicode's own compatibility-decomposition data
    (``unicodedata.normalize("NFKC", ...)``), one codepoint at a time and
    scoped to the block range above -- never a blanket NFKC pass over
    arbitrary text (see module docstring). A codepoint is included only if
    its NFKC form differs from itself; most of the CJK Radicals Supplement
    has no recorded compatibility decomposition (those are historical
    variant forms with no single agreed unified ideograph) and is therefore
    correctly excluded -- an absent table entry means "pass through
    unchanged" at lookup time. Every included codepoint's NFKC form is a
    single character, never a multi-character expansion, so the table is a
    plain str -> str mapping safe to feed to ``str.maketrans``.
    """
    table: dict[str, str] = {}
    for codepoint in range(_CJK_RADICALS_SUPPLEMENT_START, _KANGXI_RADICALS_END + 1):
        ch = chr(codepoint)
        mapped = unicodedata.normalize("NFKC", ch)
        if mapped != ch:
            table[ch] = mapped
    return table


KANGXI_RADICAL_MAP: dict[str, str] = _build_radical_map()

_TRANSLATION_TABLE = str.maketrans(KANGXI_RADICAL_MAP)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_kangxi_radicals(text: str) -> str:
    """Replace Kangxi-radical codepoints with their corresponding CJK ideographs.

    Scoped to exactly ``KANGXI_RADICAL_MAP`` (CJK Radicals Supplement
    U+2E80-2EFF and Kangxi Radicals U+2F00-2FDF). Every other character --
    including ordinary CJK ideographs, fullwidth forms, and compatibility
    ligatures -- passes through unchanged; this is deliberately narrower than
    a blanket ``unicodedata.normalize("NFKC", text)`` call.
    """
    return text.translate(_TRANSLATION_TABLE)
