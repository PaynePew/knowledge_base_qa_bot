"""Anti-drift guard: CODING_STANDARD.md must contain no `.py` references.

Per ADR-0007: CODING_STANDARD is reviewer-only abstract rules.  Module names
are banned to prevent the drift class observed in the Phase 3 → 4 transition.
If you need to anchor a rule to a code site, put the anchor in the relevant
ADR's § Consequences instead.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODING_STANDARD = REPO / "project-docs" / "CODING_STANDARD.md"


def test_coding_standard_has_no_py_module_references():
    text = CODING_STANDARD.read_text(encoding="utf-8")
    assert ".py" not in text, (
        "CODING_STANDARD.md must not reference any `.py` module per ADR-0007. "
        "If you need to anchor a rule to a code site, put it in the relevant "
        "ADR's § Consequences instead."
    )
