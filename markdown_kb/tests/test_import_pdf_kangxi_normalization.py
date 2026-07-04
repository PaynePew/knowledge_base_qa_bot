"""Unit + integration tests for Kangxi-radical codepoint normalization (issue #425).

AC coverage (issue #425 / PRD #424):
  - Table-driven unit tests over representative mappings (Kangxi Radicals
    block: ⽬→目, ⼀→一, ⽤→用, ⾃→自; CJK Radicals Supplement block: two
    real cases) and non-mappings (fullwidth digit, a compatibility ligature,
    a normal CJK ideograph — all pass through unchanged).
  - End-to-end: the ``kangxi_contamination.pdf`` fixture (planted radical
    codepoints) imports to a docs Source containing only the corrected
    unified-ideograph forms (verbatim substring assertions), with none of
    the planted radical codepoints surviving.
  - Existing PDF fixtures (English, CJK) are unaffected — they contain no
    Kangxi-radical/CJK-Radicals-Supplement codepoints, so normalization is a
    no-op regression guard on their output.
  - Re-import hash-skip / hash-drift semantics on the contaminated fixture
    are unchanged: the hash keys on raw PDF bytes, not normalized output.

Fixture: markdown_kb/tests/fixtures/raw_import/kangxi_contamination.pdf,
regenerable via markdown_kb/tests/fixtures/generate_pdf_fixtures.py (dev-only,
hand-assembled — see that script's module docstring for why a reportlab-drawn
PDF cannot plant these codepoints).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.kangxi_normalize import KANGXI_RADICAL_MAP, normalize_kangxi_radicals

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"

# ---------------------------------------------------------------------------
# Unit tests: table-driven mapping / non-mapping cases
# ---------------------------------------------------------------------------

# Representative Kangxi Radicals block (U+2F00-2FDF) mappings named in the
# issue's acceptance criteria.
_KANGXI_RADICAL_CASES = [
    ("⽬", "目"),  # ⽬ KANGXI RADICAL EYE -> 目
    ("⼀", "一"),  # ⼀ KANGXI RADICAL ONE -> 一
    ("⽤", "用"),  # ⽤ KANGXI RADICAL USE -> 用
    ("⾃", "自"),  # ⾃ KANGXI RADICAL SELF -> 自
]

# Representative CJK Radicals Supplement block (U+2E80-2EFF) mappings — only
# two codepoints in this block carry a Unicode compatibility decomposition at
# all (verified against KANGXI_RADICAL_MAP below); both are exercised here.
_CJK_RADICALS_SUPPLEMENT_CASES = [
    ("⺟", "母"),  # CJK RADICAL MOTHER -> 母
    ("⻳", "龟"),  # CJK RADICAL C-SIMPLIFIED TURTLE -> 龟
]


@pytest.mark.parametrize("radical, corrected", _KANGXI_RADICAL_CASES)
def test_normalize_kangxi_radicals_block(radical, corrected):
    """Each representative Kangxi Radicals codepoint maps to its ideograph."""
    assert normalize_kangxi_radicals(radical) == corrected
    assert KANGXI_RADICAL_MAP[radical] == corrected


@pytest.mark.parametrize("radical, corrected", _CJK_RADICALS_SUPPLEMENT_CASES)
def test_normalize_cjk_radicals_supplement_block(radical, corrected):
    """Each representative CJK Radicals Supplement codepoint maps to its ideograph."""
    assert normalize_kangxi_radicals(radical) == corrected
    assert KANGXI_RADICAL_MAP[radical] == corrected


def test_normalize_kangxi_radicals_mixed_string():
    """Radicals embedded in ordinary text are corrected; surrounding text is untouched."""
    contaminated = "如果您需要⼀些⽤法或⾃我說明"
    corrected = normalize_kangxi_radicals(contaminated)
    assert corrected == "如果您需要一些用法或自我說明"


@pytest.mark.parametrize(
    "text",
    [
        "４",  # fullwidth digit 4 — must NOT be rewritten by a scoped pass
        "ﬁ",  # LATIN SMALL LIGATURE FI — compatibility ligature, out of scope
        "目",  # 目 — an ordinary CJK Unified Ideograph, already correct
        "hello world 123",  # plain ASCII
    ],
)
def test_normalize_kangxi_radicals_passes_through_non_mappings(text):
    """Characters outside the two scoped blocks are never rewritten."""
    assert normalize_kangxi_radicals(text) == text


def test_kangxi_radical_map_targets_never_land_back_in_scope():
    """Every mapped-to character falls outside the two source blocks.

    Guards against a degenerate table where a radical maps to another
    radical instead of a true CJK Unified Ideograph.
    """
    for radical, ideograph in KANGXI_RADICAL_MAP.items():
        assert not (0x2E80 <= ord(ideograph) <= 0x2FDF), (
            f"{radical!r} maps back into the radical blocks: {ideograph!r}"
        )


# ---------------------------------------------------------------------------
# Integration tests: POST /import over the contaminated fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def import_env(tmp_path, monkeypatch):
    """Wire raw_dir and docs_dir into importer.py for isolation."""
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


def test_import_pdf_kangxi_contamination_corrected_end_to_end(import_env):
    """The contaminated fixture imports to a docs Source with only corrected forms."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "kangxi_contamination.pdf", raw_dir / "kangxi_contamination.pdf")

    resp = client.post("/import")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["original_format"] == "pdf"

    content = (docs_dir / "kangxi_contamination.md").read_text(encoding="utf-8")

    # Corrected forms present (verbatim substrings).
    for _radical, corrected in _KANGXI_RADICAL_CASES:
        assert corrected in content, f"Expected corrected form {corrected!r} in:\n{content}"

    # No planted radical codepoint survives normalization.
    for radical, _corrected in _KANGXI_RADICAL_CASES:
        assert radical not in content, f"Uncorrected radical {radical!r} leaked into:\n{content}"


def test_import_pdf_kangxi_reimport_unchanged_hash_skips(import_env):
    """Re-importing the unchanged contaminated PDF hash-skips (hash keys on raw bytes)."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / "kangxi_contamination.pdf", raw_dir / "kangxi_contamination.pdf")

    resp1 = client.post("/import")
    assert len(resp1.json()["imported_sources"]) == 1

    docs_file = docs_dir / "kangxi_contamination.md"
    mtime_after_first = docs_file.stat().st_mtime

    resp2 = client.post("/import")
    data2 = resp2.json()
    assert data2["imported_sources"] == []
    assert len(data2["skipped_sources"]) == 1
    assert data2["skipped_sources"][0]["status"] == "skipped"
    assert docs_file.stat().st_mtime == mtime_after_first, (
        "Docs file must not be rewritten on hash-match skip"
    )


def test_import_pdf_kangxi_byte_modified_overwrites_as_updated(import_env):
    """A byte-modified contaminated PDF re-import overwrites (status='updated')."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    raw_path = raw_dir / "kangxi_contamination.pdf"
    shutil.copy(FIXTURES / "kangxi_contamination.pdf", raw_path)

    client.post("/import")  # first import: created

    raw_path.write_bytes(raw_path.read_bytes() + b"%stray-comment-bytes")

    resp = client.post("/import")
    data = resp.json()
    assert data["skipped_sources"] == []
    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["status"] == "updated"


# ---------------------------------------------------------------------------
# Regression guard: existing PDF fixtures are unaffected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["sample_english.pdf", "sample_cjk.pdf"])
def test_existing_pdf_fixtures_have_no_radical_codepoints(import_env, fixture_name):
    """Pre-existing PDF fixtures contain no in-scope radical codepoints.

    Confirms Kangxi-radical normalization is a no-op regression guard on
    fixtures that predate issue #425 — their converted output is unaffected
    by this slice.
    """
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    shutil.copy(FIXTURES / fixture_name, raw_dir / fixture_name)
    resp = client.post("/import")
    assert resp.status_code == 200

    docs_filename = Path(fixture_name).stem + ".md"
    content = (docs_dir / docs_filename).read_text(encoding="utf-8")

    for ch in content:
        assert not (0x2E80 <= ord(ch) <= 0x2FDF), (
            f"Unexpected in-scope radical codepoint {ch!r} in pre-existing fixture output:\n{content}"
        )
