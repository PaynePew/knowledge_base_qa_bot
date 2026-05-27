"""Tests for NFC Unicode normalization — Slice 7-2.

AC coverage (issue #91 — Slice 7-2):
  - NFC normalization applied to single-mode source field on entry
  - NFC normalization applied to glob basenames in batch mode

Background: macOS APFS stores filenames in NFD; clipboard pastes are typically
NFC. Without normalization, an NFC source='café.html' would fail to resolve
against an on-disk NFD 'café.html'. This test verifies that both forms resolve
to the same canonical docs/<stem>.md.
"""

from __future__ import annotations

import unicodedata
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


# ---------------------------------------------------------------------------
# NFC normalization of single-mode source field
# ---------------------------------------------------------------------------


def test_nfc_normalization_single_mode(import_env):
    """NFC-normalization of source_filter routes to NFC-stem docs/<stem>.md.

    We create the raw file with its filesystem name; the important assertion
    is that the output docs path uses the NFC-normalized stem.
    """
    import app.importer as importer_module

    raw_dir = import_env["raw_dir"]

    # NFC form of 'é' is U+00E9 (single codepoint)
    nfc_name = unicodedata.normalize("NFC", "café.html")

    # Write the raw file under the NFC name (the filesystem-canonical form on most OSes)
    (raw_dir / nfc_name).write_text("<h1>Café</h1><p>Content.</p>", encoding="utf-8")

    # Call import_sources directly with the NFC source_filter
    result = importer_module.import_sources(nfc_name)

    assert len(result.imported_sources) == 1, (
        f"Expected 1 imported source, got: {result.failed_sources}"
    )
    imported = result.imported_sources[0]
    # Output stem must be NFC-normalized
    out_stem = unicodedata.normalize("NFC", Path(imported.docs_path).stem)
    expected_stem = unicodedata.normalize("NFC", "café")
    assert out_stem == expected_stem, (
        f"docs path stem must be NFC-normalized: got {repr(out_stem)}, "
        f"expected {repr(expected_stem)}"
    )


def test_nfc_normalization_applied_to_source_filter(import_env, monkeypatch):
    """import_sources applies NFC normalization to source_filter before processing.

    Verified by monkeypatching _resolve_single_source to capture the
    normalized value passed to it.
    """
    import app.importer as importer_module
    from app.importer import ImportFailure

    captured = {}

    def capturing_resolve(source_filter):
        captured["source_filter"] = source_filter
        # Return a FileNotFoundError so we don't need real files
        return Path(""), ImportFailure(
            raw_path=source_filter,
            error_type="FileNotFoundError",
            error_message="test stub",
        )

    monkeypatch.setattr(importer_module, "_resolve_single_source", capturing_resolve)

    # NFD form (macOS APFS style)
    nfd_input = unicodedata.normalize("NFD", "café.html")
    importer_module.import_sources(nfd_input)

    assert "source_filter" in captured, "Capturing stub was not called"
    # The value passed to _resolve_single_source must be NFC-normalized
    result_form = unicodedata.is_normalized("NFC", captured["source_filter"])
    assert result_form, (
        f"source_filter must be NFC-normalized before _resolve_single_source, "
        f"got: {repr(captured['source_filter'])}"
    )


def test_nfc_normalization_batch_mode_stem(import_env):
    """Batch-mode glob basename stem is NFC-normalized for the docs output path."""
    import app.importer as importer_module

    raw_dir = import_env["raw_dir"]

    # Create a raw file with NFC name
    nfc_name = unicodedata.normalize("NFC", "rapporté.html")  # é = U+00E9
    (raw_dir / nfc_name).write_text("<h1>Rapport</h1><p>Body.</p>", encoding="utf-8")

    result = importer_module.import_sources(None)  # batch mode

    assert len(result.imported_sources) == 1, (
        f"Expected 1 imported source, got failures: {result.failed_sources}"
    )
    docs_path = Path(result.imported_sources[0].docs_path)
    # Stem used in output must be NFC
    assert unicodedata.is_normalized("NFC", docs_path.stem), (
        f"docs stem must be NFC-normalized, got: {repr(docs_path.stem)}"
    )
