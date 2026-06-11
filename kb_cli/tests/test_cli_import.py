"""Tests for ``kb import <path>`` CLI subcommand — Slice 227.

AC coverage (issue #227):
  - ``kb import <path>`` drives the deep module; success prints a concise result
  - Any failure exits non-zero with a stderr message and no traceback (ADR-0015)
  - Traversal/invalid path failures exit non-zero with a clear message
  - PDF (extractor unavailable) exits non-zero with a clear message

Uses Typer CliRunner (same pattern as test_cli.py) so no subprocess is spawned.
Mocking follows the project pattern: we monkeypatch ``import_path`` at the
``markdown_kb.app.importer`` module level so no real raw/docs directories are
written.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# AC-1: success path — exit 0, concise result printed
# ---------------------------------------------------------------------------


def test_kb_import_exits_zero_on_success(monkeypatch, tmp_path):
    """``kb import <path>`` exits with code 0 when import_path succeeds."""
    import markdown_kb.app.importer as importer_module
    from markdown_kb.app.importer import ImportSourceResult

    from kb_cli.main import app

    src = tmp_path / "notes.txt"
    src.write_text("Some content.", encoding="utf-8")

    fake_result = ImportSourceResult(
        raw_path=str(tmp_path / "raw" / "notes.txt"),
        docs_path=str(tmp_path / "docs" / "notes.md"),
        original_format="txt",
        content_sha256="abc123",
        status="created",
    )

    monkeypatch.setattr(importer_module, "import_path", lambda p: fake_result)

    result = runner.invoke(app, ["import", str(src)])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"


def test_kb_import_prints_concise_result(monkeypatch, tmp_path):
    """``kb import <path>`` prints the outcome (file name or status)."""
    import markdown_kb.app.importer as importer_module
    from markdown_kb.app.importer import ImportSourceResult

    from kb_cli.main import app

    src = tmp_path / "notes.txt"
    src.write_text("Some content.", encoding="utf-8")

    fake_result = ImportSourceResult(
        raw_path=str(tmp_path / "raw" / "notes.txt"),
        docs_path=str(tmp_path / "docs" / "notes.md"),
        original_format="txt",
        content_sha256="abc123",
        status="created",
    )

    monkeypatch.setattr(importer_module, "import_path", lambda p: fake_result)

    result = runner.invoke(app, ["import", str(src)])
    output = result.output
    # Must print something about the import (filename, status, or path)
    assert "notes" in output.lower() or "created" in output.lower() or "import" in output.lower(), (
        f"Expected concise result in output, got:\n{output}"
    )


def test_kb_import_prints_status(monkeypatch, tmp_path):
    """``kb import <path>`` prints the status (created/updated/skipped)."""
    import markdown_kb.app.importer as importer_module
    from markdown_kb.app.importer import ImportSourceResult

    from kb_cli.main import app

    src = tmp_path / "article.html"
    src.write_text("<h1>T</h1>", encoding="utf-8")

    fake_result = ImportSourceResult(
        raw_path=str(tmp_path / "raw" / "article.html"),
        docs_path=str(tmp_path / "docs" / "article.md"),
        original_format="html",
        content_sha256="deadbeef",
        status="updated",
    )

    monkeypatch.setattr(importer_module, "import_path", lambda p: fake_result)

    result = runner.invoke(app, ["import", str(src)])
    assert "updated" in result.output.lower() or "article" in result.output.lower(), (
        f"Expected status/filename in output:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# AC-2: failure exits non-zero with stderr message (ADR-0015 CLI contract)
# ---------------------------------------------------------------------------


def test_kb_import_nonzero_on_import_path_error(monkeypatch, tmp_path):
    """ImportPathError from import_path renders as non-zero exit."""
    import markdown_kb.app.importer as importer_module
    from markdown_kb.app.importer import ImportPathError

    from kb_cli.main import app

    src = tmp_path / "bad.pdf"
    src.write_bytes(b"%PDF-1.4 fake")

    def _raise(p):
        raise ImportPathError(
            "PDF extractor not yet available: bad.pdf",
            error_type="UnsupportedExtension",
        )

    monkeypatch.setattr(importer_module, "import_path", _raise)

    result = runner.invoke(app, ["import", str(src)])
    assert result.exit_code != 0, (
        f"Expected non-zero exit for ImportPathError, got {result.exit_code}"
    )


def test_kb_import_error_message_in_output(monkeypatch, tmp_path):
    """ImportPathError message appears in output (no traceback)."""
    import markdown_kb.app.importer as importer_module
    from markdown_kb.app.importer import ImportPathError

    from kb_cli.main import app

    src = tmp_path / "bad.pdf"
    src.write_bytes(b"%PDF-1.4 fake")

    def _raise(p):
        raise ImportPathError(
            "PDF extractor not yet available: bad.pdf",
            error_type="UnsupportedExtension",
        )

    monkeypatch.setattr(importer_module, "import_path", _raise)

    result = runner.invoke(app, ["import", str(src)])
    combined = result.output or ""
    assert (
        "pdf" in combined.lower() or "extractor" in combined.lower() or "error" in combined.lower()
    ), f"Expected error message in output, got:\n{combined}"
    # No traceback — "Traceback" must not appear
    assert "Traceback" not in combined, f"Traceback must not appear in output:\n{combined}"


def test_kb_import_missing_file_nonzero(tmp_path):
    """``kb import`` on a non-existent path exits non-zero."""
    from kb_cli.main import app

    result = runner.invoke(app, ["import", str(tmp_path / "nonexistent.txt")])
    assert result.exit_code != 0, (
        f"Expected non-zero exit for missing file, got {result.exit_code}\n{result.output}"
    )
