"""Tests for ``kb transcribe <path>`` CLI subcommand (issue #426, ADR-0032).

AC coverage:
  - ``kb transcribe <path>`` drives the deep module; success prints a concise
    result naming the model and status.
  - Any ``TranscribePathError`` exits non-zero with a stderr message and no
    traceback (ADR-0015 CLI contract), mirroring ``kb import``'s pattern.

Uses Typer CliRunner (same pattern as test_cli_import.py) so no subprocess is
spawned; ``transcribe_path`` is monkeypatched at the
``markdown_kb.app.transcriber`` module level so no real raw/docs directories
are written and no model is called.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

runner = CliRunner()


def test_kb_transcribe_exits_zero_on_success(monkeypatch, tmp_path):
    import markdown_kb.app.transcriber as transcriber_module
    from markdown_kb.app.transcriber import TranscribeSourceResult

    from kb_cli.main import app

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"%PDF-1.4 fake bytes")

    fake_result = TranscribeSourceResult(
        raw_path=str(tmp_path / "raw" / "scan.pdf"),
        docs_path=str(tmp_path / "docs" / "scan.md"),
        content_sha256="abc123",
        transcribe_model="gpt-5-mini",
        status="created",
    )
    monkeypatch.setattr(transcriber_module, "transcribe_path", lambda p: fake_result)

    result = runner.invoke(app, ["transcribe", str(src)])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}\n{result.output}"


def test_kb_transcribe_prints_model_and_status(monkeypatch, tmp_path):
    import markdown_kb.app.transcriber as transcriber_module
    from markdown_kb.app.transcriber import TranscribeSourceResult

    from kb_cli.main import app

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"%PDF-1.4 fake bytes")

    fake_result = TranscribeSourceResult(
        raw_path=str(tmp_path / "raw" / "scan.pdf"),
        docs_path=str(tmp_path / "docs" / "scan.md"),
        content_sha256="abc123",
        transcribe_model="gpt-5-mini",
        status="created",
    )
    monkeypatch.setattr(transcriber_module, "transcribe_path", lambda p: fake_result)

    result = runner.invoke(app, ["transcribe", str(src)])
    output = result.output.lower()
    assert "gpt-5-mini" in output
    assert "created" in output


def test_kb_transcribe_nonzero_on_transcribe_path_error(monkeypatch, tmp_path):
    import markdown_kb.app.transcriber as transcriber_module
    from markdown_kb.app.transcriber import TranscribePathError

    from kb_cli.main import app

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"%PDF-1.4 fake bytes")

    def raise_unavailable(p: Path):
        raise TranscribePathError(
            "Transcribe is unavailable: missing OPENAI_API_KEY",
            error_type="TranscribeUnavailable",
        )

    monkeypatch.setattr(transcriber_module, "transcribe_path", raise_unavailable)

    result = runner.invoke(app, ["transcribe", str(src)])
    assert result.exit_code == 1
    assert "TranscribeUnavailable" in result.output
    assert "Traceback" not in result.output


def test_kb_transcribe_missing_argument_exits_two(tmp_path):
    from kb_cli.main import app

    result = runner.invoke(app, ["transcribe"])
    assert result.exit_code == 2
