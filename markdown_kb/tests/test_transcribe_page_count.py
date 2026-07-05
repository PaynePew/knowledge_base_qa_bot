"""Tests for ``page_count_for_source`` + ``GET /transcribe/page-count`` (issue #447).

A mechanical preflight — no model call, no ``OPENAI_API_KEY`` needed — that
lets the Console's guarded Transcribe action name a staged raw/ PDF's real
page count (plus the configured page cap) before the operator confirms a
forced transcription (CODING_STANDARD §12.5 — no guessed bound client-side).

AC coverage:
  - Returns the real page count of a staged raw/ PDF, no model call required.
  - Returns the configured ``KB_TRANSCRIBE_MAX_PAGES`` cap alongside it.
  - Shares ``/transcribe``'s validation chain: not-found / bad-extension map
    to the same typed failures and HTTP status codes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


@pytest.fixture()
def page_count_env(tmp_path, monkeypatch):
    """Wire RAW_DIR into transcriber.py — no OPENAI_API_KEY needed (mechanical only)."""
    import app.transcriber as transcriber_module

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(transcriber_module, "RAW_DIR", raw_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KB_TRANSCRIBE_ENABLED", raising=False)
    return {"raw_dir": raw_dir}


# ---------------------------------------------------------------------------
# Unit: page_count_for_source() — deep-module function
# ---------------------------------------------------------------------------


def test_page_count_for_source_returns_real_count_no_model_call(page_count_env):
    import app.transcriber as transcriber_module

    (page_count_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    page_count, max_pages = transcriber_module.page_count_for_source("sample_english.pdf")

    assert page_count == 1
    assert max_pages == transcriber_module._DEFAULT_MAX_PAGES


def test_page_count_for_source_reads_configured_max_pages(page_count_env, monkeypatch):
    import app.transcriber as transcriber_module

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "80")
    (page_count_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    _, max_pages = transcriber_module.page_count_for_source("sample_english.pdf")
    assert max_pages == 80


def test_page_count_for_source_not_found_raises_typed_error(page_count_env):
    import app.transcriber as transcriber_module

    with pytest.raises(transcriber_module.TranscribePathError) as exc_info:
        transcriber_module.page_count_for_source("does_not_exist.pdf")
    assert exc_info.value.error_type == "FileNotFoundError"


def test_page_count_for_source_works_when_transcribe_unavailable(page_count_env):
    """Mechanical, unlike the force entry — works even with no OPENAI_API_KEY
    and KB_TRANSCRIBE_ENABLED unset (page counting is not a Transcribe call)."""
    import app.transcriber as transcriber_module

    assert transcriber_module.transcribe_available() is False
    (page_count_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    page_count, _ = transcriber_module.page_count_for_source("sample_english.pdf")
    assert page_count == 1


# ---------------------------------------------------------------------------
# Route: GET /transcribe/page-count
# ---------------------------------------------------------------------------


@pytest.fixture()
def page_count_route_env(tmp_path, monkeypatch):
    import app.transcriber as transcriber_module

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(transcriber_module, "RAW_DIR", raw_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KB_TRANSCRIBE_ENABLED", raising=False)

    from fastapi.testclient import TestClient

    from app.main import app

    return {"client": TestClient(app), "raw_dir": raw_dir}


def test_route_returns_page_count_and_max_pages(page_count_route_env):
    client = page_count_route_env["client"]
    (page_count_route_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    resp = client.get("/transcribe/page-count", params={"source": "sample_english.pdf"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "sample_english.pdf"
    assert data["page_count"] == 1
    assert data["max_pages"] == 50


def test_route_not_found_maps_to_404(page_count_route_env):
    client = page_count_route_env["client"]
    resp = client.get("/transcribe/page-count", params={"source": "does_not_exist.pdf"})
    assert resp.status_code == 404


def test_route_bad_extension_maps_to_400(page_count_route_env):
    client = page_count_route_env["client"]
    (page_count_route_env["raw_dir"] / "not_a_pdf.txt").write_text("hello")

    resp = client.get("/transcribe/page-count", params={"source": "not_a_pdf.txt"})
    assert resp.status_code == 400
