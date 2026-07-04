"""Live integration smoke test for Transcribe (issue #426, ADR-0032).

Makes ONE real OpenAI vision-model call to confirm the transcription
pipeline end-to-end against the committed scanned-CJK fixture. Opt-in only:
skipped by default; run with:

    pytest -m live   (from markdown_kb/)

Requirements:
    OPENAI_API_KEY must be set in the environment; the test fails with a
    clear message if it is absent rather than silently passing or skipping.

This is the ONE authorised @pytest.mark.live test for the Transcribe surface
(ADR-0005 Sec"LLM-facing surface enumeration" / CODING_STANDARD Sec6.4) --
do not add a second.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


@pytest.mark.live
def test_transcribe_scanned_cjk_fixture_live(tmp_path, monkeypatch):
    """POST /transcribe against the real vision model, forcing the committed
    scanned-CJK fixture end-to-end.

    Assertions are SHAPE-only (CODING_STANDARD Sec6.2/Sec6.4):
      - HTTP 200, status="created", origin="transcribed"
      - transcribe_model matches the resolved model name
      - the written docs/ file contains at least one CJK codepoint (readable
        Chinese content transcribed from the page image) -- never asserting
        specific words, since models update and tests must outlive them.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail(
            "OPENAI_API_KEY is not set. "
            "Export your key before running live tests: "
            "export OPENAI_API_KEY=sk-..."
        )

    import app.logger as logger_module
    import app.transcriber as transcriber_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(transcriber_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(transcriber_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")

    # Reset any cached LLM singleton so we get a fresh real ChatOpenAI instance
    # with the current OPENAI_API_KEY from the environment.
    monkeypatch.setattr(transcriber_module, "_transcribe_llm", None)

    (raw_dir / "transcribe_scanned_cjk.pdf").write_bytes(
        (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    )

    from app.main import app

    client = TestClient(app)
    resp = client.post("/transcribe", json={"source": "transcribe_scanned_cjk.pdf"})

    assert resp.status_code == 200, f"Expected HTTP 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "created"
    assert body["origin"] == "transcribed"
    assert body["transcribe_model"] == transcriber_module._transcribe_model_name()

    content = (docs_dir / "transcribe_scanned_cjk.md").read_text(encoding="utf-8")
    has_cjk = any("一" <= ch <= "鿿" for ch in content)
    assert has_cjk, f"Expected at least one CJK codepoint in transcribed output, got: {content!r}"
