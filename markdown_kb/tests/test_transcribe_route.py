"""Integration tests for POST /transcribe -- the force entry (issue #426, ADR-0032).

AC coverage:
  - POST /transcribe force-transcribes a digital-native PDF (bypass proven --
    a PDF WITH a text layer still calls the model when hit via this route).
  - Unavailable / page-limit / not-found / bad-extension map to the documented
    HTTP status codes.
  - Unchanged re-transcribe hash-skips (status="skipped", no second model call).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from .conftest import FakeLLMResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


class FakeTranscribeLLM:
    def __init__(self, body: str = "# forced\ntranscribed via POST /transcribe."):
        self.call_count = 0
        self.body = body

    def invoke(self, messages):
        self.call_count += 1
        return FakeLLMResponse(content=self.body)


@pytest.fixture()
def transcribe_route_env(tmp_path, monkeypatch):
    import app.logger as logger_module
    import app.transcriber as transcriber_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(transcriber_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(transcriber_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")
    monkeypatch.setenv("OPENAI_TRANSCRIBE_MODEL", "gpt-5-mini-test")

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    from app.main import app

    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    return {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir, "fake_llm": fake_llm}


def test_transcribe_forces_digital_native_pdf(transcribe_route_env):
    """A PDF WITH a text layer still gets transcribed when forced via /transcribe."""
    client = transcribe_route_env["client"]
    (transcribe_route_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    resp = client.post("/transcribe", json={"source": "sample_english.pdf"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "created"
    assert data["origin"] == "transcribed"
    assert data["transcribe_model"] == "gpt-5-mini-test"
    assert transcribe_route_env["fake_llm"].call_count == 1, (
        "The force entry must call the model even for a digital-native PDF"
    )

    content = (transcribe_route_env["docs_dir"] / "sample_english.md").read_text(encoding="utf-8")
    assert "transcribed via POST /transcribe" in content


def test_transcribe_hash_skip_no_second_model_call(transcribe_route_env):
    client = transcribe_route_env["client"]
    (transcribe_route_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    first = client.post("/transcribe", json={"source": "sample_english.pdf"})
    assert first.json()["status"] == "created"

    second = client.post("/transcribe", json={"source": "sample_english.pdf"})
    assert second.status_code == 200
    assert second.json()["status"] == "skipped"
    assert transcribe_route_env["fake_llm"].call_count == 1


def test_transcribe_not_found_returns_404(transcribe_route_env):
    client = transcribe_route_env["client"]
    resp = client.post("/transcribe", json={"source": "missing.pdf"})
    assert resp.status_code == 404


def test_transcribe_unsupported_extension_returns_400(transcribe_route_env):
    client = transcribe_route_env["client"]
    (transcribe_route_env["raw_dir"] / "notes.txt").write_text("hello", encoding="utf-8")

    resp = client.post("/transcribe", json={"source": "notes.txt"})
    assert resp.status_code == 400


def test_transcribe_unavailable_returns_503(transcribe_route_env, monkeypatch):
    monkeypatch.delenv("KB_TRANSCRIBE_ENABLED", raising=False)
    client = transcribe_route_env["client"]
    (transcribe_route_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    resp = client.post("/transcribe", json={"source": "sample_english.pdf"})
    assert resp.status_code == 503


def test_transcribe_page_limit_exceeded_returns_413(transcribe_route_env, monkeypatch):
    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "0")
    client = transcribe_route_env["client"]
    (transcribe_route_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    resp = client.post("/transcribe", json={"source": "sample_english.pdf"})
    assert resp.status_code == 413
    assert transcribe_route_env["fake_llm"].call_count == 0
