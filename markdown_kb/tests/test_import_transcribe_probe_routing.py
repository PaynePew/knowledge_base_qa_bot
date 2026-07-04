"""Integration tests for POST /import's Transcribe probe-routing (issue #426, ADR-0032).

AC coverage:
  - A digital-native PDF keeps routing to the mechanical path (probe finds
    text; no model call, Kangxi normalization still applies) -- the deterministic
    "no auto-route in either direction unless warranted" half of the AC.
  - A text-less PDF auto-routes to Transcribe when available, landing in
    docs/ with the standard envelope plus origin: transcribed + transcribe_model.
  - When Transcribe is unavailable, the pre-existing NoTextLayer path fires
    with an updated message naming Transcribe and the missing prerequisite
    (covered already by test_import_pdf_failure_modes.py /
    test_import_pdf_hardening.py -- not re-asserted here).
  - TranscribePageLimitExceeded / TranscribeError surface as typed
    ImportFailure entries when auto-routed transcription itself fails.

The LLM is mocked at transcriber.get_transcribe_llm (CODING_STANDARD SS6.3);
the probe itself makes no model call (asserted via a call counter).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from .conftest import FakeLLMResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


class FakeTranscribeLLM:
    def __init__(
        self, body: str = "# scanned doc\ntranscribed content.", raise_on_call: bool = False
    ):
        self.call_count = 0
        self.body = body
        self.raise_on_call = raise_on_call

    def invoke(self, messages):
        self.call_count += 1
        if self.raise_on_call:
            raise RuntimeError("simulated model failure")
        return FakeLLMResponse(content=self.body)


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
    return {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir}


@pytest.fixture()
def stub_transcribe_model(monkeypatch):
    """Configure Transcribe as available and stub the model at the lazy getter."""
    import app.transcriber as transcriber_module

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")
    monkeypatch.setenv("OPENAI_TRANSCRIBE_MODEL", "gpt-5-mini-test")

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)
    return fake_llm


# ---------------------------------------------------------------------------
# Digital-native PDF: probe finds text -> mechanical path, unchanged
# ---------------------------------------------------------------------------


def test_digital_native_pdf_stays_on_mechanical_path_no_model_call(
    import_env, stub_transcribe_model
):
    """Even with Transcribe available, a PDF WITH a text layer never calls the model."""
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    raw_dir_pdf = raw_dir / "sample_english.pdf"
    raw_dir_pdf.write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    resp = client.post("/import", json={"source": "sample_english.pdf"})
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["original_format"] == "pdf"
    assert stub_transcribe_model.call_count == 0, (
        "A digital-native PDF must never reach the vision model"
    )

    docs_content = (import_env["docs_dir"] / "sample_english.md").read_text(encoding="utf-8")
    assert "origin: transcribed" not in docs_content


# ---------------------------------------------------------------------------
# Text-less PDF, Transcribe available -> auto-routes
# ---------------------------------------------------------------------------


def test_text_less_pdf_auto_routes_to_transcribe_when_available(import_env, stub_transcribe_model):
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    raw_dir_pdf = raw_dir / "transcribe_scanned_cjk.pdf"
    raw_dir_pdf.write_bytes((FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes())

    resp = client.post("/import", json={"source": "transcribe_scanned_cjk.pdf"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["failed_sources"] == []
    assert len(data["imported_sources"]) == 1
    assert data["imported_sources"][0]["original_format"] == "pdf"
    assert data["imported_sources"][0]["status"] == "created"
    assert stub_transcribe_model.call_count == 1

    content = (docs_dir / "transcribe_scanned_cjk.md").read_text(encoding="utf-8")
    fm_end = content.index("---\n", 4)
    frontmatter = yaml.safe_load(content[4:fm_end])
    assert frontmatter["origin"] == "transcribed"
    assert frontmatter["transcribe_model"] == "gpt-5-mini-test"
    assert frontmatter["original_format"] == "pdf"
    assert "transcribed content" in content


def test_text_less_pdf_auto_route_logs_transcribe_source_not_import_source(
    import_env, stub_transcribe_model
):
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]

    (raw_dir / "transcribe_scanned_cjk.pdf").write_bytes(
        (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    )
    resp = client.post("/import", json={"source": "transcribe_scanned_cjk.pdf"})
    assert resp.status_code == 200

    from app.logger import LOG_PATH

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    kinds = [line.split("|")[0].split("] ")[-1].strip() for line in lines if line.startswith("##")]
    assert kinds == ["import_batch_started", "transcribe_source", "import_batch_completed"]


# ---------------------------------------------------------------------------
# Text-less PDF, Transcribe fails mid-conversion -> typed ImportFailure
# ---------------------------------------------------------------------------


def test_text_less_pdf_page_limit_exceeded_surfaces_as_import_failure(
    import_env, stub_transcribe_model, monkeypatch
):
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "0")
    (raw_dir / "transcribe_scanned_cjk.pdf").write_bytes(
        (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    )

    resp = client.post("/import", json={"source": "transcribe_scanned_cjk.pdf"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["imported_sources"] == []
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "TranscribePageLimitExceeded"
    assert stub_transcribe_model.call_count == 0
    assert not (docs_dir / "transcribe_scanned_cjk.md").exists()


def test_text_less_pdf_model_failure_surfaces_as_transcribe_error(import_env, monkeypatch):
    import app.transcriber as transcriber_module

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")
    monkeypatch.setattr(
        transcriber_module, "get_transcribe_llm", lambda: FakeTranscribeLLM(raise_on_call=True)
    )

    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    (raw_dir / "transcribe_scanned_cjk.pdf").write_bytes(
        (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    )

    resp = client.post("/import", json={"source": "transcribe_scanned_cjk.pdf"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["imported_sources"] == []
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "TranscribeError"
    assert not (docs_dir / "transcribe_scanned_cjk.md").exists()


# ---------------------------------------------------------------------------
# Batch mode: mixed digital-native + text-less, Transcribe available
# ---------------------------------------------------------------------------


def test_batch_mixed_mechanical_and_transcribed_both_succeed(import_env, stub_transcribe_model):
    client = import_env["client"]
    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]

    (raw_dir / "sample_english.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (raw_dir / "transcribe_scanned_cjk.pdf").write_bytes(
        (FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes()
    )

    resp = client.post("/import")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["imported_sources"]) == 2
    assert data["failed_sources"] == []
    assert stub_transcribe_model.call_count == 1, "Only the text-less PDF calls the model"
    assert (docs_dir / "sample_english.md").exists()
    assert (docs_dir / "transcribe_scanned_cjk.md").exists()


# ---------------------------------------------------------------------------
# Budget hook (issue #460) — a batch stops billing once the cap trips
# ---------------------------------------------------------------------------


def test_budget_hook_trips_mid_batch_remaining_files_never_billed(
    import_env, stub_transcribe_model, monkeypatch
):
    """Once the (mocked) daily cap trips, later files in the SAME batch are
    rejected before any vision-model call — proving a multi-scan batch cannot
    silently blow past the ceiling under one flat charge (issue #460)."""
    import app.transcriber as transcriber_module

    raw_dir = import_env["raw_dir"]
    docs_dir = import_env["docs_dir"]
    client = import_env["client"]

    # Two independent text-less PDFs (batch mode globs both).
    (raw_dir / "scan_a.pdf").write_bytes((FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes())
    (raw_dir / "scan_b.pdf").write_bytes((FIXTURES / "transcribe_scanned_cjk.pdf").read_bytes())

    # Simulate the Gateway's hook: reject every call after the first, as if
    # the first file's charge already crossed the daily cap.
    calls: list[int] = []

    def _hook(page_count: int) -> None:
        calls.append(page_count)
        if len(calls) > 1:
            raise transcriber_module.TranscribeBudgetExceeded("daily demo budget reached")

    monkeypatch.setattr(transcriber_module, "_page_budget_hook", _hook)

    resp = client.post("/import")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["imported_sources"]) == 1, "one file bills through before the cap trips"
    assert len(data["failed_sources"]) == 1
    assert data["failed_sources"][0]["error_type"] == "TranscribeBudgetExceeded"
    assert stub_transcribe_model.call_count == 1, (
        "the rejected file's pages must never reach the vision model"
    )
    # Exactly one of the two docs targets was written — the other was rejected
    # before any content existed to write.
    assert sum(1 for name in ("scan_a.md", "scan_b.md") if (docs_dir / name).exists()) == 1
