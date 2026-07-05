"""Integration tests for POST /transcribe/batch + GET /transcribe/jobs/{job_id}
(issue #459 AC5 — async submit/poll for a batch of Transcribe scans).

AC coverage:
  - POST /transcribe/batch returns a job_id immediately (HTTP 200), without
    waiting for the batch to finish.
  - GET /transcribe/jobs/{job_id} reports submitted -> working -> completed,
    with per-source results landing in docs/ exactly as a synchronous
    POST /transcribe call would.
  - A per-source failure (e.g. not found) is recorded in results without
    aborting the rest of the batch.
  - An unknown job_id returns 404.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from .conftest import FakeLLMResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


class FakeTranscribeLLM:
    def __init__(self, body: str = "# batch route\ntranscribed via /transcribe/batch."):
        self.call_count = 0
        self.body = body

    def invoke(self, messages):
        self.call_count += 1
        return FakeLLMResponse(content=self.body)


@pytest.fixture(autouse=True)
def _reset_job_registry():
    from app import transcribe_jobs

    transcribe_jobs._reset_jobs()
    yield
    transcribe_jobs._reset_jobs()


@pytest.fixture()
def batch_route_env(tmp_path, monkeypatch):
    """Yields inside ``with TestClient(app) as client`` — REQUIRED here (unlike
    the plain-``TestClient(app)`` used by the synchronous /transcribe route
    tests): without an entered context, Starlette's TestClient spins up a
    fresh, throwaway event loop per request (``_portal_factory``), so the
    ``asyncio.Task`` a submit POST schedules would be abandoned the instant
    that POST's response completes — before the background batch finishes.
    One persistent portal across submit + polling GETs mirrors how a real
    long-lived uvicorn process actually runs this code.
    """
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

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir, "fake_llm": fake_llm}


def _wait_for_terminal(client, job_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        resp = client.get(f"/transcribe/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} did not complete in time: {data}")
        time.sleep(0.01)


def test_submit_batch_returns_job_id_without_blocking(batch_route_env):
    client = batch_route_env["client"]
    (batch_route_env["raw_dir"] / "a.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    resp = client.post("/transcribe/batch", json={"sources": ["a.pdf"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data and data["job_id"]


def test_batch_job_completes_and_writes_docs(batch_route_env):
    client = batch_route_env["client"]
    (batch_route_env["raw_dir"] / "a.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    submit = client.post("/transcribe/batch", json={"sources": ["a.pdf"]})
    job_id = submit.json()["job_id"]

    final = _wait_for_terminal(client, job_id)

    assert final["status"] == "completed"
    assert final["pages_done"] == 1
    assert final["pages_total"] == 1
    assert len(final["results"]) == 1
    assert final["results"][0]["source"] == "a.pdf"
    assert final["results"][0]["status"] == "created"
    assert (batch_route_env["docs_dir"] / "a.md").exists()


def test_batch_job_records_per_source_failure_without_aborting_batch(batch_route_env):
    client = batch_route_env["client"]
    (batch_route_env["raw_dir"] / "a.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )

    submit = client.post("/transcribe/batch", json={"sources": ["a.pdf", "does_not_exist.pdf"]})
    job_id = submit.json()["job_id"]

    final = _wait_for_terminal(client, job_id)

    assert final["status"] == "completed"
    by_source = {r["source"]: r for r in final["results"]}
    assert by_source["a.pdf"]["status"] == "created"
    assert by_source["does_not_exist.pdf"]["status"] == "failed"
    assert by_source["does_not_exist.pdf"]["error_type"] == "FileNotFoundError"


def test_unknown_job_id_returns_404(batch_route_env):
    client = batch_route_env["client"]
    resp = client.get("/transcribe/jobs/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Resource-exhaustion hardening (issue #474)
# ---------------------------------------------------------------------------


def test_submit_over_concurrent_job_cap_returns_503(batch_route_env, monkeypatch):
    """Sub-issue A: submitting past the concurrent-job cap is a clear 503, not a hang.

    Pre-seeds one "working" job directly in the registry rather than racing a
    real background task's completion time against the second HTTP call —
    deterministic, and exercises exactly the cap check + its 503 mapping.
    """
    from app import transcribe_jobs

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_CONCURRENT_JOBS", "1")
    transcribe_jobs._JOBS["seed"] = transcribe_jobs.TranscribeJob(job_id="seed", status="working")

    client = batch_route_env["client"]
    resp = client.post("/transcribe/batch", json={"sources": ["a.pdf"]})
    assert resp.status_code == 503


def test_batch_request_rejects_over_long_sources_list(batch_route_env):
    """Sub-issue B: an over-long ``sources`` list is a 422, not accepted."""
    from app.schemas import MAX_BATCH_SOURCES

    client = batch_route_env["client"]
    sources = [f"file_{i}.pdf" for i in range(MAX_BATCH_SOURCES + 1)]

    resp = client.post("/transcribe/batch", json={"sources": sources})
    assert resp.status_code == 422


def test_batch_request_accepts_sources_list_at_the_cap(batch_route_env):
    """The cap itself is still a valid request (off-by-one guard)."""
    from app.schemas import MAX_BATCH_SOURCES

    client = batch_route_env["client"]
    sources = [f"file_{i}.pdf" for i in range(MAX_BATCH_SOURCES)]

    resp = client.post("/transcribe/batch", json={"sources": sources})
    assert resp.status_code == 200
