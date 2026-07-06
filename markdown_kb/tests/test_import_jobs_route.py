"""Integration tests for POST /import/jobs + GET /import/jobs/{job_id}
(issue #497 — async submit/poll for a whole Import run).

AC coverage:
  - POST /import/jobs returns a job_id immediately (HTTP 200), without
    waiting for the run to finish.
  - GET /import/jobs/{job_id} reports submitted -> working -> completed with
    file/page progress, and its terminal ``result`` matches the shape the
    synchronous POST /import returns for the same inputs.
  - Single-source mode (body ``{"source": ...}``) passes through unchanged.
  - An unknown job_id returns 404.
  - The concurrent-job cap returns a clear 503 at submit time.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


@pytest.fixture(autouse=True)
def _reset_job_registry():
    from app import import_jobs

    import_jobs._reset_jobs()
    yield
    import_jobs._reset_jobs()


@pytest.fixture()
def jobs_route_env(tmp_path, monkeypatch):
    """Yields inside ``with TestClient(app) as client`` — REQUIRED here (same
    reason as test_transcribe_batch_route.py): without an entered context,
    Starlette's TestClient spins up a throwaway event loop per request, so
    the ``asyncio.Task`` a submit POST schedules would be abandoned the
    instant that POST's response completes.
    """
    import app.importer as importer_module
    import app.logger as logger_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield {"client": client, "raw_dir": raw_dir, "docs_dir": docs_dir}


def _wait_for_terminal(client, job_id: str, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        resp = client.get(f"/import/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} did not complete in time: {data}")
        time.sleep(0.01)


def test_submit_returns_job_id_without_blocking(jobs_route_env):
    client = jobs_route_env["client"]
    (jobs_route_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")

    resp = client.post("/import/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data and data["job_id"]


def test_job_completes_with_the_sync_routes_response_shape(jobs_route_env):
    """The terminal ``result`` must be the SAME shape POST /import returns —
    the Console renders both through one renderImportCard."""
    client = jobs_route_env["client"]
    (jobs_route_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")

    submit = client.post("/import/jobs")
    job_id = submit.json()["job_id"]

    final = _wait_for_terminal(client, job_id)

    assert final["status"] == "completed"
    assert final["error"] is None
    assert final["files_done"] == 1 and final["files_total"] == 1
    result = final["result"]
    assert result is not None
    assert len(result["imported_sources"]) == 1
    assert result["imported_sources"][0]["original_format"] == "txt"
    assert result["skipped_sources"] == []
    assert result["failed_sources"] == []
    assert (jobs_route_env["docs_dir"] / "a.md").exists()
    # Same keys as the synchronous ImportResponse — one Console renderer.
    sync = client.post("/import").json()
    assert set(result.keys()) == set(sync.keys())


def test_single_source_mode_passes_through(jobs_route_env):
    client = jobs_route_env["client"]
    (jobs_route_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")
    (jobs_route_env["raw_dir"] / "b.txt").write_text("beta", encoding="utf-8")

    submit = client.post("/import/jobs", json={"source": "b.txt"})
    final = _wait_for_terminal(client, submit.json()["job_id"])

    assert final["status"] == "completed"
    assert len(final["result"]["imported_sources"]) == 1
    assert final["result"]["imported_sources"][0]["docs_path"].endswith("b.md")


def test_unknown_job_id_returns_404(jobs_route_env):
    resp = jobs_route_env["client"].get("/import/jobs/no-such-job")
    assert resp.status_code == 404


def test_capacity_cap_returns_503(jobs_route_env, monkeypatch):
    """Deterministic cap test: the first job's worker blocks on an Event, so
    it is guaranteed still active when the second submit lands (no reliance
    on the real import being slow enough)."""
    import threading

    from app import import_jobs
    from app.importer import ImportBatchResult

    monkeypatch.setenv("KB_IMPORT_MAX_CONCURRENT_JOBS", "1")
    release = threading.Event()

    def _blocked_import(source_filter, **_kwargs):
        release.wait(timeout=10)
        return ImportBatchResult()

    monkeypatch.setattr(import_jobs, "_import_sources", _blocked_import)
    client = jobs_route_env["client"]

    first = client.post("/import/jobs")
    assert first.status_code == 200
    try:
        second = client.post("/import/jobs")
        assert second.status_code == 503
        assert "retry later" in second.json()["detail"]
    finally:
        release.set()
    # Drain the first job so the TestClient context exits with no live task.
    _wait_for_terminal(client, first.json()["job_id"])
