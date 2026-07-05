"""Unit tests for ``app.import_jobs`` — async submit/poll Import registry (issue #497).

Mirrors ``test_transcribe_jobs.py``'s pattern (itself mirroring
``kb_mcp/tests/test_kb_ingest_jobs.py``), adapted for a whole Import run
(glob + convert + ADR-0032 auto-route) instead of a named list of PDFs.

The LLM is mocked at ``app.transcriber.get_transcribe_llm`` (CODING_STANDARD
§6.3) — never ``import_jobs`` or the deep-module entry points.

Isolation: ``import_jobs._reset_jobs()`` runs autouse per test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from .conftest import FakeLLMResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


class FakeTranscribeLLM:
    def __init__(self, body: str = "# import job\ntranscribed body."):
        self.call_count = 0
        self.body = body

    def invoke(self, messages):
        self.call_count += 1
        return FakeLLMResponse(content=self.body)


@pytest.fixture(autouse=True)
def _reset_job_registry():
    from app import import_jobs

    import_jobs._reset_jobs()
    yield
    import_jobs._reset_jobs()


@pytest.fixture()
def jobs_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log path into importer.py for isolation,
    and stub Transcribe as available so the auto-route path is exercisable."""
    import app.importer as importer_module
    import app.logger as logger_module
    import app.transcriber as transcriber_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(importer_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(importer_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")

    fake_llm = FakeTranscribeLLM()
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_llm)

    return {"raw_dir": raw_dir, "docs_dir": docs_dir, "fake_llm": fake_llm}


async def _poll_until_terminal(job_id: str, *, timeout: float = 10.0) -> None:
    from app import import_jobs

    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        job = import_jobs.status(job_id)
        assert job is not None
        if job.status in ("completed", "failed"):
            return
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"job {job_id} did not reach a terminal state: {job}")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# importer callback seams (the plumbing import_jobs folds progress from)
# ---------------------------------------------------------------------------


def test_import_sources_fires_on_source_done_per_file(jobs_env):
    from app.importer import import_sources

    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")
    (jobs_env["raw_dir"] / "b.txt").write_text("beta", encoding="utf-8")

    calls: list[tuple[int, int]] = []
    import_sources(None, on_source_done=lambda done, total: calls.append((done, total)))

    assert calls == [(1, 2), (2, 2)]


def test_import_sources_fires_on_transcribe_page_only_for_auto_routed_scans(jobs_env):
    from app.importer import import_sources

    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")
    (jobs_env["raw_dir"] / "scan.pdf").write_bytes((FIXTURES / "image_only.pdf").read_bytes())

    pages: list[tuple[str, int, int]] = []
    batch = import_sources(
        None, on_transcribe_page=lambda src, done, total: pages.append((src, done, total))
    )

    assert len(batch.imported_sources) == 2
    assert pages, "the auto-routed scan must report per-page progress"
    assert all(src == "scan.pdf" for src, _done, _total in pages), (
        "only the scanned PDF transcribes — the mechanical .txt must not report pages"
    )


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def test_submit_returns_job_id_immediately(jobs_env):
    """submit must not block for the run's duration."""
    from app import import_jobs

    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")

    async def _run():
        job = import_jobs.submit(None)
        assert job.job_id
        assert job.status in ("submitted", "working")

    asyncio.run(_run())


def test_status_unknown_job_id_returns_none(jobs_env):
    from app import import_jobs

    assert import_jobs.status("does-not-exist") is None


def test_job_completes_with_result_and_progress(jobs_env):
    """A mixed run (mechanical txt + auto-routed scan) completes with the same
    ImportBatchResult the synchronous route would return, plus file and page
    progress counters at their final values."""
    from app import import_jobs

    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")
    (jobs_env["raw_dir"] / "scan.pdf").write_bytes((FIXTURES / "image_only.pdf").read_bytes())

    async def _run():
        job = import_jobs.submit(None)
        await _poll_until_terminal(job.job_id)
        return import_jobs.status(job.job_id)

    final = asyncio.run(_run())

    assert final.status == "completed"
    assert final.result is not None
    assert len(final.result.imported_sources) == 2
    assert (jobs_env["docs_dir"] / "a.md").exists()
    assert (jobs_env["docs_dir"] / "scan.md").exists()
    assert final.files_done == 2
    assert final.files_total == 2
    assert final.pages_total >= 1, "the scan's page count must land in pages_total"
    assert final.pages_done == final.pages_total


def test_job_records_per_source_failure_without_failing(jobs_env):
    """import_sources is continue-on-error — a missing single source lands in
    result.failed_sources with the job still 'completed'."""
    from app import import_jobs

    async def _run():
        job = import_jobs.submit("no-such-file.txt")
        await _poll_until_terminal(job.job_id)
        return import_jobs.status(job.job_id)

    final = asyncio.run(_run())

    assert final.status == "completed"
    assert final.error is None
    assert final.result is not None
    assert len(final.result.failed_sources) == 1
    assert final.result.failed_sources[0].error_type == "FileNotFoundError"


# ---------------------------------------------------------------------------
# Concurrent-job cap (mirrors issue #474 sub-issue A)
# ---------------------------------------------------------------------------


def test_submit_rejects_when_concurrent_job_cap_reached(jobs_env, monkeypatch):
    from app import import_jobs

    monkeypatch.setenv("KB_IMPORT_MAX_CONCURRENT_JOBS", "1")
    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")

    async def _run():
        first = import_jobs.submit(None)
        # No await has run yet, so the scheduled Task has not started —
        # `first` is still status="submitted" and must count as active.
        assert first.status == "submitted"
        with pytest.raises(import_jobs.ImportJobCapacityExceeded):
            import_jobs.submit(None)

    asyncio.run(_run())


def test_submit_allows_new_job_once_previous_completes(jobs_env, monkeypatch):
    """The cap only counts ACTIVE jobs — a terminal job frees a slot."""
    from app import import_jobs

    monkeypatch.setenv("KB_IMPORT_MAX_CONCURRENT_JOBS", "1")
    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")

    async def _run():
        first = import_jobs.submit(None)
        await _poll_until_terminal(first.job_id)
        second = import_jobs.submit(None)
        return second

    second = asyncio.run(_run())
    assert second.job_id


# ---------------------------------------------------------------------------
# Terminal-job eviction (mirrors issue #474 sub-issue B)
# ---------------------------------------------------------------------------


def test_terminal_job_evicted_once_ttl_elapses(jobs_env, monkeypatch):
    """A completed job past its TTL is swept on the next submit call."""
    from app import import_jobs

    monkeypatch.setenv("KB_IMPORT_JOB_TTL_SECONDS", "10")
    fake_now = [0.0]
    # Patch the job-TTL clock seam, NOT time.monotonic — patching the latter
    # also freezes the asyncio loop clock and deadlocks _poll_until_terminal's
    # asyncio.sleep (the #474 ubuntu CI hang). See import_jobs._now.
    monkeypatch.setattr(import_jobs, "_now", lambda: fake_now[0])

    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")

    async def _run():
        first = import_jobs.submit(None)
        await _poll_until_terminal(first.job_id)
        fake_now[0] = 100.0  # well past the 10s TTL
        import_jobs.submit(None)  # sweeps at the top of submit
        return first.job_id

    first_job_id = asyncio.run(_run())
    assert import_jobs.status(first_job_id) is None


def test_terminal_job_not_evicted_before_ttl_elapses(jobs_env, monkeypatch):
    """A completed job still inside its TTL window survives a sweep."""
    from app import import_jobs

    monkeypatch.setenv("KB_IMPORT_JOB_TTL_SECONDS", "10")
    fake_now = [0.0]
    monkeypatch.setattr(import_jobs, "_now", lambda: fake_now[0])

    (jobs_env["raw_dir"] / "a.txt").write_text("alpha", encoding="utf-8")

    async def _run():
        first = import_jobs.submit(None)
        await _poll_until_terminal(first.job_id)
        fake_now[0] = 5.0  # inside the 10s TTL
        import_jobs.submit(None)
        return first.job_id

    first_job_id = asyncio.run(_run())
    assert import_jobs.status(first_job_id) is not None
