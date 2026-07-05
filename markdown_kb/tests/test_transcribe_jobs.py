"""Unit tests for ``app.transcribe_jobs`` — async submit/poll batch registry (issue #459 AC5).

Mirrors ``kb_mcp/tests/test_kb_ingest_jobs.py``'s pattern, adapted for a
list-of-sources Transcribe batch with per-page progress instead of a single
ingest Source.

The LLM is mocked at ``app.transcriber.get_transcribe_llm`` (CODING_STANDARD
§6.3) — never ``transcribe_jobs`` or the deep-module entry points.

Isolation: ``transcribe_jobs._reset_jobs()`` runs autouse per test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from .conftest import FakeLLMResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"


class FakeTranscribeLLM:
    def __init__(self, body: str = "# batch job\ntranscribed body."):
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
def jobs_env(tmp_path, monkeypatch):
    """Wire raw_dir, docs_dir, and log path into transcriber.py for isolation."""
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

    return {"raw_dir": raw_dir, "docs_dir": docs_dir, "fake_llm": fake_llm}


async def _poll_until_terminal(job_id: str, *, timeout: float = 5.0) -> None:
    from app import transcribe_jobs

    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        job = transcribe_jobs.status(job_id)
        assert job is not None
        if job.status in ("completed", "failed"):
            return
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"job {job_id} did not reach a terminal state: {job}")
        await asyncio.sleep(0.01)


def test_submit_batch_returns_job_id_immediately(jobs_env):
    """submit_batch must not block for the batch's duration."""
    from app import transcribe_jobs

    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        job = transcribe_jobs.submit_batch(["a.pdf"])
        assert job.job_id
        assert job.status in ("submitted", "working")

    asyncio.run(_run())


def test_status_unknown_job_id_returns_none(jobs_env):
    from app import transcribe_jobs

    assert transcribe_jobs.status("does-not-exist") is None


def test_batch_completes_and_writes_docs(jobs_env):
    from app import transcribe_jobs

    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (jobs_env["raw_dir"] / "b.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        job = transcribe_jobs.submit_batch(["a.pdf", "renamed_b.pdf"])
        await _poll_until_terminal(job.job_id)
        return transcribe_jobs.status(job.job_id)

    # b.pdf staged under a name that doesn't exist on disk to exercise the
    # continue-on-error path in the SAME batch as a real success.
    final = asyncio.run(_run())

    assert final.status == "completed"
    assert len(final.results) == 2
    by_source = {r.source: r for r in final.results}
    assert by_source["a.pdf"].status == "created"
    assert (jobs_env["docs_dir"] / "a.md").exists()
    assert by_source["renamed_b.pdf"].status == "failed"
    assert by_source["renamed_b.pdf"].error_type == "FileNotFoundError"


def test_batch_progress_reaches_pages_total_at_completion(jobs_env):
    from app import transcribe_jobs

    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        job = transcribe_jobs.submit_batch(["a.pdf"])
        await _poll_until_terminal(job.job_id)
        return transcribe_jobs.status(job.job_id)

    final = asyncio.run(_run())

    assert final.pages_total == 1
    assert final.pages_done == 1


def test_batch_continues_after_one_source_fails(jobs_env, monkeypatch):
    """A page-limit failure on one source must not abort the rest of the batch."""
    from app import transcribe_jobs

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_PAGES", "0")
    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (jobs_env["raw_dir"] / "b.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        job = transcribe_jobs.submit_batch(["a.pdf", "b.pdf"])
        await _poll_until_terminal(job.job_id)
        return transcribe_jobs.status(job.job_id)

    final = asyncio.run(_run())

    assert final.status == "completed"
    assert len(final.results) == 2
    assert all(r.status == "failed" for r in final.results)
    assert all(r.error_type == "TranscribePageLimitExceeded" for r in final.results)


# ---------------------------------------------------------------------------
# Concurrent-job cap (issue #474 sub-issue A)
# ---------------------------------------------------------------------------


def test_submit_batch_rejects_when_concurrent_job_cap_reached(jobs_env, monkeypatch):
    """A second submit while the cap's worth of jobs are still active is rejected clearly."""
    from app import transcribe_jobs
    from app.transcriber import TranscribePathError

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_CONCURRENT_JOBS", "1")
    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (jobs_env["raw_dir"] / "b.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        first = transcribe_jobs.submit_batch(["a.pdf"])
        # No await has run yet, so the scheduled Task has not started —
        # `first` is still status="submitted" and must count as active.
        assert first.status == "submitted"
        with pytest.raises(TranscribePathError) as exc_info:
            transcribe_jobs.submit_batch(["b.pdf"])
        return exc_info.value

    exc = asyncio.run(_run())
    assert exc.error_type == "TranscribeJobCapacityExceeded"


def test_submit_batch_allows_second_job_under_cap(jobs_env, monkeypatch):
    """A cap of 2 admits two concurrently-active jobs without rejection."""
    from app import transcribe_jobs

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_CONCURRENT_JOBS", "2")
    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (jobs_env["raw_dir"] / "b.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        first = transcribe_jobs.submit_batch(["a.pdf"])
        second = transcribe_jobs.submit_batch(["b.pdf"])
        return first, second

    first, second = asyncio.run(_run())
    assert first.job_id != second.job_id


def test_submit_batch_allows_new_job_once_previous_completes(jobs_env, monkeypatch):
    """The cap only counts ACTIVE jobs — a terminal job frees a slot."""
    from app import transcribe_jobs

    monkeypatch.setenv("KB_TRANSCRIBE_MAX_CONCURRENT_JOBS", "1")
    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (jobs_env["raw_dir"] / "b.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        first = transcribe_jobs.submit_batch(["a.pdf"])
        await _poll_until_terminal(first.job_id)
        second = transcribe_jobs.submit_batch(["b.pdf"])
        return second

    second = asyncio.run(_run())
    assert second.job_id


# ---------------------------------------------------------------------------
# Terminal-job eviction (issue #474 sub-issue B)
# ---------------------------------------------------------------------------


def test_terminal_job_evicted_once_ttl_elapses(jobs_env, monkeypatch):
    """A completed job past its TTL is swept on the next submit_batch call."""
    from app import transcribe_jobs

    monkeypatch.setenv("KB_TRANSCRIBE_JOB_TTL_SECONDS", "10")
    fake_now = [0.0]
    # Patch the job-TTL clock seam, NOT time.monotonic — patching the latter
    # also freezes the asyncio loop clock and deadlocks _poll_until_terminal's
    # asyncio.sleep (the #474 ubuntu CI hang). See transcribe_jobs._now.
    monkeypatch.setattr(transcribe_jobs, "_now", lambda: fake_now[0])

    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (jobs_env["raw_dir"] / "b.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        first = transcribe_jobs.submit_batch(["a.pdf"])
        await _poll_until_terminal(first.job_id)
        fake_now[0] = 100.0  # well past the 10s TTL
        transcribe_jobs.submit_batch(["b.pdf"])  # sweeps at the top of submit_batch
        return first.job_id

    first_job_id = asyncio.run(_run())
    assert transcribe_jobs.status(first_job_id) is None


def test_terminal_job_not_evicted_before_ttl_elapses(jobs_env, monkeypatch):
    """A completed job still inside its TTL window survives a sweep."""
    from app import transcribe_jobs

    monkeypatch.setenv("KB_TRANSCRIBE_JOB_TTL_SECONDS", "10")
    fake_now = [0.0]
    # Patch the job-TTL clock seam, NOT time.monotonic — patching the latter
    # also freezes the asyncio loop clock and deadlocks _poll_until_terminal's
    # asyncio.sleep (the #474 ubuntu CI hang). See transcribe_jobs._now.
    monkeypatch.setattr(transcribe_jobs, "_now", lambda: fake_now[0])

    (jobs_env["raw_dir"] / "a.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())
    (jobs_env["raw_dir"] / "b.pdf").write_bytes((FIXTURES / "sample_english.pdf").read_bytes())

    async def _run():
        first = transcribe_jobs.submit_batch(["a.pdf"])
        await _poll_until_terminal(first.job_id)
        fake_now[0] = 5.0  # inside the 10s TTL
        transcribe_jobs.submit_batch(["b.pdf"])
        return first.job_id

    first_job_id = asyncio.run(_run())
    assert transcribe_jobs.status(first_job_id) is not None
