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
