"""Tests for the submit/poll async ingest job registry (Fix 1b).

Issue: large Sources that would exceed the MCP host tool-call timeout (-32001)
need an escape hatch.  Fix 1b adds a submit/poll pattern:

  kb_ingest_start_v1(source)  → {job_id, status}  (returns immediately)
  kb_ingest_status_v1(job_id) → {job_id, status, progress, result|error}

The underlying work runs as a background asyncio.Task inside the MCP server's
event loop.  The registry (ingest_jobs.py) keeps a strong reference to the Task
so the GC cannot cancel it mid-run.

Scenarios covered:
  test_start_returns_job_id_immediately     — start tool does not block
  test_status_reports_completed             — job completes, carries pages_created
  test_status_unknown_job_id                — bogus id → not-found shape, never raises
  test_failed_job_carries_llm_error         — LLMError → job.status=="failed", error dict
  test_job_lifecycle_transitions            — submit/working/completed transitions
  test_start_status_strict_schema           — both tools have additionalProperties:false
  test_large_source_routed_to_start         — over-soft-cap Source sent to kb_ingest_v1
                                              returns routed_async payload immediately

Isolation: ingest_jobs._reset_jobs() is called as autouse per test.
Path isolation is handled by the kb_mcp conftest _isolate_module_state.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers (shared with other kb_mcp test files)
# ---------------------------------------------------------------------------


def _parse_result(raw: Any) -> dict:
    """Extract the dict payload from a FastMCP tool call result."""
    from mcp.types import CallToolResult

    if isinstance(raw, CallToolResult):
        return json.loads(raw.content[0].text)
    if isinstance(raw, list):
        item = raw[0]
        return json.loads(item.text)
    return json.loads(raw)


def _is_error_result(raw: Any) -> bool:
    from mcp.types import CallToolResult

    return isinstance(raw, CallToolResult) and raw.isError is True


# ---------------------------------------------------------------------------
# Stubs: LLM / grounding  (identical to test_kb_ingest.py pattern)
# ---------------------------------------------------------------------------


class _FakeLLMStructured:
    def __init__(self, schema: Any) -> None:
        self._schema = schema

    def invoke(self, messages: Any) -> Any:
        schema = self._schema
        fields = set(getattr(schema, "model_fields", {}).keys())
        if "type" in fields and "body" not in fields:
            return schema(type="concept")
        elif "body" in fields:
            return schema(
                body="A stub concept page body.",
                open_questions=[],
            )
        else:
            try:
                return schema(type="concept")
            except Exception:
                return schema()


class _FakeLLM:
    def with_structured_output(self, schema: Any, **kwargs: Any) -> _FakeLLMStructured:
        return _FakeLLMStructured(schema=schema)


def _grounding_passed():
    from markdown_kb.app.grounding import GroundingOutcome

    return GroundingOutcome(passed=True, reason="claim_supported", retries_attempted=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    """A tmp docs/ directory with one minimal Source (within soft cap)."""
    d = tmp_path / "docs"
    d.mkdir(parents=True)
    (d / "stub_jobs.md").write_text(
        "# Jobs Stub\n\nThis is stub content for the job registry test.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def docs_dir_large(tmp_path: Path) -> Path:
    """A tmp docs/ directory with a Source whose token estimate exceeds the soft cap.

    Writes a Source large enough that _estimate_tokens(content) > _max_ingest_tokens()
    even with the default 64 000 token cap.  Uses a 1-token-per-3-chars estimate,
    so we need > 64 000 * 3 = 192 000 chars.  We write 200 000 'a' chars split
    across 2 sections.

    NOTE: uses a separate subdir (large_docs/) to avoid colliding with the
    docs/ created by docs_dir in the _patch_ingest_paths autouse fixture.
    """
    d = tmp_path / "large_docs"
    d.mkdir(parents=True)
    large_body = "a " * 96_001  # ~192 002 chars → ~64 001 token estimate
    content = f"# Big Section One\n\n{large_body}\n\n# Big Section Two\n\n{large_body}\n"
    (d / "large_source.md").write_text(content, encoding="utf-8")
    return d


@pytest.fixture()
def wiki_dir(tmp_path: Path) -> Path:
    w = tmp_path / "wiki"
    w.mkdir(parents=True)
    (w / "concepts").mkdir(parents=True)
    (w / "entities").mkdir(parents=True)
    return w


@pytest.fixture(autouse=True)
def _patch_ingest_paths(
    monkeypatch: pytest.MonkeyPatch, docs_dir: Path, wiki_dir: Path
) -> None:
    """Redirect ingest deep module paths to tmp dirs (mirrors test_kb_ingest.py)."""
    import markdown_kb.app._paths as paths_mod
    import markdown_kb.app.indexer as indexer_mod
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.logger as logger_mod

    monkeypatch.setattr(paths_mod, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(ingest_mod, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(indexer_mod, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_mod, "LOG_PATH", wiki_dir / "log.md")


@pytest.fixture(autouse=True)
def _reset_job_registry() -> None:
    """Clear the job registry before each test so jobs don't leak."""
    from kb_mcp import ingest_jobs

    ingest_jobs._reset_jobs()
    yield
    ingest_jobs._reset_jobs()


# ---------------------------------------------------------------------------
# Helpers: poll until a job reaches a terminal status
# ---------------------------------------------------------------------------


async def _poll_status(job_id: str, *, timeout: float = 5.0) -> dict:
    """Poll kb_ingest_status_v1 until the job is in a terminal state.

    Returns the final status payload dict.  Raises TimeoutError when the
    job does not complete within ``timeout`` seconds.
    """
    from kb_mcp.server import mcp

    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        raw = await mcp.call_tool("kb_ingest_status_v1", {"job_id": job_id})
        result = _parse_result(raw)
        if result.get("status") in ("completed", "failed"):
            return result
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s: {result}")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_start_returns_job_id_immediately(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kb_ingest_start_v1 returns {job_id, status} immediately without blocking.

    The status at return time must be in {submitted, working} — not completed.
    The tool must NOT await the full pipeline before returning.
    """
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda *_a, **_kw: _grounding_passed())

    # Use a slow-ish fake aingest to ensure the tool returns before work completes.
    async def _slow_aingest(*args, **kwargs):
        from markdown_kb.app.ingest import IngestBatchResult

        await asyncio.sleep(0.1)  # slow enough that we can observe submitted status
        return IngestBatchResult()

    monkeypatch.setattr(ingest_mod, "aingest_sources", _slow_aingest)

    async def _run():
        raw = await mcp.call_tool("kb_ingest_start_v1", {"source": "stub_jobs.md"})
        assert not _is_error_result(raw), f"Expected success, got isError: {raw}"
        result = _parse_result(raw)
        assert "job_id" in result, f"Missing job_id: {result}"
        assert result.get("status") in ("submitted", "working"), (
            f"Expected submitted/working at return time, got: {result}"
        )
        assert result["job_id"], "job_id must be non-empty"

    asyncio.run(_run())


def test_status_reports_completed(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After start, polling kb_ingest_status_v1 eventually reports completed.

    The completed payload must include pages_created.
    """
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda *_a, **_kw: _grounding_passed())

    async def _run():
        # Start the job
        raw = await mcp.call_tool("kb_ingest_start_v1", {"source": "stub_jobs.md"})
        assert not _is_error_result(raw), f"Start failed: {raw}"
        result = _parse_result(raw)
        job_id = result["job_id"]

        # Poll until completed
        final = await _poll_status(job_id)
        assert final["status"] == "completed", f"Expected completed: {final}"
        assert "result" in final, f"completed job must have 'result': {final}"
        assert "pages_created" in final["result"], (
            f"result must carry pages_created: {final['result']}"
        )

    asyncio.run(_run())


def test_status_unknown_job_id(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kb_ingest_status_v1 on a bogus job_id returns a not-found shape, never raises."""
    from kb_mcp.server import mcp

    async def _run():
        raw = await mcp.call_tool(
            "kb_ingest_status_v1", {"job_id": "does-not-exist-00000000"}
        )
        # Must not be an exception — just a not-found shape
        result = _parse_result(raw)
        assert result.get("status") == "unknown", (
            f"Expected status='unknown' for bogus job_id, got: {result}"
        )

    asyncio.run(_run())


def test_failed_job_carries_llm_error(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the background ingest raises LLMError, the job ends status=='failed'.

    The error dict must carry the error code (LLM_UNAVAILABLE or LLM_ERROR).
    """
    import markdown_kb.app.templates as templates_mod
    from markdown_kb.app.errors import LLMError

    from kb_mcp.server import mcp

    def _raise_llm_error():
        raise LLMError(retryable=True, message="LLM service down (test stub).")

    monkeypatch.setattr(templates_mod, "get_ingest_llm", _raise_llm_error)

    async def _run():
        raw = await mcp.call_tool("kb_ingest_start_v1", {"source": "stub_jobs.md"})
        assert not _is_error_result(raw), f"Start must succeed (job submitted): {raw}"
        result = _parse_result(raw)
        job_id = result["job_id"]

        final = await _poll_status(job_id)
        assert final["status"] == "failed", f"Expected failed: {final}"
        assert "error" in final, f"failed job must carry 'error': {final}"
        error = final["error"]
        assert "code" in error, f"error dict must have 'code': {error}"
        assert error["code"] in ("LLM_UNAVAILABLE", "LLM_ERROR"), (
            f"code must be LLM_UNAVAILABLE or LLM_ERROR: {error}"
        )

    asyncio.run(_run())


def test_job_lifecycle_transitions(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a slow fake aingest, observe job moving through submitted/working then completed."""
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda *_a, **_kw: _grounding_passed())

    statuses_seen: list[str] = []

    async def _slow_aingest(*args, **kwargs):
        from markdown_kb.app.ingest import IngestBatchResult

        await asyncio.sleep(0.15)  # slow enough to observe intermediate states
        return IngestBatchResult()

    monkeypatch.setattr(ingest_mod, "aingest_sources", _slow_aingest)

    async def _run():
        from kb_mcp import ingest_jobs

        raw = await mcp.call_tool("kb_ingest_start_v1", {"source": "stub_jobs.md"})
        result = _parse_result(raw)
        job_id = result["job_id"]

        # Immediately check — should be submitted or working
        job = ingest_jobs.status(job_id)
        assert job is not None, "Job must be registered immediately after start"
        statuses_seen.append(job.status)

        # Let the background task run
        final = await _poll_status(job_id)
        statuses_seen.append(final["status"])

    asyncio.run(_run())

    # Must have seen at least one non-terminal status before completed
    assert "completed" in statuses_seen, f"Must reach completed: {statuses_seen}"
    # The first observation must be a non-terminal state
    assert statuses_seen[0] in ("submitted", "working"), (
        f"First status must be submitted or working: {statuses_seen}"
    )


def test_start_status_strict_schema(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both kb_ingest_start_v1 and kb_ingest_status_v1 have additionalProperties:false.

    Parity with test_kb_ingest_v1_strict_schema (ADR-0016).
    """
    from kb_mcp.server import mcp

    for tool_name in ("kb_ingest_start_v1", "kb_ingest_status_v1"):
        tool = mcp._tool_manager.get_tool(tool_name)
        assert tool is not None, f"{tool_name} not registered in server"
        assert tool.parameters.get("additionalProperties") is False, (
            f"{tool_name}: expected additionalProperties:false, "
            f"got: {tool.parameters}"
        )


def test_large_source_routed_to_start(
    docs_dir_large: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Source exceeding the soft token cap sent to kb_ingest_v1 returns routed_async.

    The tool must return immediately with {status: 'routed_async', job_id: ..., note: ...}
    rather than running the full pipeline inline (which would block and risk -32001).
    """
    import markdown_kb.app._paths as paths_mod
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    # Override the docs dir for THIS test to point at docs_dir_large
    monkeypatch.setattr(paths_mod, "DOCS_DIR", docs_dir_large)
    monkeypatch.setattr(ingest_mod, "DOCS_DIR", docs_dir_large)

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda *_a, **_kw: _grounding_passed())

    # Patch aingest_sources to track whether it was called (it should NOT
    # complete inline for a large source — the routing should intercept first).
    aingest_called = []

    async def _fake_aingest(*args, **kwargs):
        from markdown_kb.app.ingest import IngestBatchResult

        aingest_called.append(True)
        # Simulate slow ingest — this should never block the tool return
        await asyncio.sleep(10)  # if this runs inline, the test would hang
        return IngestBatchResult()

    monkeypatch.setattr(ingest_mod, "aingest_sources", _fake_aingest)

    async def _run():
        # Use a very tight timeout — if the tool blocks, the test fails fast
        raw = await asyncio.wait_for(
            mcp.call_tool("kb_ingest_v1", {"source": "large_source.md"}),
            timeout=3.0,
        )
        assert not _is_error_result(raw), f"Expected success result: {raw}"
        result = _parse_result(raw)
        assert result.get("status") == "routed_async", (
            f"Expected status='routed_async' for over-cap Source, got: {result}"
        )
        assert "job_id" in result, f"routed_async result must carry job_id: {result}"
        assert result["job_id"], "job_id must be non-empty"
        assert "note" in result, f"routed_async result must carry note: {result}"

    asyncio.run(_run())
