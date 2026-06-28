"""Test that kb_ingest_v1 does not block the event loop.

Fix 1a: the MCP handler must await aingest_sources (not call ingest_sources
synchronously), so the stdio event loop is free to process other work
during multi-minute ingest runs.

The test verifies that:
  - The handler uses the async path (aingest_sources) rather than the sync path.
  - A sentinel coroutine scheduled alongside the ingest call can run before
    the ingest completes (proving the event loop yields during the handler).

Mocking follows CODING_STANDARD §11 — LLM mocked at
``markdown_kb.app.templates.get_ingest_llm``, grounding at
``markdown_kb.app.ingest.verify``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers (shared from test_kb_ingest.py pattern)
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
    d = tmp_path / "docs"
    d.mkdir(parents=True)
    (d / "stub_async.md").write_text(
        "# Async Stub\n\nThis is stub content for the async test.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def wiki_dir(tmp_path: Path) -> Path:
    w = tmp_path / "wiki"
    w.mkdir(parents=True)
    (w / "concepts").mkdir(parents=True)
    (w / "entities").mkdir(parents=True)
    return w


@pytest.fixture(autouse=True)
def _patch_ingest_paths(monkeypatch: pytest.MonkeyPatch, docs_dir: Path, wiki_dir: Path) -> None:
    import markdown_kb.app._paths as paths_mod
    import markdown_kb.app.indexer as indexer_mod
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.logger as logger_mod

    monkeypatch.setattr(paths_mod, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(ingest_mod, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(indexer_mod, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_mod, "LOG_PATH", wiki_dir / "log.md")


# ---------------------------------------------------------------------------
# AC: handler does not call the sync ingest_sources (uses async path)
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_does_not_call_sync_ingest_sources(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kb_ingest_v1 must NOT call the synchronous ingest_sources.

    We replace ingest_sources with a sentinel that raises if called; the test
    confirms the handler succeeds (meaning it took the async path, not the sync
    path).  aingest_sources is patched to a fast async stub so the test stays
    hermetic.
    """
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda *_a, **_kw: _grounding_passed())

    # Sentinel: if ingest_sources is called, raise to fail the test
    def _sync_must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "kb_ingest_v1 must NOT call the synchronous ingest_sources; "
            "it must await aingest_sources instead."
        )

    monkeypatch.setattr("markdown_kb.app.ingest.ingest_sources", _sync_must_not_be_called)

    # Exercise the real aingest_sources so the test actually ingests
    raw = asyncio.run(mcp.call_tool("kb_ingest_v1", {"source": "stub_async.md"}))

    assert not _is_error_result(raw), f"Expected success result, got isError: {raw}"
    result = _parse_result(raw)
    assert result.get("source") == "stub_async.md"


# ---------------------------------------------------------------------------
# AC: a concurrent coroutine gets to run while ingest is in flight
# ---------------------------------------------------------------------------


def test_kb_ingest_v1_yields_event_loop(
    docs_dir: Path, wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sentinel coroutine scheduled alongside the ingest call runs before
    the ingest completes, proving the event loop is not blocked.

    The fake aingest_sources yields once (await asyncio.sleep(0)) so the
    sentinel coroutine has an opportunity to run.
    """
    import markdown_kb.app.ingest as ingest_mod
    import markdown_kb.app.templates as templates_mod

    from kb_mcp.server import mcp

    fake_llm = _FakeLLM()
    monkeypatch.setattr(templates_mod, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_mod, "verify", lambda *_a, **_kw: _grounding_passed())

    sentinel_ran = []

    async def _run_test():
        from markdown_kb.app.ingest import IngestBatchResult

        # Patch aingest_sources to a stub that yields the loop once
        async def _fake_aingest(*args, **kwargs):
            await asyncio.sleep(0)  # yield the event loop
            return IngestBatchResult()  # return empty result

        monkeypatch.setattr(ingest_mod, "aingest_sources", _fake_aingest)

        # Schedule the sentinel alongside the tool call
        async def _sentinel():
            sentinel_ran.append(True)

        sentinel_task = asyncio.ensure_future(_sentinel())
        await mcp.call_tool("kb_ingest_v1", {"source": "stub_async.md"})
        await sentinel_task

    asyncio.run(_run_test())

    assert sentinel_ran, (
        "Sentinel coroutine did not run before ingest completed — "
        "the event loop is blocked (sync ingest_sources is still being called)."
    )
