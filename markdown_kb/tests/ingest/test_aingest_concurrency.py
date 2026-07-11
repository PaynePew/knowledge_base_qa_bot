"""Tests for aingest_sources bounded-concurrency async ingest path.

All tests hermetic — no OPENAI_API_KEY, no network.

LLM mocked at ``app.templates.get_ingest_llm`` (lazy-singleton getter).
Grounding mocked at ``app.ingest.verify`` (via conftest autouse default).

Tests:
1. test_aingest_produces_same_pages_as_sync
2. test_aingest_slug_determinism_matches_sync
3. test_aingest_respects_semaphore
4. test_aingest_runs_sections_concurrently
5. test_aingest_grounding_failure_surfaced
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.indexer as indexer_module
import app.ingest as ingest_module
import app.templates as templates_module

# ---------------------------------------------------------------------------
# Shared fake LLM helpers (same pattern as test_ingest_batch_sources.py)
# ---------------------------------------------------------------------------

FIXED_BODY = "Synthesised body text for testing."


class _FakeSynthesisOutput:
    body: str
    open_questions: list

    def __init__(self, body: str = FIXED_BODY, open_questions: list | None = None):
        self.body = body
        self.open_questions = open_questions or []


class _FakeClassifierOutput:
    type: str

    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_schema_aware_fake_llm(classifier_type: str = "concept") -> MagicMock:
    """Return a fake ChatOpenAI whose with_structured_output is schema-aware."""
    from app.templates import _ClassifierOutput

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            chain.invoke.return_value = _FakeClassifierOutput(classifier_type)
        else:
            chain.invoke.return_value = _FakeSynthesisOutput()
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


# ---------------------------------------------------------------------------
# Test 1: aingest_sources produces the same pages as ingest_sources
# ---------------------------------------------------------------------------


def test_aingest_produces_same_pages_as_sync(tmp_path, monkeypatch):
    """aingest_sources and ingest_sources over the same multi-section concept
    Source produce identical slugs and identical page set.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir_sync = tmp_path / "wiki_sync"
    wiki_dir_async = tmp_path / "wiki_async"

    source = docs_dir / "multi.md"
    source.write_text(
        "## Introduction\n\nIntro content.\n\n"
        "## Background\n\nBackground content.\n\n"
        "## Summary\n\nSummary content.\n",
        encoding="utf-8",
    )

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir_sync)

    sync_result = ingest_module.ingest_sources(
        ["multi.md"], docs_dir=docs_dir, wiki_dir=wiki_dir_sync
    )

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir_async)

    async_result = asyncio.run(
        ingest_module.aingest_sources(["multi.md"], docs_dir=docs_dir, wiki_dir=wiki_dir_async)
    )

    assert sync_result.failed_sources == [], f"Sync failures: {sync_result.failed_sources}"
    assert async_result.failed_sources == [], f"Async failures: {async_result.failed_sources}"

    sync_pages = sorted(p for r in sync_result.results for p in r.pages_written)
    async_pages = sorted(p for r in async_result.results for p in r.pages_written)

    assert sync_pages == async_pages, f"Slug mismatch: sync={sync_pages} async={async_pages}"


# ---------------------------------------------------------------------------
# Test 2: aingest_sources slug determinism matches sync (collision regression)
# ---------------------------------------------------------------------------


def test_aingest_slug_determinism_matches_sync(tmp_path, monkeypatch):
    """A Source with 3 sections all titled 'Overview' yields overview /
    overview-2 / overview-3 in section order, identical to the sync path.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    source = docs_dir / "overview.md"
    source.write_text(
        "## Overview\n\nFirst overview.\n\n"
        "## Overview\n\nSecond overview.\n\n"
        "## Overview\n\nThird overview.\n",
        encoding="utf-8",
    )

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)

    wiki_dir_sync = tmp_path / "wiki_sync"
    wiki_dir_async = tmp_path / "wiki_async"
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir_sync)

    sync_result = ingest_module.ingest_sources(
        ["overview.md"], docs_dir=docs_dir, wiki_dir=wiki_dir_sync
    )

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir_async)

    async_result = asyncio.run(
        ingest_module.aingest_sources(["overview.md"], docs_dir=docs_dir, wiki_dir=wiki_dir_async)
    )

    assert sync_result.failed_sources == []
    assert async_result.failed_sources == []

    sync_pages = sorted(p for r in sync_result.results for p in r.pages_written)
    async_pages = sorted(p for r in async_result.results for p in r.pages_written)

    # Both must produce exactly overview, overview-2, overview-3
    expected = sorted(["concepts/overview.md", "concepts/overview-2.md", "concepts/overview-3.md"])
    assert sync_pages == expected, f"Sync produced unexpected slugs: {sync_pages}"
    assert async_pages == expected, f"Async produced unexpected slugs: {async_pages}"


# ---------------------------------------------------------------------------
# Test 3: aingest_sources respects semaphore (KB_INGEST_CONCURRENCY)
# ---------------------------------------------------------------------------


def test_aingest_respects_semaphore(tmp_path, monkeypatch):
    """KB_INGEST_CONCURRENCY=2 limits peak in-flight LLM calls to 2.

    The fake generate_page uses threading.Lock + time.sleep to measure
    peak concurrency (it runs in a worker thread via to_thread).
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # 6 sections to give the semaphore room to prove itself
    content = "".join(f"## Section {i}\n\nContent {i}.\n\n" for i in range(1, 7))
    (docs_dir / "source.md").write_text(content, encoding="utf-8")

    monkeypatch.setenv("KB_INGEST_CONCURRENCY", "2")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    peak_concurrent = [0]
    current_concurrent = [0]
    counter_lock = threading.Lock()

    original_generate_page = ingest_module.generate_page

    def _tracked_generate_page(section, source_type, **kwargs):
        with counter_lock:
            current_concurrent[0] += 1
            if current_concurrent[0] > peak_concurrent[0]:
                peak_concurrent[0] = current_concurrent[0]
        time.sleep(0.02)  # hold slot briefly so concurrency can be observed
        try:
            return original_generate_page(section, source_type, **kwargs)
        finally:
            with counter_lock:
                current_concurrent[0] -= 1

    # Patch generate_page on both the ingest module namespace and the original
    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_module, "generate_page", _tracked_generate_page)

    asyncio.run(ingest_module.aingest_sources(["source.md"], docs_dir=docs_dir, wiki_dir=wiki_dir))

    assert peak_concurrent[0] <= 2, (
        f"Expected peak concurrency <= 2 (KB_INGEST_CONCURRENCY=2), got peak={peak_concurrent[0]}"
    )


# ---------------------------------------------------------------------------
# Test 4: aingest_sources runs sections concurrently (rendezvous check)
# ---------------------------------------------------------------------------


def test_aingest_runs_sections_concurrently(tmp_path, monkeypatch):
    """At least two generate_page calls overlap in flight (#568).

    Deterministic rendezvous instead of a wall-clock bound: each fake call
    increments an in-flight counter, then holds its slot until a second call
    is in flight (bounded wait). With real fan-out the peer arrives while the
    first call is still holding, the event fires, and every call proceeds
    immediately — machine load only delays the rendezvous, never breaks it.
    A serial implementation can never have two calls in flight, so the first
    call times out alone and the peak assertion fails.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    content = "".join(f"## Section {i}\n\nContent {i}.\n\n" for i in range(1, 9))
    (docs_dir / "concurrent.md").write_text(content, encoding="utf-8")

    monkeypatch.setenv("KB_INGEST_CONCURRENCY", "8")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    original_generate_page = ingest_module.generate_page

    counter_lock = threading.Lock()
    in_flight = [0]
    peak_in_flight = [0]
    overlap = threading.Event()  # set the moment 2 calls are in flight together
    gave_up = threading.Event()  # first timeout short-circuits later waits

    def _rendezvous_generate_page(section, source_type, **kwargs):
        with counter_lock:
            in_flight[0] += 1
            if in_flight[0] > peak_in_flight[0]:
                peak_in_flight[0] = in_flight[0]
            if in_flight[0] >= 2:
                overlap.set()
        try:
            if not gave_up.is_set() and not overlap.wait(timeout=10):
                gave_up.set()
            return original_generate_page(section, source_type, **kwargs)
        finally:
            with counter_lock:
                in_flight[0] -= 1

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(ingest_module, "generate_page", _rendezvous_generate_page)

    asyncio.run(
        ingest_module.aingest_sources(["concurrent.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    )

    assert peak_in_flight[0] >= 2, (
        f"Expected >= 2 generate_page calls in flight simultaneously "
        f"(KB_INGEST_CONCURRENCY=8), got peak={peak_in_flight[0]} — "
        "parallel fan-out may be missing"
    )


# ---------------------------------------------------------------------------
# Test 5: aingest grounding failure is surfaced through the async path
# ---------------------------------------------------------------------------


def test_aingest_grounding_failure_surfaced(tmp_path, monkeypatch):
    """A section whose verify returns claim_unsupported appears in
    pages_with_failed_grounding (fail-soft preserved through the async path).
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    (docs_dir / "fail.md").write_text(
        "## Topic\n\nSome content that will fail grounding.\n",
        encoding="utf-8",
    )

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.grounding import GroundingOutcome

    failed_outcome = GroundingOutcome(passed=False, reason="claim_unsupported", result=None)
    monkeypatch.setattr(ingest_module, "verify", lambda *_a, **_kw: failed_outcome)

    result = asyncio.run(
        ingest_module.aingest_sources(["fail.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    )

    assert result.failed_sources == [], (
        f"Source should not be in failed_sources: {result.failed_sources}"
    )
    assert result.pages_with_failed_grounding, (
        f"Expected pages_with_failed_grounding to be non-empty: {result}"
    )
