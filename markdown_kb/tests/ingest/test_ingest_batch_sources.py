"""Tests for the multi-source batch path via POST /ingest with sources=[...].

Phase 15 S3 (issue #172): the Operator Console sends drop batches as a
SINGLE POST /ingest call with ``{"sources": ["a.md", "b.md", ...]}`` so
the cross-source ``used_slugs`` set is shared and slug collisions are
disambiguated with -2 suffixes.

AC covered:
- POST /ingest with sources=[...] resolves to the named sources (not all-docs).
- Two Sources producing a colliding slug are disambiguated (-2) through the
  batch path — the slug-collision regression test required by issue #172.
- Back-compat: existing ``source`` (singular) and no-body modes still work
  after the schema change.
- ``sources=[]`` (empty list) falls back to all-docs batch mode.

All tests are hermetic (no OPENAI_API_KEY required). The autouse conftest
fixture mocks ``app.ingest.verify`` and redirects WIKI_DIR / INDEX_PATH.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.templates as templates_module

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_ingest_integration.py pattern)
# ---------------------------------------------------------------------------

FIXED_BODY = "Synthesised body text for testing."
FIXED_TS = "2026-05-26T14:30:00Z"


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
    from app.templates import _ClassifierOutput  # private but accessible in tests

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
# Fixture: TestClient with patched LLM + tmp wiki dir
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_and_wiki(tmp_path, monkeypatch):
    """Return (TestClient, wiki_dir) with fake LLM and redirected paths.

    Does NOT redirect DOCS_DIR — each test creates its own docs dir and
    passes it via ``ingest_sources`` directly when testing the module, or
    sets it up when testing via the HTTP client.
    """
    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)

    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    return client, tmp_path / "wiki"


# ---------------------------------------------------------------------------
# AC: sources=[...] ingests only the named sources (not all-docs)
# ---------------------------------------------------------------------------


def test_ingest_sources_list_ingests_only_named_sources(tmp_path, monkeypatch):
    """POST /ingest with sources=[a.md] ingests only a.md, ignores b.md."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("## Alpha Section\n\nAlpha content.\n", encoding="utf-8")
    (docs_dir / "b.md").write_text("## Beta Section\n\nBeta content.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["a.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    # Only a.md should be processed
    processed = [r.source for r in result.results] + result.failed_sources
    assert "a.md" in processed, f"Expected a.md in processed, got {processed}"
    assert "b.md" not in processed, f"b.md should NOT be in processed, got {processed}"


def test_ingest_sources_list_via_http(tmp_path, monkeypatch):
    """POST /ingest with body {sources: [X, Y]} ingests those two files."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "x.md").write_text("## X Section\n\nX content.\n", encoding="utf-8")
    (docs_dir / "y.md").write_text("## Y Section\n\nY content.\n", encoding="utf-8")
    (docs_dir / "z.md").write_text("## Z Section\n\nZ content.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post("/ingest", json={"sources": ["x.md", "y.md"]})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    body = resp.json()
    assert body["failed_sources"] == [], f"Unexpected failures: {body['failed_sources']}"
    sources_processed = [r["source"] for r in body["results"]]
    assert "x.md" in sources_processed, f"Expected x.md in results, got {sources_processed}"
    assert "y.md" in sources_processed, f"Expected y.md in results, got {sources_processed}"
    assert "z.md" not in sources_processed, f"z.md should NOT be processed, got {sources_processed}"


# ---------------------------------------------------------------------------
# AC: slug collision regression — two Sources producing a colliding slug
# are disambiguated (-2) when ingested through the console batch path
# (POST /ingest with sources=[a.md, b.md]).
# ---------------------------------------------------------------------------


def test_batch_sources_colliding_slugs_disambiguated(tmp_path, monkeypatch):
    """Regression: two Sources in sources=[...] producing the same slug get -2.

    Both alpha.md and beta.md have a single "## Overview" section.
    When sent in ONE POST /ingest call via sources=[alpha.md, beta.md],
    the shared used_slugs set within ingest_sources() must guarantee that
    alpha gets 'overview' and beta gets 'overview-2' (or vice versa, since
    alphabetical ordering puts alpha.md first).

    This is the critical correctness invariant: a per-file loop of separate
    POST /ingest calls would RESET used_slugs between calls and silently
    overwrite 'overview' with beta's version, violating "a Section is never
    silently overwritten" (#54 / ADR-0011).
    """
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.md").write_text("## Overview\n\nAlpha overview text.\n", encoding="utf-8")
    (docs_dir / "beta.md").write_text("## Overview\n\nBeta overview text.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    # Single POST with both sources — used_slugs shared across the call
    resp = client.post("/ingest", json={"sources": ["alpha.md", "beta.md"]})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    body = resp.json()
    assert body["failed_sources"] == [], f"Unexpected failures: {body['failed_sources']}"

    all_pages = [p for r in body["results"] for p in r["pages_written"]]

    # One source gets "overview", the other gets "overview-2"
    assert "concepts/overview.md" in all_pages, f"Expected concepts/overview.md in {all_pages}"
    assert "concepts/overview-2.md" in all_pages, (
        f"Expected concepts/overview-2.md (disambiguation suffix) in {all_pages}\n"
        "If only 'overview' appears, the sources list was sent as two separate "
        "calls (resetting used_slugs), violating the single-batch-call rule (#54)."
    )

    # Both files must exist on disk
    assert (wiki_dir / "concepts" / "overview.md").exists(), (
        "concepts/overview.md not found on disk"
    )
    assert (wiki_dir / "concepts" / "overview-2.md").exists(), (
        "concepts/overview-2.md not found on disk"
    )


# ---------------------------------------------------------------------------
# Back-compat: source (singular) still works after schema change
# ---------------------------------------------------------------------------


def test_single_source_backcompat_still_works(tmp_path, monkeypatch):
    """POST /ingest with {source: x.md} still works (back-compat)."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "x.md").write_text("## X Section\n\nX content.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post("/ingest", json={"source": "x.md"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["source"] == "x.md"


# ---------------------------------------------------------------------------
# Back-compat: empty sources list falls through to all-docs batch mode
# ---------------------------------------------------------------------------


def test_empty_sources_list_falls_through_to_all_docs(tmp_path, monkeypatch):
    """POST /ingest with sources=[] falls back to all-docs batch mode.

    An empty list is treated as "not provided" (the route's priority check
    ``if req.sources`` is falsy for []) and falls through to all-docs mode.
    """
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("## A Section\n\nContent A.\n", encoding="utf-8")
    (docs_dir / "b.md").write_text("## B Section\n\nContent B.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post("/ingest", json={"sources": []})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    # All-docs mode should process both a.md and b.md
    sources_processed = [r["source"] for r in body["results"]]
    assert len(sources_processed) == 2, (
        f"Expected 2 sources processed (all-docs fallback), got {sources_processed}"
    )
    assert set(sources_processed) == {"a.md", "b.md"}


# ---------------------------------------------------------------------------
# sources takes priority over source when both are provided
# ---------------------------------------------------------------------------


def test_sources_list_takes_priority_over_source_singular(tmp_path, monkeypatch):
    """When both sources=[...] and source=... are provided, sources wins."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "priority.md").write_text("## Priority\n\nPriority content.\n", encoding="utf-8")
    (docs_dir / "ignored.md").write_text("## Ignored\n\nIgnored content.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    # sources wins over source
    resp = client.post("/ingest", json={"sources": ["priority.md"], "source": "ignored.md"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    sources_processed = [r["source"] for r in body["results"]] + body["failed_sources"]
    assert "priority.md" in sources_processed, (
        f"Expected priority.md to be processed, got {sources_processed}"
    )
    assert "ignored.md" not in sources_processed, (
        f"ignored.md should NOT be processed when sources=[...] is provided, got {sources_processed}"
    )
