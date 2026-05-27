"""Tests for backward compatibility with legacy wiki pages without source_hashes.

Phase 6 left 9 wiki concept pages on disk without source_hashes. These pages
have default-empty dicts for source_hashes. The ingest pipeline must NOT skip
on empty source_hashes — treat as "unknown drift state" and proceed with ingest.

All tests hermetic — no OPENAI_API_KEY required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import app.templates as templates_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-05-26T14:30:00Z"
FIXED_BODY = "Test body for legacy test."


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


def _make_schema_aware_fake_llm(
    synthesis_body: str = FIXED_BODY,
    classifier_type: str = "concept",
) -> MagicMock:
    from app.templates import _ClassifierOutput

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            chain.invoke.return_value = _FakeClassifierOutput(classifier_type)
        else:
            chain.invoke.return_value = _FakeSynthesisOutput(synthesis_body)
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


def _make_legacy_wiki_page(
    wiki_dir: Path,
    subdir: str,
    slug: str,
    source_name: str,
) -> Path:
    """Create a Phase 6 legacy wiki page WITHOUT source_hashes field."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    # Legacy page: no source_hashes field (as Phase 6 pages look)
    fm_data = {
        "id": slug,
        "type": "concept" if subdir == "concepts" else "entity",
        "created": FIXED_TS,
        "updated": FIXED_TS,
        "sources": [f"{source_name}#{slug}"],
        "status": "live",
        "open_questions": [],
        # NOTE: no source_hashes field — this is the legacy state
    }
    fm_yaml = yaml.dump(fm_data, default_flow_style=False, allow_unicode=True)
    page_path.write_text(
        f"<!-- sentinel -->\n\n---\n{fm_yaml}---\n\n# {slug.title()}\n\nLegacy body text.\n\n[Source: {source_name}#{slug}]\n",
        encoding="utf-8",
    )
    return page_path


def _make_legacy_wiki_page_empty_source_hashes(
    wiki_dir: Path,
    subdir: str,
    slug: str,
    source_name: str,
) -> Path:
    """Create a wiki page with EXPLICIT empty source_hashes dict (Pydantic default)."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    # Page with explicit empty source_hashes
    fm_data = {
        "id": slug,
        "type": "concept" if subdir == "concepts" else "entity",
        "created": FIXED_TS,
        "updated": FIXED_TS,
        "sources": [f"{source_name}#{slug}"],
        "status": "live",
        "open_questions": [],
        "source_hashes": {},  # explicit empty dict — should NOT skip
    }
    fm_yaml = yaml.dump(fm_data, default_flow_style=False, allow_unicode=True)
    page_path.write_text(
        f"<!-- sentinel -->\n\n---\n{fm_yaml}---\n\n# {slug.title()}\n\nLegacy empty body.\n\n[Source: {source_name}#{slug}]\n",
        encoding="utf-8",
    )
    return page_path


# ---------------------------------------------------------------------------
# Tests: empty/missing source_hashes → NOT skip
# ---------------------------------------------------------------------------


def test_legacy_page_without_source_hashes_is_not_skipped(tmp_path, monkeypatch):
    """Page without source_hashes field triggers ingest (unknown drift state)."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Legacy Section\n\nLegacy content to be re-ingested.\n"
    source_path = docs_dir / "legacy.md"
    source_path.write_text(source_content, encoding="utf-8")

    # Plant legacy page with NO source_hashes
    _make_legacy_wiki_page(wiki_dir, "concepts", "legacy-section", "legacy.md")

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["legacy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    # MUST NOT skip — legacy pages always get re-ingested
    assert len(result.skipped_sources) == 0, (
        f"Legacy page without source_hashes should NOT be skipped, got: {result.skipped_sources}"
    )
    assert len(result.results) == 1, f"Expected 1 result, got: {result.results}"
    # LLM was called for synthesis
    assert fake_llm.with_structured_output.called, (
        "Expected LLM to be called for legacy page (no source_hashes = unknown drift)"
    )


def test_empty_source_hashes_dict_is_not_skipped(tmp_path, monkeypatch):
    """Page with explicit empty source_hashes dict triggers ingest (unknown drift state)."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Empty Hash Section\n\nContent with empty source_hashes.\n"
    source_path = docs_dir / "empty_hash.md"
    source_path.write_text(source_content, encoding="utf-8")

    # Plant page with EMPTY source_hashes dict
    _make_legacy_wiki_page_empty_source_hashes(
        wiki_dir, "concepts", "empty-hash-section", "empty_hash.md"
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["empty_hash.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    assert len(result.skipped_sources) == 0, (
        f"Empty source_hashes should NOT trigger skip, got: {result.skipped_sources}"
    )
    assert len(result.results) == 1


def test_wiki_page_frontmatter_default_source_hashes_is_empty_dict():
    """WikiPageFrontmatter.source_hashes defaults to an empty dict."""
    from app.schemas import WikiPageFrontmatter

    fm = WikiPageFrontmatter(
        id="test",
        type="concept",
        created=FIXED_TS,
        updated=FIXED_TS,
        sources=["test.md#test"],
        status="live",
        open_questions=[],
    )
    assert hasattr(fm, "source_hashes"), "WikiPageFrontmatter should have source_hashes"
    assert fm.source_hashes == {}, (
        f"Expected empty dict default for source_hashes, got: {fm.source_hashes!r}"
    )


def test_ingest_response_skipped_sources_empty_by_default():
    """IngestResponse.skipped_sources defaults to empty list."""
    from app.schemas import IngestResponse

    resp = IngestResponse(results=[], failed_sources=[])
    assert hasattr(resp, "skipped_sources"), "IngestResponse should have skipped_sources"
    assert resp.skipped_sources == [], f"Expected empty list default, got: {resp.skipped_sources!r}"


def test_ingest_source_result_status_field():
    """IngestSourceResult.status can be 'created', 'updated', or 'skipped'."""
    from app.schemas import IngestSourceResult

    created = IngestSourceResult(source="foo.md", pages_written=[], status="created")
    updated = IngestSourceResult(source="foo.md", pages_written=[], status="updated")
    skipped = IngestSourceResult(source="foo.md", pages_written=[], status="skipped")

    assert created.status == "created"
    assert updated.status == "updated"
    assert skipped.status == "skipped"
