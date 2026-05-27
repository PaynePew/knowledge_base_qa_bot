"""Tests for source_hashes written to wiki frontmatter — Phase 3 amendment (issue #93).

Verifies that after ingest:
- wiki page frontmatter contains `source_hashes` dict
- `source_hashes[source_filename]["docs_body"]` is the SHA-256 of the source file bytes
- `source_hashes[source_filename]["raw"]` is the content_sha256 from docs frontmatter when present
- `source_hashes[source_filename]["raw"]` is null when docs frontmatter has no content_sha256

All tests hermetic — no OPENAI_API_KEY required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import app.templates as templates_module
from app.schemas import WikiPageDraft, WikiPageFrontmatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-05-26T14:30:00Z"
FIXED_BODY = "Test body content."


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


def _compute_docs_body_hash(source_path: Path) -> str:
    """Compute SHA-256 of the docs file content as utf-8 bytes."""
    return hashlib.sha256(source_path.read_text(encoding="utf-8").encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tests: source_hashes written to wiki frontmatter
# ---------------------------------------------------------------------------


def test_source_hashes_written_to_wiki_frontmatter(tmp_path, monkeypatch):
    """After ingest, wiki page frontmatter contains source_hashes for the source file."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Policy Overview\n\nThis policy describes our refund terms.\n"
    source_path = docs_dir / "policy.md"
    source_path.write_text(source_content, encoding="utf-8")

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert len(result.results) == 1

    # Find the written wiki page
    written_pages = result.results[0].pages_written
    assert len(written_pages) >= 1

    page_path = wiki_dir / written_pages[0]
    assert page_path.exists()
    content = page_path.read_text(encoding="utf-8")

    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    parsed = yaml.safe_load(fm_text)

    assert "source_hashes" in parsed, (
        f"Expected 'source_hashes' in frontmatter, got keys: {list(parsed.keys())}"
    )
    source_hashes = parsed["source_hashes"]
    assert isinstance(source_hashes, dict), f"Expected dict, got: {type(source_hashes)}"
    assert "policy.md" in source_hashes, (
        f"Expected 'policy.md' key in source_hashes, got: {source_hashes}"
    )

    entry = source_hashes["policy.md"]
    assert "docs_body" in entry, f"Expected 'docs_body' in entry: {entry}"


def test_source_hashes_docs_body_hash_is_correct(tmp_path, monkeypatch):
    """docs_body hash in source_hashes is SHA-256 of source file utf-8 content."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Shipping Policy\n\nWe ship worldwide in 3-5 business days.\n"
    source_path = docs_dir / "shipping.md"
    source_path.write_text(source_content, encoding="utf-8")

    expected_hash = _compute_docs_body_hash(source_path)

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["shipping.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    page_path = wiki_dir / result.results[0].pages_written[0]
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    parsed = yaml.safe_load(fm_text)

    actual_hash = parsed["source_hashes"]["shipping.md"]["docs_body"]
    assert actual_hash == expected_hash, (
        f"Expected docs_body hash {expected_hash!r}, got {actual_hash!r}"
    )


def test_source_hashes_raw_is_none_when_no_content_sha256_in_docs(tmp_path, monkeypatch):
    """raw field is null when docs frontmatter has no content_sha256 (hand-authored doc)."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Source without content_sha256 frontmatter (hand-authored)
    source_content = "## Hand Authored\n\nThis doc has no import frontmatter.\n"
    source_path = docs_dir / "hand_authored.md"
    source_path.write_text(source_content, encoding="utf-8")

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(
        ["hand_authored.md"], docs_dir=docs_dir, wiki_dir=wiki_dir
    )

    assert result.failed_sources == []
    page_path = wiki_dir / result.results[0].pages_written[0]
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    parsed = yaml.safe_load(fm_text)

    raw_value = parsed["source_hashes"]["hand_authored.md"].get("raw")
    assert raw_value is None, f"Expected raw=null, got: {raw_value!r}"


def test_source_hashes_raw_is_populated_from_docs_frontmatter(tmp_path, monkeypatch):
    """raw field is populated from content_sha256 in docs frontmatter (imported doc)."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    fake_sha256 = "abc123def456" * 4 + "abc123de"  # 56 chars, not real but valid
    # Source with content_sha256 in frontmatter (as written by importer.py)
    source_content = f"---\nimported_from: raw/imported.html\noriginal_format: html\nimported_at: 2026-05-26T12:00:00Z\ncontent_sha256: {fake_sha256}\n---\n## Imported Content\n\nThis was imported.\n"
    source_path = docs_dir / "imported.md"
    source_path.write_text(source_content, encoding="utf-8")

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["imported.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    page_path = wiki_dir / result.results[0].pages_written[0]
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    parsed = yaml.safe_load(fm_text)

    raw_value = parsed["source_hashes"]["imported.md"]["raw"]
    assert raw_value == fake_sha256, f"Expected raw={fake_sha256!r}, got {raw_value!r}"


def test_source_hashes_entity_page_has_correct_key(tmp_path, monkeypatch):
    """Entity pages also have source_hashes with the source filename as key."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = (
        "# My Entity\n\n## Overview\n\nThis is an entity.\n## Features\n\nMany features.\n"
    )
    source_path = docs_dir / "my_entity.md"
    source_path.write_text(source_content, encoding="utf-8")

    fake_llm = _make_schema_aware_fake_llm(classifier_type="entity")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["my_entity.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    page_path = wiki_dir / result.results[0].pages_written[0]
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    parsed = yaml.safe_load(fm_text)

    assert "source_hashes" in parsed
    assert "my_entity.md" in parsed["source_hashes"]
