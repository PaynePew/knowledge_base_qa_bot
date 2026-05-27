"""Tests for hash drift (non-matching hash) behavior in /ingest — Phase 3 amendment (issue #93).

When a wiki page's source_hashes shows a different docs_body_hash than the
current source content, ingest must proceed (overwrite), not skip.

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
FIXED_BODY = "Test body for hash drift test."


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


def _make_wiki_page_with_stale_hash(
    wiki_dir: Path,
    subdir: str,
    slug: str,
    source_name: str,
    stale_hash: str,
) -> Path:
    """Create a wiki page with a source_hashes hash that does NOT match current source."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    source_hashes: dict = {
        source_name: {
            "docs_body": stale_hash,
            "raw": None,
        }
    }
    fm_data = {
        "id": slug,
        "type": "concept" if subdir == "concepts" else "entity",
        "created": FIXED_TS,
        "updated": FIXED_TS,
        "sources": [f"{source_name}#{slug}"],
        "status": "live",
        "open_questions": [],
        "source_hashes": source_hashes,
    }
    fm_yaml = yaml.dump(fm_data, default_flow_style=False, allow_unicode=True)
    page_path.write_text(
        f"<!-- sentinel -->\n\n---\n{fm_yaml}---\n\n# {slug.title()}\n\nOld body text.\n\n[Source: {source_name}#{slug}]\n",
        encoding="utf-8",
    )
    return page_path


# ---------------------------------------------------------------------------
# Tests: hash differs → ingest (overwrite)
# ---------------------------------------------------------------------------


def test_hash_drift_triggers_ingest(tmp_path, monkeypatch):
    """When existing wiki page has a DIFFERENT docs_body_hash, ingest proceeds."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Drift Section\n\nUpdated content after source was modified.\n"
    source_path = docs_dir / "drift.md"
    source_path.write_text(source_content, encoding="utf-8")

    # Plant wiki page with a STALE (non-matching) hash
    stale_hash = "a" * 64  # definitely not the real hash
    old_page = _make_wiki_page_with_stale_hash(
        wiki_dir, "concepts", "drift-section", "drift.md", stale_hash
    )
    old_content = old_page.read_text(encoding="utf-8")
    assert "Old body text." in old_content

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["drift.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    assert len(result.skipped_sources) == 0, (
        f"Expected 0 skipped_sources on drift, got: {result.skipped_sources}"
    )
    assert len(result.results) == 1, f"Expected 1 result, got: {result.results}"
    assert result.results[0].source == "drift.md"

    # LLM was called for synthesis
    assert fake_llm.with_structured_output.called, "Expected LLM to be called on hash drift"


def test_hash_drift_result_has_updated_status(tmp_path, monkeypatch):
    """On drift, the written page has status='updated' (or 'created' if slug changed)."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Drift Status\n\nContent that drifted.\n"
    source_path = docs_dir / "drift_status.md"
    source_path.write_text(source_content, encoding="utf-8")

    # Plant a wiki page for the expected slug with stale hash
    stale_hash = "b" * 64
    _make_wiki_page_with_stale_hash(
        wiki_dir, "concepts", "drift-status", "drift_status.md", stale_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["drift_status.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    assert len(result.results) == 1
    source_result = result.results[0]
    # The page existed, so it should be updated
    assert len(source_result.pages_updated) >= 1 or len(source_result.pages_created) >= 1


def test_hash_drift_page_is_overwritten_with_new_hash(tmp_path, monkeypatch):
    """After drift ingest, wiki page frontmatter has the new (correct) docs_body_hash."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## New Hash Section\n\nContent with updated hash.\n"
    source_path = docs_dir / "new_hash.md"
    source_path.write_text(source_content, encoding="utf-8")

    correct_hash = hashlib.sha256(source_content.encode()).hexdigest()
    stale_hash = "c" * 64  # wrong
    _make_wiki_page_with_stale_hash(
        wiki_dir, "concepts", "new-hash-section", "new_hash.md", stale_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["new_hash.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    assert len(result.results) == 1

    page_path = wiki_dir / result.results[0].pages_written[0]
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    parsed = yaml.safe_load(fm_text)

    new_hash = parsed["source_hashes"]["new_hash.md"]["docs_body"]
    assert new_hash == correct_hash, (
        f"Expected updated docs_body_hash {correct_hash!r}, got {new_hash!r}"
    )
    assert new_hash != stale_hash, "Stale hash should have been replaced"
