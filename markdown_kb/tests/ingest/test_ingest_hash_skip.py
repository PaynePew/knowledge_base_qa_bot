"""Tests for hash-skip idempotency in /ingest — Phase 3 amendment (issue #93).

When a wiki page already has source_hashes with a matching docs_body_hash,
ingest should skip the LLM call and write no new page — emit ingest_skipped
Wiki Log event instead.

All tests hermetic — no OPENAI_API_KEY required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import app.templates as templates_module
from app.schemas import WikiPageDraft, WikiPageFrontmatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-05-26T14:30:00Z"
FIXED_BODY = "Test body for hash skip test."


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


def _make_wiki_page_with_source_hashes(
    wiki_dir: Path,
    subdir: str,
    slug: str,
    source_name: str,
    docs_body_hash: str,
    raw_hash: str | None = None,
) -> Path:
    """Create a wiki page with source_hashes frontmatter for testing skip logic."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    source_hashes: dict = {
        source_name: {
            "docs_body": docs_body_hash,
            "raw": raw_hash,
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
        f"<!-- sentinel -->\n\n---\n{fm_yaml}---\n\n# {slug.title()}\n\nBody text.\n\n[Source: {source_name}#{slug}]\n",
        encoding="utf-8",
    )
    return page_path


# ---------------------------------------------------------------------------
# Tests: hash match → skip
# ---------------------------------------------------------------------------


def test_hash_match_skips_llm_call(tmp_path, monkeypatch):
    """When docs_body_hash matches existing wiki page, LLM is NOT called."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Policy Overview\n\nOur standard policy.\n"
    source_path = docs_dir / "policy.md"
    source_path.write_text(source_content, encoding="utf-8")

    # Pre-compute hash that matches current source
    docs_body_hash = _compute_docs_body_hash(source_path)

    # Plant wiki page with matching hash
    _make_wiki_page_with_source_hashes(
        wiki_dir,
        "concepts",
        "policy-overview",
        "policy.md",
        docs_body_hash,
    )

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    # Source should be in skipped_sources, not results
    assert len(result.skipped_sources) == 1, (
        f"Expected 1 skipped source, got: {result.skipped_sources}"
    )
    assert result.skipped_sources[0].source == "policy.md"
    # LLM synthesis was NOT called (classify + generate both require LLM)
    assert not fake_llm.with_structured_output.called, "Expected LLM to NOT be called on hash-skip"


def test_hash_match_result_has_skipped_status(tmp_path, monkeypatch):
    """Skipped source result has status='skipped'."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Skipped Section\n\nThis will be skipped.\n"
    source_path = docs_dir / "skipped.md"
    source_path.write_text(source_content, encoding="utf-8")

    docs_body_hash = _compute_docs_body_hash(source_path)
    _make_wiki_page_with_source_hashes(
        wiki_dir, "concepts", "skipped-section", "skipped.md", docs_body_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["skipped.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert len(result.skipped_sources) == 1
    skipped = result.skipped_sources[0]
    assert skipped.status == "skipped", f"Expected status='skipped', got: {skipped.status!r}"
    assert skipped.source == "skipped.md"


def test_hash_match_emits_ingest_skipped_log_event(tmp_path, monkeypatch):
    """Hash-match skip emits ingest_skipped Wiki Log event."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module
    import app.logger as logger_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Log Test\n\nFor log event testing.\n"
    source_path = docs_dir / "log_test.md"
    source_path.write_text(source_content, encoding="utf-8")

    docs_body_hash = _compute_docs_body_hash(source_path)
    _make_wiki_page_with_source_hashes(
        wiki_dir, "concepts", "log-test", "log_test.md", docs_body_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    # Redirect log to tmp so we can read it
    log_path = tmp_path / "wiki" / "log.md"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_path)

    result = ingest_module.ingest_sources(["log_test.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert len(result.skipped_sources) == 1

    # Read log and verify ingest_skipped was emitted
    log_content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    assert "ingest_skipped" in log_content, f"Expected 'ingest_skipped' in log, got:\n{log_content}"
    assert "log_test.md" in log_content


def test_hash_match_ingest_response_has_skipped_sources(tmp_path, monkeypatch):
    """IngestResponse.skipped_sources is populated on hash-match skip via the route."""
    from fastapi.testclient import TestClient

    import app.indexer as indexer_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Route Test\n\nFor route test.\n"
    source_path = docs_dir / "route_test.md"
    source_path.write_text(source_content, encoding="utf-8")

    docs_body_hash = _compute_docs_body_hash(source_path)
    _make_wiki_page_with_source_hashes(
        wiki_dir, "concepts", "route-test", "route_test.md", docs_body_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    # Point route's docs_dir to our tmp docs
    import app.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from app.main import app

    client = TestClient(app)
    resp = client.post("/ingest", json={"source": "route_test.md"})

    assert resp.status_code == 200
    body = resp.json()
    assert "skipped_sources" in body, f"Expected 'skipped_sources' in response: {body}"
    assert len(body["skipped_sources"]) == 1, (
        f"Expected 1 skipped_sources entry, got: {body['skipped_sources']}"
    )
    assert body["skipped_sources"][0]["source"] == "route_test.md"
