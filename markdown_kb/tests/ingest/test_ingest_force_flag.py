"""Tests for force=True flag bypassing hash-skip — Phase 3 amendment (issue #93).

When force=True is passed in IngestRequest, hash-match skip is bypassed
and normal ingest proceeds even if the hash matches.

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
FIXED_BODY = "Test body for force flag test."


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


def _make_wiki_page_with_source_hashes(
    wiki_dir: Path,
    subdir: str,
    slug: str,
    source_name: str,
    docs_body_hash: str,
) -> Path:
    """Create a wiki page with matching source_hashes for testing force bypass."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    source_hashes: dict = {
        source_name: {
            "docs_body": docs_body_hash,
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
        f"<!-- sentinel -->\n\n---\n{fm_yaml}---\n\n# {slug.title()}\n\nExisting body.\n\n[Source: {source_name}#{slug}]\n",
        encoding="utf-8",
    )
    return page_path


def _compute_docs_body_hash(source_path: Path) -> str:
    return hashlib.sha256(source_path.read_text(encoding="utf-8").encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tests: force=True bypasses hash-skip
# ---------------------------------------------------------------------------


def test_force_true_bypasses_hash_skip(tmp_path, monkeypatch):
    """force=True causes ingest to proceed even when docs_body_hash matches."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Force Section\n\nContent to be force-reingested.\n"
    source_path = docs_dir / "force.md"
    source_path.write_text(source_content, encoding="utf-8")

    # Matching hash — would normally skip
    docs_body_hash = _compute_docs_body_hash(source_path)
    _make_wiki_page_with_source_hashes(
        wiki_dir, "concepts", "force-section", "force.md", docs_body_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    # force=True should bypass skip
    result = ingest_module.ingest_sources(
        ["force.md"], docs_dir=docs_dir, wiki_dir=wiki_dir, force=True
    )

    assert result.failed_sources == []
    assert len(result.skipped_sources) == 0, (
        f"Expected 0 skipped_sources with force=True, got: {result.skipped_sources}"
    )
    assert len(result.results) == 1, f"Expected 1 result with force=True, got: {result.results}"
    # LLM was invoked
    assert fake_llm.with_structured_output.called, "Expected LLM to be called with force=True"


def test_force_false_still_skips_on_match(tmp_path, monkeypatch):
    """force=False (default) still skips when hash matches."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## No Force Section\n\nContent with matching hash.\n"
    source_path = docs_dir / "no_force.md"
    source_path.write_text(source_content, encoding="utf-8")

    docs_body_hash = _compute_docs_body_hash(source_path)
    _make_wiki_page_with_source_hashes(
        wiki_dir, "concepts", "no-force-section", "no_force.md", docs_body_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    # force=False (default) — should skip
    result = ingest_module.ingest_sources(
        ["no_force.md"], docs_dir=docs_dir, wiki_dir=wiki_dir, force=False
    )

    assert len(result.skipped_sources) == 1
    assert result.skipped_sources[0].source == "no_force.md"
    assert not fake_llm.with_structured_output.called


def test_ingest_request_has_force_field(tmp_path):
    """IngestRequest schema has force field with default=False."""
    from app.schemas import IngestRequest

    req = IngestRequest()
    assert hasattr(req, "force"), "IngestRequest should have 'force' field"
    assert req.force is False, f"Expected force=False default, got: {req.force}"

    req_forced = IngestRequest(force=True)
    assert req_forced.force is True


def test_force_true_via_route(tmp_path, monkeypatch):
    """POST /ingest with force=True bypasses hash-skip."""
    import app.indexer as indexer_module
    import app.ingest as ingest_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    source_content = "## Route Force Section\n\nContent for route force test.\n"
    source_path = docs_dir / "route_force.md"
    source_path.write_text(source_content, encoding="utf-8")

    docs_body_hash = _compute_docs_body_hash(source_path)
    _make_wiki_page_with_source_hashes(
        wiki_dir, "concepts", "route-force-section", "route_force.md", docs_body_hash
    )

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    # Without force — should skip
    resp = client.post("/ingest", json={"source": "route_force.md"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body.get("skipped_sources", [])) == 1

    # With force=True — should NOT skip
    resp2 = client.post("/ingest", json={"source": "route_force.md", "force": True})
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert len(body2.get("skipped_sources", [])) == 0, (
        f"Expected 0 skipped_sources with force=True, got: {body2}"
    )
    assert len(body2.get("results", [])) >= 1
