"""Single-source ingest resolves citation-shaped refs like the C3 view path.

Issue #475: ingest-produced wiki pages always cite ``{source_name}#{slug}``
(templates.py), and Sources may live nested under ``docs/**`` — but
single-source mode looked refs up verbatim as ``docs_dir / name`` (flat-only,
anchor included). The Console's C3 "Re-ingest (retry)" therefore dead-ended
in ``failed_sources`` on exactly the refs lint's view path (#467) reports as
``resolved``. These tests pin the agreement contract:

- ``#anchor`` is stripped before lookup (flat and nested).
- A nested Source resolves by basename (mirrors ``lint._resolve_c3_source_path``
  / ``_resolve_docs_files``).
- A basename matching 2+ files is NEVER silently guessed (same contract as
  the view side, issue #445) — it fails into ``failed_sources``.
- The route-level shape from the issue AC: ``POST /ingest
  {"source": "x.md#anchor", "force": true}`` succeeds for a Source the view
  path resolves.
- The async twin ``aingest_sources`` behaves identically (drift guard).

All tests hermetic (no OPENAI_API_KEY). LLM mocked at
``app.templates.get_ingest_llm``; grounding mocked by the conftest autouse
``_mock_ingest_verifier_supported``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

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


def _patch_llm_and_wiki(tmp_path, monkeypatch):
    """Fake the ingest LLM and redirect WIKI_DIR; return (docs_dir, wiki_dir)."""
    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return docs_dir, wiki_dir


# ---------------------------------------------------------------------------
# Anchor stripping — flat Source
# ---------------------------------------------------------------------------


def test_anchored_flat_source_resolves(tmp_path, monkeypatch):
    """ingest_sources(["x.md#some-anchor"]) resolves flat docs/x.md."""
    docs_dir, wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)
    (docs_dir / "x.md").write_text("## X Section\n\nX content.\n", encoding="utf-8")

    result = ingest_module.ingest_sources(
        ["x.md#some-anchor"], docs_dir=docs_dir, wiki_dir=wiki_dir
    )

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert [r.source for r in result.results] == ["x.md"]


# ---------------------------------------------------------------------------
# Nested Source — with and without anchor (the #475 reproduction shape)
# ---------------------------------------------------------------------------


def test_nested_source_resolves(tmp_path, monkeypatch):
    """A Source nested under docs/** resolves by basename in single-source mode."""
    docs_dir, wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)
    nested = docs_dir / "fake-docs"
    nested.mkdir()
    (nested / "product_care.md").write_text(
        "## Cleaning Instructions\n\nWipe gently.\n", encoding="utf-8"
    )

    result = ingest_module.ingest_sources(["product_care.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert [r.source for r in result.results] == ["product_care.md"]


def test_anchored_nested_source_resolves(tmp_path, monkeypatch):
    """The exact #475 shape: anchored citation of a nested Source resolves."""
    docs_dir, wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)
    nested = docs_dir / "fake-docs"
    nested.mkdir()
    (nested / "product_care.md").write_text(
        "## Cleaning Instructions\n\nWipe gently.\n", encoding="utf-8"
    )

    result = ingest_module.ingest_sources(
        ["product_care.md#cleaning-instructions"], docs_dir=docs_dir, wiki_dir=wiki_dir
    )

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert [r.source for r in result.results] == ["product_care.md"]
    assert (wiki_dir / "concepts" / "cleaning-instructions.md").exists()


# ---------------------------------------------------------------------------
# Ambiguity — 2+ files sharing the basename are never silently guessed
# ---------------------------------------------------------------------------


def test_ambiguous_basename_fails_without_guessing(tmp_path, monkeypatch):
    """dup.md existing both flat and nested fails instead of guessing the flat one."""
    docs_dir, wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)
    (docs_dir / "dup.md").write_text("## Flat Dup\n\nFlat content.\n", encoding="utf-8")
    nested = docs_dir / "sub"
    nested.mkdir()
    (nested / "dup.md").write_text("## Nested Dup\n\nNested content.\n", encoding="utf-8")

    result = ingest_module.ingest_sources(["dup.md#flat-dup"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == ["dup.md"], (
        f"Ambiguous basename must fail, got failed={result.failed_sources} "
        f"results={[r.source for r in result.results]}"
    )
    assert result.results == []
    assert not (wiki_dir / "concepts" / "flat-dup.md").exists(), (
        "Ambiguous ref must not silently ingest the flat match (view side "
        "reports 'ambiguous' for this citation — issue #445 contract)"
    )


# ---------------------------------------------------------------------------
# Missing Source still fails (reported under the stripped basename)
# ---------------------------------------------------------------------------


def test_missing_source_still_fails_with_basename(tmp_path, monkeypatch):
    """An anchored ref to a nonexistent Source fails as its stripped basename."""
    docs_dir, wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)

    result = ingest_module.ingest_sources(
        ["nope.md#anything"], docs_dir=docs_dir, wiki_dir=wiki_dir
    )

    assert result.failed_sources == ["nope.md"]
    assert result.results == []


# ---------------------------------------------------------------------------
# Route-level AC shape: POST /ingest {"source": "x.md#anchor", "force": true}
# ---------------------------------------------------------------------------


def test_route_anchored_nested_source_resolves(tmp_path, monkeypatch):
    """POST /ingest with an anchored nested citation succeeds end-to-end."""
    docs_dir, _wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)
    nested = docs_dir / "fake-docs"
    nested.mkdir()
    (nested / "product_care.md").write_text(
        "## Cleaning Instructions\n\nWipe gently.\n", encoding="utf-8"
    )
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/ingest", json={"source": "product_care.md#cleaning-instructions", "force": True}
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    body = resp.json()
    assert body["failed_sources"] == [], f"Unexpected failures: {body['failed_sources']}"
    assert [r["source"] for r in body["results"]] == ["product_care.md"]


# ---------------------------------------------------------------------------
# Async twin — aingest_sources resolves identically (drift guard)
# ---------------------------------------------------------------------------


def test_aingest_anchored_nested_source_resolves(tmp_path, monkeypatch):
    """aingest_sources resolves the anchored nested ref exactly like the sync path."""
    docs_dir, wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)
    nested = docs_dir / "fake-docs"
    nested.mkdir()
    (nested / "product_care.md").write_text(
        "## Cleaning Instructions\n\nWipe gently.\n", encoding="utf-8"
    )

    result = asyncio.run(
        ingest_module.aingest_sources(
            ["product_care.md#cleaning-instructions"], docs_dir=docs_dir, wiki_dir=wiki_dir
        )
    )

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert [r.source for r in result.results] == ["product_care.md"]


def test_aingest_ambiguous_basename_fails_without_guessing(tmp_path, monkeypatch):
    """Async twin of the ambiguity contract."""
    docs_dir, wiki_dir = _patch_llm_and_wiki(tmp_path, monkeypatch)
    (docs_dir / "dup.md").write_text("## Flat Dup\n\nFlat content.\n", encoding="utf-8")
    nested = docs_dir / "sub"
    nested.mkdir()
    (nested / "dup.md").write_text("## Nested Dup\n\nNested content.\n", encoding="utf-8")

    result = asyncio.run(
        ingest_module.aingest_sources(["dup.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    )

    assert result.failed_sources == ["dup.md"]
    assert result.results == []
