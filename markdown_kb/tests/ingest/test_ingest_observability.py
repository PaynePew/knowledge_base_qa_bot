"""Tests for issue #511 / ADR-0033 "Ingest observability": POST /ingest gains
per-source sections_count / uncarried_chars / enriched_chars.

The 63-page-book incident (ADR-0033) showed a degenerate parse can report
plain success (``pages_created=1``) while most of a Source's text never
reached a Section — the response and log carried no signal of that. This
slice makes the parse's structure visible: every ``IngestSourceResult``
(both ``results`` and hash-match ``skipped_sources``) now carries the Section
count and the non-whitespace character count the parse did not carry into
any Section (normally 0 after issue #509 — see
``test_indexer_uncarried_chars.py`` for the invariant itself). All tests
hermetic — no OPENAI_API_KEY required.

Tests:
- test_multi_section_source_reports_sections_count: refund_policy.md (3
  headings) -> sections_count=3, uncarried_chars=0, enriched_chars=0 on
  every one of its 3 result rows.
- test_single_section_source_reports_sections_count: a zero-heading Source
  -> sections_count=1, uncarried_chars=0, enriched_chars=0.
- test_skipped_source_still_reports_sections_count: a hash-match skip still
  carries the same fields (parse_markdown already ran before the skip check).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import app.templates as templates_module

FIXED_BODY = "Customers can cancel within 24 hours of purchase if the order has not shipped."


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


_TESTS_DIR = Path(__file__).resolve().parent.parent
_FIXTURE_DOCS_DIR = _TESTS_DIR / "fixtures" / "docs"


def _client(monkeypatch, tmp_path, docs_dir: Path) -> TestClient:
    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)

    import app.indexer as indexer_module
    import app.ingest as ingest_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from app.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# AC: multi-section source -> sections_count == 3, both char counts 0
# ---------------------------------------------------------------------------


def test_multi_section_source_reports_sections_count(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, _FIXTURE_DOCS_DIR)

    resp = client.post("/ingest", json={"source": "refund_policy.md"})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["sections_count"] == 3, result
    assert result["uncarried_chars"] == 0, result
    assert result["enriched_chars"] == 0, result


# ---------------------------------------------------------------------------
# AC: single-section (zero-heading) source -> sections_count == 1
# ---------------------------------------------------------------------------


def test_single_section_source_reports_sections_count(monkeypatch, tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "flat_notice.md").write_text(
        "A single flat Source with no headings at all — the whole body is one Section.\n",
        encoding="utf-8",
    )

    client = _client(monkeypatch, tmp_path, docs_dir)

    resp = client.post("/ingest", json={"source": "flat_notice.md"})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["sections_count"] == 1, result
    assert result["uncarried_chars"] == 0, result
    assert result["enriched_chars"] == 0, result


# ---------------------------------------------------------------------------
# AC: a hash-match skip still carries sections_count / uncarried_chars
# (parse_markdown already ran before the skip decision is made).
# ---------------------------------------------------------------------------


def test_skipped_source_still_reports_sections_count(monkeypatch, tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_text = "## Only Section\n\nSome body text.\n"
    source_path = docs_dir / "single_concept.md"
    source_path.write_text(source_text, encoding="utf-8")
    docs_body_hash = hashlib.sha256(source_text.encode()).hexdigest()

    wiki_dir = tmp_path / "wiki"
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True)
    (concepts_dir / "only-section.md").write_text(
        "---\n"
        "id: only-section\n"
        "type: concept\n"
        "sources:\n"
        "  - single_concept.md#only-section\n"
        "source_hashes:\n"
        "  single_concept.md:\n"
        f"    docs_body: {docs_body_hash}\n"
        "    raw: null\n"
        "---\n\n"
        "Some body text.\n",
        encoding="utf-8",
    )

    client = _client(monkeypatch, tmp_path, docs_dir)

    resp = client.post("/ingest", json={"source": "single_concept.md"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"] == [], "hash-match must skip, not land in results"
    assert len(body["skipped_sources"]) == 1, body
    skipped = body["skipped_sources"][0]
    assert skipped["status"] == "skipped"
    assert skipped["sections_count"] == 1, skipped
    assert skipped["uncarried_chars"] == 0, skipped
