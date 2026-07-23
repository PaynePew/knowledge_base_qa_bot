"""Tests for issue #657's ingest LLM call-undercount fix.

Before this fix, ``ingest_batch_completed``'s ``llm_calls=`` count only summed
classify + synthesis calls — the grounding-verify call ``_finalise_source_drafts``
makes for every draft (``grounding.verify()``, ADR-0004) never reached
``batch._llm_call_count`` at all, so the cost ledger's build-phase call count
(PRD #654 user story 22) was systematically low for any Source with drafts.

Tests:
- test_llm_calls_includes_grounding_verify_calls: a 3-section concept Source
  makes 1 classify + 3 synthesis + 3 grounding-verify calls = 7 total, pinning
  the corrected count (pre-fix this would have read 4).
- test_llm_calls_zero_when_hash_skip_avoids_all_llm_calls: an unchanged
  Source hash-skips before any LLM call, so the corrected accounting still
  reports 0 — the fix must not double-count or misfire on the skip path.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import app.templates as templates_module

LOG_LINE_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\] (\S+) \| (.+)$")


class _FakeSynthesisOutput:
    def __init__(self, body: str = "Synthesised content.", open_questions: list | None = None):
        self.body = body
        self.open_questions = open_questions or []


class _FakeClassifierOutput:
    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_schema_aware_fake_llm() -> MagicMock:
    """Schema-aware fake ``get_ingest_llm()`` client (mirrors the established
    pattern in ``test_ingest_observability.py`` / ``test_ingest_grounding_failure.py``)."""
    from app.templates import _ClassifierOutput

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            chain.invoke.return_value = _FakeClassifierOutput("concept")
        else:
            chain.invoke.return_value = _FakeSynthesisOutput()
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


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


def _batch_completed_summary(log_path: Path) -> str:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        m = LOG_LINE_RE.match(line)
        if m and m.group(2) == "ingest_batch_completed":
            return m.group(3)
    raise AssertionError(f"ingest_batch_completed not found in log: {lines}")


def _llm_calls_from_summary(summary: str) -> int:
    m = re.search(r"llm_calls=(\d+)", summary)
    assert m is not None, f"llm_calls= missing from summary: {summary}"
    return int(m.group(1))


def test_llm_calls_includes_grounding_verify_calls(monkeypatch, tmp_path):
    """3 headings -> 1 classify + 3 synthesis + 3 grounding-verify = 7 total.

    ``_mock_ingest_verifier_supported`` (repo-wide autouse conftest fixture)
    fakes ``app.ingest.verify`` to always return ``claim_supported``, but
    each call still runs through ``_finalise_source_drafts`` and must be
    counted — the autouse fake proves the *counting* path, not the real
    LLM call, is what's under test here.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "three_sections.md").write_text(
        "## First Heading\n\nFirst section body.\n\n"
        "## Second Heading\n\nSecond section body.\n\n"
        "## Third Heading\n\nThird section body.\n",
        encoding="utf-8",
    )

    client = _client(monkeypatch, tmp_path, docs_dir)
    resp = client.post("/ingest", json={"source": "three_sections.md"})
    assert resp.status_code == 200, resp.text

    import app.logger as logger_module

    summary = _batch_completed_summary(logger_module.LOG_PATH)
    assert _llm_calls_from_summary(summary) == 7, summary


def test_llm_calls_zero_when_hash_skip_avoids_all_llm_calls(monkeypatch, tmp_path):
    """A hash-match skip makes zero LLM calls of any kind (classify, synthesis,
    or grounding-verify) — the fix must not add phantom grounding calls to a
    Source that never reached the finalise tail."""
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
    assert resp.json()["results"] == [], "hash-match must skip, not land in results"

    import app.logger as logger_module

    summary = _batch_completed_summary(logger_module.LOG_PATH)
    assert _llm_calls_from_summary(summary) == 0, summary
