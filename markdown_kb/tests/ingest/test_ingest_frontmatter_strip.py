"""Integration test for POST /ingest frontmatter stripping (issue #106).

Hermetic: TestClient + mock LLM, no OPENAI_API_KEY required. The mock pattern
mirrors test_ingest_integration.py — a schema-aware fake ``ChatOpenAI`` patched
into ``app.templates``. Here the classifier chain additionally records the
HumanMessage content it is invoked with, so the test can assert that the text
handed to the ingest LLM for a Source EXCLUDES the leading YAML frontmatter
block (provenance metadata written by importer.py — imported_from /
original_format / imported_at / content_sha256).

AC coverage (issue #106):
  - A Source WITH provenance frontmatter produces a Wiki Page whose synthesis
    input does not treat the frontmatter fields as Source facts: the text fed to
    the classifier LLM contains none of the frontmatter keys/values.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import app.templates as templates_module

# Provenance frontmatter fields written by importer.py (slice 7-3). These are
# metadata ABOUT the file, not content OF the Source — they must never reach the
# ingest LLM prompt.
_FRONTMATTER_MARKERS = [
    "imported_from",
    "raw/customer_handbook.html",
    "original_format",
    "imported_at",
    "content_sha256",
    "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
]

_SOURCE_WITH_FRONTMATTER = (
    "---\n"
    "imported_from: raw/customer_handbook.html\n"
    "original_format: html\n"
    "imported_at: '2026-05-28T10:00:00Z'\n"
    "content_sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n"
    "---\n"
    "\n"
    "## Cancellation Window\n"
    "\n"
    "Customers can cancel within 24 hours of purchase if the order has not shipped.\n"
)


class _FakeSynthesisOutput:
    def __init__(self, body: str = "Synthesised body.", open_questions=None):
        self.body = body
        self.open_questions = open_questions or []


class _FakeClassifierOutput:
    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_recording_fake_llm(captured: list[str]) -> MagicMock:
    """Return a schema-aware fake ChatOpenAI that records classifier input.

    The classifier chain appends the HumanMessage content of each invoke() call
    to ``captured`` so the test can assert what text the ingest LLM received.
    """
    from app.templates import _ClassifierOutput

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:

            def _classify_invoke(messages):
                # messages == [SystemMessage(...), HumanMessage(content=...)]
                captured.append(messages[-1].content)
                return _FakeClassifierOutput("concept")

            chain.invoke.side_effect = _classify_invoke
        else:
            chain.invoke.return_value = _FakeSynthesisOutput()
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


def test_ingest_strips_frontmatter_before_classifier(tmp_path, monkeypatch):
    """A Source carrying provenance frontmatter must not feed it to the LLM.

    The text handed to the ingest classifier excludes the leading YAML
    frontmatter block; the Wiki Page is still generated from the real body.
    """
    import app.indexer as indexer_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "customer_handbook.md").write_text(_SOURCE_WITH_FRONTMATTER, encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    captured_classifier_input: list[str] = []
    fake_llm = _make_recording_fake_llm(captured_classifier_input)
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.ingest import ingest_sources

    result = ingest_sources(["customer_handbook.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    # The Source was processed successfully into a page.
    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert len(result.results) == 1
    assert result.results[0].pages_written, "Expected at least one page written"

    # The classifier LLM was actually invoked (proving the fake path ran).
    assert captured_classifier_input, "Classifier LLM was never invoked"

    # AC: none of the provenance frontmatter markers reached the LLM prompt.
    classifier_text = captured_classifier_input[0]
    for marker in _FRONTMATTER_MARKERS:
        assert marker not in classifier_text, (
            f"Frontmatter marker {marker!r} leaked into the ingest LLM prompt:\n{classifier_text}"
        )

    # The real Source body still reached the LLM (strip removed only metadata).
    assert "Cancellation Window" in classifier_text
    assert "cancel within 24 hours" in classifier_text


def test_ingest_no_frontmatter_source_unaffected(tmp_path, monkeypatch):
    """A Source with NO frontmatter is fed to the LLM byte-identically.

    Regression guard for the unchanged path: when there is no leading YAML
    block, the text handed to the classifier equals the full file content.
    """
    import app.indexer as indexer_module

    plain_source = (
        "## Cancellation Window\n"
        "\n"
        "Customers can cancel within 24 hours of purchase if the order has not shipped.\n"
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "plain.md").write_text(plain_source, encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    captured_classifier_input: list[str] = []
    fake_llm = _make_recording_fake_llm(captured_classifier_input)
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.ingest import ingest_sources

    result = ingest_sources(["plain.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert captured_classifier_input, "Classifier LLM was never invoked"
    # The classifier user message wraps the content; the full plain body is
    # present verbatim (no bytes stripped when there is no frontmatter).
    assert plain_source in captured_classifier_input[0]
