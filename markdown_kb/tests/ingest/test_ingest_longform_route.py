"""Tests for the longform ingest route (ADR-0033 decision 3, issue #513).

A Longform Source — identified by its `structure: enriched` frontmatter,
written by Structure Enrichment (issue #512) at Import/Transcribe time —
takes a THIRD `/ingest` route beside `entity` and `concept`: one entity-style
Hub Page ("about this document", wikilinked to every chapter) plus one
concept-style chapter page per Section, through the SAME synthesis,
Grounding Check, slug-collision, orphan-delete, and hash-skip machinery as
the existing routes.

All tests hermetic — no OPENAI_API_KEY required. The grounding verifier is
mocked to `claim_supported` by the repo-wide autouse fixture
(`tests/conftest.py::_mock_ingest_verifier_supported`), so every page in
these tests is expected to land with `status: live`.

Tests:
- test_longform_source_writes_hub_and_chapter_pages: 1 hub + 3 chapter pages
  written, all `status: live`, `sections_count` reflects the chapter count
  (AC 1, AC 4).
- test_hub_wikilinks_resolve_to_every_chapter_no_red_links: the hub body
  cites every chapter slug and every cited slug resolves to a real file on
  disk (AC 1).
- test_reingest_after_source_change_orphans_stale_chapter: a Source edit that
  drops a chapter orphan-deletes the stale chapter page and the hub is
  rewritten without a dangling link to it (AC 2).
- test_unchanged_reingest_hash_skips_zero_llm_calls: an unchanged Source
  hash-skips with no LLM calls at all (AC 2).
- test_non_longform_source_uses_existing_classify_route: a Source with NO
  `structure: enriched` marker still takes the untouched classify_source
  route — one page, classifier invoked (AC 3).
- test_build_index_includes_hub_and_chapter_pages: the wiki BM25 index picks
  up the hub + chapter pages via the existing SOURCE_DIRS scan (AC 5).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import app.indexer as indexer_module
import app.ingest as ingest_module
import app.templates as templates_module

FIXED_BODY = (
    "This book follows its protagonist across three chapters, from the opening "
    "setup through the central conflict to its resolution."
)

_BOOK_FRONTMATTER = (
    "---\n"
    "imported_from: raw/my_book.pdf\n"
    "original_format: pdf\n"
    "imported_at: '2026-07-01T00:00:00Z'\n"
    "content_sha256: deadbeef\n"
    "origin: transcribed\n"
    "transcribe_model: gpt-4o\n"
    "structure: enriched\n"
    "---\n"
)

_BOOK_BODY_3_CHAPTERS = (
    "\n"
    "## Chapter One\n\n"
    "The opening chapter introduces the protagonist and the setting.\n\n"
    "## Chapter Two\n\n"
    "The second chapter develops the central conflict.\n\n"
    "## Chapter Three\n\n"
    "The final chapter resolves the conflict.\n"
)

_BOOK_BODY_2_CHAPTERS = (
    "\n"
    "## Chapter One\n\n"
    "The opening chapter introduces the protagonist and the setting, now with "
    "a longer revised passage that changes the docs_body_hash.\n\n"
    "## Chapter Two\n\n"
    "The second chapter develops the central conflict.\n"
)

# ---------------------------------------------------------------------------
# Fake LLM helpers
# ---------------------------------------------------------------------------


class _FakeSynthesisOutput:
    def __init__(self, body: str = FIXED_BODY, open_questions: list | None = None):
        self.body = body
        self.open_questions = open_questions or []


class _FakeClassifierOutput:
    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_fake_llm() -> MagicMock:
    """A fake ingest LLM: every with_structured_output() call returns a fixed
    synthesis body. Sufficient for longform tests — classify_source is never
    invoked on the longform route, so no schema differentiation is needed."""
    fake_llm = MagicMock()
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = _FakeSynthesisOutput()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


def _make_schema_aware_fake_llm(classifier_type: str = "concept") -> MagicMock:
    """A fake ingest LLM that also answers the classifier schema — needed for
    the non-longform bypass regression test."""
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


def _write_book(
    docs_dir: Path, body: str = _BOOK_BODY_3_CHAPTERS, name: str = "my_book.md"
) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / name
    path.write_text(_BOOK_FRONTMATTER + body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# AC 1 / AC 4 — hub + chapter pages written, observability reflects them
# ---------------------------------------------------------------------------


def test_longform_source_writes_hub_and_chapter_pages(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    _write_book(docs_dir)

    fake_llm = _make_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    batch = ingest_module.ingest_sources(["my_book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert batch.failed_sources == [], batch.failed_reasons
    assert len(batch.results) == 1
    result = batch.results[0]

    # AC 4: observability reflects hub + chapter counts.
    assert result.sections_count == 3, result
    assert result.status == "created"
    assert len(result.pages_written) == 4, result.pages_written
    assert len(result.pages_created) == 4, result.pages_created
    assert result.pages_deleted == []

    hub_path = wiki_dir / "entities" / "my-book.md"
    chapter_paths = [
        wiki_dir / "concepts" / f"{slug}.md"
        for slug in ("chapter-one", "chapter-two", "chapter-three")
    ]
    assert hub_path.exists(), "Expected the Hub Page at wiki/entities/my-book.md"
    for p in chapter_paths:
        assert p.exists(), f"Expected chapter page at {p}"

    # Every page passed the (mocked-supported) Grounding Check.
    for p in [hub_path, *chapter_paths]:
        content = p.read_text(encoding="utf-8")
        assert "status: live" in content, f"{p} did not land as status: live:\n{content}"

    assert batch.pages_with_failed_grounding == []


# ---------------------------------------------------------------------------
# AC 1 — hub wikilinks resolve to every chapter page, no Red Links among them
# ---------------------------------------------------------------------------


def test_hub_wikilinks_resolve_to_every_chapter_no_red_links(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    _write_book(docs_dir)

    fake_llm = _make_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    ingest_module.ingest_sources(["my_book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    hub_content = (wiki_dir / "entities" / "my-book.md").read_text(encoding="utf-8")
    for slug in ("chapter-one", "chapter-two", "chapter-three"):
        wikilink = f"[[{slug}]]"
        assert wikilink in hub_content, f"Expected {wikilink} in hub body:\n{hub_content}"
        # No Red Links: every cited chapter slug resolves to a real file.
        assert (wiki_dir / "concepts" / f"{slug}.md").exists()


# ---------------------------------------------------------------------------
# AC 2 — Source change orphan-deletes the stale chapter page
# ---------------------------------------------------------------------------


def test_reingest_after_source_change_orphans_stale_chapter(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    book_path = _write_book(docs_dir, body=_BOOK_BODY_3_CHAPTERS)

    fake_llm = _make_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    first = ingest_module.ingest_sources(["my_book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    assert len(first.results) == 1
    assert first.results[0].sections_count == 3
    assert (wiki_dir / "concepts" / "chapter-three.md").exists()

    # Edit the Source: drop Chapter Three (changes docs_body_hash so this is
    # NOT a hash-skip).
    book_path.write_text(_BOOK_FRONTMATTER + _BOOK_BODY_2_CHAPTERS, encoding="utf-8")

    second = ingest_module.ingest_sources(["my_book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    assert len(second.results) == 1
    result = second.results[0]
    assert result.sections_count == 2, result
    assert result.pages_deleted == ["concepts/chapter-three.md"], result.pages_deleted
    assert not (wiki_dir / "concepts" / "chapter-three.md").exists()
    assert (wiki_dir / "concepts" / "chapter-one.md").exists()
    assert (wiki_dir / "concepts" / "chapter-two.md").exists()
    assert (wiki_dir / "entities" / "my-book.md").exists()

    # Hub is rewritten without a dangling link to the deleted chapter.
    hub_content = (wiki_dir / "entities" / "my-book.md").read_text(encoding="utf-8")
    assert "[[chapter-three]]" not in hub_content
    assert "[[chapter-one]]" in hub_content
    assert "[[chapter-two]]" in hub_content


# ---------------------------------------------------------------------------
# AC 2 — unchanged Source hash-skips with zero LLM calls
# ---------------------------------------------------------------------------


def test_unchanged_reingest_hash_skips_zero_llm_calls(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    _write_book(docs_dir)

    fake_llm = _make_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    first = ingest_module.ingest_sources(["my_book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    assert len(first.results) == 1
    assert fake_llm.with_structured_output.called

    fake_llm.reset_mock()

    second = ingest_module.ingest_sources(["my_book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    assert second.results == []
    assert len(second.skipped_sources) == 1
    skipped = second.skipped_sources[0]
    assert skipped.status == "skipped"
    assert skipped.sections_count == 3, skipped
    assert not fake_llm.with_structured_output.called, (
        "Expected zero LLM calls on an unchanged-hash longform re-ingest"
    )


# ---------------------------------------------------------------------------
# AC 3 — non-longform Sources take the existing entity/concept routes
# ---------------------------------------------------------------------------


def test_non_longform_source_uses_existing_classify_route(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    docs_dir.mkdir()
    # No `structure: enriched` frontmatter at all — an ordinary hand-authored
    # zero-heading Source.
    (docs_dir / "flat_notice.md").write_text(
        "A single flat Source with no headings — the whole body is one Section.\n",
        encoding="utf-8",
    )

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    batch = ingest_module.ingest_sources(["flat_notice.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert batch.failed_sources == [], batch.failed_reasons
    assert len(batch.results) == 1
    result = batch.results[0]
    # Byte-identical to the pre-#513 route: exactly ONE page (not hub + N).
    assert len(result.pages_written) == 1, result.pages_written
    assert result.sections_count == 1
    # classify_source's _ClassifierOutput schema WAS requested — proves the
    # existing classify route ran, not the longform bypass.
    from app.templates import _ClassifierOutput

    schema_calls = [c.args[0] for c in fake_llm.with_structured_output.call_args_list]
    assert _ClassifierOutput in schema_calls, (
        "Expected classify_source's schema to be requested for a non-longform Source"
    )


# ---------------------------------------------------------------------------
# AC 5 — Wiki BM25 index rebuild includes the hub + chapter pages
# ---------------------------------------------------------------------------


def test_build_index_includes_hub_and_chapter_pages(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    wiki_dir = tmp_path / "wiki"
    _write_book(docs_dir)

    fake_llm = _make_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    ingest_module.ingest_sources(["my_book.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    # build_index's default SOURCE_DIRS is pre-baked at module load from the
    # real WIKI_DIR; rebuild it against the patched tmp WIKI_DIR (existing
    # convention — see test_indexing.py::test_build_index_scans_wiki_subdirs).
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts"],
    )

    files_indexed, sections_indexed = indexer_module.build_index()

    assert files_indexed == 4, "Expected the hub page + 3 chapter pages to be indexed"
    assert sections_indexed == 4
