"""Tests for the qa-question tokenization exception (issue #570, rule 2a).

The filing writer (POST /chat) stores the question ONLY in frontmatter
(``question:``) — the page body is the answer text alone, with no heading.
Under rule 2 ("do NOT tokenize frontmatter values") such a page can never be
retrieved BY its own question: the heading falls back to the slug (one
hyphenated blob for English) and the body rarely repeats the question's
words. Observed live as 3/8 dead starter presets (issue #570).

Rule 2a carves the one principled exception: for ``type: qa`` pages the
``question:`` value joins the page's BM25 tokens. It is reader-visible
content that merely *lives* in frontmatter; ids/dates/sources remain
untokenized. Display fields (heading, id) are untouched — this is a
retrieval-only exception.

Follows the conventions of ``test_indexer_qa_filter.py`` (tmp wiki dirs,
patched indexer paths, external behaviour only).
"""

from __future__ import annotations

from pathlib import Path


def _setup_wiki(tmp_path: Path) -> tuple[Path, Path, Path]:
    wiki_dir = tmp_path / "wiki"
    concepts_dir = wiki_dir / "concepts"
    qa_dir = wiki_dir / "qa"
    (wiki_dir / "entities").mkdir(parents=True)
    concepts_dir.mkdir(parents=True)
    qa_dir.mkdir(parents=True)
    return wiki_dir, concepts_dir, qa_dir


def _patch_indexer(monkeypatch, tmp_path: Path, wiki_dir: Path) -> None:
    import app.indexer as indexer_module
    import app.logger as logger_module

    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", wiki_dir / "log.md")
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts", wiki_dir / "qa"],
    )


def _write_filed_qa_page(qa_dir: Path, slug: str, question: str, body: str) -> Path:
    """Mirror the REAL filing writer's byte-shape (§6.5): sentinel HTML
    comment, question in frontmatter only, answer-only body, no heading."""
    content = (
        "<!-- Auto-filed by POST /chat. -->\n\n"
        "---\n"
        f"count: 1\n"
        f"created: '2026-07-11T00:00:00Z'\n"
        f"id: {slug}\n"
        "open_questions: []\n"
        f"question: {question}\n"
        "sources:\n"
        "- destinations#destinations\n"
        "status: live\n"
        "type: qa\n"
        "updated: '2026-07-11T00:00:00Z'\n"
        "---\n\n"
        f"{body}\n"
    )
    path = qa_dir / f"{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_qa_question_tokens_join_bm25(tmp_path, monkeypatch):
    """A filed qa page is retrievable BY its question (rule 2a).

    The body deliberately shares no distinctive token with the question
    ("countries", "ship") so retrieval can only succeed via the
    frontmatter question.
    """
    import app.indexer as indexer_module

    wiki_dir, _, qa_dir = _setup_wiki(tmp_path)
    _write_filed_qa_page(
        qa_dir,
        "which-countries-does-acme-shop-ship-to-abc123",
        question="Which countries do you ship to?",
        body=(
            "ACME currently delivers to Japan, South Korea, Hong Kong, Macau, "
            "Singapore, Malaysia, the United States, and Canada "
            "[Source: destinations#destinations]."
        ),
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    qa_sections = [s for s in indexer_module.sections if s.file.startswith("which-countries")]
    assert qa_sections, "the live qa page must be indexed"
    tokens = qa_sections[0].tokens
    assert "countries" in tokens and "ship" in tokens, (
        f"question tokens must join the qa section's BM25 tokens, got: {tokens}"
    )

    hits = indexer_module.search("Which countries do you ship to?", 3)
    top_files = [sec.file for sec, _ in hits]
    assert any(f.startswith("which-countries") for f in top_files), (
        f"the qa page must rank for its own question, got top files: {top_files}"
    )


def test_qa_question_display_fields_unchanged(tmp_path, monkeypatch):
    """Rule 2a is retrieval-only: heading and id keep the slug fallback."""
    import app.indexer as indexer_module

    wiki_dir, _, qa_dir = _setup_wiki(tmp_path)
    slug = "which-countries-does-acme-shop-ship-to-abc123"
    _write_filed_qa_page(
        qa_dir,
        slug,
        question="Which countries do you ship to?",
        body="Delivered worldwide, allegedly.",
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    qa_sections = [s for s in indexer_module.sections if s.file.startswith("which-countries")]
    assert qa_sections
    sec = qa_sections[0]
    assert sec.heading == slug, f"heading must keep the rule-7 slug fallback, got: {sec.heading}"
    assert sec.id == slug, f"id must keep the rule-7 no-anchor shape, got: {sec.id}"


def test_non_qa_page_question_field_stays_untokenized(tmp_path, monkeypatch):
    """The exception is scoped to ``type: qa`` — a concept page carrying a
    stray ``question:`` key keeps rule 2 (frontmatter never tokenized)."""
    import app.indexer as indexer_module

    wiki_dir, concepts_dir, _ = _setup_wiki(tmp_path)
    content = (
        "---\n"
        "id: delivery-zones\n"
        "type: concept\n"
        "question: Which countries do you ship to?\n"
        "---\n\n"
        "## Delivery Zones\n\n"
        "Zones are assigned per warehouse.\n"
    )
    (concepts_dir / "delivery-zones.md").write_text(content, encoding="utf-8")

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    zone_sections = [s for s in indexer_module.sections if s.file == "delivery-zones"]
    assert zone_sections, "concept page must be indexed"
    for sec in zone_sections:
        assert "countries" not in sec.tokens, (
            f"non-qa frontmatter must stay untokenized (rule 2), got: {sec.tokens}"
        )
