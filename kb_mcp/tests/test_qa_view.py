"""Tests for ``kb_mcp.qa_view`` — the read-only Filed Answer display helper (issue #377).

Path isolation: the conftest autouse fixture redirects
``markdown_kb.app.indexer.WIKI_DIR`` to ``tmp_path / "wiki"``, and ``qa_view``
resolves the qa dir from that module attribute at call time, so every test
here operates purely under ``tmp_path`` without further redirection.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _write_qa_page(
    wiki_dir: Path,
    slug: str,
    *,
    question: str = "How do refunds work?",
    status: str = "draft",
    count: int = 1,
    sources: list[str] | None = None,
    body: str = "Refunds are processed within 5 business days.",
) -> Path:
    """Write a real ``wiki/qa/<slug>.md`` fixture mirroring ``qa._render_qa_page``.

    Mirrors the real producer's shape (CODING_STANDARD §6.5 fixture fidelity):
    a leading sentinel HTML comment, then YAML frontmatter, then the body.
    """
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "id": slug,
        "type": "qa",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": sources or ["refund-policy#cancellation-window"],
        "status": status,
        "open_questions": [],
        "question": question,
        "count": count,
    }
    fm_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    content = f"<!-- Auto-filed by POST /chat. -->\n\n---\n{fm_yaml}---\n\n{body}\n"
    page_path = qa_dir / f"{slug}.md"
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ---------------------------------------------------------------------------
# qa_page_path / display_path
# ---------------------------------------------------------------------------


def test_qa_page_path_resolves_under_patched_wiki_dir(tmp_path: Path) -> None:
    """qa_page_path derives from the (monkeypatched) indexer.WIKI_DIR at call time."""
    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp import qa_view

    assert tmp_path / "wiki" == indexer_mod.WIKI_DIR
    assert qa_view.qa_page_path("some-slug") == tmp_path / "wiki" / "qa" / "some-slug.md"


def test_display_path_is_pure_string_formatting(tmp_path: Path) -> None:
    """display_path never touches disk — safe to call for a nonexistent slug."""
    from kb_mcp import qa_view

    assert qa_view.display_path("never-written") == "wiki/qa/never-written.md"


# ---------------------------------------------------------------------------
# read_qa_page
# ---------------------------------------------------------------------------


def test_read_qa_page_returns_none_when_missing(tmp_path: Path) -> None:
    from kb_mcp import qa_view

    assert qa_view.read_qa_page("does-not-exist") is None


def test_read_qa_page_returns_none_for_unparseable_frontmatter(tmp_path: Path) -> None:
    """A file with no frontmatter fence at all yields no usable metadata -> None."""
    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp import qa_view

    qa_dir = indexer_mod.WIKI_DIR / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "broken.md").write_text("just some text, no frontmatter\n", encoding="utf-8")

    assert qa_view.read_qa_page("broken") is None


def test_read_qa_page_reads_question_status_sources_body(tmp_path: Path) -> None:
    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp import qa_view

    _write_qa_page(
        indexer_mod.WIKI_DIR,
        "popular-question",
        question="How do refunds work?",
        status="draft",
        count=3,
        sources=["refund-policy#cancellation-window", "refund-policy#eligibility"],
        body="Refunds are processed within 5 business days.",
    )

    page = qa_view.read_qa_page("popular-question")

    assert page is not None
    assert page.slug == "popular-question"
    assert page.status == "draft"
    assert page.question == "How do refunds work?"
    assert page.count == 3
    assert page.sources == [
        "refund-policy#cancellation-window",
        "refund-policy#eligibility",
    ]
    assert page.body == "Refunds are processed within 5 business days."
    assert page.path == "wiki/qa/popular-question.md"


def test_read_qa_page_question_is_untruncated(tmp_path: Path) -> None:
    """Unlike the C8 lint finding's 80-char report truncation, show/read is full-length."""
    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp import qa_view

    long_question = "Why " + ("x" * 100) + "?"
    _write_qa_page(indexer_mod.WIKI_DIR, "long-question", question=long_question)

    page = qa_view.read_qa_page("long-question")

    assert page is not None
    assert page.question == long_question


def test_read_qa_page_coerces_missing_count_to_none(tmp_path: Path) -> None:
    import markdown_kb.app.indexer as indexer_mod

    from kb_mcp import qa_view

    qa_dir = indexer_mod.WIKI_DIR / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "id": "no-count",
        "type": "qa",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": [],
        "status": "draft",
        "open_questions": [],
        "question": "A question with no count field?",
    }
    fm_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    (qa_dir / "no-count.md").write_text(f"---\n{fm_yaml}---\n\nBody text.\n", encoding="utf-8")

    page = qa_view.read_qa_page("no-count")

    assert page is not None
    assert page.count is None
