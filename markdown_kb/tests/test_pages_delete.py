"""TestClient-seam tests for DELETE /pages/{slug} (tier-B S5, issue #381, ADR-0025).

C11 Confirmed Remediation: deletes an entities/concepts page only when the
full-orphan predicate — ``sources`` non-empty and every citation's file
missing under docs/** — holds, RECOMPUTED server-side at delete time (never
trusts a client-supplied lint finding). No LLM anywhere in this path
(ADR-0024 Invariant) — fully hermetic, no OPENAI_API_KEY needed.

Hermetic: ``app.indexer.WIKI_DIR`` is redirected to a tmp wiki/ pre-populated
with fixture pages; ``app.pages.DOCS_DIR`` is redirected to a tmp docs/ dir
this file controls directly (so "a Source exists" / "a Source is missing" is
asserted precisely, not via the shared fixtures/docs/ corpus).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer_module
import app.pages as pages_module

_FULL_ORPHAN_SLUG = "full-orphan-page"
_PARTIAL_ORPHAN_SLUG = "partial-orphan-page"
_GROUNDED_SLUG = "grounded-page"


def _page_text(slug: str, sources: list[str]) -> str:
    sources_block = "".join(f"  - {s}\n" for s in sources) if sources else "  []\n"
    return (
        "---\n"
        f"id: {slug}\n"
        "type: concept\n"
        "created: '2026-07-03T00:00:00Z'\n"
        "updated: '2026-07-03T00:00:00Z'\n"
        f"sources:\n{sources_block}"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes: {}\n"
        "---\n\n"
        f"# {slug}\n\nSome content.\n"
    )


@pytest.fixture()
def pages_wiki_dir(tmp_path: Path) -> Path:
    """A tmp wiki/concepts/ with a full orphan, a partial orphan, and a grounded page."""
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "concepts" / f"{_FULL_ORPHAN_SLUG}.md").write_text(
        _page_text(_FULL_ORPHAN_SLUG, ["gone_a.md#s", "gone_b.md#s"]),
        encoding="utf-8",
    )
    (wiki_dir / "concepts" / f"{_PARTIAL_ORPHAN_SLUG}.md").write_text(
        _page_text(_PARTIAL_ORPHAN_SLUG, ["gone.md#s", "present.md#s"]),
        encoding="utf-8",
    )
    (wiki_dir / "concepts" / f"{_GROUNDED_SLUG}.md").write_text(
        _page_text(_GROUNDED_SLUG, ["present.md#s"]),
        encoding="utf-8",
    )
    return wiki_dir


@pytest.fixture()
def pages_docs_dir(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "present.md").write_text("# Present\n", encoding="utf-8")
    return docs_dir


@pytest.fixture()
def pages_client(pages_wiki_dir, pages_docs_dir, monkeypatch):
    monkeypatch.setattr(indexer_module, "WIKI_DIR", pages_wiki_dir)
    # SOURCE_DIRS is pre-baked at module load from the real WIKI_DIR; without
    # realigning it, build_index() scans the committed wiki instead of the tmp
    # fixture and the reindex assertions below would not see the tmp pages.
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [pages_wiki_dir / "entities", pages_wiki_dir / "concepts", pages_wiki_dir / "qa"],
    )
    monkeypatch.setattr(pages_module, "DOCS_DIR", pages_docs_dir)

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Refusal paths (409 / 404 / 500) — nothing deleted, no reindex
# ---------------------------------------------------------------------------


def test_delete_refuses_partial_orphan(pages_client, pages_wiki_dir):
    path = pages_wiki_dir / "concepts" / f"{_PARTIAL_ORPHAN_SLUG}.md"
    assert path.exists()

    resp = pages_client.delete(f"/pages/{_PARTIAL_ORPHAN_SLUG}")

    assert resp.status_code == 409, resp.text
    assert path.exists(), "a partial orphan must never be deleted"


def test_delete_refuses_grounded_page(pages_client, pages_wiki_dir):
    path = pages_wiki_dir / "concepts" / f"{_GROUNDED_SLUG}.md"

    resp = pages_client.delete(f"/pages/{_GROUNDED_SLUG}")

    assert resp.status_code == 409, resp.text
    assert path.exists(), "a fully-grounded page must never be deleted"


def test_delete_404_when_slug_missing(pages_client):
    resp = pages_client.delete("/pages/no-such-page")
    assert resp.status_code == 404


def test_delete_refuses_when_source_restored_since_lint(
    pages_client, pages_wiki_dir, pages_docs_dir
):
    """The stale-report scenario ADR-0025 exists to guard against: a lint
    report calls this a full orphan, but the Source was restored/re-imported
    before the curator clicks delete — the server re-verifies NOW and refuses."""
    path = pages_wiki_dir / "concepts" / f"{_FULL_ORPHAN_SLUG}.md"
    # Restore one of the two cited Sources between "lint ran" and "delete clicked".
    (pages_docs_dir / "gone_a.md").write_text("# Restored\n", encoding="utf-8")

    resp = pages_client.delete(f"/pages/{_FULL_ORPHAN_SLUG}")

    assert resp.status_code == 409, resp.text
    assert path.exists(), "a Source restored since the lint report must block the delete"


# ---------------------------------------------------------------------------
# Path-shape guard (issue #397): %5C (backslash) / drive-relative traversal
# ---------------------------------------------------------------------------
#
# A FastAPI ``{slug}`` path segment cannot contain "/" but CAN contain "\\"
# or ":" (route matching is unaffected), which act as path separators once
# joined into ``wiki_dir / subdir_name / f"{slug}.md"`` on Windows.


def test_delete_rejects_pathlike_slug_returns_404_before_filesystem_touch(pages_client, tmp_path):
    """A traversal-shaped slug returns 404 (same mapping a missing slug
    already gets) and never touches a file outside entities/ or concepts/."""
    escape_dir = tmp_path / "wiki" / "qa"
    escape_dir.mkdir(parents=True, exist_ok=True)
    outside = escape_dir / "escape-target.md"
    before = "---\nstatus: live\n---\n\nnot an entity/concept page.\n"
    outside.write_text(before, encoding="utf-8")

    # NUL is exercised directly against ``is_bare_slug`` /
    # ``qa.delete`` etc. in ``test_paths_is_bare_slug.py`` / ``test_qa_delete.py``
    # — httpx refuses to put a raw NUL byte on the wire (InvalidURL), so it
    # cannot reach this route-level seam at all. A forward slash is likewise
    # not a route-level case here: "/pages/../qa/x" normalizes to "/qa/x"
    # before routing (a DIFFERENT endpoint), which is exactly why "/" needs
    # no guard — a FastAPI path segment can never contain it. The forward-
    # slash shape is exercised directly against ``is_bare_slug`` instead.
    for bad in (
        "..\\qa\\escape-target",
        "D:drive-relative",
        "..",
        ".",
    ):
        resp = pages_client.delete(f"/pages/{bad}")
        assert resp.status_code == 404, f"slug={bad!r} got {resp.status_code}: {resp.text}"

    assert outside.read_text(encoding="utf-8") == before, (
        "a path-shaped slug must never reach the filesystem"
    )


def test_delete_full_orphan_cjk_slug_is_not_over_rejected(pages_client, pages_wiki_dir):
    """Real corpus slugs include CJK — the path-shape guard must not
    treat them as invalid."""
    slug = "你們接受哪些付款方式-fb0f2e"
    path = pages_wiki_dir / "concepts" / f"{slug}.md"
    path.write_text(
        "---\n"
        f"id: {slug}\n"
        "type: concept\n"
        "created: '2026-07-03T00:00:00Z'\n"
        "updated: '2026-07-03T00:00:00Z'\n"
        "sources:\n  - gone_a.md#s\n  - gone_b.md#s\n"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes: {}\n"
        "---\n\n"
        f"# {slug}\n\nSome content.\n",
        encoding="utf-8",
    )

    resp = pages_client.delete(f"/pages/{slug}")

    assert resp.status_code == 204, resp.text
    assert not path.exists()


# ---------------------------------------------------------------------------
# Pass path — full orphan deletes, exactly one reindex, page not retrievable
# ---------------------------------------------------------------------------


def test_delete_full_orphan_succeeds_and_reindexes_exactly_once(pages_client, pages_wiki_dir):
    path = pages_wiki_dir / "concepts" / f"{_FULL_ORPHAN_SLUG}.md"
    assert path.exists()

    import app.routes as routes_module

    real_build_index = routes_module.build_index
    spy = MagicMock(wraps=real_build_index)

    with patch.object(routes_module, "build_index", spy):
        resp = pages_client.delete(f"/pages/{_FULL_ORPHAN_SLUG}")

    assert resp.status_code == 204, resp.text
    assert resp.content == b""
    assert not path.exists(), "the full orphan's file must be gone"
    spy.assert_called_once()


def test_delete_full_orphan_removes_page_from_bm25_corpus(pages_client):
    """After a successful delete, the reindex actually ran end to end — the
    in-memory BM25 Section index carries no Section sourced from the deleted
    page's file (not just 'the file vanished from disk')."""
    resp = pages_client.delete(f"/pages/{_FULL_ORPHAN_SLUG}")
    assert resp.status_code == 204, resp.text

    assert not any(s.file == f"concepts/{_FULL_ORPHAN_SLUG}.md" for s in indexer_module.sections), (
        "the BM25 Section index must not retain any Section from the deleted page"
    )
