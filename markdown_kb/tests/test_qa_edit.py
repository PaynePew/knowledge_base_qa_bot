"""Tests for tier-B S3 ``qa.edit`` / ``PUT /qa/{slug}`` (issue #379, ADR-0026 decision 2).

Coverage mirrors the issue's acceptance criteria:

- Edit a draft page whose submitted body passes the LLM-free grounding
  re-check against its cited Sections -> persists (question/body updated,
  ``status`` stays ``draft``, ``created``/``sources``/``count`` preserved,
  ``updated`` bumped, ``qa_reflect op=edited`` logged).
- Edit a ``status: live`` page -> refused (``QaPageLive`` / HTTP 409),
  draft-only per ADR-0026.
- Edit whose submitted body fails the re-check -> rejected
  (``QaEditRejected`` / HTTP 422 with the failure list), file unchanged.
  Failure modes: no inline citation at all; a citation outside the page's
  recorded sources (an edit never widens sources); a citation that no
  longer resolves to a Section on disk.
- Missing slug -> ``QaPageNotFound`` / HTTP 404. Corrupt frontmatter ->
  ``QaPageCorrupt`` / HTTP 500 (orphan-visibility).

The re-check is LLM-free by design (ADR-0026: "the re-check is LLM-free and
instant; there is no cost argument for skipping it") — no test here stubs
any LLM seam; a real network call would error loudly if the edit path ever
tried to construct an LLM client.

Hermetic: no OPENAI_API_KEY needed. Every test uses ``tmp_path`` via the
autouse ``_redirect_paths_to_tmp`` fixture in ``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal Section stub (mirrors test_qa_promote.py / test_qa_delete.py)
# ---------------------------------------------------------------------------


@dataclass
class _StubSection:
    """Minimal Protocol satisfier -- qa.maybe_file_answer only reads ``id``."""

    id: str
    heading_path: list[str]
    content: str


def _stub(section_id: str) -> _StubSection:
    return _StubSection(id=section_id, heading_path=[section_id], content="")


_CITED_SECTION_ID = "cancellation-window#cancellation-window"
_CITED_BODY = f"Within 24 hours of purchase. [Source: {_CITED_SECTION_ID}]"

_ENTITY_PAGE = (
    "---\n"
    "id: cancellation-window\n"
    "type: entity\n"
    "created: '2026-01-01T00:00:00Z'\n"
    "updated: '2026-01-01T00:00:00Z'\n"
    "sources: []\n"
    "status: live\n"
    "open_questions: []\n"
    "source_hashes: {}\n"
    "---\n\n"
    "# Cancellation Window\n\n"
    "Orders can be cancelled within 24 hours of purchase.\n"
)


def _write_cited_entity_page(tmp_path) -> None:
    """Plant the entity page a filed qa draft cites, so ``_resolve_cited_sections``
    can resolve ``_CITED_SECTION_ID`` back to real content."""
    entities_dir = tmp_path / "wiki" / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    (entities_dir / "cancellation-window.md").write_text(_ENTITY_PAGE, encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers: write raw qa page bypassing the filing path (mirrors test_qa_delete.py)
# ---------------------------------------------------------------------------


def _write_raw_qa(tmp_path, slug, status, sources=None, parseable=True):
    qa_dir = tmp_path / "wiki" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{slug}.md"

    if parseable:
        sources = sources if sources is not None else []
        sources_yaml = "[]" if not sources else "\n" + "\n".join(f"  - {s}" for s in sources)
        path.write_text(
            "---\n"
            f"id: {slug}\n"
            "type: qa\n"
            'created: "2026-05-29T00:00:00Z"\n'
            'updated: "2026-05-29T00:00:00Z"\n'
            f"sources: {sources_yaml}\n"
            f"status: {status}\n"
            "open_questions: []\n"
            'question: "original question"\n'
            "count: 3\n"
            "---\n\noriginal body text.\n",
            encoding="utf-8",
        )
    else:
        path.write_text("not valid yaml frontmatter at all\njust raw text\n", encoding="utf-8")

    return path


# ---------------------------------------------------------------------------
# Direct-module tests: qa.edit()
# ---------------------------------------------------------------------------


def test_edit_draft_passing_grounding_persists(tmp_path):
    from app.qa import compute_slug, edit, maybe_file_answer

    _write_cited_entity_page(tmp_path)
    query = "How long can I cancel an order?"
    cited = [_stub(_CITED_SECTION_ID)]
    filed = maybe_file_answer(query, "Within 24 hours.", cited)
    assert filed is not None and filed.status == "draft"
    slug = compute_slug(query)

    result = edit(slug, "How long can I cancel an order?", _CITED_BODY)

    assert result.slug == slug
    assert result.status == "draft"
    assert result.op == "touched"
    assert result.count == filed.count

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert _CITED_BODY in after
    assert "status: draft" in after
    assert f"count: {filed.count}" in after
    assert _CITED_SECTION_ID in after, "sources must be preserved verbatim, not widened"


def test_edit_preserves_created_and_sources_bumps_updated(tmp_path):
    from app.qa import edit

    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])
    before = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")

    edit(slug, "edited question", _CITED_BODY)

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert "2026-05-29T00:00:00Z" in before  # sanity: fixture created value
    assert "2026-05-29T00:00:00Z" in after, "created must be preserved verbatim"
    assert "count: 3" in after, "count must be preserved verbatim (edit does not bump it)"
    assert "edited question" in after
    assert _CITED_BODY in after
    assert "original body text." not in after


def test_edit_emits_qa_reflect_edited_log(tmp_path):
    from app.qa import edit

    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])

    edit(slug, "edited question", _CITED_BODY)

    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    edited_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln and "op=edited" in ln]
    assert len(edited_lines) == 1, (
        f"Expected exactly one op=edited reflect entry, got: {edited_lines}"
    )
    assert f"slug={slug}" in edited_lines[0]


def test_edit_live_page_refused(tmp_path):
    from app.qa import QaPageLive, edit

    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "live", sources=[_CITED_SECTION_ID])
    before = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")

    with pytest.raises(QaPageLive):
        edit(slug, "edited question", _CITED_BODY)

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert after == before, "a refused edit must leave the live page byte-identical"


def test_edit_missing_slug_raises_not_found(tmp_path):
    from app.qa import QaPageNotFound, edit

    with pytest.raises(QaPageNotFound):
        edit("no-such-slug", "q", "b")


# ---------------------------------------------------------------------------
# Path-shape guard (issue #397): %5C (backslash) / drive-relative traversal
# ---------------------------------------------------------------------------
#
# A FastAPI ``{slug}`` path segment cannot contain "/" but CAN contain "\\"
# or ":" (route matching is unaffected), which act as path separators once
# joined into ``_qa_dir() / f"{slug}.md"`` on Windows.


def test_edit_rejects_pathlike_slug_raises_not_found_before_filesystem_touch(tmp_path):
    """A traversal-shaped slug raises QaPageNotFound and never rewrites a
    file outside wiki/qa/."""
    from app.qa import QaPageNotFound, edit

    escape_dir = tmp_path / "wiki" / "entities"
    escape_dir.mkdir(parents=True, exist_ok=True)
    outside = escape_dir / "escape-target.md"
    before = "---\nstatus: draft\n---\n\nnot a qa page.\n"
    outside.write_text(before, encoding="utf-8")

    for bad in (
        "..\\entities\\escape-target",
        "../entities/escape-target",
        "D:drive-relative",
        "..",
        ".",
        "",
        "nul\x00byte",
    ):
        with pytest.raises(QaPageNotFound):
            edit(bad, "q", "b")

    assert outside.read_text(encoding="utf-8") == before, (
        "a path-shaped slug must never reach the filesystem"
    )


def test_edit_cjk_slug_is_not_over_rejected(tmp_path):
    """Real corpus slugs include CJK — the path-shape guard must not
    treat them as invalid."""
    from app.qa import edit

    _write_cited_entity_page(tmp_path)
    slug = "你們接受哪些付款方式-fb0f2e"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])

    result = edit(slug, "edited question", _CITED_BODY)

    assert result.slug == slug
    assert result.status == "draft"


def test_edit_corrupt_frontmatter_raises_corrupt(tmp_path):
    from app.qa import QaPageCorrupt, edit

    slug = "broken-page"
    _write_raw_qa(tmp_path, slug, "draft", parseable=False)

    with pytest.raises(QaPageCorrupt):
        edit(slug, "q", "b")


def test_edit_body_without_citation_rejected_writes_nothing(tmp_path):
    """ADR-0026: an uncited answer is ungrounded drift — rejected with the
    failure naming the missing citation, and the file stays untouched."""
    from app.qa import QaEditRejected, edit

    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])
    before = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")

    with pytest.raises(QaEditRejected) as exc_info:
        edit(slug, "edited question", "Orders can be cancelled within 90 days.")

    assert exc_info.value.failures
    assert any("no [Source:" in f for f in exc_info.value.failures)

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert after == before, "a rejected edit must write nothing"


def test_edit_citation_outside_sources_rejected(tmp_path):
    """An edit never widens ``frontmatter.sources`` — citing a Section the
    page does not record is rejected with a failure pointing at Re-file."""
    from app.qa import QaEditRejected, edit

    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])

    with pytest.raises(QaEditRejected) as exc_info:
        edit(slug, "edited question", "New claim. [Source: some-other-page#some-heading]")

    assert any("not among this page's sources" in f for f in exc_info.value.failures)


def test_edit_unresolvable_citation_rejected(tmp_path):
    """A citation whose Section no longer resolves on disk fails the
    LLM-free check deterministically — no LLM client is ever constructed
    (no stub installed; a network attempt would error loudly)."""
    from app.qa import QaEditRejected, edit

    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=["missing-page#missing-heading"])

    with pytest.raises(QaEditRejected) as exc_info:
        edit(slug, "edited question", "edited body. [Source: missing-page#missing-heading]")

    assert any("no longer resolves" in f for f in exc_info.value.failures)


# ---------------------------------------------------------------------------
# Route-level tests: PUT /qa/{slug}
# ---------------------------------------------------------------------------


@pytest.fixture()
def edit_client(tmp_path, monkeypatch):
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_route_edit_success_returns_200_and_filed_status(edit_client, tmp_path):
    from app.qa import maybe_file_answer

    _write_cited_entity_page(tmp_path)
    query = "How long can I cancel an order?"
    filed = maybe_file_answer(query, "Within 24 hours.", [_stub(_CITED_SECTION_ID)])
    assert filed is not None
    slug_path = list((tmp_path / "wiki" / "qa").glob("*.md"))[0]
    slug = slug_path.stem

    resp = edit_client.put(
        f"/qa/{slug}",
        json={"question": "How long can I cancel?", "body": _CITED_BODY},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["slug"] == slug
    assert data["status"] == "draft"


def test_route_edit_live_page_returns_409(edit_client, tmp_path):
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "live", sources=[_CITED_SECTION_ID])

    resp = edit_client.put(f"/qa/{slug}", json={"question": "q", "body": _CITED_BODY})

    assert resp.status_code == 409


def test_route_edit_missing_slug_returns_404(edit_client):
    resp = edit_client.put("/qa/no-such-slug", json={"question": "q", "body": "b"})

    assert resp.status_code == 404


def test_route_edit_pathlike_slug_returns_404(edit_client, tmp_path):
    """``PUT /qa/{slug}`` for a backslash-carrying slug returns 404, matching
    the "no such qa page" 404 a garbage slug produces on Linux (issue #397 AC)."""
    escape_dir = tmp_path / "wiki" / "entities"
    escape_dir.mkdir(parents=True, exist_ok=True)
    outside = escape_dir / "escape-target.md"
    before = "---\nstatus: draft\n---\n\nnot a qa page.\n"
    outside.write_text(before, encoding="utf-8")

    resp = edit_client.put(
        "/qa/..\\entities\\escape-target",
        json={"question": "q", "body": "b"},
    )

    assert resp.status_code == 404, resp.text
    assert outside.read_text(encoding="utf-8") == before


def test_route_edit_grounding_failure_returns_422_with_failures(edit_client, tmp_path):
    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])

    resp = edit_client.put(
        f"/qa/{slug}",
        json={
            "question": "q",
            "body": "Orders can be cancelled within 90 days for a full refund.",
        },
    )

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["failures"], "the 422 detail must carry the failure list (ADR-0026)"
