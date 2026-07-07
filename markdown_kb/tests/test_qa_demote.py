"""Tests for issue #535 (ADR-0037) ``qa.demote`` / ``POST /qa/{slug}/demote``.

Coverage mirrors the issue's acceptance criteria:

- ``qa.demote`` flips a live page ``live -> draft`` in place, preserving
  question/body/count/created, bumping ``updated``, and logs ``qa_demoted``.
- Idempotent on an already-draft page: no rewrite, no log entry.
- Unknown slug -> ``QaPageNotFound``. Corrupt frontmatter / invalid status ->
  ``QaPageCorrupt`` (orphan-visibility).
- Route mapping: ``POST /qa/{slug}/demote`` -> 200 (live or draft), 404
  (missing), 500 (corrupt); the route reindexes exactly once on success so
  the demoted page leaves the live BM25 corpus immediately.

External-behaviour testing only: nothing reaches into ``qa._filing_lock`` or
private write helpers. Hermetic — no LLM calls, no production wiki — every
test uses ``tmp_path`` via the autouse ``_redirect_paths_to_tmp`` fixture in
``conftest.py`` (mirrors ``test_qa_delete.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _write_raw_qa(tmp_path: Path, slug: str, status: str, parseable: bool = True) -> Path:
    """Write a wiki/qa/<slug>.md file directly, bypassing maybe_file_answer.

    Mirrors ``test_qa_delete.py``'s helper of the same name — used to
    construct pages with controlled status values (live, invalid, etc.)
    that the normal filing path would not produce.
    """
    qa_dir = tmp_path / "wiki" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{slug}.md"

    if parseable:
        path.write_text(
            "---\n"
            f"id: {slug}\n"
            "type: qa\n"
            'created: "2026-05-29T00:00:00Z"\n'
            'updated: "2026-05-29T00:00:00Z"\n'
            "sources:\n"
            "  - acme-shop#acme-shop\n"
            f"status: {status}\n"
            "open_questions: []\n"
            'question: "test question"\n'
            "count: 3\n"
            "---\n\nbody text.\n",
            encoding="utf-8",
        )
    else:
        path.write_text("not valid yaml frontmatter at all\njust raw text\n", encoding="utf-8")

    return path


# ---------------------------------------------------------------------------
# Direct-module tests: qa.demote()
# ---------------------------------------------------------------------------


def test_demote_live_flips_to_draft_preserving_content(tmp_path):
    """Demote a live page -> status: draft, question/body/count/created preserved."""
    from app.qa import demote

    slug = "count-zero-page-abc123"
    qa_path = _write_raw_qa(tmp_path, slug, status="live")

    result = demote(slug)

    assert result.slug == slug
    assert result.status == "draft"
    assert result.count == 3

    after = qa_path.read_text(encoding="utf-8")
    assert "status: draft" in after
    assert "body text." in after, "body must be preserved verbatim"
    assert "test question" in after, "question must be preserved verbatim"
    assert "created: '2026-05-29T00:00:00Z'" in after, "created must be preserved verbatim"
    assert "acme-shop#acme-shop" in after, "sources must be preserved verbatim"


def test_demote_bumps_updated_timestamp(tmp_path):
    from app.qa import demote

    slug = "bump-updated-page"
    qa_path = _write_raw_qa(tmp_path, slug, status="live")

    demote(slug)

    after = qa_path.read_text(encoding="utf-8")
    assert "updated: '2026-05-29T00:00:00Z'" not in after, "updated timestamp must be bumped"
    assert "created: '2026-05-29T00:00:00Z'" in after, "created must be preserved verbatim"


def test_demote_emits_qa_demoted_log(tmp_path):
    from app.qa import demote

    slug = "log-demoted-page"
    _write_raw_qa(tmp_path, slug, status="live")

    demote(slug)

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    demoted_lines = [ln for ln in log.splitlines() if "qa_demoted" in ln]
    assert len(demoted_lines) == 1, (
        f"Expected exactly one qa_demoted log entry, got: {demoted_lines}"
    )
    assert f"slug={slug}" in demoted_lines[0]
    assert "prev_status=live" in demoted_lines[0]


def test_demote_already_draft_is_idempotent_no_rewrite_no_log(tmp_path):
    """Demoting an already-draft page is a no-op: file byte-identical, no log entry."""
    from app.qa import demote

    slug = "already-draft-page"
    qa_path = _write_raw_qa(tmp_path, slug, status="draft")
    before = qa_path.read_text(encoding="utf-8")

    result = demote(slug)

    assert result.status == "draft"
    assert result.count == 3
    after = qa_path.read_text(encoding="utf-8")
    assert after == before, "an already-draft page must be left byte-identical"

    log_path = tmp_path / "wiki" / "log.md"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8")
        assert "qa_demoted" not in log, "demoting an already-draft page must not log"


def test_demote_missing_slug_raises_not_found(tmp_path):
    from app.qa import QaPageNotFound, demote

    with pytest.raises(QaPageNotFound):
        demote("does-not-exist-zz9999")


def test_demote_corrupt_frontmatter_raises_corrupt(tmp_path):
    from app.qa import QaPageCorrupt, demote

    slug = "broken-frontmatter-page"
    _write_raw_qa(tmp_path, slug, status="live", parseable=False)

    with pytest.raises(QaPageCorrupt):
        demote(slug)


def test_demote_invalid_status_raises_corrupt(tmp_path):
    """A status value outside {draft, live} (e.g. curator typo) is corrupt,
    not silently coerced — orphan-visibility (mirrors promote/edit)."""
    from app.qa import QaPageCorrupt, demote

    slug = "typo-status-page"
    _write_raw_qa(tmp_path, slug, status="Live")

    with pytest.raises(QaPageCorrupt):
        demote(slug)


def test_demote_rejects_pathlike_slug_raises_not_found_before_filesystem_touch(tmp_path):
    """A traversal-shaped slug raises QaPageNotFound and never touches a file
    outside wiki/qa/ (issue #397 pattern)."""
    from app.qa import QaPageNotFound, demote

    escape_dir = tmp_path / "wiki" / "entities"
    escape_dir.mkdir(parents=True, exist_ok=True)
    outside = escape_dir / "escape-target.md"
    before = "---\nstatus: live\n---\n\nnot a qa page.\n"
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
            demote(bad)

    assert outside.read_text(encoding="utf-8") == before


def test_demote_cjk_slug_is_not_over_rejected(tmp_path):
    """Real corpus slugs include CJK — the path-shape guard must not
    treat them as invalid."""
    from app.qa import demote

    slug = "你們接受哪些付款方式-fb0f2e"
    _write_raw_qa(tmp_path, slug, status="live")

    result = demote(slug)

    assert result.slug == slug
    assert result.status == "draft"


# ---------------------------------------------------------------------------
# Route-level tests: POST /qa/{slug}/demote
# ---------------------------------------------------------------------------


@pytest.fixture()
def demote_client():
    """TestClient — WIKI_DIR/INDEX_PATH/LOG_PATH are already redirected to
    tmp_path by the autouse ``_redirect_paths_to_tmp`` fixture in conftest."""
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_route_demote_missing_slug_returns_404(demote_client):
    resp = demote_client.post("/qa/no-such-slug/demote")

    assert resp.status_code == 404


def test_route_demote_pathlike_slug_returns_404(demote_client):
    resp = demote_client.post("/qa/..\\entities\\escape-target/demote")

    assert resp.status_code == 404, resp.text


def test_route_demote_corrupt_frontmatter_returns_500(demote_client, tmp_path):
    slug = "corrupt-route-page"
    _write_raw_qa(tmp_path, slug, status="live", parseable=False)

    resp = demote_client.post(f"/qa/{slug}/demote")

    assert resp.status_code == 500, resp.text


def test_route_demote_live_returns_200_and_reindexes(demote_client, tmp_path, monkeypatch):
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts", wiki_dir / "qa"],
    )
    slug = "live-route-page"
    _write_raw_qa(tmp_path, slug, status="live")
    indexer_module.build_index()
    assert any(s.file == slug for s in indexer_module.sections), "live: in corpus before demote"

    resp = demote_client.post(f"/qa/{slug}/demote")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["slug"] == slug
    assert data["status"] == "draft"
    assert not any(s.file == slug for s in indexer_module.sections), (
        "the demoted (now-draft) page must leave the live BM25 corpus after reindex"
    )


def test_route_demote_already_draft_returns_200_idempotent(demote_client, tmp_path):
    slug = "already-draft-route-page"
    _write_raw_qa(tmp_path, slug, status="draft")

    resp = demote_client.post(f"/qa/{slug}/demote")

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "draft"
