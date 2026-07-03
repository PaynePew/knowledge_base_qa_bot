"""Tests for Phase 15 Slice 6 ``qa.delete`` — curator delete of inert Filed Answers.

Coverage mirrors issue #174 acceptance criteria for the qa.delete function:

- Delete a draft page → file gone, ``qa_deleted`` log entry emitted.
- Delete a schema-invalid (unparseable frontmatter) page → file gone, log emitted.
- Delete a page with invalid status (not live, not draft) → file gone, log emitted.
- Refuse a live page → raises ``QaPageLive``, file unchanged, no log entry.
- Unknown slug → raises ``QaPageNotFound``.
- Route mapping: ``DELETE /qa/{slug}`` → 204 (draft), 409 (live), 404 (missing).
- Log event: ``qa_deleted slug=<slug> prev_status=<status>`` emitted on success.

External-behaviour testing only: nothing reaches into ``qa._filing_lock`` or
private write helpers. All assertions go through ``qa.delete`` + filesystem +
``wiki/log.md`` reads.

Hermetic: no LLM calls, no production wiki — every test uses ``tmp_path`` via
the autouse ``_redirect_paths_to_tmp`` fixture in ``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Minimal Section stub (mirrors test_qa_promote.py)
# ---------------------------------------------------------------------------


@dataclass
class _StubSection:
    """Minimal Protocol satisfier — qa.maybe_file_answer only reads ``id``."""

    id: str
    heading_path: list[str]
    content: str


def _stub(section_id: str) -> _StubSection:
    return _StubSection(id=section_id, heading_path=[section_id], content="")


# ---------------------------------------------------------------------------
# Helpers: write raw qa page bypassing the filing path
# ---------------------------------------------------------------------------


def _write_raw_qa(tmp_path, slug, status, parseable=True):
    """Write a wiki/qa/<slug>.md file directly, bypassing maybe_file_answer.

    Used to construct pages with controlled status values (live, invalid, etc.)
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
            "sources: []\n"
            f"status: {status}\n"
            "open_questions: []\n"
            'question: "test question"\n'
            "count: 1\n"
            "---\n\nbody text.\n",
            encoding="utf-8",
        )
    else:
        # Schema-invalid / unparseable frontmatter — no valid YAML fences
        path.write_text("not valid yaml frontmatter at all\njust raw text\n", encoding="utf-8")

    return path


# ---------------------------------------------------------------------------
# Delete draft page: happy path
# ---------------------------------------------------------------------------


def test_delete_draft_removes_file(tmp_path):
    """Delete a draft qa page → file gone from disk."""
    from app.qa import compute_slug, delete, maybe_file_answer

    query = "How do I cancel my subscription?"
    cited = [_stub("refund-policy#cancellation")]
    filed = maybe_file_answer(query, "You can cancel anytime.", cited)
    assert filed is not None, "Filing setup failed"
    assert filed.status == "draft"

    slug = compute_slug(query)
    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    assert qa_path.exists(), "File must exist before delete"

    result = delete(slug)

    assert not qa_path.exists(), "File must be gone after delete"
    assert result.slug == slug
    assert result.prev_status == "draft"


def test_delete_draft_emits_qa_deleted_log(tmp_path):
    """Delete emits a ``qa_deleted slug=... prev_status=draft`` log entry."""
    from app.qa import compute_slug, delete, maybe_file_answer

    query = "How do I cancel my subscription?"
    cited = [_stub("refund-policy#cancellation")]
    maybe_file_answer(query, "You can cancel anytime.", cited)
    slug = compute_slug(query)

    delete(slug)

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    deleted_lines = [ln for ln in log.splitlines() if "qa_deleted" in ln]
    assert len(deleted_lines) == 1, (
        f"Expected exactly one qa_deleted log entry, got: {deleted_lines}"
    )
    line = deleted_lines[0]
    assert f"slug={slug}" in line, f"Log must include slug={slug}, got: {line}"
    assert "prev_status=draft" in line, f"Log must include prev_status=draft, got: {line}"


# ---------------------------------------------------------------------------
# Delete schema-invalid (unparseable) page
# ---------------------------------------------------------------------------


def test_delete_unparseable_frontmatter_removes_file(tmp_path):
    """Delete a qa page with completely unparseable frontmatter → file gone."""
    from app.qa import delete

    slug = "some-qa-page-abc123"
    qa_path = _write_raw_qa(tmp_path, slug, status="draft", parseable=False)
    assert qa_path.exists()

    result = delete(slug)

    assert not qa_path.exists(), "File must be gone after delete of unparseable page"
    assert result.slug == slug
    assert result.prev_status == "<unparseable>"


def test_delete_unparseable_emits_log_with_unparseable_status(tmp_path):
    """Delete of an unparseable page logs prev_status=<unparseable>."""
    from app.qa import delete

    slug = "some-qa-page-abc123"
    _write_raw_qa(tmp_path, slug, status="draft", parseable=False)

    delete(slug)

    log_path = tmp_path / "wiki" / "log.md"
    log = log_path.read_text(encoding="utf-8")
    deleted_lines = [ln for ln in log.splitlines() if "qa_deleted" in ln]
    assert len(deleted_lines) == 1
    assert "prev_status=<unparseable>" in deleted_lines[0]


# ---------------------------------------------------------------------------
# Delete page with invalid (non-live, non-draft) status
# ---------------------------------------------------------------------------


def test_delete_invalid_status_removes_file(tmp_path):
    """Delete a qa page with status=Live (capital L) → file gone (inert, not live)."""
    from app.qa import delete

    slug = "corrupt-status-page-xyz456"
    qa_path = _write_raw_qa(tmp_path, slug, status="Live")  # invalid capital L
    assert qa_path.exists()

    result = delete(slug)

    assert not qa_path.exists(), "Schema-invalid page must be deletable"
    assert result.slug == slug
    assert result.prev_status == "Live"


def test_delete_stale_status_removes_file(tmp_path):
    """Delete a qa page with status=stale (forward-compat value) → file gone."""
    from app.qa import delete

    slug = "stale-status-page-def789"
    qa_path = _write_raw_qa(tmp_path, slug, status="stale")
    assert qa_path.exists()

    result = delete(slug)

    assert not qa_path.exists()
    assert result.prev_status == "stale"


# ---------------------------------------------------------------------------
# Refuse live page: QaPageLive + 409
# ---------------------------------------------------------------------------


def test_delete_live_page_raises_qa_page_live(tmp_path):
    """Delete a live qa page → raises QaPageLive; file unchanged."""
    from app.qa import QaPageLive, delete

    slug = "live-answer-ghi012"
    qa_path = _write_raw_qa(tmp_path, slug, status="live")
    before = qa_path.read_text(encoding="utf-8")

    with pytest.raises(QaPageLive):
        delete(slug)

    assert qa_path.exists(), "Live page must NOT be removed on refused delete"
    after = qa_path.read_text(encoding="utf-8")
    assert before == after, "Live page content must be unchanged after refused delete"


def test_delete_live_page_does_not_emit_log(tmp_path):
    """Refused delete of a live page must NOT emit a qa_deleted log entry."""
    from app.qa import QaPageLive, delete

    slug = "live-answer-ghi012"
    _write_raw_qa(tmp_path, slug, status="live")

    with pytest.raises(QaPageLive):
        delete(slug)

    log_path = tmp_path / "wiki" / "log.md"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8")
        assert "qa_deleted" not in log, (
            "Refused delete of a live page must NOT emit a qa_deleted entry"
        )


def test_delete_live_via_route_returns_409(tmp_path):
    """``DELETE /qa/{slug}`` for a live page returns HTTP 409 via the route."""
    from fastapi.testclient import TestClient

    from app.main import app

    slug = "live-answer-route-test"
    _write_raw_qa(tmp_path, slug, status="live")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.delete(f"/qa/{slug}")
    assert response.status_code == 409, (
        f"Expected 409 for live-page delete, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# 404 on missing slug
# ---------------------------------------------------------------------------


def test_delete_missing_slug_raises_qa_page_not_found(tmp_path):
    """No file on disk for the given slug → QaPageNotFound."""
    from app.qa import QaPageNotFound, delete

    with pytest.raises(QaPageNotFound):
        delete("does-not-exist-zz9999")


def test_delete_missing_slug_does_not_emit_log(tmp_path):
    """Failed delete of a missing slug must NOT pollute the log."""
    from app.qa import QaPageNotFound, delete

    with pytest.raises(QaPageNotFound):
        delete("does-not-exist-zz9999")

    log_path = tmp_path / "wiki" / "log.md"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8")
        assert "qa_deleted" not in log


def test_delete_missing_via_route_returns_404(tmp_path):
    """``DELETE /qa/{slug}`` for an unknown slug returns HTTP 404."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.delete("/qa/no-such-slug-aaaaaa")
    assert response.status_code == 404, (
        f"Expected 404 for missing-slug delete, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Path-shape guard (issue #397): %5C (backslash) / drive-relative traversal
# ---------------------------------------------------------------------------
#
# A FastAPI ``{slug}`` path segment cannot contain "/" but CAN contain "\\"
# or ":" (route matching is unaffected), which act as path separators once
# joined into ``_qa_dir() / f"{slug}.md"`` on Windows.


def test_delete_rejects_pathlike_slug_raises_not_found_before_filesystem_touch(tmp_path):
    """A traversal-shaped slug raises QaPageNotFound and never deletes (or
    even touches) a file outside wiki/qa/."""
    from app.qa import QaPageNotFound, delete

    escape_dir = tmp_path / "wiki" / "entities"
    escape_dir.mkdir(parents=True, exist_ok=True)
    outside = escape_dir / "escape-target.md"
    outside.write_text("---\nstatus: live\n---\n\nnot a qa page.\n", encoding="utf-8")

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
            delete(bad)

    assert outside.exists(), "a path-shaped slug must never reach the filesystem"


def test_delete_cjk_slug_is_not_over_rejected(tmp_path):
    """Real corpus slugs include CJK — the path-shape guard must not
    treat them as invalid."""
    from app.qa import delete

    slug = "你們接受哪些付款方式-fb0f2e"
    qa_path = _write_raw_qa(tmp_path, slug, "draft")
    assert qa_path.exists()

    result = delete(slug)

    assert not qa_path.exists()
    assert result.slug == slug


def test_route_delete_pathlike_slug_returns_404(tmp_path):
    """``DELETE /qa/{slug}`` for a backslash-carrying slug returns 404,
    matching the "no such qa page" 404 a garbage slug produces on Linux
    (issue #397 AC)."""
    from fastapi.testclient import TestClient

    from app.main import app

    escape_dir = tmp_path / "wiki" / "entities"
    escape_dir.mkdir(parents=True, exist_ok=True)
    outside = escape_dir / "escape-target.md"
    outside.write_text("---\nstatus: live\n---\n\nnot a qa page.\n", encoding="utf-8")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.delete("/qa/..\\entities\\escape-target")

    assert response.status_code == 404, response.text
    assert outside.exists()


# ---------------------------------------------------------------------------
# Route: 204 on successful delete
# ---------------------------------------------------------------------------


def test_delete_draft_via_route_returns_204(tmp_path):
    """``DELETE /qa/{slug}`` for a draft page returns HTTP 204 No Content."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.qa import compute_slug, maybe_file_answer

    query = "How do I cancel my subscription?"
    cited = [_stub("refund-policy#cancellation")]
    maybe_file_answer(query, "You can cancel anytime.", cited)
    slug = compute_slug(query)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.delete(f"/qa/{slug}")
    assert response.status_code == 204, f"Expected 204 for draft delete, got {response.status_code}"

    # File must be gone
    qa_path = tmp_path / "wiki" / "qa" / f"{slug}.md"
    assert not qa_path.exists(), "Draft file must be removed after 204 response"
