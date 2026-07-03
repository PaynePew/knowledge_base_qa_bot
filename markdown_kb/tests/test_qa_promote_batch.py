"""Tests for tier-B S6 ``qa.promote_batch`` / ``POST /qa/promote-batch``
(issue #382, ADR-0023 Consequences).

Coverage mirrors the issue's acceptance criteria:

- A batch of all-valid drafts promotes every slug; the response's
  ``promoted`` list matches submission order and ``skipped`` is empty.
- Per-slug validation is independent — a missing slug, an already-live
  slug, a corrupt-frontmatter page, and an invalid-status page are each
  skipped with a distinct reason; a batch mixing valid and invalid slugs
  still promotes every valid one (non-transactional, ADR-0023).
- An empty slugs list is a well-formed no-op (``promoted=[]``,
  ``skipped=[]``), not an error.
- Exactly one ``build_index()`` call regardless of how many slugs were
  submitted (issue #382 AC).
- Promoted pages are retrievable immediately after the route's single
  reindex.

Hermetic: no LLM calls anywhere on this path (Direct Remediation). Every
test uses ``tmp_path`` via the autouse ``_redirect_paths_to_tmp`` fixture in
``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal Section stub implementing the CitableContent Protocol
# (mirrors test_qa_promote.py / test_qa_edit.py)
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
# Helper: write a raw qa page bypassing the filing path (mirrors
# test_qa_edit.py / test_qa_delete.py)
# ---------------------------------------------------------------------------


def _write_raw_qa(tmp_path, slug, status, parseable=True):
    qa_dir = tmp_path / "wiki" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{slug}.md"

    if parseable:
        path.write_text(
            "---\n"
            f"id: {slug}\n"
            "type: qa\n"
            'created: "2026-07-03T00:00:00Z"\n'
            'updated: "2026-07-03T00:00:00Z"\n'
            "sources: []\n"
            f"status: {status}\n"
            "open_questions: []\n"
            f'question: "question for {slug}"\n'
            "count: 1\n"
            "---\n\noriginal body text.\n",
            encoding="utf-8",
        )
    else:
        path.write_text("not valid yaml frontmatter at all\njust raw text\n", encoding="utf-8")

    return path


# ---------------------------------------------------------------------------
# Direct-module tests: qa.promote_batch()
# ---------------------------------------------------------------------------


def test_promote_batch_all_valid_drafts_promotes_all(tmp_path):
    from app.qa import promote_batch

    _write_raw_qa(tmp_path, "draft-one", "draft")
    _write_raw_qa(tmp_path, "draft-two", "draft")

    result = promote_batch(["draft-one", "draft-two"])

    assert result.promoted == ["draft-one", "draft-two"]
    assert result.skipped == []

    for slug in ("draft-one", "draft-two"):
        content = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
        assert "status: live" in content


def test_promote_batch_skips_missing_slug(tmp_path):
    from app.qa import promote_batch

    result = promote_batch(["does-not-exist-zzz"])

    assert result.promoted == []
    assert len(result.skipped) == 1
    assert result.skipped[0].slug == "does-not-exist-zzz"
    assert result.skipped[0].reason == "not_found"


def test_promote_batch_skips_already_live(tmp_path):
    from app.qa import promote_batch

    path = _write_raw_qa(tmp_path, "already-live", "live")
    before = path.read_text(encoding="utf-8")

    result = promote_batch(["already-live"])

    assert result.promoted == []
    assert len(result.skipped) == 1
    assert result.skipped[0].slug == "already-live"
    assert result.skipped[0].reason == "already_live"
    assert path.read_text(encoding="utf-8") == before, (
        "a slug already live must NOT be rewritten — nothing to promote"
    )


def test_promote_batch_skips_corrupt_frontmatter(tmp_path):
    from app.qa import promote_batch

    _write_raw_qa(tmp_path, "broken-page", "draft", parseable=False)

    result = promote_batch(["broken-page"])

    assert result.promoted == []
    assert len(result.skipped) == 1
    assert result.skipped[0].slug == "broken-page"
    assert result.skipped[0].reason == "corrupt_frontmatter"


def test_promote_batch_skips_invalid_status(tmp_path):
    from app.qa import promote_batch

    _write_raw_qa(tmp_path, "stale-status-page", "stale")

    result = promote_batch(["stale-status-page"])

    assert result.promoted == []
    assert len(result.skipped) == 1
    assert result.skipped[0].slug == "stale-status-page"
    assert result.skipped[0].reason == "invalid_status:stale"


def test_promote_batch_mixed_valid_and_invalid_promotes_only_valid(tmp_path):
    """Non-transactional (ADR-0023): a bad slug never aborts the batch —
    every OTHER valid slug in the same call still promotes."""
    from app.qa import promote_batch

    _write_raw_qa(tmp_path, "good-draft", "draft")
    _write_raw_qa(tmp_path, "live-already", "live")

    result = promote_batch(["good-draft", "missing-slug", "live-already"])

    assert result.promoted == ["good-draft"]
    reasons = {s.slug: s.reason for s in result.skipped}
    assert reasons == {
        "missing-slug": "not_found",
        "live-already": "already_live",
    }

    content = (tmp_path / "wiki" / "qa" / "good-draft.md").read_text(encoding="utf-8")
    assert "status: live" in content


def test_promote_batch_empty_list_returns_empty_result(tmp_path):
    from app.qa import promote_batch

    result = promote_batch([])

    assert result.promoted == []
    assert result.skipped == []


def test_promote_batch_emits_one_qa_reflect_promoted_line_per_promoted_slug(tmp_path):
    from app.qa import promote_batch

    _write_raw_qa(tmp_path, "draft-one", "draft")
    _write_raw_qa(tmp_path, "draft-two", "draft")

    promote_batch(["draft-one", "draft-two", "missing-slug"])

    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    promoted_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln and "op=promoted" in ln]
    assert len(promoted_lines) == 2, (
        f"Expected one reflect line per promoted slug, got: {promoted_lines}"
    )
    assert any("slug=draft-one" in ln and "by=curator" in ln for ln in promoted_lines)
    assert any("slug=draft-two" in ln and "by=curator" in ln for ln in promoted_lines)
    assert "missing-slug" not in log, "a skipped slug must not emit a phantom reflect entry"


def test_promote_batch_preserves_body_and_other_frontmatter(tmp_path):
    from app.qa import compute_slug, maybe_file_answer, promote_batch

    query = "How do I cancel my order?"
    cited = [_stub("refund-policy#cancellation-window")]
    maybe_file_answer(query, "Within 24h. Original body wording.", cited)
    slug = compute_slug(query)

    promote_batch([slug])

    content = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert "Within 24h. Original body wording." in content
    assert "How do I cancel my order" in content
    assert "refund-policy#cancellation-window" in content
    assert "count: 1" in content
    assert "status: live" in content


# ---------------------------------------------------------------------------
# Route-level tests: POST /qa/promote-batch
# ---------------------------------------------------------------------------


@pytest.fixture()
def promote_batch_client(tmp_path, monkeypatch):
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [
            tmp_path / "wiki" / "entities",
            tmp_path / "wiki" / "concepts",
            tmp_path / "wiki" / "qa",
        ],
    )

    from app.main import app

    # indexer.sections is a module-level singleton that persists across tests
    # in this file (mirrors test_routes_filing.py's built_corpus fixture) —
    # clear it so a prior test's in-memory index never leaks into this one.
    indexer_module.sections.clear()
    yield TestClient(app, raise_server_exceptions=False)
    indexer_module.sections.clear()


def test_route_promote_batch_returns_promoted_and_skipped(promote_batch_client, tmp_path):
    _write_raw_qa(tmp_path, "draft-one", "draft")

    resp = promote_batch_client.post(
        "/qa/promote-batch", json={"slugs": ["draft-one", "missing-slug"]}
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["promoted"] == ["draft-one"]
    assert body["skipped"] == [{"slug": "missing-slug", "reason": "not_found"}]


def test_route_promote_batch_empty_list_returns_200_empty(promote_batch_client):
    resp = promote_batch_client.post("/qa/promote-batch", json={"slugs": []})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"promoted": [], "skipped": []}


def test_route_promote_batch_reindexes_exactly_once(promote_batch_client, tmp_path):
    """Exactly one BM25 rebuild per batch call, regardless of N (issue #382 AC)."""
    _write_raw_qa(tmp_path, "draft-one", "draft")
    _write_raw_qa(tmp_path, "draft-two", "draft")
    _write_raw_qa(tmp_path, "draft-three", "draft")

    import app.routes as routes_module

    real_build_index = routes_module.build_index
    spy = MagicMock(wraps=real_build_index)

    with patch.object(routes_module, "build_index", spy):
        resp = promote_batch_client.post(
            "/qa/promote-batch",
            json={"slugs": ["draft-one", "draft-two", "draft-three", "missing-slug"]},
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["promoted"] == ["draft-one", "draft-two", "draft-three"]
    spy.assert_called_once()


def test_route_promote_batch_promoted_pages_retrievable_immediately(promote_batch_client, tmp_path):
    """After the batch's single auto-reindex, every promoted page is in the
    in-memory BM25 corpus without a separate POST /index call (mirrors
    test_promote_auto_reindexes_without_explicit_index_call for the single-
    item promote route)."""
    _write_raw_qa(tmp_path, "draft-one", "draft")
    _write_raw_qa(tmp_path, "draft-two", "draft")

    import app.indexer as indexer_module

    qa_sections_before = [s for s in indexer_module.sections if s.metadata.get("type") == "qa"]
    assert not qa_sections_before

    resp = promote_batch_client.post(
        "/qa/promote-batch", json={"slugs": ["draft-one", "draft-two"]}
    )
    assert resp.status_code == 200, resp.text

    qa_ids = [s.id for s in indexer_module.sections if s.metadata.get("type") == "qa"]
    assert any("draft-one" in sid for sid in qa_ids)
    assert any("draft-two" in sid for sid in qa_ids)


def test_route_promote_batch_skipped_slug_file_unchanged(promote_batch_client, tmp_path):
    path = _write_raw_qa(tmp_path, "already-live", "live")
    before = path.read_text(encoding="utf-8")

    resp = promote_batch_client.post("/qa/promote-batch", json={"slugs": ["already-live"]})

    assert resp.status_code == 200, resp.text
    assert resp.json()["skipped"] == [{"slug": "already-live", "reason": "already_live"}]
    assert path.read_text(encoding="utf-8") == before
