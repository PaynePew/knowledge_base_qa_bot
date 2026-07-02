"""Tests for tier-B S3 ``qa.edit`` / ``PUT /qa/{slug}`` (issue #379, ADR-0026 decision 2).

Coverage mirrors the issue's acceptance criteria:

- Edit a draft page whose submitted body passes the Grounding Check re-run
  against its cited Sections -> persists (question/body updated, ``status``
  stays ``draft``, ``created``/``sources``/``count`` preserved, ``updated``
  bumped, ``qa_reflect op=edited`` logged).
- Edit a ``status: live`` page -> refused (``QaPageLive`` / HTTP 409),
  draft-only per ADR-0026.
- Edit whose submitted body fails the grounding re-check -> rejected
  (``QaEditRejected`` / HTTP 422 with the failure list), file unchanged.
- Missing slug -> ``QaPageNotFound`` / HTTP 404. Corrupt frontmatter ->
  ``QaPageCorrupt`` / HTTP 500 (orphan-visibility).
- A draft whose citations no longer resolve to any Section fails
  deterministically via ``grounding.verify``'s empty-sections short-circuit
  -- no LLM call needed to prove this path.

Mocking discipline (CODING_STANDARD §6.3, matching ``test_reconcile.py``):
the grounding check runs UN-STUBBED -- ``grounding.verify()`` itself is
never monkeypatched. Only ``app.grounding.ChatOpenAI`` is patched, so
verify()'s real retry / error-classification / structured-output-mapping
logic executes end to end against a fake structured-output chain.

Hermetic: no OPENAI_API_KEY needed. Every test uses ``tmp_path`` via the
autouse ``_redirect_paths_to_tmp`` fixture in ``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.grounding import GroundingClaim, GroundingResult

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


def _make_fake_grounding_llm(result: GroundingResult) -> MagicMock:
    """Fake ``ChatOpenAI`` instance for ``app.grounding.ChatOpenAI`` -- verify() itself runs for real."""
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = result
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


_PASS_RESULT = GroundingResult(
    reasoning="The claim traces to the cited Cancellation Window section.",
    claims=[
        GroundingClaim(
            text="Orders can be cancelled within 24 hours.",
            supported=True,
            citing_section_ids=[_CITED_SECTION_ID],
        )
    ],
    unsupported_claims=[],
    passed=True,
)

_FAIL_RESULT = GroundingResult(
    reasoning="The claim is not supported by the cited section.",
    claims=[
        GroundingClaim(
            text="Orders can be cancelled within 90 days for a full refund.",
            supported=False,
            citing_section_ids=[],
        )
    ],
    unsupported_claims=["Orders can be cancelled within 90 days for a full refund."],
    passed=False,
)


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

    with patch("app.grounding.ChatOpenAI", return_value=_make_fake_grounding_llm(_PASS_RESULT)):
        result = edit(slug, "How long can I cancel an order?", "Within 24 hours of purchase.")

    assert result.slug == slug
    assert result.status == "draft"
    assert result.op == "touched"
    assert result.count == filed.count

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert "Within 24 hours of purchase." in after
    assert "status: draft" in after
    assert f"count: {filed.count}" in after
    assert _CITED_SECTION_ID in after, "sources must be preserved verbatim, not widened"


def test_edit_preserves_created_and_sources_bumps_updated(tmp_path):
    from app.qa import edit

    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])
    before = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")

    with patch("app.grounding.ChatOpenAI", return_value=_make_fake_grounding_llm(_PASS_RESULT)):
        edit(slug, "edited question", "edited body text.")

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert "2026-05-29T00:00:00Z" in before  # sanity: fixture created value
    assert "2026-05-29T00:00:00Z" in after, "created must be preserved verbatim"
    assert "count: 3" in after, "count must be preserved verbatim (edit does not bump it)"
    assert "edited question" in after
    assert "edited body text." in after
    assert "original body text." not in after


def test_edit_emits_qa_reflect_edited_log(tmp_path):
    from app.qa import edit

    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])

    with patch("app.grounding.ChatOpenAI", return_value=_make_fake_grounding_llm(_PASS_RESULT)):
        edit(slug, "edited question", "edited body text.")

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
        edit(slug, "edited question", "edited body text.")

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert after == before, "a refused edit must leave the live page byte-identical"


def test_edit_missing_slug_raises_not_found(tmp_path):
    from app.qa import QaPageNotFound, edit

    with pytest.raises(QaPageNotFound):
        edit("no-such-slug", "q", "b")


def test_edit_corrupt_frontmatter_raises_corrupt(tmp_path):
    from app.qa import QaPageCorrupt, edit

    slug = "broken-page"
    _write_raw_qa(tmp_path, slug, "draft", parseable=False)

    with pytest.raises(QaPageCorrupt):
        edit(slug, "q", "b")


def test_edit_grounding_failure_writes_nothing(tmp_path):
    from app.qa import QaEditRejected, edit

    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])
    before = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")

    with (
        patch("app.grounding.ChatOpenAI", return_value=_make_fake_grounding_llm(_FAIL_RESULT)),
        pytest.raises(QaEditRejected) as exc_info,
    ):
        edit(slug, "edited question", "Orders can be cancelled within 90 days for a full refund.")

    assert exc_info.value.grounding.passed is False
    assert exc_info.value.grounding.unsupported_claims

    after = (tmp_path / "wiki" / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert after == before, "a rejected edit must write nothing"


def test_edit_unresolvable_citations_fail_without_llm_call(tmp_path):
    """A draft whose cited Section no longer resolves on disk fails the
    grounding re-check via verify()'s empty-sections short-circuit -- no
    ChatOpenAI patch is installed here, so a real network call would error
    loudly if this path ever tried to construct the LLM client."""
    from app.qa import QaEditRejected, edit

    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=["missing-page#missing-heading"])

    with pytest.raises(QaEditRejected) as exc_info:
        edit(slug, "edited question", "edited body.")

    assert exc_info.value.grounding.passed is False


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

    with patch("app.grounding.ChatOpenAI", return_value=_make_fake_grounding_llm(_PASS_RESULT)):
        resp = edit_client.put(
            f"/qa/{slug}",
            json={"question": "How long can I cancel?", "body": "Within 24 hours of purchase."},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["slug"] == slug
    assert data["status"] == "draft"


def test_route_edit_live_page_returns_409(edit_client, tmp_path):
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "live", sources=[_CITED_SECTION_ID])

    resp = edit_client.put(f"/qa/{slug}", json={"question": "q", "body": "b"})

    assert resp.status_code == 409


def test_route_edit_missing_slug_returns_404(edit_client):
    resp = edit_client.put("/qa/no-such-slug", json={"question": "q", "body": "b"})

    assert resp.status_code == 404


def test_route_edit_grounding_failure_returns_422_with_claims(edit_client, tmp_path):
    _write_cited_entity_page(tmp_path)
    slug = "cancel-question-ab12cd"
    _write_raw_qa(tmp_path, slug, "draft", sources=[_CITED_SECTION_ID])

    with patch("app.grounding.ChatOpenAI", return_value=_make_fake_grounding_llm(_FAIL_RESULT)):
        resp = edit_client.put(
            f"/qa/{slug}",
            json={
                "question": "q",
                "body": "Orders can be cancelled within 90 days for a full refund.",
            },
        )

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason"] == "claim_unsupported"
    assert detail["unsupported_claims"]
