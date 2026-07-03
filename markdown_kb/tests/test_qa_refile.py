"""Tests for tier-B S4 ``qa.refile`` / ``POST /qa/{slug}/refile`` (issue #380,
ADR-0026 decision 1).

Coverage mirrors the issue's acceptance criteria:

- Failed re-ground writes nothing: no demote, no draft, no reindex; the old
  live page stays byte-identical and the response is honest (422, no LLM
  needed here — the pre-LLM below_threshold gate is cheaper and exercises
  the same "nothing written" invariant).
- Successful refile: same slug, ``status: draft``, updated timestamp bumped,
  fresh cited Sources, one reindex (route-level), old answer out of the
  BM25 corpus.
- Retrieval for the refile call provably excludes qa pages: the stale qa
  page's own body is written densely enough that BM25 would otherwise rank
  it above the entity page for its own question (empirically proven in
  ``test_indexer_exclude_qa.py``); the refiled page's fresh ``sources``
  cite only the entity, never the qa page itself.
- Missing slug -> ``QaPageNotFound`` / HTTP 404. Corrupt frontmatter /
  missing question -> ``QaPageCorrupt`` / HTTP 500 (orphan-visibility).

Hermetic: FakeLLM + ``grounding_module.verify`` stub for the success path
(mirrors ``test_chat_grounded.py``); the rejection tests use the pre-LLM
below_threshold gate so no LLM client is ever constructed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer_module
import app.retrieval as retrieval_module
from app.grounding import GroundingOutcome

from .conftest import FakeLLMResponse

# ---------------------------------------------------------------------------
# Fixtures — an entity page (current truth) + a stale live qa page whose own
# body is dense enough in the question's terms to out-rank the entity absent
# exclude_qa=True (mirrors test_indexer_exclude_qa.py's proven fixture shape).
# ---------------------------------------------------------------------------

ENTITY_SECTION_ID = "acme-shop#acme-shop"
# Same word forms as test_indexer_exclude_qa.py's proven fixture (no stemmer
# in this tokenizer — "cancel"/"cancelled" and "order"/"orders" do NOT match).
ENTITY_BODY = "Orders may be cancelled within the cancellation window for a full refund."

STALE_QA_BODY = (
    "You can cancel your order within the cancellation window. To cancel "
    "your order within the cancellation window, contact support before the "
    "cancellation window closes. Cancel order cancellation window."
)

QUESTION = "how do i cancel my order within the cancellation window"

FRESH_ANSWER = f"You may cancel within the cancellation window. [Source: {ENTITY_SECTION_ID}]"


class FakeLLM:
    """Minimal LLM stub returning a canned, entity-grounded answer.

    Records the prompt it received so a test can assert the stale qa page's
    own content never reached the LLM context (exclude_qa proof, belt and
    suspenders on top of the BM25-level fixture check).
    """

    def __init__(self, answer: str = FRESH_ANSWER) -> None:
        self.answer = answer
        self.last_messages: list = []

    def invoke(self, messages: list):
        self.last_messages = messages
        return FakeLLMResponse(content=self.answer)


def _write_entity(wiki_dir: Path) -> None:
    entities_dir = wiki_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    (entities_dir / "acme-shop.md").write_text(
        "---\n"
        "id: acme-shop\n"
        "type: entity\n"
        "created: '2026-01-01T00:00:00Z'\n"
        "updated: '2026-07-01T00:00:00Z'\n"
        "sources: []\n"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes: {}\n"
        "---\n\n"
        "# Acme Shop\n\n"
        f"{ENTITY_BODY}\n",
        encoding="utf-8",
    )


def _write_qa(
    wiki_dir: Path,
    slug: str,
    status: str = "live",
    question: str | None = QUESTION,
    parseable: bool = True,
) -> Path:
    qa_dir = wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{slug}.md"
    if not parseable:
        path.write_text("not valid yaml frontmatter at all\njust raw text\n", encoding="utf-8")
        return path
    question_line = f'question: "{question}"\n' if question else ""
    path.write_text(
        "---\n"
        f"id: {slug}\n"
        "type: qa\n"
        'created: "2026-05-01T00:00:00Z"\n'
        'updated: "2026-05-01T00:00:00Z"\n'
        "sources:\n"
        f"  - {slug}\n"  # deliberately cites itself (bare-filename Section id)
        f"status: {status}\n"
        "open_questions: []\n"
        f"{question_line}"
        "count: 4\n"
        "---\n\n"
        f"{STALE_QA_BODY}\n",
        encoding="utf-8",
    )
    return path


def _patch_source_dirs(monkeypatch, wiki_dir: Path) -> None:
    """Point indexer.SOURCE_DIRS at the tmp wiki tree (mirrors
    test_indexer_qa_filter.py's ``_patch_indexer`` helper)."""
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts", wiki_dir / "qa"],
    )


def _build_corpus(tmp_path, monkeypatch, qa_status: str = "live") -> Path:
    """Write entity + stale qa fixtures, patch SOURCE_DIRS, and build the
    in-memory BM25 index. Returns the wiki_dir."""
    wiki_dir = tmp_path / "wiki"
    _write_entity(wiki_dir)
    _write_qa(wiki_dir, "cancel-order-abc123", status=qa_status)
    _patch_source_dirs(monkeypatch, wiki_dir)
    indexer_module.build_index()
    return wiki_dir


def _stub_passing_llm(monkeypatch) -> FakeLLM:
    fake_llm = FakeLLM()
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: GroundingOutcome(
            passed=True, reason="claim_supported", result=None
        ),
    )
    return fake_llm


# ---------------------------------------------------------------------------
# Direct-module tests: qa.refile()
# ---------------------------------------------------------------------------


def test_refile_missing_slug_raises_not_found(tmp_path, monkeypatch):
    from app.qa import QaPageNotFound, refile

    _patch_source_dirs(monkeypatch, tmp_path / "wiki")

    with pytest.raises(QaPageNotFound):
        refile("no-such-slug")


def test_refile_corrupt_frontmatter_raises_corrupt(tmp_path, monkeypatch):
    from app.qa import QaPageCorrupt, refile

    wiki_dir = tmp_path / "wiki"
    _write_qa(wiki_dir, "broken-page", parseable=False)
    _patch_source_dirs(monkeypatch, wiki_dir)

    with pytest.raises(QaPageCorrupt):
        refile("broken-page")


def test_refile_missing_question_raises_corrupt(tmp_path, monkeypatch):
    from app.qa import QaPageCorrupt, refile

    wiki_dir = tmp_path / "wiki"
    _write_qa(wiki_dir, "no-question", question=None)
    _patch_source_dirs(monkeypatch, wiki_dir)

    with pytest.raises(QaPageCorrupt):
        refile("no-question")


def test_refile_failed_reground_writes_nothing(tmp_path, monkeypatch):
    """Invariant (ADR-0026 § Consequences): a failed re-ground writes nothing.

    Forces the pre-LLM below_threshold gate (no LLM client ever constructed)
    so the rejection path is exercised as cheaply and deterministically as
    possible."""
    from app.qa import QaRefileRejected, refile

    wiki_dir = _build_corpus(tmp_path, monkeypatch)
    slug = "cancel-order-abc123"
    before = (wiki_dir / "qa" / f"{slug}.md").read_text(encoding="utf-8")

    # Force below_threshold: no score can clear an effectively-infinite bar.
    monkeypatch.setattr(retrieval_module, "_SCORE_THRESHOLD", 10_000.0)

    with pytest.raises(QaRefileRejected) as exc_info:
        refile(slug)

    assert exc_info.value.grounding.passed is False
    assert exc_info.value.grounding.reason == "below_threshold"

    after = (wiki_dir / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert after == before, "a rejected refile must write nothing — old live page byte-identical"

    log = (wiki_dir / "log.md").read_text(encoding="utf-8")
    assert "qa_refile_rejected" in log
    assert f"slug={slug}" in log
    assert "op=refiled" not in log, "no qa_reflect op=refiled entry on a rejected refile"


def test_refile_success_demotes_in_place_with_fresh_sources(tmp_path, monkeypatch):
    from app.qa import refile

    wiki_dir = _build_corpus(tmp_path, monkeypatch)
    slug = "cancel-order-abc123"
    fake_llm = _stub_passing_llm(monkeypatch)

    result = refile(slug)

    assert result.filed.slug == slug
    assert result.filed.status == "draft"
    assert result.filed.count == 4, "count must be preserved verbatim"
    assert result.grounding.passed is True

    after = (wiki_dir / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert "status: draft" in after
    assert FRESH_ANSWER.split(" [Source")[0] in after, "fresh answer body must be written"
    assert STALE_QA_BODY not in after, "stale body must be fully replaced, not appended"
    assert "2026-05-01T00:00:00Z" in after, "created must be preserved verbatim"
    assert ENTITY_SECTION_ID in after, "fresh sources must cite the entity"
    assert f"- {slug}\n" not in after, "refile must never re-cite the stale qa page itself"

    # Exclude-qa proof at the retrieval boundary: the stale qa page's dense
    # body text must never have reached the LLM prompt.
    assert fake_llm.last_messages, "FakeLLM must have been invoked"
    prompt_text = fake_llm.last_messages[1].content
    assert "contact support before the cancellation window closes" not in prompt_text, (
        "the stale qa page's own body must be excluded from the re-synthesis "
        "prompt (ADR-0026 decision 1 step 1)"
    )


def test_refile_emits_qa_reflect_refiled_log(tmp_path, monkeypatch):
    from app.qa import refile

    wiki_dir = _build_corpus(tmp_path, monkeypatch)
    slug = "cancel-order-abc123"
    _stub_passing_llm(monkeypatch)

    refile(slug)

    log = (wiki_dir / "log.md").read_text(encoding="utf-8")
    refiled_lines = [ln for ln in log.splitlines() if "qa_reflect" in ln and "op=refiled" in ln]
    assert len(refiled_lines) == 1, (
        f"Expected exactly one op=refiled reflect entry, got: {refiled_lines}"
    )
    assert f"slug={slug}" in refiled_lines[0]


# ---------------------------------------------------------------------------
# Route-level tests: POST /qa/{slug}/refile
# ---------------------------------------------------------------------------


@pytest.fixture()
def refile_client():
    """TestClient — WIKI_DIR/INDEX_PATH/LOG_PATH are already redirected to
    tmp_path by the autouse ``_redirect_paths_to_tmp`` fixture in conftest."""
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_route_refile_missing_slug_returns_404(refile_client, tmp_path, monkeypatch):
    _patch_source_dirs(monkeypatch, tmp_path / "wiki")

    resp = refile_client.post("/qa/no-such-slug/refile")

    assert resp.status_code == 404


def test_route_refile_grounding_failure_returns_422_and_writes_nothing(
    refile_client, tmp_path, monkeypatch
):
    wiki_dir = _build_corpus(tmp_path, monkeypatch)
    slug = "cancel-order-abc123"
    before = (wiki_dir / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    monkeypatch.setattr(retrieval_module, "_SCORE_THRESHOLD", 10_000.0)

    resp = refile_client.post(f"/qa/{slug}/refile")

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason"] == "below_threshold"

    after = (wiki_dir / "qa" / f"{slug}.md").read_text(encoding="utf-8")
    assert after == before


def test_route_refile_success_returns_200_demotes_and_reindexes(
    refile_client, tmp_path, monkeypatch
):
    _build_corpus(tmp_path, monkeypatch)
    slug = "cancel-order-abc123"
    _stub_passing_llm(monkeypatch)

    # Sanity: before refile, the stale qa page is part of the live BM25 corpus.
    assert any(s.file == slug for s in indexer_module.sections)

    resp = refile_client.post(f"/qa/{slug}/refile")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["filed"]["slug"] == slug
    assert data["filed"]["status"] == "draft"
    assert data["grounding"]["passed"] is True
    assert data["sections_indexed"] >= 1

    # The route's own build_index() call must have removed the now-draft qa
    # page from the live BM25 corpus (ADR-0026: "the stale answer leaves the
    # corpus").
    assert not any(s.file == slug for s in indexer_module.sections), (
        "the demoted (now-draft) qa page must no longer be in the BM25 corpus "
        "after the route's reindex"
    )
