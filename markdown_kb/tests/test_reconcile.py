"""TestClient-seam tests for POST /pages/reconcile + POST /pages/reconcile/apply
(tier-B S1, issue #376, ADR-0028).

Mocking discipline (CODING_STANDARD §6.3):
- The DRAFTING LLM (``app.lint.get_lint_llm``) is stubbed via the SAME
  schema-aware-fake pattern ``test_lint_e2e.py`` / ``test_ingest_integration.py``
  already use — no real OpenAI call for the synthesis step.
- The grounding check runs UN-STUBBED: ``grounding.verify()`` itself is never
  monkeypatched. Only ``app.grounding.ChatOpenAI`` is patched (the
  ``test_verifier.py`` / ``test_retry.py`` pattern), so verify()'s real retry
  / error-classification / structured-output-mapping logic executes end to
  end against a fake structured-output chain.

Hermetic: no OPENAI_API_KEY needed. ``app.indexer.WIKI_DIR`` is redirected to
a tmp wiki/ pre-populated with two contradicting concept pages;
``app.reconcile.DOCS_DIR`` is redirected to the existing 3-Source hermetic
fixture under ``tests/fixtures/docs/`` (same fixture ``test_ingest_integration.py``
uses) so grounding has real Section content to check against.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer_module
import app.lint as lint_module
import app.reconcile as reconcile_module
from app.grounding import GroundingClaim, GroundingResult
from app.schemas import ReconcileDraft

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_DOCS_DIR = _TESTS_DIR / "fixtures" / "docs"

_SOURCE_CITATION = "refund_policy.md#cancellation-window"

_PAGE_A_BODY = (
    "# Cancellation Window A\n\n"
    "Orders can be cancelled within 24 hours of purchase.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)
_PAGE_B_BODY = (
    "# Cancellation Window B\n\n"
    "Orders can be cancelled within 48 hours of purchase.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)

_DRAFT_CONTENT_A = (
    "# Cancellation Window A\n\n"
    "Orders can be cancelled within 24 hours of purchase, per the Refund Policy.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)
_DRAFT_CONTENT_B = (
    "# Cancellation Window B\n\n"
    "Orders can be cancelled within 24 hours of purchase, matching the Refund Policy.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)


def _page_text(slug: str, updated: str, sources: list[str], body: str) -> str:
    sources_yaml = "\n".join(f"- {s}" for s in sources)
    return (
        "---\n"
        f"id: {slug}\n"
        "type: concept\n"
        "created: '2026-01-01T00:00:00Z'\n"
        f"updated: '{updated}'\n"
        f"sources:\n{sources_yaml}\n"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes: {}\n"
        "---\n\n"
        f"{body}"
    )


@pytest.fixture()
def reconcile_wiki_dir(tmp_path: Path) -> Path:
    """A tmp wiki/concepts/ with two pages that disagree about the cancellation window."""
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "concepts" / "cancellation-window-a.md").write_text(
        _page_text(
            "cancellation-window-a", "2026-01-01T00:00:00Z", [_SOURCE_CITATION], _PAGE_A_BODY
        ),
        encoding="utf-8",
    )
    (wiki_dir / "concepts" / "cancellation-window-b.md").write_text(
        _page_text(
            "cancellation-window-b", "2026-01-01T00:00:00Z", [_SOURCE_CITATION], _PAGE_B_BODY
        ),
        encoding="utf-8",
    )
    return wiki_dir


def _make_fake_lint_llm(
    content_a: str = _DRAFT_CONTENT_A, content_b: str = _DRAFT_CONTENT_B
) -> MagicMock:
    """Schema-aware fake for ``get_lint_llm().with_structured_output(ReconcileDraft)``."""
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = ReconcileDraft(content_a=content_a, content_b=content_b)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


def _make_fake_grounding_llm(result: GroundingResult) -> MagicMock:
    """Fake ``ChatOpenAI`` instance for ``app.grounding.ChatOpenAI`` — verify() itself runs for real."""
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = result
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


_PASS_RESULT = GroundingResult(
    reasoning="All claims trace to the cited Refund Policy section.",
    claims=[
        GroundingClaim(
            text="Orders can be cancelled within 24 hours of purchase.",
            supported=True,
            citing_section_ids=[_SOURCE_CITATION],
        )
    ],
    unsupported_claims=[],
    passed=True,
)

_FAIL_RESULT = GroundingResult(
    reasoning="The claim is not supported by the cited section.",
    claims=[
        GroundingClaim(
            text="Orders can be cancelled within 24 hours of purchase.",
            supported=False,
            citing_section_ids=[],
        )
    ],
    unsupported_claims=["Orders can be cancelled within 24 hours of purchase."],
    passed=False,
)


@pytest.fixture()
def reconcile_client(reconcile_wiki_dir, monkeypatch):
    """TestClient with the drafting LLM stubbed + wiki/docs dirs redirected.

    Grounding is NOT stubbed here — each test patches ``app.grounding.ChatOpenAI``
    itself for the outcome it needs (pass or fail).
    """
    fake_lint_llm = _make_fake_lint_llm()
    monkeypatch.setattr(lint_module, "get_lint_llm", lambda: fake_lint_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", reconcile_wiki_dir)
    # SOURCE_DIRS is pre-baked at module load from the real WIKI_DIR; without
    # realigning it, build_index() scans the committed wiki instead of the tmp
    # fixture and the reindex assertions below would not see the tmp pages.
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [
            reconcile_wiki_dir / "entities",
            reconcile_wiki_dir / "concepts",
            reconcile_wiki_dir / "qa",
        ],
    )
    monkeypatch.setattr(reconcile_module, "DOCS_DIR", _FIXTURE_DOCS_DIR)

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /pages/reconcile — generate (writes nothing to disk)
# ---------------------------------------------------------------------------


def test_generate_returns_draft_grounding_and_hashes(reconcile_client, reconcile_wiki_dir):
    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile",
            json={"page_a": "cancellation-window-a", "page_b": "cancellation-window-b"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["page_a"] == "cancellation-window-a"
    assert data["page_b"] == "cancellation-window-b"
    assert data["content_a"] == _DRAFT_CONTENT_A
    assert data["content_b"] == _DRAFT_CONTENT_B
    assert _PAGE_A_BODY in data["old_content_a"]
    assert _PAGE_B_BODY in data["old_content_b"]
    assert data["grounding"]["passed"] is True
    assert data["grounding"]["reason"] == "claim_supported"
    assert isinstance(data["hash_a"], str) and len(data["hash_a"]) == 64
    assert isinstance(data["hash_b"], str) and len(data["hash_b"]) == 64
    assert data["hash_a"] != data["hash_b"]


def test_generate_wiki_rooted_pair_carries_cited_sections_unchanged(
    reconcile_client, reconcile_wiki_dir
):
    """issue #534 regression guard: a wiki-rooted pair's existing generate
    behavior (asserted above) is unaffected — the new cited_sections_a/b
    fields are a strict ADDITION, populated from the SAME shared citation
    both fixture pages already carry."""
    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile",
            json={"page_a": "cancellation-window-a", "page_b": "cancellation-window-b"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["cited_sections_a"]) == 1
    assert len(data["cited_sections_b"]) == 1
    section_a = data["cited_sections_a"][0]
    section_b = data["cited_sections_b"][0]
    assert section_a["id"] == _SOURCE_CITATION
    assert section_b["id"] == _SOURCE_CITATION
    assert section_a["source_resolution"] == "resolved"
    assert section_a["source_path"] == "docs/refund_policy.md"
    assert section_a["content"]  # actual Section content, not just the id


def test_generate_writes_nothing_to_disk(reconcile_client, reconcile_wiki_dir):
    """ADR-0028 Invariant: POST /pages/reconcile writes nothing to disk."""
    path_a = reconcile_wiki_dir / "concepts" / "cancellation-window-a.md"
    path_b = reconcile_wiki_dir / "concepts" / "cancellation-window-b.md"
    before_a = path_a.read_text(encoding="utf-8")
    before_b = path_b.read_text(encoding="utf-8")

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile",
            json={"page_a": "cancellation-window-a", "page_b": "cancellation-window-b"},
        )

    assert resp.status_code == 200, resp.text
    assert path_a.read_text(encoding="utf-8") == before_a
    assert path_b.read_text(encoding="utf-8") == before_b


def test_generate_surfaces_failed_grounding_without_refusing(reconcile_client):
    """Generate always returns the draft + report, even when grounding fails
    (ADR-0028: only apply refuses; generate is diagnostic)."""
    fake_grounding_llm = _make_fake_grounding_llm(_FAIL_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile",
            json={"page_a": "cancellation-window-a", "page_b": "cancellation-window-b"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grounding"]["passed"] is False
    assert data["grounding"]["unsupported_claims"]


def test_generate_404_when_page_missing(reconcile_client):
    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile",
            json={"page_a": "cancellation-window-a", "page_b": "no-such-page"},
        )
    assert resp.status_code == 404


def test_generate_400_when_pages_identical(reconcile_client):
    resp = reconcile_client.post(
        "/pages/reconcile",
        json={"page_a": "cancellation-window-a", "page_b": "cancellation-window-a"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# C5 Source-comparison payload (issue #534, ADR-0036 decision 3)
#
# The fixture above gives both pages the SAME citation — sufficient to prove
# the field exists (test_generate_wiki_rooted_pair_carries_cited_sections_unchanged
# above), but not to prove cited_sections_a/b are each page's OWN citations
# rather than a shared/union view. This fixture cites two DIFFERENT Source
# files per page instead.
# ---------------------------------------------------------------------------

_PAGE_ALPHA_CITATION = "refund_policy.md#cancellation-window"
_PAGE_BETA_CITATION = "shipping_faq.md#standard-shipping"


@pytest.fixture()
def two_source_wiki_dir(tmp_path: Path) -> Path:
    """Two pages, each citing a DIFFERENT Source file."""
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "concepts" / "page-alpha.md").write_text(
        _page_text(
            "page-alpha",
            "2026-01-01T00:00:00Z",
            [_PAGE_ALPHA_CITATION],
            "# Page Alpha\n\nOrders can be cancelled within 24 hours.\n\n"
            f"[Source: {_PAGE_ALPHA_CITATION}]\n",
        ),
        encoding="utf-8",
    )
    (wiki_dir / "concepts" / "page-beta.md").write_text(
        _page_text(
            "page-beta",
            "2026-01-01T00:00:00Z",
            [_PAGE_BETA_CITATION],
            "# Page Beta\n\nStandard shipping takes 3-5 business days.\n\n"
            f"[Source: {_PAGE_BETA_CITATION}]\n",
        ),
        encoding="utf-8",
    )
    (wiki_dir / "concepts" / "page-gamma.md").write_text(
        _page_text(
            "page-gamma",
            "2026-01-01T00:00:00Z",
            ["no_such_source.md#some-anchor"],
            "# Page Gamma\n\nCites a Source that does not exist under docs_dir.\n\n"
            "[Source: no_such_source.md#some-anchor]\n",
        ),
        encoding="utf-8",
    )
    return wiki_dir


@pytest.fixture()
def two_source_client(two_source_wiki_dir, monkeypatch):
    """Mirrors ``reconcile_client`` but redirected at ``two_source_wiki_dir``.

    ``content_a``/``content_b`` are placeholder draft text — reused across
    this fixture's three tests regardless of which two of the three fixture
    pages a given test reconciles, since none of them assert on draft
    content (only on cited_sections_a/b and grounding.passed)."""
    fake_lint_llm = _make_fake_lint_llm(content_a="Draft A.", content_b="Draft B.")
    monkeypatch.setattr(lint_module, "get_lint_llm", lambda: fake_lint_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", two_source_wiki_dir)
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [
            two_source_wiki_dir / "entities",
            two_source_wiki_dir / "concepts",
            two_source_wiki_dir / "qa",
        ],
    )
    monkeypatch.setattr(reconcile_module, "DOCS_DIR", _FIXTURE_DOCS_DIR)

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_generate_cited_sections_are_per_page_not_union(two_source_client):
    """cited_sections_a carries ONLY page-alpha's citation, cited_sections_b
    ONLY page-beta's — proving this is per-page data, distinct from the
    (unchanged, whole-file) grounding union both pages are checked against."""
    fake_grounding_llm = _make_fake_grounding_llm(_FAIL_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = two_source_client.post(
            "/pages/reconcile", json={"page_a": "page-alpha", "page_b": "page-beta"}
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [s["id"] for s in data["cited_sections_a"]] == [_PAGE_ALPHA_CITATION]
    assert [s["id"] for s in data["cited_sections_b"]] == [_PAGE_BETA_CITATION]
    assert data["cited_sections_a"][0]["source_path"] == "docs/refund_policy.md"
    assert data["cited_sections_b"][0]["source_path"] == "docs/shipping_faq.md"
    assert "24 hours" in data["cited_sections_a"][0]["content"]
    assert "3-5 business days" in data["cited_sections_b"][0]["content"]


def test_generate_cited_sections_degrade_honestly_on_missing_source(two_source_client):
    """A citation naming a Source that does not exist under docs_dir still
    produces an entry (never silently dropped) with heading/content left
    unset and source_resolution="missing" — mirrors C3's honesty convention
    for an unresolvable citation."""
    fake_grounding_llm = _make_fake_grounding_llm(_FAIL_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = two_source_client.post(
            "/pages/reconcile", json={"page_a": "page-alpha", "page_b": "page-gamma"}
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    # page-alpha's own citation still resolves normally.
    assert data["cited_sections_a"][0]["source_resolution"] == "resolved"
    # page-gamma cites a Source with no matching file under docs_dir.
    gamma_section = data["cited_sections_b"][0]
    assert gamma_section["id"] == "no_such_source.md#some-anchor"
    assert gamma_section["source_resolution"] == "missing"
    assert gamma_section["source_path"] is None
    assert gamma_section["heading"] is None
    assert gamma_section["content"] is None


def test_union_collector_still_combines_both_pages_sources_whole_file():
    """ADR-0036 decision 7 (unaffected by issue #534): the grounding union
    ``_collect_union_sections`` feeds to ``verify()`` stays WHOLE-FILE across
    both pages' Sources — narrowing it to the new per-page
    cited_sections_a/b payload would hide a sibling contradicting section,
    exactly the regression ADR-0036 §7 rejects. A pure function-level check,
    independent of the HTTP round trip above."""
    fm_a = {"sources": [_PAGE_ALPHA_CITATION]}
    fm_b = {"sources": [_PAGE_BETA_CITATION]}

    sections = reconcile_module._collect_union_sections(fm_a, fm_b, _FIXTURE_DOCS_DIR)

    files_covered = {s.file for s in sections}
    assert {"refund_policy.md", "shipping_faq.md"} <= files_covered
    # Whole-file: every Section from refund_policy.md is present, not just
    # the one anchor page-alpha actually cites.
    refund_headings = {s.heading for s in sections if s.file == "refund_policy.md"}
    assert "Refund Timeline" in refund_headings, (
        f"union must include sibling sections beyond the cited anchor (whole-file, "
        f"ADR-0036 §7): got {refund_headings}"
    )


# ---------------------------------------------------------------------------
# POST /pages/reconcile/apply — apply (writes both pages, once, on pass)
# ---------------------------------------------------------------------------


def _generate(reconcile_client) -> dict:
    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile",
            json={"page_a": "cancellation-window-a", "page_b": "cancellation-window-b"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_apply_success_rewrites_both_pages_and_reindexes_exactly_once(
    reconcile_client, reconcile_wiki_dir
):
    draft = _generate(reconcile_client)

    import app.routes as routes_module

    real_build_index = routes_module.build_index
    spy = MagicMock(wraps=real_build_index)

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm),
        patch.object(routes_module, "build_index", spy),
    ):
        resp = reconcile_client.post(
            "/pages/reconcile/apply",
            json={
                "page_a": draft["page_a"],
                "page_b": draft["page_b"],
                "content_a": draft["content_a"],
                "content_b": draft["content_b"],
                "hash_a": draft["hash_a"],
                "hash_b": draft["hash_b"],
            },
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grounding"]["passed"] is True
    assert data["sections_indexed"] == 2, "reindex must cover exactly the two tmp fixture pages"
    assert spy.call_count == 1, "apply must trigger exactly one BM25 reindex"

    # AC: both slugs still retrievable after apply — the rebuilt index serves
    # the reconciled pages (BM25 over the tmp corpus, not the committed wiki).
    hit_pages = {
        section.id.split("#", 1)[0] for section, _ in indexer_module.search("cancelled", k=4)
    }
    assert {"cancellation-window-a", "cancellation-window-b"} <= hit_pages, (
        f"both reconciled slugs must be retrievable post-apply; got {hit_pages}"
    )

    # Both slugs still resolve — no deletion, both pages rewritten in place.
    path_a = reconcile_wiki_dir / "concepts" / "cancellation-window-a.md"
    path_b = reconcile_wiki_dir / "concepts" / "cancellation-window-b.md"
    assert path_a.exists()
    assert path_b.exists()
    text_a = path_a.read_text(encoding="utf-8")
    text_b = path_b.read_text(encoding="utf-8")
    assert _DRAFT_CONTENT_A.strip() in text_a
    assert _DRAFT_CONTENT_B.strip() in text_b
    assert "status: live" in text_a
    assert "status: live" in text_b
    assert "updated: '2026-01-01T00:00:00Z'" not in text_a, "updated timestamp must be bumped"
    assert "updated: '2026-01-01T00:00:00Z'" not in text_b, "updated timestamp must be bumped"


def test_apply_409_on_hash_mismatch_leaves_pages_untouched(reconcile_client, reconcile_wiki_dir):
    draft = _generate(reconcile_client)

    # Simulate a concurrent edit to page_a AFTER generate — its on-disk hash
    # no longer matches draft["hash_a"].
    path_a = reconcile_wiki_dir / "concepts" / "cancellation-window-a.md"
    mutated_a = path_a.read_text(encoding="utf-8").replace(
        "2026-01-01T00:00:00Z", "2026-02-02T00:00:00Z", 1
    )
    path_a.write_text(mutated_a, encoding="utf-8")
    before_b = (reconcile_wiki_dir / "concepts" / "cancellation-window-b.md").read_text(
        encoding="utf-8"
    )

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile/apply",
            json={
                "page_a": draft["page_a"],
                "page_b": draft["page_b"],
                "content_a": draft["content_a"],
                "content_b": draft["content_b"],
                "hash_a": draft["hash_a"],
                "hash_b": draft["hash_b"],
            },
        )

    assert resp.status_code == 409, resp.text
    # Neither page was rewritten — page_a keeps the mutated (not reconciled) content.
    assert path_a.read_text(encoding="utf-8") == mutated_a
    assert (reconcile_wiki_dir / "concepts" / "cancellation-window-b.md").read_text(
        encoding="utf-8"
    ) == before_b


def test_apply_422_on_grounding_failure_lists_unsupported_claims(
    reconcile_client, reconcile_wiki_dir
):
    draft = _generate(reconcile_client)
    path_a = reconcile_wiki_dir / "concepts" / "cancellation-window-a.md"
    path_b = reconcile_wiki_dir / "concepts" / "cancellation-window-b.md"
    before_a = path_a.read_text(encoding="utf-8")
    before_b = path_b.read_text(encoding="utf-8")

    fake_grounding_llm = _make_fake_grounding_llm(_FAIL_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile/apply",
            json={
                "page_a": draft["page_a"],
                "page_b": draft["page_b"],
                "content_a": draft["content_a"],
                "content_b": draft["content_b"],
                "hash_a": draft["hash_a"],
                "hash_b": draft["hash_b"],
            },
        )

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["unsupported_claims"], "422 detail must list the offending claims"
    # Neither page was rewritten.
    assert path_a.read_text(encoding="utf-8") == before_a
    assert path_b.read_text(encoding="utf-8") == before_b


def test_apply_404_when_page_missing(reconcile_client):
    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = reconcile_client.post(
            "/pages/reconcile/apply",
            json={
                "page_a": "cancellation-window-a",
                "page_b": "no-such-page",
                "content_a": _DRAFT_CONTENT_A,
                "content_b": _DRAFT_CONTENT_B,
                "hash_a": "x" * 64,
                "hash_b": "y" * 64,
            },
        )
    assert resp.status_code == 404
