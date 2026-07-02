"""TestClient-seam tests for the C4 dual-resolution endpoints:
POST /pages/collision/merge(+/apply) and
POST /pages/collision/differentiate(+/apply) (tier-B S2, issue #378, ADR-0028).

Mocking discipline (CODING_STANDARD §6.3) — mirrors ``test_reconcile.py``:
- The DRAFTING LLM (``app.lint.get_lint_llm``) is stubbed via a schema-aware
  fake: ``with_structured_output`` dispatches on the requested schema class
  (``CollisionMergeDraft`` vs ``CollisionDifferentiateDraft``) so one fake
  singleton serves both call sites.
- The grounding check runs UN-STUBBED: ``grounding.verify()`` itself is never
  monkeypatched. Only ``app.grounding.ChatOpenAI`` is patched, so verify()'s
  real retry / error-classification / structured-output-mapping logic
  executes end to end against a fake structured-output chain.

Hermetic: no OPENAI_API_KEY needed. ``app.indexer.WIKI_DIR`` is redirected to
a tmp wiki/ pre-populated with a merge group (widget/widget-2) and a
differentiate group (gizmo/gizmo-2/gizmo-3); ``app.reconcile.DOCS_DIR`` is
redirected to the existing 3-Source hermetic fixture under
``tests/fixtures/docs/`` (same fixture ``test_reconcile.py`` uses).
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
from app.schemas import CollisionDifferentiateDraft, CollisionMergeDraft, CollisionPageDraft

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_DOCS_DIR = _TESTS_DIR / "fixtures" / "docs"

_SOURCE_CITATION = "refund_policy.md#cancellation-window"

_WIDGET_BASE_BODY = f"# Widget\n\nA widget costs $10.\n\n[Source: {_SOURCE_CITATION}]\n"
_WIDGET_VARIANT_BODY = (
    "# Widget (variant)\n\nA widget can be cancelled within 24 hours.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)
_MERGE_DRAFT_CONTENT = (
    "# Widget\n\nA widget costs $10 and can be cancelled within 24 hours of purchase.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)

_GIZMO_BODY = f"# Gizmo\n\nGizmos are refundable within 24 hours.\n\n[Source: {_SOURCE_CITATION}]\n"
_GIZMO2_BODY = (
    "# Gizmo (variant 2)\n\nGizmos ship within a day of purchase.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)
_GIZMO3_BODY = (
    "# Gizmo (variant 3)\n\nGizmos become non-refundable after 24 hours.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)
_GIZMO_DRAFT = (
    "# Gizmo\n\nGizmos are refundable within 24 hours of purchase (differentiated).\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)
_GIZMO2_DRAFT = (
    "# Gizmo (variant 2)\n\nGizmos ship within a day, separate from the refund window.\n\n"
    f"[Source: {_SOURCE_CITATION}]\n"
)
_GIZMO3_DRAFT = (
    "# Gizmo (variant 3)\n\nAfter 24 hours a Gizmo purchase becomes final.\n\n"
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


def _write_page(wiki_dir: Path, slug: str, body: str) -> None:
    (wiki_dir / "concepts" / f"{slug}.md").write_text(
        _page_text(slug, "2026-01-01T00:00:00Z", [_SOURCE_CITATION], body),
        encoding="utf-8",
    )


@pytest.fixture()
def collision_wiki_dir(tmp_path: Path) -> Path:
    """A tmp wiki/concepts/ pre-populated with a merge group
    (widget / widget-2, no inbound references) and a differentiate group
    (gizmo / gizmo-2 / gizmo-3)."""
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    _write_page(wiki_dir, "widget", _WIDGET_BASE_BODY)
    _write_page(wiki_dir, "widget-2", _WIDGET_VARIANT_BODY)
    _write_page(wiki_dir, "gizmo", _GIZMO_BODY)
    _write_page(wiki_dir, "gizmo-2", _GIZMO2_BODY)
    _write_page(wiki_dir, "gizmo-3", _GIZMO3_BODY)
    return wiki_dir


def _make_fake_lint_llm(
    merge_content_base: str = _MERGE_DRAFT_CONTENT,
    differentiate_pages: list[CollisionPageDraft] | None = None,
) -> MagicMock:
    """Schema-aware fake for ``get_lint_llm().with_structured_output(...)`` —
    dispatches on the requested Pydantic schema so ONE fake singleton serves
    both the merge and differentiate drafting call sites."""
    if differentiate_pages is None:
        differentiate_pages = [
            CollisionPageDraft(slug="gizmo", content=_GIZMO_DRAFT),
            CollisionPageDraft(slug="gizmo-2", content=_GIZMO2_DRAFT),
            CollisionPageDraft(slug="gizmo-3", content=_GIZMO3_DRAFT),
        ]

    merge_chain = MagicMock()
    merge_chain.invoke.return_value = CollisionMergeDraft(content_base=merge_content_base)

    differentiate_chain = MagicMock()
    differentiate_chain.invoke.return_value = CollisionDifferentiateDraft(pages=differentiate_pages)

    def _with_structured_output(schema):
        if schema is CollisionMergeDraft:
            return merge_chain
        if schema is CollisionDifferentiateDraft:
            return differentiate_chain
        raise AssertionError(f"unexpected structured-output schema requested: {schema}")

    fake_llm = MagicMock()
    fake_llm.with_structured_output.side_effect = _with_structured_output
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
            text="A widget can be cancelled within 24 hours of purchase.",
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
            text="A widget can be cancelled within 24 hours of purchase.",
            supported=False,
            citing_section_ids=[],
        )
    ],
    unsupported_claims=["A widget can be cancelled within 24 hours of purchase."],
    passed=False,
)


@pytest.fixture()
def collision_client(collision_wiki_dir, monkeypatch):
    """TestClient with the drafting LLM stubbed + wiki/docs dirs redirected.

    Grounding is NOT stubbed here — each test patches ``app.grounding.ChatOpenAI``
    itself for the outcome it needs (pass or fail).
    """
    fake_lint_llm = _make_fake_lint_llm()
    monkeypatch.setattr(lint_module, "get_lint_llm", lambda: fake_lint_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", collision_wiki_dir)
    # SOURCE_DIRS is pre-baked at module load from the real WIKI_DIR; without
    # realigning it, build_index() scans the committed wiki instead of the tmp
    # fixture and the reindex assertions below would not see the tmp pages.
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [
            collision_wiki_dir / "entities",
            collision_wiki_dir / "concepts",
            collision_wiki_dir / "qa",
        ],
    )
    monkeypatch.setattr(reconcile_module, "DOCS_DIR", _FIXTURE_DOCS_DIR)

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def _generate_merge(collision_client) -> dict:
    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/merge",
            json={"base_slug": "widget", "variant_slugs": ["widget-2"]},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _generate_differentiate(collision_client) -> dict:
    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/differentiate",
            json={"slugs": ["gizmo", "gizmo-2", "gizmo-3"]},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# POST /pages/collision/merge — generate (writes nothing to disk)
# ---------------------------------------------------------------------------


def test_merge_generate_returns_draft_grounding_and_hashes(collision_client, collision_wiki_dir):
    data = _generate_merge(collision_client)

    assert data["base_slug"] == "widget"
    assert data["variant_slugs"] == ["widget-2"]
    assert data["content_base"] == _MERGE_DRAFT_CONTENT
    assert _WIDGET_BASE_BODY in data["old_content_base"]
    assert data["grounding"]["passed"] is True
    assert isinstance(data["hash_base"], str) and len(data["hash_base"]) == 64
    assert set(data["hash_variants"]) == {"widget-2"}
    assert len(data["hash_variants"]["widget-2"]) == 64


def test_merge_generate_writes_nothing_to_disk(collision_client, collision_wiki_dir):
    """ADR-0028 Invariant: POST /pages/collision/merge writes nothing to disk."""
    base_path = collision_wiki_dir / "concepts" / "widget.md"
    variant_path = collision_wiki_dir / "concepts" / "widget-2.md"
    before_base = base_path.read_text(encoding="utf-8")
    before_variant = variant_path.read_text(encoding="utf-8")

    _generate_merge(collision_client)

    assert base_path.read_text(encoding="utf-8") == before_base
    assert variant_path.exists()
    assert variant_path.read_text(encoding="utf-8") == before_variant


def test_merge_generate_400_when_base_in_variants(collision_client):
    resp = collision_client.post(
        "/pages/collision/merge",
        json={"base_slug": "widget", "variant_slugs": ["widget"]},
    )
    assert resp.status_code == 400


def test_merge_generate_404_when_base_missing(collision_client):
    resp = collision_client.post(
        "/pages/collision/merge",
        json={"base_slug": "no-such-page", "variant_slugs": ["widget-2"]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /pages/collision/merge/apply — guard pass (reference-free)
# ---------------------------------------------------------------------------


def test_merge_apply_success_deletes_variant_and_reindexes_exactly_once(
    collision_client, collision_wiki_dir
):
    draft = _generate_merge(collision_client)

    import app.routes as routes_module

    real_build_index = routes_module.build_index
    spy = MagicMock(wraps=real_build_index)

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm),
        patch.object(routes_module, "build_index", spy),
    ):
        resp = collision_client.post(
            "/pages/collision/merge/apply",
            json={
                "base_slug": draft["base_slug"],
                "variant_slugs": draft["variant_slugs"],
                "content_base": draft["content_base"],
                "hash_base": draft["hash_base"],
                "hash_variants": draft["hash_variants"],
            },
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grounding"]["passed"] is True
    assert data["deleted_variants"] == ["widget-2"]
    assert spy.call_count == 1, "apply must trigger exactly one BM25 reindex"

    base_path = collision_wiki_dir / "concepts" / "widget.md"
    variant_path = collision_wiki_dir / "concepts" / "widget-2.md"
    assert base_path.exists()
    assert not variant_path.exists(), "reference-free variant must be deleted"
    text_base = base_path.read_text(encoding="utf-8")
    assert _MERGE_DRAFT_CONTENT.strip() in text_base
    assert "status: live" in text_base

    # AC: base slug still retrievable post-apply.
    hit_pages = {section.id.split("#", 1)[0] for section, _ in indexer_module.search("widget", k=4)}
    assert "widget" in hit_pages


# ---------------------------------------------------------------------------
# POST /pages/collision/merge/apply — inbound-reference guard refusal
# ---------------------------------------------------------------------------


def test_merge_apply_409_guard_refusal_lists_wiki_referrer(collision_client, collision_wiki_dir):
    draft = _generate_merge(collision_client)

    # A third page links to the variant AFTER generate — the guard must
    # refuse and list it as a referrer.
    (collision_wiki_dir / "concepts" / "other.md").write_text(
        _page_text(
            "other",
            "2026-01-01T00:00:00Z",
            [_SOURCE_CITATION],
            f"# Other\n\nSee also [[widget-2]].\n\n[Source: {_SOURCE_CITATION}]\n",
        ),
        encoding="utf-8",
    )

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/merge/apply",
            json={
                "base_slug": draft["base_slug"],
                "variant_slugs": draft["variant_slugs"],
                "content_base": draft["content_base"],
                "hash_base": draft["hash_base"],
                "hash_variants": draft["hash_variants"],
            },
        )

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict), "guard refusal must be a structured dict, not a plain string"
    referrers = detail["referrers"]
    assert len(referrers) == 1
    assert referrers[0]["variant_slug"] == "widget-2"
    assert referrers[0]["wiki_referrers"] == ["other"]
    assert referrers[0]["qa_referrers"] == []

    # Nothing was written or deleted.
    assert (collision_wiki_dir / "concepts" / "widget-2.md").exists()
    assert (collision_wiki_dir / "concepts" / "widget.md").read_text(encoding="utf-8") == (
        _page_text("widget", "2026-01-01T00:00:00Z", [_SOURCE_CITATION], _WIDGET_BASE_BODY)
    )


def test_merge_apply_409_guard_refusal_lists_qa_referrer(collision_client, collision_wiki_dir):
    draft = _generate_merge(collision_client)

    qa_dir = collision_wiki_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "widget-question.md").write_text(
        "<!-- Auto-filed by POST /chat. -->\n"
        "\n"
        "---\n"
        "count: 1\n"
        "created: '2026-01-01T00:00:00Z'\n"
        "id: widget-question\n"
        "open_questions: []\n"
        "question: How long can I cancel a widget?\n"
        "sources:\n"
        "- widget-2#widget-variant\n"
        "status: live\n"
        "type: qa\n"
        "updated: '2026-01-01T00:00:00Z'\n"
        "---\n"
        "\n"
        "Body text.\n",
        encoding="utf-8",
    )

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/merge/apply",
            json={
                "base_slug": draft["base_slug"],
                "variant_slugs": draft["variant_slugs"],
                "content_base": draft["content_base"],
                "hash_base": draft["hash_base"],
                "hash_variants": draft["hash_variants"],
            },
        )

    assert resp.status_code == 409, resp.text
    referrers = resp.json()["detail"]["referrers"]
    assert len(referrers) == 1
    assert referrers[0]["variant_slug"] == "widget-2"
    assert referrers[0]["wiki_referrers"] == []
    assert referrers[0]["qa_referrers"] == ["widget-question"]
    assert (collision_wiki_dir / "concepts" / "widget-2.md").exists()


# ---------------------------------------------------------------------------
# POST /pages/collision/merge/apply — hash mismatch / grounding failure
# ---------------------------------------------------------------------------


def test_merge_apply_409_on_hash_mismatch_leaves_pages_untouched(
    collision_client, collision_wiki_dir
):
    draft = _generate_merge(collision_client)

    base_path = collision_wiki_dir / "concepts" / "widget.md"
    mutated = base_path.read_text(encoding="utf-8").replace(
        "2026-01-01T00:00:00Z", "2026-02-02T00:00:00Z", 1
    )
    base_path.write_text(mutated, encoding="utf-8")

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/merge/apply",
            json={
                "base_slug": draft["base_slug"],
                "variant_slugs": draft["variant_slugs"],
                "content_base": draft["content_base"],
                "hash_base": draft["hash_base"],
                "hash_variants": draft["hash_variants"],
            },
        )

    assert resp.status_code == 409, resp.text
    assert isinstance(resp.json()["detail"], str), "hash mismatch stays a plain-string detail"
    assert base_path.read_text(encoding="utf-8") == mutated
    assert (collision_wiki_dir / "concepts" / "widget-2.md").exists()


def test_merge_apply_422_on_grounding_failure_lists_unsupported_claims(
    collision_client, collision_wiki_dir
):
    draft = _generate_merge(collision_client)
    base_path = collision_wiki_dir / "concepts" / "widget.md"
    before = base_path.read_text(encoding="utf-8")

    fake_grounding_llm = _make_fake_grounding_llm(_FAIL_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/merge/apply",
            json={
                "base_slug": draft["base_slug"],
                "variant_slugs": draft["variant_slugs"],
                "content_base": draft["content_base"],
                "hash_base": draft["hash_base"],
                "hash_variants": draft["hash_variants"],
            },
        )

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["unsupported_claims"]
    assert base_path.read_text(encoding="utf-8") == before
    assert (collision_wiki_dir / "concepts" / "widget-2.md").exists()


# ---------------------------------------------------------------------------
# POST /pages/collision/differentiate — generate (writes nothing to disk)
# ---------------------------------------------------------------------------


def test_differentiate_generate_returns_drafts_grounding_and_hashes(collision_client):
    data = _generate_differentiate(collision_client)

    assert data["slugs"] == ["gizmo", "gizmo-2", "gizmo-3"]
    assert data["content"]["gizmo"] == _GIZMO_DRAFT
    assert data["content"]["gizmo-2"] == _GIZMO2_DRAFT
    assert data["content"]["gizmo-3"] == _GIZMO3_DRAFT
    assert _GIZMO_BODY in data["old_content"]["gizmo"]
    assert data["grounding"]["passed"] is True
    assert set(data["hashes"]) == {"gizmo", "gizmo-2", "gizmo-3"}
    assert all(len(h) == 64 for h in data["hashes"].values())


def test_differentiate_generate_400_when_fewer_than_two_slugs(collision_client):
    resp = collision_client.post("/pages/collision/differentiate", json={"slugs": ["gizmo"]})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /pages/collision/differentiate/apply — nobody dies, all rewritten
# ---------------------------------------------------------------------------


def test_differentiate_apply_success_rewrites_all_pages_and_reindexes_exactly_once(
    collision_client, collision_wiki_dir
):
    draft = _generate_differentiate(collision_client)

    import app.routes as routes_module

    real_build_index = routes_module.build_index
    spy = MagicMock(wraps=real_build_index)

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm),
        patch.object(routes_module, "build_index", spy),
    ):
        resp = collision_client.post(
            "/pages/collision/differentiate/apply",
            json={
                "slugs": draft["slugs"],
                "content": draft["content"],
                "hashes": draft["hashes"],
            },
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grounding"]["passed"] is True
    assert spy.call_count == 1, "apply must trigger exactly one BM25 reindex"

    for slug, expected in (
        ("gizmo", _GIZMO_DRAFT),
        ("gizmo-2", _GIZMO2_DRAFT),
        ("gizmo-3", _GIZMO3_DRAFT),
    ):
        path = collision_wiki_dir / "concepts" / f"{slug}.md"
        assert path.exists(), f"{slug} must survive differentiate — nobody dies"
        text = path.read_text(encoding="utf-8")
        assert expected.strip() in text
        assert "status: live" in text
        assert "updated: '2026-01-01T00:00:00Z'" not in text, "updated timestamp must be bumped"

    hit_pages = {section.id.split("#", 1)[0] for section, _ in indexer_module.search("gizmo", k=6)}
    assert {"gizmo", "gizmo-2", "gizmo-3"} <= hit_pages, (
        f"every differentiated slug must be retrievable post-apply; got {hit_pages}"
    )

    # Issue #378 AC "apply → re-lint clears the finding": the sentinel stamped
    # on every member exempts the group from C4-a on the next lint run.
    from app.lint import _check_c4a_slug_collision

    bases = {f.base_slug for f in _check_c4a_slug_collision(collision_wiki_dir)}
    assert "gizmo" not in bases, "re-lint must clear the C4 finding after differentiate apply"
    assert "widget" in bases, "the untouched merge group must still fire"


def test_differentiate_apply_409_on_hash_mismatch_leaves_pages_untouched(
    collision_client, collision_wiki_dir
):
    draft = _generate_differentiate(collision_client)

    gizmo2_path = collision_wiki_dir / "concepts" / "gizmo-2.md"
    mutated = gizmo2_path.read_text(encoding="utf-8").replace(
        "2026-01-01T00:00:00Z", "2026-02-02T00:00:00Z", 1
    )
    gizmo2_path.write_text(mutated, encoding="utf-8")
    before_gizmo3 = (collision_wiki_dir / "concepts" / "gizmo-3.md").read_text(encoding="utf-8")

    fake_grounding_llm = _make_fake_grounding_llm(_PASS_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/differentiate/apply",
            json={"slugs": draft["slugs"], "content": draft["content"], "hashes": draft["hashes"]},
        )

    assert resp.status_code == 409, resp.text
    assert gizmo2_path.read_text(encoding="utf-8") == mutated
    assert (collision_wiki_dir / "concepts" / "gizmo-3.md").read_text(
        encoding="utf-8"
    ) == before_gizmo3


def test_differentiate_apply_422_on_grounding_failure(collision_client, collision_wiki_dir):
    draft = _generate_differentiate(collision_client)
    before = {
        slug: (collision_wiki_dir / "concepts" / f"{slug}.md").read_text(encoding="utf-8")
        for slug in draft["slugs"]
    }

    fake_grounding_llm = _make_fake_grounding_llm(_FAIL_RESULT)
    with patch("app.grounding.ChatOpenAI", return_value=fake_grounding_llm):
        resp = collision_client.post(
            "/pages/collision/differentiate/apply",
            json={"slugs": draft["slugs"], "content": draft["content"], "hashes": draft["hashes"]},
        )

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["unsupported_claims"]
    for slug, text in before.items():
        assert (collision_wiki_dir / "concepts" / f"{slug}.md").read_text(encoding="utf-8") == text


# ---------------------------------------------------------------------------
# C4-a differentiate exemption (check seam) — issue #378 "re-lint clears"
# ---------------------------------------------------------------------------


def _stamp_differentiated(wiki_dir: Path, slug: str, group: list[str]) -> None:
    """Prepend the differentiate sentinel onto an existing fixture page,
    mirroring the byte-shape ``reconcile._write_differentiated_page`` writes."""
    from app.lint import DIFFERENTIATE_SENTINEL_TEMPLATE

    path = wiki_dir / "concepts" / f"{slug}.md"
    sentinel = DIFFERENTIATE_SENTINEL_TEMPLATE.format(
        ts="2026-07-03T00:00:00Z", group=", ".join(group)
    )
    path.write_text(sentinel + "\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8")


def test_c4a_skips_group_where_every_member_is_differentiated(collision_wiki_dir):
    from app.lint import _check_c4a_slug_collision

    group = ["gizmo", "gizmo-2", "gizmo-3"]
    for slug in group:
        _stamp_differentiated(collision_wiki_dir, slug, group)

    bases = {f.base_slug for f in _check_c4a_slug_collision(collision_wiki_dir)}
    assert "gizmo" not in bases, "fully differentiated group must be exempt from C4-a"
    assert "widget" in bases, "undifferentiated groups must still fire"


def test_c4a_refires_when_a_new_member_joins_a_differentiated_group(collision_wiki_dir):
    from app.lint import _check_c4a_slug_collision

    group = ["gizmo", "gizmo-2", "gizmo-3"]
    for slug in group:
        _stamp_differentiated(collision_wiki_dir, slug, group)
    _write_page(collision_wiki_dir, "gizmo-4", _GIZMO_BODY)

    bases = {f.base_slug for f in _check_c4a_slug_collision(collision_wiki_dir)}
    assert "gizmo" in bases, (
        "a new member outside the recorded differentiate set must re-fire the finding"
    )


def test_c4a_refires_when_one_member_lost_its_sentinel(collision_wiki_dir):
    """Simulates an ingest rewrite: the replaced file has no sentinel, so the
    group's differentiate ruling no longer covers every member."""
    from app.lint import _check_c4a_slug_collision

    group = ["gizmo", "gizmo-2", "gizmo-3"]
    for slug in group[:2]:
        _stamp_differentiated(collision_wiki_dir, slug, group)

    bases = {f.base_slug for f in _check_c4a_slug_collision(collision_wiki_dir)}
    assert "gizmo" in bases, "a member without a sentinel must keep the finding alive"
