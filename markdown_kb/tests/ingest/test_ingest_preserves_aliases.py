"""Tests for `/ingest`'s alias preservation (issue #406, ADR-0030).

ADR-0030 Invariant: "`/ingest`'s page overwrite preserves the `aliases`
frontmatter field alongside `created`." The synthesis draft NEVER carries
aliases (they are curator-authored, not LLM-generated), so without this
preserve step, re-ingesting a Source would silently wipe any alias a curator
had assigned — mirroring the existing `created`-preservation precedent
(Slice #3, `test_ingest_orphan_handling.py::test_scenario_b_...`).

Uses `_finalise_source_drafts` directly with `sections=[]` so the grounding
verifier's documented empty-sections short-circuit (`grounding.verify`)
returns deterministically without any LLM call — fully hermetic, no
OPENAI_API_KEY, no mocking required.
"""

from __future__ import annotations

from app.ingest import _finalise_source_drafts
from app.schemas import WikiPageDraft, WikiPageFrontmatter
from app.wiki_writer import read_existing_frontmatter, write_pages_for_source

FIXED_TS = "2026-05-26T14:30:00Z"


def _make_draft(slug: str, aliases: list[str] | None = None) -> WikiPageDraft:
    citation_id = f"refund_policy.md#{slug}"
    fm = WikiPageFrontmatter(
        id=slug,
        type="concept",
        created=FIXED_TS,
        updated=FIXED_TS,
        sources=[citation_id],
        status="live",
        open_questions=[],
        aliases=aliases or [],
    )
    return WikiPageDraft(
        frontmatter=fm,
        body=f"Content for {slug}.",
        citation_line=f"[Source: {citation_id}]",
        slug=slug,
        heading=slug.replace("-", " ").title(),
    )


def test_reingest_preserves_curator_authored_aliases(tmp_path):
    """A curator-assigned alias survives a re-ingest even though the fresh
    synthesis draft carries no aliases at all."""
    wiki_dir = tmp_path / "wiki"
    page_path = wiki_dir / "concepts" / "cancellation-window.md"

    # First write: a page WITHOUT aliases (as any real /ingest draft is).
    write_pages_for_source(
        "refund_policy.md", [_make_draft("cancellation-window")], wiki_dir=wiki_dir
    )
    fm_v0 = read_existing_frontmatter(page_path)
    assert fm_v0["aliases"] == []

    # Curator hand-assigns an alias (simulating the follow-up assign-alias
    # endpoint this foundation slice does not build yet).
    write_pages_for_source(
        "refund_policy.md",
        [_make_draft("cancellation-window", aliases=["retracted-order"])],
        wiki_dir=wiki_dir,
    )
    fm_v1 = read_existing_frontmatter(page_path)
    assert fm_v1["aliases"] == ["retracted-order"]

    # Re-ingest via the real pipeline tail: the fresh draft carries NO
    # aliases (the LLM never produces them) — the preserve step must copy
    # the curator's alias forward from the existing on-disk frontmatter.
    fresh_draft = _make_draft("cancellation-window")
    assert fresh_draft.frontmatter.aliases == []

    _finalise_source_drafts(
        [fresh_draft],
        sections=[],
        source_name="refund_policy.md",
        source_path=tmp_path / "docs" / "refund_policy.md",
        docs_body_hash="deadbeef",
        resolved_wiki_dir=wiki_dir,
    )

    fm_v2 = read_existing_frontmatter(page_path)
    assert fm_v2["aliases"] == ["retracted-order"], (
        f"aliases should be preserved across re-ingest, got: {fm_v2['aliases']}"
    )


def test_first_ingest_of_a_new_page_has_no_aliases_to_preserve(tmp_path):
    """No existing page on disk: preserve step is a no-op, draft's own
    (empty) aliases list is written unchanged."""
    wiki_dir = tmp_path / "wiki"
    page_path = wiki_dir / "concepts" / "brand-new-page.md"

    _finalise_source_drafts(
        [_make_draft("brand-new-page")],
        sections=[],
        source_name="refund_policy.md",
        source_path=tmp_path / "docs" / "refund_policy.md",
        docs_body_hash="deadbeef",
        resolved_wiki_dir=wiki_dir,
    )

    fm = read_existing_frontmatter(page_path)
    assert fm is not None
    assert fm["aliases"] == []
