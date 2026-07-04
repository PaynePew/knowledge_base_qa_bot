"""Tests for ``markdown_kb.app.slugs.build_alias_resolution_map`` — the shared
link-layer resolver (issue #406, ADR-0030).

ADR-0030 Invariant: "every consumer of wikilink resolution (C2, linkify,
inbound-reference computation) uses the shared resolver; no surface builds
its own slug set." This file tests the ONE shared implementation directly;
``app.lint``'s C2 / ``find_inbound_references`` tests cover their own
call-site wiring on top of it.

Deterministic collision rule (ADR-0030 decision 4 / issue #406 scope item 2):
a real page slug always wins over an alias claiming the same key; an
alias-vs-alias tie (no real page at that key) breaks lexicographically by
canonical slug.
"""

from __future__ import annotations

from pathlib import Path

from app.schemas import WikiPageDraft, WikiPageFrontmatter
from app.slugs import build_alias_resolution_map, build_slug_paths
from app.wiki_writer import write_pages_for_source

FIXED_TS = "2026-05-26T14:30:00Z"


def _write_page(
    wiki_dir: Path,
    slug: str,
    *,
    page_type: str = "concept",
    aliases: list[str] | None = None,
) -> None:
    """Write one minimal entities/concepts page carrying ``aliases``."""
    citation_id = f"source.md#{slug}"
    fm = WikiPageFrontmatter(
        id=slug,
        type=page_type,
        created=FIXED_TS,
        updated=FIXED_TS,
        sources=[citation_id],
        status="live",
        open_questions=[],
        aliases=aliases or [],
    )
    draft = WikiPageDraft(
        frontmatter=fm,
        body=f"Content for {slug}.",
        citation_line=f"[Source: {citation_id}]",
        slug=slug,
        heading=slug.replace("-", " ").title(),
    )
    write_pages_for_source("source.md", [draft], wiki_dir=wiki_dir)


# ---------------------------------------------------------------------------
# Base case: every real slug resolves to itself
# ---------------------------------------------------------------------------


def test_real_slugs_resolve_to_themselves_with_no_aliases(tmp_path):
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "cancellation-window")
    _write_page(wiki_dir, "acme-shop", page_type="entity")

    resolution = build_alias_resolution_map(wiki_dir)

    assert resolution == {"cancellation-window": "cancellation-window", "acme-shop": "acme-shop"}


def test_empty_wiki_dir_returns_empty_map(tmp_path):
    wiki_dir = tmp_path / "wiki"
    assert build_alias_resolution_map(wiki_dir) == {}


# ---------------------------------------------------------------------------
# Alias resolves to its target page's canonical slug
# ---------------------------------------------------------------------------


def test_alias_resolves_to_canonical_slug(tmp_path):
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "replacement-payment-methods", aliases=["paypal", "alt-payment"])

    resolution = build_alias_resolution_map(wiki_dir)

    assert resolution["replacement-payment-methods"] == "replacement-payment-methods"
    assert resolution["paypal"] == "replacement-payment-methods"
    assert resolution["alt-payment"] == "replacement-payment-methods"


def test_entity_page_aliases_also_resolve(tmp_path):
    """Aliases work identically for entities/ pages, not just concepts/."""
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "acme-shop", page_type="entity", aliases=["acme"])

    resolution = build_alias_resolution_map(wiki_dir)

    assert resolution["acme"] == "acme-shop"


# ---------------------------------------------------------------------------
# Deterministic collision rule 1: a real page slug always wins over an alias
# ---------------------------------------------------------------------------


def test_real_slug_wins_over_a_colliding_alias(tmp_path):
    """An alias value equal to another page's real slug never overrides it —
    the real page's own self-mapping is authoritative (ADR-0030 decision 4)."""
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "pricing")
    _write_page(wiki_dir, "other-page", aliases=["pricing"])

    resolution = build_alias_resolution_map(wiki_dir)

    assert resolution["pricing"] == "pricing"


# ---------------------------------------------------------------------------
# Deterministic collision rule 2: alias-vs-alias tie breaks lexicographically
# ---------------------------------------------------------------------------


def test_alias_vs_alias_tie_breaks_lexicographically_by_canonical_slug(tmp_path):
    """Two pages claim the SAME alias, no real page owns that slug: the
    lexicographically-first canonical slug wins, independent of directory
    iteration order (ADR-0030 decision 4)."""
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "zeta-page", aliases=["shared-alias"])
    _write_page(wiki_dir, "alpha-page", aliases=["shared-alias"])

    resolution = build_alias_resolution_map(wiki_dir)

    assert resolution["shared-alias"] == "alpha-page"


# ---------------------------------------------------------------------------
# Malformed / absent aliases degrade gracefully
# ---------------------------------------------------------------------------


def test_page_with_no_aliases_field_is_unaffected(tmp_path):
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "no-alias-page")

    resolution = build_alias_resolution_map(wiki_dir)

    assert resolution == {"no-alias-page": "no-alias-page"}


def test_blank_alias_entries_are_ignored(tmp_path):
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "page-a", aliases=["", "   ", "real-alias"])

    resolution = build_alias_resolution_map(wiki_dir)

    assert "" not in resolution
    assert resolution["real-alias"] == "page-a"


# ---------------------------------------------------------------------------
# build_slug_paths — a location lookup, not a resolution decision (issue #410)
# ---------------------------------------------------------------------------


def test_slug_paths_maps_concept_slug_to_its_relpath(tmp_path):
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "cancellation-window")

    paths = build_slug_paths(wiki_dir)

    assert paths == {"cancellation-window": "wiki/concepts/cancellation-window.md"}


def test_slug_paths_maps_entity_slug_to_its_relpath(tmp_path):
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "acme-shop", page_type="entity")

    paths = build_slug_paths(wiki_dir)

    assert paths == {"acme-shop": "wiki/entities/acme-shop.md"}


def test_slug_paths_excludes_aliases_and_unresolvable_strings(tmp_path):
    """Aliases resolve THROUGH a real slug (issue #410 AC) — they are not
    themselves keys of ``build_slug_paths``; only real page slugs are."""
    wiki_dir = tmp_path / "wiki"
    _write_page(wiki_dir, "replacement-payment-methods", aliases=["paypal"])

    paths = build_slug_paths(wiki_dir)

    assert paths == {"replacement-payment-methods": "wiki/concepts/replacement-payment-methods.md"}
    assert "paypal" not in paths


def test_slug_paths_empty_wiki_dir_returns_empty_map(tmp_path):
    wiki_dir = tmp_path / "wiki"
    assert build_slug_paths(wiki_dir) == {}
