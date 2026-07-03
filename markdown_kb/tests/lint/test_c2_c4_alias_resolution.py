"""Tests for the C2 / C4 alias-resolution amendments (issue #406, ADR-0030).

ADR-0030 decision 2: "One resolver, three consumers, link-layer only."
A ``[[wikilink]]`` now resolves iff its target matches an existing page's
slug OR one of its declared ``aliases`` — both C2's red-link judgment and
``find_inbound_references`` (the C4 merge-apply reference guard) must use
the SAME shared resolver (``slugs.build_alias_resolution_map``).

Uses ``tmp_wiki_dir`` from the lint sub-package's ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.lint import _check_c2_red_links, find_inbound_references


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    *,
    subdir: str = "concepts",
    body: str = "",
    aliases: list[str] | None = None,
) -> Path:
    """Write a minimal wiki page, optionally carrying an ``aliases`` list."""
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    frontmatter: dict = {
        "id": slug,
        "type": subdir.rstrip("s"),
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "sources": [f"source.md#{slug}"],
        "status": "live",
        "open_questions": [],
    }
    if aliases is not None:
        frontmatter["aliases"] = aliases
    if not body:
        body = f"# {slug}\n\nSome content."
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ---------------------------------------------------------------------------
# C2 — alias resolution
# ---------------------------------------------------------------------------


class TestC2AliasResolution:
    def test_link_to_alias_is_not_a_red_link(self, tmp_wiki_dir):
        """[[paypal]] resolves via replacement-payment-methods' declared alias."""
        _write_wiki_page(
            tmp_wiki_dir,
            "replacement-payment-methods",
            aliases=["paypal"],
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "referrer-page",
            body="See [[paypal]] for alternatives.",
        )

        findings = _check_c2_red_links(tmp_wiki_dir)

        assert findings == [], f"expected no red links, got: {findings}"

    def test_link_to_unknown_slug_is_still_a_red_link(self, tmp_wiki_dir):
        """An alias existing elsewhere does not make EVERY link resolve —
        only the specific declared alias string."""
        _write_wiki_page(tmp_wiki_dir, "replacement-payment-methods", aliases=["paypal"])
        _write_wiki_page(
            tmp_wiki_dir,
            "referrer-page",
            body="See [[venmo]] for alternatives.",
        )

        findings = _check_c2_red_links(tmp_wiki_dir)

        assert [f.slug for f in findings] == ["venmo"]


# ---------------------------------------------------------------------------
# C4 — find_inbound_references alias-mediation
# ---------------------------------------------------------------------------


class TestFindInboundReferencesAliasMediation:
    def test_alias_mediated_link_counts_as_inbound_reference(self, tmp_wiki_dir):
        """A [[paypal]] link counts as an inbound reference to
        replacement-payment-methods (the page that declares "paypal" as an
        alias) — the C4 merge-apply guard must not miss it."""
        _write_wiki_page(
            tmp_wiki_dir,
            "replacement-payment-methods",
            aliases=["paypal"],
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "referrer-page",
            body="See [[paypal]] for alternatives.",
        )

        wiki_refs, qa_refs = find_inbound_references("replacement-payment-methods", tmp_wiki_dir)

        assert wiki_refs == ["referrer-page"]
        assert qa_refs == []

    def test_literal_slug_link_still_counts_as_before(self, tmp_wiki_dir):
        """Regression: a plain [[slug]] link (no alias involved) still counts
        — the pre-#406 behaviour is a special case of resolver lookup."""
        _write_wiki_page(tmp_wiki_dir, "target-page")
        _write_wiki_page(tmp_wiki_dir, "referrer-page", body="See [[target-page]].")

        wiki_refs, _qa_refs = find_inbound_references("target-page", tmp_wiki_dir)

        assert wiki_refs == ["referrer-page"]

    def test_unrelated_alias_does_not_leak_into_a_different_slugs_referrers(self, tmp_wiki_dir):
        """A link resolving to page A's alias must not be counted as an
        inbound reference to unrelated page B."""
        _write_wiki_page(tmp_wiki_dir, "page-a", aliases=["alias-a"])
        _write_wiki_page(tmp_wiki_dir, "page-b")
        _write_wiki_page(tmp_wiki_dir, "referrer-page", body="See [[alias-a]].")

        wiki_refs_a, _ = find_inbound_references("page-a", tmp_wiki_dir)
        wiki_refs_b, _ = find_inbound_references("page-b", tmp_wiki_dir)

        assert wiki_refs_a == ["referrer-page"]
        assert wiki_refs_b == []
