"""Tests for the `wiki/index.md` alias projection (issue #406, ADR-0030 scope
item 6: "`wiki/index.md` projection lists each page's aliases").

``project_wiki_index`` is a pure function over ``Section`` objects, so these
tests build ``Section`` instances directly (mirroring the golden-file test's
approach in ``test_project_wiki_index.py``) rather than going through a real
``build_index()`` run — the projection change is entirely inside
``_format_bullet``, which reads ``Section.metadata["aliases"]`` (already
populated by the indexer's existing frontmatter-copy — see
``indexer._section_metadata``, unmodified by this issue).
"""

from __future__ import annotations

from app.indexer import Section
from app.wiki_index import project_wiki_index


def _section(slug: str, *, aliases: list[str] | None = None) -> Section:
    metadata: dict = {"lang": "en"}
    if aliases is not None:
        metadata["aliases"] = aliases
    return Section(
        id=f"{slug}#{slug}",
        file=slug,
        heading=slug.replace("-", " ").title(),
        heading_path=[slug],
        content="Some content.",
        tokens=["some", "content"],
        metadata=metadata,
    )


def test_page_with_aliases_shows_alias_suffix():
    content = project_wiki_index([_section("replacement-payment-methods", aliases=["paypal"])])

    assert "(aliases: `paypal`)" in content


def test_page_with_multiple_aliases_lists_all():
    content = project_wiki_index(
        [_section("replacement-payment-methods", aliases=["paypal", "alt-payment"])]
    )

    assert "(aliases: `paypal`, `alt-payment`)" in content


def test_page_with_no_aliases_field_has_no_suffix():
    content = project_wiki_index([_section("no-alias-page")])

    assert "aliases:" not in content
    assert (
        "- [No Alias Page](../docs/no-alias-page#no-alias-page) — `no-alias-page#no-alias-page`"
        in content
    )


def test_page_with_empty_aliases_list_has_no_suffix():
    """An empty `aliases: []` (the common default) renders no suffix — only
    a NON-empty list is shown."""
    content = project_wiki_index([_section("no-alias-page", aliases=[])])

    assert "aliases:" not in content


def test_malformed_aliases_field_degrades_gracefully():
    """A non-list `aliases` value (corrupt/hand-edited frontmatter) never
    raises — the suffix is simply omitted."""
    sec = _section("weird-page")
    sec.metadata["aliases"] = "not-a-list"

    content = project_wiki_index([sec])

    assert "aliases:" not in content
