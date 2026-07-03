"""Regression test for the ADR-0030 Invariant: "no alias value enters the
Section Index (BM25) ... in v1" (issue #406; CODING_STANDARD §11 "Alias &
quarantine drift" — reviewer FAIL if an alias value reaches BM25 tokens).

``indexer.parse_markdown``'s rule 2 already refuses to tokenize ANY
frontmatter value ("Do NOT tokenize frontmatter values into BM25 tokens"),
so this invariant holds by construction — this test pins it specifically for
``aliases`` against the REAL byte shape ``wiki_writer.write_pages_for_source``
produces (sentinel comment + frontmatter fence), not a hand-rolled fixture
that could drift from the real producer (§6.5 fixture fidelity).
"""

from __future__ import annotations

import app.indexer as indexer
from app.schemas import WikiPageDraft, WikiPageFrontmatter
from app.wiki_writer import write_pages_for_source

FIXED_TS = "2026-05-26T14:30:00Z"

# A deliberately distinctive alias value that would be trivially detectable
# in the tokens list if it leaked — and does NOT appear anywhere in the body.
_ALIAS_VALUE = "zzqx-distinctive-alias-marker"


def test_alias_value_never_appears_in_section_tokens(tmp_path):
    wiki_dir = tmp_path / "wiki"
    slug = "replacement-payment-methods"
    citation_id = f"source.md#{slug}"
    fm = WikiPageFrontmatter(
        id=slug,
        type="concept",
        created=FIXED_TS,
        updated=FIXED_TS,
        sources=[citation_id],
        status="live",
        open_questions=[],
        aliases=[_ALIAS_VALUE],
    )
    draft = WikiPageDraft(
        frontmatter=fm,
        body="Customers may use alternate payment options at checkout.",
        citation_line=f"[Source: {citation_id}]",
        slug=slug,
        heading="Replacement Payment Methods",
    )
    write_pages_for_source("source.md", [draft], wiki_dir=wiki_dir)
    page_path = wiki_dir / "concepts" / f"{slug}.md"

    sections = indexer.parse_markdown(page_path, source_id=slug)

    assert sections, "expected at least one Section"
    for sec in sections:
        assert _ALIAS_VALUE not in sec.tokens, (
            f"alias value leaked into BM25 tokens: {sec.tokens!r}"
        )
        # Sanity: the metadata dict DOES carry it (display-only path is intact).
        assert sec.metadata.get("aliases") == [_ALIAS_VALUE]
