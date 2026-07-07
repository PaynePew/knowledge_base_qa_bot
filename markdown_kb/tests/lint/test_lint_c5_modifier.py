"""Unit tests for the Slice 6-5 C5 modifier: filter qa pages from candidate pairs.

Phase 6 PRD #78 §"Phase 5 lint amendment" — C5 modifier.

The modifier excludes ``frontmatter.type == "qa"`` pages from
``_candidate_pairs`` BEFORE F1/F3 candidate computation so:
  - The LLM call budget does not include qa pairs (cost gate).
  - ``lint-report.md`` is not flooded with trivially-true ``overlap`` findings
    pairing every promoted qa against its source entity.

These tests verify:
  - qa pages are absent from F1 source-intersection candidate pool.
  - qa pages are absent from F3 BM25-hit candidate pool.
  - Entity-vs-entity, concept-vs-concept, and entity-vs-concept pair generation
    behaviour is preserved exactly when no qa pages are present.

Hermetic — no OPENAI_API_KEY required, no LLM invocation.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _write_wiki_page(
    wiki_dir: Path,
    slug: str,
    sources: list[str],
    body: str = "",
    *,
    subdir: str = "concepts",
    page_type: str | None = None,
    extra_frontmatter: dict | None = None,
) -> Path:
    """Write a wiki page with specified sources/body/type/subdir.

    ``page_type`` defaults to the singular of ``subdir`` ("concepts" -> "concept",
    "entities" -> "entity", "qa" -> "qa"). For qa pages, the caller may pass
    ``extra_frontmatter`` to add ``question``/``count``/``status`` overrides.
    """
    page_dir = wiki_dir / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    if page_type is None:
        page_type = "qa" if subdir == "qa" else subdir.rstrip("s")
    frontmatter: dict = {
        "id": slug,
        "type": page_type,
        "created": "2026-05-26T00:00:00Z",
        "updated": "2026-05-26T00:00:00Z",
        "sources": sources,
        "status": "live",
        "open_questions": [],
    }
    if extra_frontmatter:
        frontmatter.update(extra_frontmatter)
    body_content = body or f"# {slug}\n\nThis is a page about {slug.replace('-', ' ')}."
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body_content}\n"
    page_path.write_text(content, encoding="utf-8")
    return page_path


class TestCandidatePairsQaModifier:
    """The C5 modifier (Slice 6-5) excludes ``type=qa`` pages from candidate pairs."""

    def test_qa_page_with_shared_source_is_not_paired_with_entity(self, tmp_wiki_dir):
        """A qa page sharing a source with an entity must NOT produce an F1 candidate."""
        # Entity page with a source citation
        _write_wiki_page(
            tmp_wiki_dir,
            "refund-policy",
            ["refund_policy.md#timeline"],
            body="Refunds processed in 5 business days.",
            subdir="entities",
        )
        # qa page derived from the same entity (same source citation)
        _write_wiki_page(
            tmp_wiki_dir,
            "qa-how-long-refund-abc123",
            ["refund_policy.md#timeline"],
            body="Refunds take 5 business days based on the refund policy.",
            subdir="qa",
            extra_frontmatter={"status": "live", "question": "How long is refund?", "count": 1},
        )

        from app.lint import _candidate_pairs, _load_wiki_pages

        pages = _load_wiki_pages(tmp_wiki_dir)
        pairs = _candidate_pairs(pages, tmp_wiki_dir)

        # No pair should contain the qa slug
        for a, b in pairs:
            assert "qa-how-long-refund" not in a, f"qa slug leaked into F1 pair via {a}: {pairs}"
            assert "qa-how-long-refund" not in b, f"qa slug leaked into F1 pair via {b}: {pairs}"

    def test_qa_pages_do_not_pair_with_each_other(self, tmp_wiki_dir):
        """Two qa pages sharing a source must NOT produce an F1 candidate pair."""
        _write_wiki_page(
            tmp_wiki_dir,
            "qa-a",
            ["policy.md#a"],
            subdir="qa",
            extra_frontmatter={"question": "Q a?", "count": 2},
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "qa-b",
            ["policy.md#a"],
            subdir="qa",
            extra_frontmatter={"question": "Q b?", "count": 3},
        )

        from app.lint import _candidate_pairs, _load_wiki_pages

        pages = _load_wiki_pages(tmp_wiki_dir)
        pairs = _candidate_pairs(pages, tmp_wiki_dir)
        assert pairs == set(), f"qa-vs-qa pair leaked into candidates: {pairs}"

    def test_entity_vs_entity_f1_pair_preserved(self, tmp_wiki_dir):
        """Entity-vs-entity F1 pair-generation behaviour is unchanged by the qa modifier."""
        _write_wiki_page(
            tmp_wiki_dir,
            "alpha",
            ["shared.md#s1"],
            body="Alpha content.",
            subdir="entities",
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "beta",
            ["shared.md#s1"],
            body="Beta content.",
            subdir="entities",
        )

        from app.lint import _candidate_pairs, _load_wiki_pages

        pages = _load_wiki_pages(tmp_wiki_dir)
        pairs = _candidate_pairs(pages, tmp_wiki_dir)
        assert ("alpha", "beta") in pairs

    def test_concept_vs_concept_f1_pair_preserved(self, tmp_wiki_dir):
        """Concept-vs-concept F1 pair-generation behaviour is unchanged."""
        _write_wiki_page(tmp_wiki_dir, "cona", ["shared.md#s1"], subdir="concepts")
        _write_wiki_page(tmp_wiki_dir, "conb", ["shared.md#s1"], subdir="concepts")

        from app.lint import _candidate_pairs, _load_wiki_pages

        pages = _load_wiki_pages(tmp_wiki_dir)
        pairs = _candidate_pairs(pages, tmp_wiki_dir)
        assert ("cona", "conb") in pairs

    def test_entity_vs_concept_f1_pair_preserved(self, tmp_wiki_dir):
        """Entity-vs-concept F1 pair-generation is preserved (cross-subdir)."""
        _write_wiki_page(tmp_wiki_dir, "an-entity", ["shared.md#s1"], subdir="entities")
        _write_wiki_page(tmp_wiki_dir, "a-concept", ["shared.md#s1"], subdir="concepts")

        from app.lint import _candidate_pairs, _load_wiki_pages

        pages = _load_wiki_pages(tmp_wiki_dir)
        pairs = _candidate_pairs(pages, tmp_wiki_dir)
        # canonical sort: "a-concept" < "an-entity"
        assert ("a-concept", "an-entity") in pairs

    def test_mixed_qa_and_entity_only_entity_pairs_returned(self, tmp_wiki_dir):
        """When qa and entity pages share sources, only entity-vs-entity pairs survive."""
        # Two entity pages share a source -> F1 pair expected.
        _write_wiki_page(
            tmp_wiki_dir, "entity-a", ["shared.md#s"], subdir="entities", body="Body A."
        )
        _write_wiki_page(
            tmp_wiki_dir, "entity-b", ["shared.md#s"], subdir="entities", body="Body B."
        )
        # qa pages also share the same source - must be filtered.
        _write_wiki_page(
            tmp_wiki_dir,
            "qa-x-001abc",
            ["shared.md#s"],
            subdir="qa",
            extra_frontmatter={"question": "X?", "count": 1},
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "qa-y-002def",
            ["shared.md#s"],
            subdir="qa",
            extra_frontmatter={"question": "Y?", "count": 1},
        )

        from app.lint import _candidate_pairs, _load_wiki_pages

        pages = _load_wiki_pages(tmp_wiki_dir)
        pairs = _candidate_pairs(pages, tmp_wiki_dir)

        # Entity-vs-entity pair must be present.
        assert ("entity-a", "entity-b") in pairs

        # No qa slug must appear in any pair.
        all_slugs_in_pairs = {s for pair in pairs for s in pair}
        for slug in all_slugs_in_pairs:
            assert not slug.startswith("qa-"), (
                f"qa slug {slug} leaked into candidate pairs: {pairs}"
            )

    def test_qa_only_corpus_produces_no_pairs(self, tmp_wiki_dir):
        """A wiki containing only qa pages produces zero candidate pairs."""
        _write_wiki_page(
            tmp_wiki_dir,
            "qa-a-abc111",
            ["policy.md#s"],
            subdir="qa",
            extra_frontmatter={"question": "A?", "count": 1},
        )
        _write_wiki_page(
            tmp_wiki_dir,
            "qa-b-def222",
            ["policy.md#s"],
            subdir="qa",
            extra_frontmatter={"question": "B?", "count": 1},
        )

        from app.lint import _candidate_pairs, _load_wiki_pages

        pages = _load_wiki_pages(tmp_wiki_dir)
        pairs = _candidate_pairs(pages, tmp_wiki_dir)
        assert pairs == set()
