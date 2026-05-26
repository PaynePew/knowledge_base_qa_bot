"""Tests for cross-type slug uniqueness in ingest.py (Slice 4-3b, issue #54).

Covers:
  - Cross-type slug collision: entity page and concept page both claim slug
    ``foo`` in one batch → first keeps ``foo``, second gets ``foo-2``.
  - Same-type collision regression: two concept sources claiming ``overview``
    still resolves to ``overview`` + ``overview-2`` (unchanged behaviour).
  - Global uniqueness: ``used_slugs`` is a single set, not a per-type dict.

All tests are hermetic (no OPENAI_API_KEY required). The autouse fixture in
conftest.py mocks ``app.ingest.verify`` to ``claim_supported`` and redirects
WIKI_DIR / INDEX_PATH to tmp, so these tests cannot pollute real state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import app.templates as templates_module

# ---------------------------------------------------------------------------
# Shared helpers (inlined to keep this module self-contained)
# ---------------------------------------------------------------------------


FIXED_BODY = "Generated synthesis body."
FIXED_TS = "2026-05-26T14:30:00Z"


class _FakeSynthesisOutput:
    body: str
    open_questions: list

    def __init__(self, body: str = FIXED_BODY, open_questions: list | None = None):
        self.body = body
        self.open_questions = open_questions or []


class _FakeClassifierOutput:
    type: str

    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_schema_aware_fake_llm(classifier_type: str = "concept") -> MagicMock:
    """Return a fake ChatOpenAI whose with_structured_output is schema-aware."""
    from app.templates import _ClassifierOutput  # private but accessible in tests

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            chain.invoke.return_value = _FakeClassifierOutput(classifier_type)
        else:
            chain.invoke.return_value = _FakeSynthesisOutput()
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


def _make_per_source_type_llm(source_type_map: dict[str, str]) -> MagicMock:
    """Return a fake ChatOpenAI whose classifier returns different types per source.

    ``source_type_map`` maps a substring of the source body content (e.g. the
    heading text) to the desired source_type string.  The first matching key
    wins; fall back to "concept" if nothing matches.
    """
    from app.templates import _ClassifierOutput

    fake_llm = MagicMock()

    call_count = {"n": 0}

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            # We rely on ordering: classify calls arrive in source order.
            # Use a closure counter so call #0 → entity type, call #1 → concept.
            idx = call_count["n"]
            call_count["n"] += 1

            types_in_order = list(source_type_map.values())
            chosen_type = types_in_order[idx] if idx < len(types_in_order) else "concept"
            chain.invoke.return_value = _FakeClassifierOutput(chosen_type)
        else:
            chain.invoke.return_value = _FakeSynthesisOutput()
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


# ---------------------------------------------------------------------------
# AC: cross-type slug collision → entity "foo" + concept "foo" → foo + foo-2
# ---------------------------------------------------------------------------


def test_cross_type_slug_collision_produces_suffix(tmp_path, monkeypatch):
    """Entity page ``foo`` and concept page ``foo`` in one batch → foo + foo-2.

    This verifies that ``used_slugs`` is a single global set: after the entity
    source claims slug ``foo``, the concept source's identical slug collides and
    receives the ``-2`` suffix.
    """
    import app.indexer as indexer_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Entity source: stem "foo" → entity slug "foo"
    (docs_dir / "foo.md").write_text(
        "# Foo Entity\n\n## Overview\n\nThe foo entity overview.\n\n## Details\n\nFoo details.\n",
        encoding="utf-8",
    )
    # Concept source: heading "foo" → concept slug "foo"
    (docs_dir / "concepts_foo.md").write_text(
        "## foo\n\nThe foo concept description.\n",
        encoding="utf-8",
    )

    wiki_dir = tmp_path / "wiki"

    # foo.md → entity, concepts_foo.md → concept
    fake_llm = _make_per_source_type_llm({"foo.md": "entity", "concepts_foo.md": "concept"})
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.ingest import ingest_sources

    # Process entity first, then concept
    result = ingest_sources(
        ["foo.md", "concepts_foo.md"],
        docs_dir=docs_dir,
        wiki_dir=wiki_dir,
    )

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert len(result.results) == 2, f"Expected 2 results, got: {result.results}"

    all_pages = [p for r in result.results for p in r.pages_written]

    # Entity takes the bare slug; concept collides → gets -2
    assert "entities/foo.md" in all_pages, f"Expected entities/foo.md in pages, got: {all_pages}"
    assert "concepts/foo-2.md" in all_pages, (
        f"Expected concepts/foo-2.md (cross-type collision), got: {all_pages}"
    )

    # Both files must exist on disk
    assert (wiki_dir / "entities" / "foo.md").exists(), "entities/foo.md should exist on disk"
    assert (wiki_dir / "concepts" / "foo-2.md").exists(), "concepts/foo-2.md should exist on disk"

    # The bare concepts/foo.md must NOT exist (it was collided)
    assert not (wiki_dir / "concepts" / "foo.md").exists(), (
        "concepts/foo.md must NOT exist — slug was taken by the entity page"
    )


# ---------------------------------------------------------------------------
# Regression: same-type collision still resolves to -2, -3 (unchanged)
# ---------------------------------------------------------------------------


def test_same_type_slug_collision_still_works(tmp_path, monkeypatch):
    """Regression: two concept sources with same heading still get overview + overview-2."""
    import app.indexer as indexer_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "alpha.md").write_text("## Overview\n\nAlpha overview.\n", encoding="utf-8")
    (docs_dir / "beta.md").write_text("## Overview\n\nBeta overview.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.ingest import ingest_sources

    result = ingest_sources(["alpha.md", "beta.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"

    all_pages = [p for r in result.results for p in r.pages_written]
    assert "concepts/overview.md" in all_pages, f"Expected overview.md in {all_pages}"
    assert "concepts/overview-2.md" in all_pages, f"Expected overview-2.md in {all_pages}"

    assert (wiki_dir / "concepts" / "overview.md").exists()
    assert (wiki_dir / "concepts" / "overview-2.md").exists()
