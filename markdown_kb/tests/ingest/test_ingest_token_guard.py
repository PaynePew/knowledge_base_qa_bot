"""Tests for the token-based ingest guard — per-section hard cap + outline classify.

Replace the byte-based size guard (KB_INGEST_MAX_BYTES) with:
- _estimate_tokens(content) -> int  — CJK-pessimistic //3 estimate
- _max_ingest_tokens() -> int       — per-Source SOFT cap (env KB_INGEST_MAX_TOKENS, default 64000)
- _max_section_tokens() -> int      — per-section HARD cap (env KB_INGEST_MAX_SECTION_TOKENS, default 6000)
- _should_route_async(content) -> bool — True if content exceeds soft cap
- build_outline(content, *, max_tokens=2000) in templates.py — headings + first body chars
- classify_source operates on build_outline(content) not full content
- Large entity Sources (over soft cap) route to per-section synthesis (N pages not 1)

All tests hermetic — no OPENAI_API_KEY required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import app.indexer as indexer_module
import app.ingest as ingest_module
import app.templates as templates_module

# ---------------------------------------------------------------------------
# Tests 1-3: estimate_tokens + caps
# ---------------------------------------------------------------------------


def test_estimate_tokens_cjk_pessimistic():
    """_estimate_tokens of a 300-char CJK string equals 100 (asserts //3)."""
    cjk_300 = "中" * 300
    assert ingest_module._estimate_tokens(cjk_300) == 100


def test_default_max_tokens_is_64k(monkeypatch):
    """Unset env → default per-Source soft cap is 64000 tokens."""
    monkeypatch.delenv("KB_INGEST_MAX_TOKENS", raising=False)
    assert ingest_module._max_ingest_tokens() == 64000


def test_default_section_cap_is_6k(monkeypatch):
    """Unset env → default per-section hard cap is 6000 tokens."""
    monkeypatch.delenv("KB_INGEST_MAX_SECTION_TOKENS", raising=False)
    assert ingest_module._max_section_tokens() == 6000


# ---------------------------------------------------------------------------
# Test 4: oversized section fails without LLM call
# ---------------------------------------------------------------------------


def test_oversized_section_fails_without_llm_call(tmp_path, monkeypatch):
    """A Source with one section over KB_INGEST_MAX_SECTION_TOKENS is rejected.

    The fake LLM with_structured_output must NEVER be called; failed_reasons
    must mention the section being too large.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Make one section that exceeds the token cap.
    # _estimate_tokens = len(content) // 3; cap = 500 tokens → need > 1500 chars.
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "500")
    big_section_body = "x" * 1600  # 1600 // 3 = 533 tokens > 500 cap
    oversized = f"## Big Section\n\n{big_section_body}\n"
    (docs_dir / "big.md").write_text(oversized, encoding="utf-8")

    fake_llm = MagicMock()
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["big.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert "big.md" in result.failed_sources
    assert not fake_llm.with_structured_output.called, (
        "section token guard must fire before any LLM call"
    )
    reason = result.failed_reasons.get("big.md", "")
    assert "too large" in reason.lower()


# ---------------------------------------------------------------------------
# Test 5: under section cap reaches classifier
# ---------------------------------------------------------------------------


def test_under_section_cap_reaches_classifier(tmp_path, monkeypatch):
    """A small Source passes the section guard and reaches the classifier."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "500")
    # Small section: 90 chars → 90 // 3 = 30 tokens, well under 500
    (docs_dir / "small.md").write_text("## Topic\n\nShort body.\n", encoding="utf-8")

    class _ReachedClassifier(Exception):
        pass

    fake_llm = MagicMock()

    def _raise_on_use(_schema):
        raise _ReachedClassifier

    fake_llm.with_structured_output.side_effect = _raise_on_use
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["small.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert fake_llm.with_structured_output.called, (
        "guard must let an under-section-cap Source through to classify"
    )
    assert "too large" not in (result.failed_reasons.get("small.md") or "").lower()


# ---------------------------------------------------------------------------
# Test 6: classify_uses_outline_not_full_content
# ---------------------------------------------------------------------------


def test_classify_uses_outline_not_full_content(tmp_path, monkeypatch):
    """build_outline is used; classifier chain never sees content past outline window.

    We plant a sentinel string FAR past the outline window (after 10 000 chars of
    body), then assert the message passed to the CLASSIFIER chain does NOT contain it.
    Only the classifier chain's invoke is checked; the synthesis chain (concept page)
    is a separate with_structured_output call and intentionally sees section content.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Document: heading, 10 001 chars of body, then the sentinel.
    sentinel = "SENTINEL_PAST_OUTLINE_WINDOW_XYZ123"
    doc_content = "## Topic\n\n" + ("a" * 10_001) + "\n" + sentinel + "\n"
    (docs_dir / "outline_test.md").write_text(doc_content, encoding="utf-8")

    # No section cap concern — set it very high
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "99999")
    monkeypatch.setenv("KB_INGEST_MAX_TOKENS", "999999")

    # Only capture messages from the CLASSIFIER call (first with_structured_output call)
    classifier_messages: list[str] = []

    class _ClassifierChain:
        """Fake chain for the _ClassifierOutput call — captures what goes in."""
        def invoke(self, messages):
            for m in messages:
                content = getattr(m, "content", str(m))
                classifier_messages.append(content)
            class _Out:
                type = "concept"
            return _Out()

    class _SynthChain:
        """Fake chain for synthesis calls — not relevant to this test."""
        def invoke(self, messages):
            class _Out:
                body = "Synthesised."
                open_questions = []
            return _Out()

    def _dispatch_chain(schema):
        from app.templates import _ClassifierOutput
        if schema is _ClassifierOutput:
            return _ClassifierChain()
        return _SynthChain()

    fake_llm = MagicMock()
    fake_llm.with_structured_output.side_effect = _dispatch_chain
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr("app.ingest.verify", lambda body, sections: _make_grounding_pass())

    ingest_module.ingest_sources(["outline_test.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert classifier_messages, "Expected at least one message to be captured for the classifier"
    classifier_text = " ".join(classifier_messages)
    assert sentinel not in classifier_text, (
        "The sentinel placed past the outline window must not reach the classifier"
    )


def _make_grounding_pass():
    """Return a passing GroundingOutcome-like object."""
    from unittest.mock import MagicMock
    outcome = MagicMock()
    outcome.passed = True
    return outcome


# ---------------------------------------------------------------------------
# Test 7: build_outline includes all headings
# ---------------------------------------------------------------------------


def test_build_outline_includes_all_headings():
    """build_outline over a doc with 3 headings returns all 3 heading texts."""
    doc = (
        "# Heading One\n\nSome body.\n\n"
        "## Heading Two\n\nMore body.\n\n"
        "### Heading Three\n\nEven more.\n"
    )
    outline = templates_module.build_outline(doc)
    assert "Heading One" in outline
    assert "Heading Two" in outline
    assert "Heading Three" in outline


# ---------------------------------------------------------------------------
# Test 8: _should_route_async threshold
# ---------------------------------------------------------------------------


def test_should_route_async_threshold(monkeypatch):
    """_should_route_async returns False just under and True just over soft cap."""
    monkeypatch.setenv("KB_INGEST_MAX_TOKENS", "1000")

    # Just under: 2999 chars → 2999 // 3 = 999 tokens < 1000 cap
    under = "a" * 2999
    assert ingest_module._should_route_async(under) is False

    # Just over: 3003 chars → 3003 // 3 = 1001 tokens > 1000 cap
    over = "a" * 3003
    assert ingest_module._should_route_async(over) is True


# ---------------------------------------------------------------------------
# Test 9: large entity routes to per-section synthesis
# ---------------------------------------------------------------------------


def test_large_entity_routes_to_per_section(tmp_path, monkeypatch):
    """Classifier returns 'entity' for over-soft-cap doc; N pages written (one per section).

    generate_entity_page must NOT be called; per-section slugs must be present.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Set soft cap low: 100 tokens → 300 chars triggers async routing
    monkeypatch.setenv("KB_INGEST_MAX_TOKENS", "100")
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "99999")  # no hard cap

    # Two sections, total > 300 chars
    content = (
        "## Section Alpha\n\n" + "a" * 200 + "\n\n"
        "## Section Beta\n\n" + "b" * 200 + "\n"
    )
    (docs_dir / "entity_large.md").write_text(content, encoding="utf-8")

    # Track whether generate_entity_page was called
    entity_page_called = []

    original_generate_entity_page = templates_module.generate_entity_page

    def _fake_generate_entity_page(sections, source_stem, source_filename, **kwargs):
        entity_page_called.append(True)
        return original_generate_entity_page(sections, source_stem, source_filename, **kwargs)

    # Build a fake LLM that:
    # - first call (classify): returns entity
    # - subsequent calls (per-section synthesis): returns a valid page body
    class _ClassifierChain:
        def invoke(self, messages):
            class _Out:
                type = "entity"
            return _Out()

    class _SynthChain:
        def invoke(self, messages):
            class _Out:
                body = "Synthesised body."
                open_questions = []
            return _Out()

    def _fake_with_structured_output(schema):
        from app.templates import _ClassifierOutput
        if schema is _ClassifierOutput:
            return _ClassifierChain()
        return _SynthChain()

    fake_llm = MagicMock()
    fake_llm.with_structured_output.side_effect = _fake_with_structured_output
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(templates_module, "generate_entity_page", _fake_generate_entity_page)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr("app.ingest.verify", lambda body, sections: _make_grounding_pass())

    result = ingest_module.ingest_sources(
        ["entity_large.md"], docs_dir=docs_dir, wiki_dir=wiki_dir
    )

    assert not entity_page_called, "generate_entity_page must NOT be called for large entity"
    # Two sections → two pages written (one per section)
    pages = result.results[0].pages_written if result.results else []
    assert len(pages) == 2, f"Expected 2 per-section pages, got {pages}"
    # Pages go to entities/ subdir (pages_written contains relative paths like "entities/slug.md")
    for page_rel_path in pages:
        entity_page = wiki_dir / page_rel_path
        assert entity_page.exists(), f"Expected {page_rel_path} to exist under wiki_dir"
        assert page_rel_path.startswith("entities/"), f"Expected entity page under entities/, got {page_rel_path}"


# ---------------------------------------------------------------------------
# Test 10: under soft cap entity still produces single page (no regression)
# ---------------------------------------------------------------------------


def test_under_soft_cap_entity_still_single_page(tmp_path, monkeypatch):
    """A normal-size entity doc still produces one entities/<stem>.md page."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Generous soft cap so normal doc never triggers async routing
    monkeypatch.setenv("KB_INGEST_MAX_TOKENS", "64000")
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "6000")

    content = "## About\n\nThis entity is described here.\n"
    (docs_dir / "my_entity.md").write_text(content, encoding="utf-8")

    class _ClassifierChain:
        def invoke(self, messages):
            class _Out:
                type = "entity"
            return _Out()

    class _SynthChain:
        def invoke(self, messages):
            class _Out:
                body = "A nice entity summary."
                open_questions = []
            return _Out()

    def _fake_with_structured_output(schema):
        from app.templates import _ClassifierOutput
        if schema is _ClassifierOutput:
            return _ClassifierChain()
        return _SynthChain()

    fake_llm = MagicMock()
    fake_llm.with_structured_output.side_effect = _fake_with_structured_output
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr("app.ingest.verify", lambda body, sections: _make_grounding_pass())

    result = ingest_module.ingest_sources(
        ["my_entity.md"], docs_dir=docs_dir, wiki_dir=wiki_dir
    )

    assert result.results, f"Expected success, got failures: {result.failed_sources}"
    pages = result.results[0].pages_written
    assert len(pages) == 1, f"Expected single entity page, got {pages}"
    # pages_written contains relative paths like "entities/my-entity.md"
    page_path = pages[0]
    assert (wiki_dir / page_path).exists(), f"Expected {page_path} to exist under wiki_dir"
    assert page_path.startswith("entities/"), f"Expected entity page under entities/, got {page_path}"
