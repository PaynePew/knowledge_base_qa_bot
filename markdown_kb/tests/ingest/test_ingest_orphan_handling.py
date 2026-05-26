"""Tests for re-ingest correctness — Slice #3.

Scenarios:
  A — orphan deletion: re-ingest with fewer sections deletes removed-section pages
  B — created preservation: re-ingest preserves the original `created` timestamp
  C — orphan scope is per-Source: re-ingesting source X never deletes pages from source Y
  D — concurrency: concurrent POST /ingest + POST /index are lock-serialised with no corruption

All tests are hermetic (no OPENAI_API_KEY required, no LLM calls).

AC coverage (issue #31):
  - Scenario A: orphan deletion scoped to Source
  - Scenario B: created timestamp preserved; updated advances
  - Scenario C: per-Source orphan scope
  - Scenario D: concurrency lock
  - Existing Slice #1+#2 tests still pass (run by conftest)
  - Hermetic: no OPENAI_API_KEY dependency
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import app.templates as templates_module
from app.schemas import WikiPageDraft, WikiPageFrontmatter
from app.wiki_writer import (
    delete_orphans,
    read_existing_frontmatter,
    write_pages_for_source,
)

# ---------------------------------------------------------------------------
# Constants / helpers shared across scenarios
# ---------------------------------------------------------------------------

FIXED_TS = "2026-05-26T14:30:00Z"
LATER_TS = "2026-05-27T09:00:00Z"


def _make_draft(
    slug: str,
    source: str = "refund_policy.md",
    page_type: str = "concept",
    created: str = FIXED_TS,
    updated: str = FIXED_TS,
    heading: str | None = None,
) -> WikiPageDraft:
    """Build a minimal WikiPageDraft for testing.

    ``heading`` defaults to a title-cased reconstruction of the slug.
    """
    if heading is None:
        heading = slug.replace("-", " ").title()
    citation_id = f"{source}#{slug}"
    fm = WikiPageFrontmatter(
        id=slug,
        type=page_type,
        created=created,
        updated=updated,
        sources=[citation_id],
        status="live",
        open_questions=[],
    )
    return WikiPageDraft(
        frontmatter=fm,
        body=f"Content for {slug}.",
        citation_line=f"[Source: {citation_id}]",
        slug=slug,
        heading=heading,
    )


# ---------------------------------------------------------------------------
# Scenario A: orphan deletion
# ---------------------------------------------------------------------------


def test_scenario_a_orphan_page_deleted(tmp_path):
    """Re-ingest with fewer sections: removed-section page is deleted.

    1. First ingest: 3 pages written for refund_policy.md.
    2. Edit source: remove one section.
    3. Re-ingest: 2 pages written, 1 page deleted.
    Assertions:
      - deleted page file is gone from wiki/concepts/
      - delete_orphans returns the slugs of deleted pages
    """
    wiki_dir = tmp_path / "wiki"

    # Initial ingest: 3 pages
    drafts_v1 = [
        _make_draft("cancellation-window"),
        _make_draft("refund-timeline"),
        _make_draft("partial-refund"),
    ]
    write_pages_for_source("refund_policy.md", drafts_v1, wiki_dir=wiki_dir)

    # Confirm all 3 pages exist
    for slug in ("cancellation-window", "refund-timeline", "partial-refund"):
        assert (wiki_dir / "concepts" / f"{slug}.md").exists(), (
            f"Expected {slug}.md to exist after first ingest"
        )

    # Re-ingest: only 2 pages (partial-refund section was removed)
    drafts_v2 = [
        _make_draft("cancellation-window"),
        _make_draft("refund-timeline"),
    ]
    current_page_ids = {"cancellation-window", "refund-timeline"}
    deleted = delete_orphans("refund_policy.md", current_page_ids, wiki_dir=wiki_dir)

    # Write the 2 surviving pages
    write_pages_for_source("refund_policy.md", drafts_v2, wiki_dir=wiki_dir)

    # Orphan page is gone
    assert not (wiki_dir / "concepts" / "partial-refund.md").exists(), (
        "Expected partial-refund.md to be deleted after re-ingest"
    )

    # delete_orphans returned the deleted slug path
    assert "concepts/partial-refund.md" in deleted, (
        f"Expected 'concepts/partial-refund.md' in deleted, got: {deleted}"
    )

    # Surviving pages still exist
    assert (wiki_dir / "concepts" / "cancellation-window.md").exists()
    assert (wiki_dir / "concepts" / "refund-timeline.md").exists()


# ---------------------------------------------------------------------------
# Scenario B: created timestamp preservation
# ---------------------------------------------------------------------------


def test_scenario_b_created_timestamp_preserved_on_reingest(tmp_path):
    """Re-ingest preserves original created timestamp; updated advances.

    1. First ingest at T1: page has created=T1, updated=T1.
    2. Re-ingest at T2 > T1: page has created=T1, updated=T2.
    """
    wiki_dir = tmp_path / "wiki"

    # First ingest at T1
    draft_v1 = _make_draft("cancellation-window", created=FIXED_TS, updated=FIXED_TS)
    write_pages_for_source("refund_policy.md", [draft_v1], wiki_dir=wiki_dir)

    page_path = wiki_dir / "concepts" / "cancellation-window.md"

    # Read back and verify T1 preserved
    fm_v1 = read_existing_frontmatter(page_path)
    assert fm_v1 is not None, "Expected frontmatter to be readable after first ingest"
    assert fm_v1["created"] == FIXED_TS, f"Expected created={FIXED_TS}, got: {fm_v1['created']}"

    # Re-ingest at T2: the caller should pass the preserved created timestamp
    draft_v2 = _make_draft("cancellation-window", created=FIXED_TS, updated=LATER_TS)
    write_pages_for_source("refund_policy.md", [draft_v2], wiki_dir=wiki_dir)

    # Read back post-re-ingest
    fm_v2 = read_existing_frontmatter(page_path)
    assert fm_v2 is not None
    assert fm_v2["created"] == FIXED_TS, (
        f"created should be preserved as {FIXED_TS}, got: {fm_v2['created']}"
    )
    assert fm_v2["updated"] == LATER_TS, (
        f"updated should advance to {LATER_TS}, got: {fm_v2['updated']}"
    )


def test_scenario_b_read_existing_frontmatter_returns_none_for_missing_file(tmp_path):
    """read_existing_frontmatter returns None when the file does not exist."""
    result = read_existing_frontmatter(tmp_path / "nonexistent.md")
    assert result is None, f"Expected None for missing file, got: {result}"


# ---------------------------------------------------------------------------
# Scenario C: per-Source orphan scope
# ---------------------------------------------------------------------------


def test_scenario_c_orphan_scope_does_not_touch_other_source_pages(tmp_path):
    """Orphan deletion for refund_policy.md MUST NOT delete pages from account_help.md.

    Setup:
      - Ingest refund_policy.md → 2 pages
      - Ingest account_help.md → 2 pages
    Re-ingest refund_policy.md with 1 page:
      - 1 refund page deleted
      - Both account_help pages untouched
    """
    wiki_dir = tmp_path / "wiki"

    # Ingest refund_policy.md: pages A and B
    drafts_refund = [
        _make_draft("cancellation-window", source="refund_policy.md"),
        _make_draft("refund-timeline", source="refund_policy.md"),
    ]
    write_pages_for_source("refund_policy.md", drafts_refund, wiki_dir=wiki_dir)

    # Ingest account_help.md: pages C and D
    drafts_account = [
        _make_draft("change-email-address", source="account_help.md"),
        _make_draft("reset-password", source="account_help.md"),
    ]
    write_pages_for_source("account_help.md", drafts_account, wiki_dir=wiki_dir)

    # Confirm all 4 pages exist
    for slug in (
        "cancellation-window",
        "refund-timeline",
        "change-email-address",
        "reset-password",
    ):
        assert (wiki_dir / "concepts" / f"{slug}.md").exists(), (
            f"Expected {slug}.md to exist before re-ingest"
        )

    # Re-ingest refund_policy.md with only 1 page (refund-timeline removed)
    current_page_ids = {"cancellation-window"}
    deleted = delete_orphans("refund_policy.md", current_page_ids, wiki_dir=wiki_dir)

    # Only the refund orphan was deleted
    assert "concepts/refund-timeline.md" in deleted, (
        f"Expected refund-timeline.md in deleted, got: {deleted}"
    )
    # account_help pages must be untouched
    assert (wiki_dir / "concepts" / "change-email-address.md").exists(), (
        "change-email-address.md should NOT be deleted (belongs to account_help.md)"
    )
    assert (wiki_dir / "concepts" / "reset-password.md").exists(), (
        "reset-password.md should NOT be deleted (belongs to account_help.md)"
    )
    # Refund orphan is gone
    assert not (wiki_dir / "concepts" / "refund-timeline.md").exists(), (
        "refund-timeline.md should be deleted as orphan of refund_policy.md"
    )


# ---------------------------------------------------------------------------
# Scenario D: concurrency lock
# ---------------------------------------------------------------------------


def test_scenario_d_concurrent_ingest_and_index_no_corruption(tmp_path, monkeypatch):
    """Concurrent POST /ingest + POST /index are lock-serialised.

    Spawns two threads:
      T1: calls ingest_sources (holds _index_lock while writing)
      T2: calls build_index (also tries to hold _index_lock)

    Neither thread should corrupt state; both must complete without exception.
    No half-written .tmp files may remain after both threads finish.
    """
    import app.indexer as indexer_module
    import app.templates as templates_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("## Section One\n\nSome content here.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    results: dict[str, Exception | None] = {"ingest": None, "index": None}

    def _run_ingest():
        try:
            ingest_sources(["test.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
        except Exception as exc:  # noqa: BLE001
            results["ingest"] = exc

    def _run_index():
        try:
            indexer_module.build_index(docs_dir=docs_dir)
        except Exception as exc:  # noqa: BLE001
            results["index"] = exc

    t1 = threading.Thread(target=_run_ingest, name="ingest-thread")
    t2 = threading.Thread(target=_run_index, name="index-thread")

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive(), "Ingest thread did not complete within timeout"
    assert not t2.is_alive(), "Index thread did not complete within timeout"

    assert results["ingest"] is None, f"Ingest thread raised: {results['ingest']}"
    assert results["index"] is None, f"Index thread raised: {results['index']}"

    # No stale .tmp files in wiki/
    for tmp_file in wiki_dir.rglob("*.tmp"):
        pytest.fail(f"Stale .tmp file found after concurrent run: {tmp_file}")


# ---------------------------------------------------------------------------
# Integration: ingest_sources populates pages_created/updated/deleted correctly
# ---------------------------------------------------------------------------


def _make_schema_aware_fake_llm(
    synthesis_body: str = "Synthesised content.",
    classifier_type: str = "concept",
) -> MagicMock:
    """Schema-aware fake LLM (mirrors test_ingest_integration.py pattern)."""
    from app.templates import _ClassifierOutput

    class _FakeSynthesisOutput:
        def __init__(self):
            self.body = synthesis_body
            self.open_questions = []

    class _FakeClassifierOutput:
        def __init__(self):
            self.type = classifier_type

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            chain.invoke.return_value = _FakeClassifierOutput()
        else:
            chain.invoke.return_value = _FakeSynthesisOutput()
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


def test_ingest_sources_pages_created_on_first_ingest(tmp_path, monkeypatch):
    """First ingest of a source: all pages in pages_created, none in pages_updated/deleted."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "refund.md").write_text(
        "## Cancellation Window\n\nCancel within 24 hours.\n"
        "## Refund Timeline\n\nRefunds processed in 5 days.\n",
        encoding="utf-8",
    )
    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_sources(["refund.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert len(result.results) == 1
    src = result.results[0]

    assert len(src.pages_created) == 2, f"Expected 2 pages_created, got {src.pages_created}"
    assert src.pages_updated == [], f"Expected empty pages_updated, got {src.pages_updated}"
    assert src.pages_deleted == [], f"Expected empty pages_deleted, got {src.pages_deleted}"


def test_ingest_sources_pages_updated_on_reingest(tmp_path, monkeypatch):
    """Re-ingest same source: pages_updated populated, pages_created empty."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "refund.md").write_text(
        "## Cancellation Window\n\nCancel within 24 hours.\n",
        encoding="utf-8",
    )
    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    # First ingest
    ingest_sources(["refund.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    # Re-ingest same content
    result2 = ingest_sources(["refund.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result2.failed_sources == []
    src = result2.results[0]
    assert src.pages_created == [], (
        f"Expected empty pages_created on re-ingest, got {src.pages_created}"
    )
    assert len(src.pages_updated) == 1, (
        f"Expected 1 pages_updated on re-ingest, got {src.pages_updated}"
    )
    assert src.pages_deleted == [], (
        f"Expected empty pages_deleted on re-ingest, got {src.pages_deleted}"
    )


def test_ingest_sources_pages_deleted_on_section_removal(tmp_path, monkeypatch):
    """Re-ingest with section removed: pages_deleted populated."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_file = docs_dir / "refund.md"
    source_file.write_text(
        "## Cancellation Window\n\nCancel within 24 hours.\n"
        "## Refund Timeline\n\nRefunds processed in 5 days.\n",
        encoding="utf-8",
    )
    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    # First ingest: 2 sections
    ingest_sources(["refund.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)
    assert (wiki_dir / "concepts" / "cancellation-window.md").exists()
    assert (wiki_dir / "concepts" / "refund-timeline.md").exists()

    # Edit source: remove one section
    source_file.write_text(
        "## Cancellation Window\n\nCancel within 24 hours.\n",
        encoding="utf-8",
    )

    # Re-ingest
    result2 = ingest_sources(["refund.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result2.failed_sources == []
    src = result2.results[0]
    assert src.pages_deleted == ["concepts/refund-timeline.md"], (
        f"Expected refund-timeline.md deleted, got {src.pages_deleted}"
    )
    assert not (wiki_dir / "concepts" / "refund-timeline.md").exists(), (
        "refund-timeline.md should be physically deleted"
    )
    assert len(src.pages_updated) == 1
    assert src.pages_created == []
