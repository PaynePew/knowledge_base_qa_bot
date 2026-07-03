"""Tests for the ADR-0029 quarantine gate in ``_passes_index_filter`` (issue #405).

The indexer's status gate was qa-only (non-qa pages passed through unchanged),
so an entities/concepts page written with ``status: failed_grounding`` by the
fail-soft grounding check (ADR-0004) sat in the BM25 corpus and was
retrievable/citable by ``/chat``. This is the vertical-axis gap ADR-0029
names: the chat-time grounding check verifies answer-vs-page, never page-vs-
Source, so a machine-verified-ungrounded page was never caught downstream.

ADR-0029 decision 1 (Invariant, CODING_STANDARD §11 "Alias & quarantine
drift"): the Section Index never admits a page whose ``status`` is
``failed_grounding``, regardless of which wiki subdir it lives under. A page
with no ``status`` field at all is treated as live (legacy pass-through).

External-behaviour testing only, mirroring ``test_indexer_qa_filter.py``'s
conventions: each test sets up ``wiki/{entities,concepts,qa}`` under
``tmp_path``, runs ``build_index()`` with patched paths, and asserts on
``indexer.sections`` (no direct calls into ``_passes_index_filter``).

The concept/entity fixtures are sentinel-first byte-shape (CODING_STANDARD
§6.5 fixture fidelity): they begin with the real ``/ingest``-written HTML
comment and carry the real ``grounding_failure`` frontmatter block, mirroring
the live pages this ADR quarantines (see ``wiki/concepts/accepted-cards.md``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

WIKI_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wiki"


# ---------------------------------------------------------------------------
# Helpers (mirror test_indexer_qa_filter.py's setup/patch pattern)
# ---------------------------------------------------------------------------


def _setup_wiki(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    qa_dir = wiki_dir / "qa"
    entities_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)
    qa_dir.mkdir(parents=True)
    return wiki_dir, entities_dir, concepts_dir, qa_dir


def _patch_indexer(monkeypatch, tmp_path: Path, wiki_dir: Path) -> None:
    import app.indexer as indexer_module
    import app.logger as logger_module

    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", wiki_dir / "log.md")
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts", wiki_dir / "qa"],
    )


# ---------------------------------------------------------------------------
# AC 1: a fixture entities/concepts page with status: failed_grounding
# produces zero Sections in a fresh build_index().
# ---------------------------------------------------------------------------


def test_failed_grounding_concept_page_produces_zero_sections(tmp_path, monkeypatch):
    """A concept page quarantined by ADR-0029 must not join the BM25 corpus."""
    import app.indexer as indexer_module

    wiki_dir, _, concepts_dir, _ = _setup_wiki(tmp_path)
    src = WIKI_FIXTURES / "concepts" / "quarantined-claim.md"
    (concepts_dir / "quarantined-claim.md").write_bytes(src.read_bytes())

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "quarantined-claim" not in files, (
        f"status:failed_grounding concept page must produce zero Sections, got files: {files}"
    )


def test_failed_grounding_entity_page_produces_zero_sections(tmp_path, monkeypatch):
    """Quarantine is not concept-specific — an entity page is excluded too."""
    import app.indexer as indexer_module

    wiki_dir, entities_dir, _, _ = _setup_wiki(tmp_path)
    src = WIKI_FIXTURES / "entities" / "quarantined-entity.md"
    (entities_dir / "quarantined-entity.md").write_bytes(src.read_bytes())

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "quarantined-entity" not in files, (
        f"status:failed_grounding entity page must produce zero Sections, got files: {files}"
    )


# ---------------------------------------------------------------------------
# AC 2: a page with no status field is still indexed (legacy pass-through).
# ---------------------------------------------------------------------------


def test_concept_page_with_no_status_field_still_indexed(tmp_path, monkeypatch):
    """Legacy posture: absent ``status`` reads as live, not quarantined."""
    import app.indexer as indexer_module

    wiki_dir, _, concepts_dir, _ = _setup_wiki(tmp_path)
    (concepts_dir / "legacy-no-status.md").write_text(
        "# Legacy No Status\n\nThis page predates the status field entirely.\n",
        encoding="utf-8",
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "legacy-no-status" in files, (
        f"a page with no status field must still be indexed (legacy pass-through), "
        f"got files: {files}"
    )


def test_concept_page_with_other_status_still_indexed(tmp_path, monkeypatch):
    """Only ``failed_grounding`` quarantines — ``status: live`` is unaffected."""
    import app.indexer as indexer_module

    wiki_dir, _, concepts_dir, _ = _setup_wiki(tmp_path)
    src = WIKI_FIXTURES / "concepts" / "cancellation-window.md"
    (concepts_dir / "cancellation-window.md").write_bytes(src.read_bytes())

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "cancellation-window" in files, (
        f"status:live concept page must still be indexed, got files: {files}"
    )


# ---------------------------------------------------------------------------
# AC 3: qa gate behavior is unchanged (live in, draft out, invalid logged) —
# covered by test_indexer_qa_filter.py already; this test only proves the new
# quarantine check composes correctly ahead of the qa-specific gate: a qa page
# with status: live still passes even after the failed_grounding check runs.
# ---------------------------------------------------------------------------


def test_qa_live_page_unaffected_by_quarantine_check(tmp_path, monkeypatch):
    import app.indexer as indexer_module

    wiki_dir, _, _, qa_dir = _setup_wiki(tmp_path)
    content = (
        "---\n"
        "id: still-live\n"
        "type: qa\n"
        'created: "2026-05-27T00:00:00Z"\n'
        'updated: "2026-05-27T00:00:00Z"\n'
        "sources:\n"
        "  - refund-policy#cancellation-window\n"
        "status: live\n"
        "open_questions: []\n"
        'question: "Still live after the quarantine check runs first?"\n'
        "count: 1\n"
        "---\n\n"
        "## Still Live\n\nAnswer body.\n"
    )
    (qa_dir / "still-live.md").write_text(content, encoding="utf-8")

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "still-live" in files, f"qa status:live must still be indexed, got files: {files}"


@pytest.mark.parametrize("subdir", ["entities", "concepts", "qa"])
def test_failed_grounding_excluded_regardless_of_subdir(subdir, tmp_path, monkeypatch):
    """ADR-0029's invariant is corpus-wide, not qa-specific — parametrized proof
    that ``status: failed_grounding`` is excluded from every whitelisted subdir."""
    import app.indexer as indexer_module

    wiki_dir, entities_dir, concepts_dir, qa_dir = _setup_wiki(tmp_path)
    target_dir = {"entities": entities_dir, "concepts": concepts_dir, "qa": qa_dir}[subdir]
    content = (
        "---\n"
        "id: subdir-quarantine-probe\n"
        "status: failed_grounding\n"
        "---\n\n"
        "## Probe\n\nBody text.\n"
    )
    (target_dir / "subdir-quarantine-probe.md").write_text(content, encoding="utf-8")

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "subdir-quarantine-probe" not in files, (
        f"status:failed_grounding must be excluded from wiki/{subdir}/, got files: {files}"
    )
