"""Tests for the Phase 6 Slice 6-1 indexer qa-filter (PRD #78 Q8d).

These tests cover the 8-case truth table from issue #79: ``wiki/qa/*.md``
pages join the BM25 corpus only when ``frontmatter.status == "live"``; any
other value (or missing status) is treated as an orphan and either silently
skipped (``draft``) or skipped with a ``qa_invalid_status`` log entry (every
other value, including curator typos like ``"Live"`` with a capital L).
Entity and concept pages bypass the filter to preserve Phase 3 behaviour.

The orphan-visibility defence is the **indexer layer** of the three-layer
defence (PRD #78 §"Orphan-visibility three-layer defence"); the filing and
lint layers ship in slices 6-2 / 6-4.

External-behaviour testing only: each test sets up ``wiki/entities``,
``wiki/concepts``, and ``wiki/qa`` under ``tmp_path`` with the relevant
fixtures, runs ``build_index()`` with patched paths, and asserts on the
indexed Section list (``indexer.sections``) plus the contents of
``wiki/log.md`` (no direct calls into ``_passes_index_filter``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

WIKI_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wiki"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_wiki(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create ``wiki/{entities,concepts,qa}`` under ``tmp_path``.

    Returns ``(wiki_dir, entities_dir, concepts_dir, qa_dir)``.
    """
    wiki_dir = tmp_path / "wiki"
    entities_dir = wiki_dir / "entities"
    concepts_dir = wiki_dir / "concepts"
    qa_dir = wiki_dir / "qa"
    entities_dir.mkdir(parents=True)
    concepts_dir.mkdir(parents=True)
    qa_dir.mkdir(parents=True)
    return wiki_dir, entities_dir, concepts_dir, qa_dir


def _patch_indexer(monkeypatch, tmp_path: Path, wiki_dir: Path) -> None:
    """Point the indexer + logger at the tmp wiki and rebuild SOURCE_DIRS.

    Mirrors the pattern used in existing ``test_indexing.py`` tests so the
    new filter tests follow the same conventions (no machine-specific paths,
    full isolation under ``tmp_path``).
    """
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


def _write_qa_page(
    qa_dir: Path, slug: str, frontmatter_yaml: str, body: str = "Answer body.\n"
) -> Path:
    """Write a wiki/qa/<slug>.md fixture with the given frontmatter YAML block.

    ``frontmatter_yaml`` is the content **between** the ``---\\n`` fences (no
    fences). Each fixture has a single H2 so the section has indexable content.
    """
    content = f"---\n{frontmatter_yaml}---\n\n## {slug.replace('-', ' ').title()}\n\n{body}"
    path = qa_dir / f"{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Case 1: entity page with valid frontmatter is indexed
# ---------------------------------------------------------------------------


def test_entity_page_bypasses_filter(tmp_path, monkeypatch):
    """Entity pages always pass the filter regardless of ``status``."""
    import app.indexer as indexer_module

    wiki_dir, entities_dir, _, _ = _setup_wiki(tmp_path)
    src = WIKI_FIXTURES / "entities" / "acme-shop.md"
    (entities_dir / "acme-shop.md").write_bytes(src.read_bytes())

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "acme-shop" in files, f"Entity page must be indexed (bypass filter), got files: {files}"


# ---------------------------------------------------------------------------
# Case 2: concept page with valid frontmatter is indexed
# ---------------------------------------------------------------------------


def test_concept_page_bypasses_filter(tmp_path, monkeypatch):
    """Concept pages always pass the filter."""
    import app.indexer as indexer_module

    wiki_dir, _, concepts_dir, _ = _setup_wiki(tmp_path)
    src = WIKI_FIXTURES / "concepts" / "cancellation-window.md"
    (concepts_dir / "cancellation-window.md").write_bytes(src.read_bytes())

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "cancellation-window" in files, (
        f"Concept page must be indexed (bypass filter), got files: {files}"
    )


# ---------------------------------------------------------------------------
# Case 3: qa page with status: live is indexed
# ---------------------------------------------------------------------------


def test_qa_live_is_indexed(tmp_path, monkeypatch):
    """``wiki/qa/*.md`` with ``status: live`` joins the BM25 corpus."""
    import app.indexer as indexer_module

    wiki_dir, _, _, qa_dir = _setup_wiki(tmp_path)
    _write_qa_page(
        qa_dir,
        "how-to-cancel",
        frontmatter_yaml=(
            "id: how-to-cancel\n"
            "type: qa\n"
            'created: "2026-05-27T00:00:00Z"\n'
            'updated: "2026-05-27T00:00:00Z"\n'
            "sources:\n"
            "  - refund-policy#cancellation-window\n"
            "status: live\n"
            "open_questions: []\n"
            'question: "How do I cancel my order?"\n'
            "count: 1\n"
        ),
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "how-to-cancel" in files, f"qa page with status:live must be indexed, got files: {files}"


# ---------------------------------------------------------------------------
# Case 4: qa page with status: draft is silently skipped
# ---------------------------------------------------------------------------


def test_qa_draft_is_skipped_silently(tmp_path, monkeypatch):
    """``status: draft`` qa pages are skipped and produce NO ``qa_invalid_status`` log.

    Draft pages are a healthy intermediate state in the two-stage curation
    lifecycle (PRD #78 Q1) — they accumulate from /chat side-effects and wait
    for curator promotion. They must NOT clutter the log.
    """
    import app.indexer as indexer_module

    wiki_dir, _, _, qa_dir = _setup_wiki(tmp_path)
    _write_qa_page(
        qa_dir,
        "draft-question",
        frontmatter_yaml=(
            "id: draft-question\n"
            "type: qa\n"
            'created: "2026-05-27T00:00:00Z"\n'
            'updated: "2026-05-27T00:00:00Z"\n'
            "sources: []\n"
            "status: draft\n"
            "open_questions: []\n"
            'question: "Some pending question?"\n'
            "count: 1\n"
        ),
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "draft-question" not in files, (
        f"qa page with status:draft must NOT be indexed, got files: {files}"
    )

    log_path = wiki_dir / "log.md"
    log_content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    assert "qa_invalid_status" not in log_content, (
        "draft is a valid lifecycle state; no qa_invalid_status log expected.\n"
        f"Log content:\n{log_content}"
    )


# ---------------------------------------------------------------------------
# Case 5: qa page with status: Live (capital L) is skipped + logged
# ---------------------------------------------------------------------------


def test_qa_capital_live_is_skipped_and_logged(tmp_path, monkeypatch):
    """Curator typo (capital L) is the canonical orphan zombie case (PRD #78 Q8d).

    Must be skipped from indexing AND surfaced via ``qa_invalid_status``
    so the curator can find the broken page from log inspection.
    """
    import app.indexer as indexer_module

    wiki_dir, _, _, qa_dir = _setup_wiki(tmp_path)
    _write_qa_page(
        qa_dir,
        "typo-status",
        frontmatter_yaml=(
            "id: typo-status\n"
            "type: qa\n"
            'created: "2026-05-27T00:00:00Z"\n'
            'updated: "2026-05-27T00:00:00Z"\n'
            "sources: []\n"
            "status: Live\n"  # <-- curator typo: capital L
            "open_questions: []\n"
            'question: "Will this be indexed?"\n'
            "count: 1\n"
        ),
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "typo-status" not in files, f"capital-L status must NOT be indexed, got files: {files}"

    log_path = wiki_dir / "log.md"
    log_content = log_path.read_text(encoding="utf-8")
    assert "qa_invalid_status" in log_content, (
        f"Expected qa_invalid_status log entry for capital-L status.\nLog content:\n{log_content}"
    )
    assert "file=typo-status.md" in log_content, (
        f"qa_invalid_status entry must include file=<name>.\nLog content:\n{log_content}"
    )
    # repr-of-value: 'Live' -> "'Live'"
    assert "status='Live'" in log_content, (
        f"qa_invalid_status entry must include repr of the invalid status value.\n"
        f"Log content:\n{log_content}"
    )


# ---------------------------------------------------------------------------
# Case 6: qa page with missing status field is skipped + logged
# ---------------------------------------------------------------------------


def test_qa_missing_status_is_skipped_and_logged(tmp_path, monkeypatch):
    """Missing ``status`` key is a structural defect; gets the same defence."""
    import app.indexer as indexer_module

    wiki_dir, _, _, qa_dir = _setup_wiki(tmp_path)
    _write_qa_page(
        qa_dir,
        "no-status",
        frontmatter_yaml=(
            "id: no-status\n"
            "type: qa\n"
            'created: "2026-05-27T00:00:00Z"\n'
            'updated: "2026-05-27T00:00:00Z"\n'
            "sources: []\n"
            "open_questions: []\n"
            'question: "Status field missing entirely?"\n'
            "count: 1\n"
            # NB: status: <value> is intentionally absent
        ),
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "no-status" not in files, f"missing status must NOT be indexed, got files: {files}"

    log_path = wiki_dir / "log.md"
    log_content = log_path.read_text(encoding="utf-8")
    assert "qa_invalid_status" in log_content, (
        f"Expected qa_invalid_status log entry for missing status.\nLog content:\n{log_content}"
    )
    assert "file=no-status.md" in log_content, (
        f"qa_invalid_status entry must include file=<name>.\nLog content:\n{log_content}"
    )
    # repr of None is the literal string "None"
    assert "status=None" in log_content, (
        f"qa_invalid_status entry must include repr(None) when status absent.\n"
        f"Log content:\n{log_content}"
    )


# ---------------------------------------------------------------------------
# Case 7: qa page with malformed frontmatter (page_sections == []) is
#         skipped without a qa_invalid_status entry (parse_warning covers it)
# ---------------------------------------------------------------------------


def test_qa_malformed_frontmatter_skipped_without_duplicate_log(tmp_path, monkeypatch):
    """Empty/zero-metadata pages skip silently — ``parse_warning`` already covers.

    PRD #78 §"Orphan-visibility three-layer defence" — the indexer layer must
    not double-log when ``parse_markdown`` already emitted a ``parse_warning``
    for the same file. The malformed-frontmatter scenario produces empty
    metadata (``yaml.YAMLError`` path in ``parse_markdown``), so the qa-filter
    must skip silently to avoid noise.

    Contract under test: when ``metadata == {}`` (because frontmatter parse
    failed or no frontmatter was present), the filter returns False without
    emitting ``qa_invalid_status`` for that file.
    """
    import app.indexer as indexer_module

    wiki_dir, _, _, qa_dir = _setup_wiki(tmp_path)
    # Malformed YAML: missing closing fence makes parse_markdown's
    # raw.index("\n---\n", 4) raise ValueError -> parse_warning, metadata={}
    bad = qa_dir / "broken.md"
    bad.write_text("---\nkey: value\n\nbody but no closing fence\n", encoding="utf-8")

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    log_path = wiki_dir / "log.md"
    log_content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    # parse_warning is expected (the YAML failed); qa_invalid_status MUST NOT
    # be emitted for the same file — that's the double-log we are preventing.
    parse_warning_lines = [
        ln for ln in log_content.splitlines() if "parse_warning" in ln and "broken.md" in ln
    ]
    invalid_status_lines = [
        ln for ln in log_content.splitlines() if "qa_invalid_status" in ln and "broken.md" in ln
    ]
    assert parse_warning_lines, (
        f"Expected parse_warning entry for malformed frontmatter.\nLog:\n{log_content}"
    )
    assert not invalid_status_lines, (
        "qa_invalid_status must NOT also fire for a file that parse_warning "
        "already covered.\n"
        f"Log:\n{log_content}"
    )


# ---------------------------------------------------------------------------
# Case 8: qa page with status: stale is skipped + logged
# ---------------------------------------------------------------------------


def test_qa_stale_is_skipped_and_logged(tmp_path, monkeypatch):
    """``status: stale`` is a forward-compat value but NOT a live indexable state.

    Phase 6 reserves ``stale`` and ``superseded`` in the schema Literal but
    only ``live`` actually admits the page to the corpus. The indexer-layer
    defence logs every non-live/non-draft value so the curator can spot a
    re-ingest that left the qa page stranded.
    """
    import app.indexer as indexer_module

    wiki_dir, _, _, qa_dir = _setup_wiki(tmp_path)
    _write_qa_page(
        qa_dir,
        "stale-answer",
        frontmatter_yaml=(
            "id: stale-answer\n"
            "type: qa\n"
            'created: "2026-05-20T00:00:00Z"\n'
            'updated: "2026-05-20T00:00:00Z"\n'
            "sources:\n"
            "  - refund-policy#cancellation-window\n"
            "status: stale\n"
            "open_questions: []\n"
            'question: "Stale forward-compat?"\n'
            "count: 1\n"
        ),
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    files = {s.file for s in indexer_module.sections}
    assert "stale-answer" not in files, f"status:stale must NOT be indexed, got files: {files}"

    log_path = wiki_dir / "log.md"
    log_content = log_path.read_text(encoding="utf-8")
    assert "qa_invalid_status" in log_content, (
        f"Expected qa_invalid_status log entry for stale.\nLog content:\n{log_content}"
    )
    assert "file=stale-answer.md" in log_content, (
        f"qa_invalid_status entry must include file=<name>.\nLog content:\n{log_content}"
    )
    assert "status='stale'" in log_content, (
        f"qa_invalid_status entry must include repr('stale').\nLog content:\n{log_content}"
    )


# ---------------------------------------------------------------------------
# Cross-cutting: mixed scenario — wiki_layer_empty should NOT fire when only
# entities + concepts have content, regardless of qa-page filtering.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qa_status", ["live", "draft", "Live", "stale"])
def test_wiki_layer_empty_unaffected_by_qa_filter(qa_status, tmp_path, monkeypatch):
    """The Phase 4 ``wiki_layer_empty`` log entry semantics must not regress.

    Issue #79 AC: "Existing ``wiki_layer_empty`` emission semantics unchanged".
    With one entity page present and one qa page (whatever the status) the
    indexer should index at least the entity and NOT emit wiki_layer_empty.
    """
    import app.indexer as indexer_module

    wiki_dir, entities_dir, _, qa_dir = _setup_wiki(tmp_path)

    src = WIKI_FIXTURES / "entities" / "acme-shop.md"
    (entities_dir / "acme-shop.md").write_bytes(src.read_bytes())

    _write_qa_page(
        qa_dir,
        "probe",
        frontmatter_yaml=(
            "id: probe\n"
            "type: qa\n"
            'created: "2026-05-27T00:00:00Z"\n'
            'updated: "2026-05-27T00:00:00Z"\n'
            "sources: []\n"
            f"status: {qa_status}\n"
            "open_questions: []\n"
            'question: "Probe?"\n'
            "count: 1\n"
        ),
    )

    _patch_indexer(monkeypatch, tmp_path, wiki_dir)
    indexer_module.build_index()

    log_path = wiki_dir / "log.md"
    log_content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    assert "wiki_layer_empty" not in log_content, (
        "wiki_layer_empty must NOT fire when entities has at least one page, "
        "even if all qa pages are filtered out.\n"
        f"Log content:\n{log_content}"
    )
