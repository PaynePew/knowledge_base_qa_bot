"""Tests for the Operator Console live artifact-node counts (issue #559 A1).

Covers two cheap, read-only indexer helpers:
  - ``wiki_page_count()`` — recursive file count over entities/concepts/qa,
    re-reading ``WIKI_DIR`` at call time (not the frozen ``SOURCE_DIRS``
    snapshot) so a monkeypatched ``WIKI_DIR`` alone is enough for isolation.
  - ``indexed_sections_count()`` — a pure read of the persisted
    ``.kb/index.json``'s ``stats.sections_indexed`` field; never mutates the
    in-memory ``sections`` list and never logs (unlike ``load_index_json``).

Hermetic: no OPENAI_API_KEY, no real docs/wiki/.kb directories.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# wiki_page_count
# ---------------------------------------------------------------------------


def test_wiki_page_count_empty_wiki_is_zero(tmp_path, monkeypatch):
    """No entities/concepts/qa subdirs on disk yet -> 0."""
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")

    assert indexer_module.wiki_page_count() == 0


def test_wiki_page_count_sums_all_three_subdirs(tmp_path, monkeypatch):
    """Counts files across entities/, concepts/, and qa/ together."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "qa").mkdir(parents=True)
    (wiki_dir / "entities" / "acme.md").write_text("# Acme\n", encoding="utf-8")
    (wiki_dir / "concepts" / "refunds.md").write_text("# Refunds\n", encoding="utf-8")
    (wiki_dir / "qa" / "q1.md").write_text("# Q1\n", encoding="utf-8")
    (wiki_dir / "qa" / "q2.md").write_text("# Q2\n", encoding="utf-8")

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.wiki_page_count() == 4


def test_wiki_page_count_excludes_root_meta_files(tmp_path, monkeypatch):
    """wiki/index.md, log.md etc. at the wiki root are never counted (not scanned)."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    wiki_dir_files = ["index.md", "log.md", "hot.md", "lint-report.md", "README.md"]
    for name in wiki_dir_files:
        (wiki_dir / name).write_text("meta", encoding="utf-8")
    (wiki_dir / "entities" / "acme.md").write_text("# Acme\n", encoding="utf-8")

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.wiki_page_count() == 1


def test_wiki_page_count_does_not_gate_on_status(tmp_path, monkeypatch):
    """A draft/quarantined qa page still counts — no frontmatter parsing (cheap-path scope)."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "qa").mkdir(parents=True)
    (wiki_dir / "qa" / "draft.md").write_text(
        "---\nstatus: draft\n---\n# Draft\n", encoding="utf-8"
    )

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.wiki_page_count() == 1


def test_wiki_page_count_follows_monkeypatched_wiki_dir_not_frozen_source_dirs(
    tmp_path, monkeypatch
):
    """WIKI_DIR monkeypatch alone (no SOURCE_DIRS patch) is enough for isolation."""
    import app.indexer as indexer_module

    real_source_dirs = indexer_module.SOURCE_DIRS  # left untouched deliberately
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "concepts" / "one.md").write_text("# One\n", encoding="utf-8")

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.wiki_page_count() == 1
    assert real_source_dirs == indexer_module.SOURCE_DIRS  # never mutated


# ---------------------------------------------------------------------------
# indexed_sections_count
# ---------------------------------------------------------------------------


def test_indexed_sections_count_missing_file_is_zero(tmp_path):
    """No .kb/index.json on disk yet -> 0, no error."""
    import app.indexer as indexer_module

    assert indexer_module.indexed_sections_count(tmp_path / ".kb" / "index.json") == 0


def test_indexed_sections_count_reads_persisted_stat(tmp_path):
    """Reads stats.sections_indexed straight from the persisted JSON."""
    import app.indexer as indexer_module

    index_path = tmp_path / ".kb" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps({"sections": [], "stats": {"sections_indexed": 7, "files_indexed": 3}}),
        encoding="utf-8",
    )

    assert indexer_module.indexed_sections_count(index_path) == 7


def test_indexed_sections_count_does_not_mutate_in_memory_sections(tmp_path, monkeypatch):
    """Unlike load_index_json, this never touches the module-level sections list."""
    import app.indexer as indexer_module

    index_path = tmp_path / ".kb" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "sections": [
                    {
                        "id": "x#h",
                        "file": "x.md",
                        "heading": "H",
                        "heading_path": "H",
                        "content": "c",
                        "tokens": ["c"],
                    }
                ],
                "stats": {"sections_indexed": 1, "files_indexed": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(indexer_module, "sections", [])

    indexer_module.indexed_sections_count(index_path)

    assert indexer_module.sections == []


def test_indexed_sections_count_uses_module_index_path_by_default(tmp_path, monkeypatch):
    """Called with no argument, reads the module-level INDEX_PATH (like load_index_json)."""
    import app.indexer as indexer_module

    index_path = tmp_path / ".kb" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps({"sections": [], "stats": {"sections_indexed": 5}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(indexer_module, "INDEX_PATH", index_path)

    assert indexer_module.indexed_sections_count() == 5
