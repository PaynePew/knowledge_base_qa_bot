"""Tests for the Section Index staleness signal (issue #559 A2).

``index_stale()`` derives the Operator Console's Index-node fresh/stale
badge from **artifact state**, not click history (contrast the pre-existing
#173 ``indexStale`` Console flag, which flips on Ingest-click / clears on
Index-click and is untouched by this slice). It implements the CONTEXT.md
"Section Index" staleness semantic verbatim: the index "[g]oes stale when
[the whitelisted wiki subdirectories] change and /index has not been
re-run" — checked here via filesystem mtimes only (no LLM, no content diff).

Hermetic: no OPENAI_API_KEY, no real wiki/.kb directories (monkeypatches
WIKI_DIR / uses an explicit index_path arg, mirroring
test_indexer_artifact_counts.py's pattern for the sibling A1 helpers).
"""

from __future__ import annotations

import os
import time

# ---------------------------------------------------------------------------
# No wiki content at all — nothing to be stale relative to.
# ---------------------------------------------------------------------------


def test_index_stale_false_when_wiki_empty_and_no_index(tmp_path, monkeypatch):
    """No entities/concepts/qa subdirs, no .kb/index.json -> not stale."""
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")

    assert indexer_module.index_stale(tmp_path / ".kb" / "index.json") is False


def test_index_stale_false_when_wiki_dirs_exist_but_are_empty(tmp_path, monkeypatch):
    """entities/concepts/qa subdirs exist but hold no files -> not stale."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "qa").mkdir(parents=True)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.index_stale(tmp_path / ".kb" / "index.json") is False


# ---------------------------------------------------------------------------
# Wiki has content — the index-missing / index-present + mtime cases.
# ---------------------------------------------------------------------------


def test_index_stale_true_when_wiki_has_content_but_index_never_built(tmp_path, monkeypatch):
    """wiki/ has a page but .kb/index.json does not exist yet -> stale."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    (wiki_dir / "entities" / "acme.md").write_text("# Acme\n", encoding="utf-8")
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.index_stale(tmp_path / ".kb" / "index.json") is True


def test_index_stale_false_when_index_newer_than_every_wiki_file(tmp_path, monkeypatch):
    """index.json's mtime is after the newest wiki file's mtime -> fresh."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    page = wiki_dir / "entities" / "acme.md"
    page.write_text("# Acme\n", encoding="utf-8")

    index_path = tmp_path / ".kb" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{}", encoding="utf-8")

    # Force an unambiguous ordering regardless of filesystem mtime resolution.
    now = time.time()
    os.utime(page, (now - 10, now - 10))
    os.utime(index_path, (now, now))

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.index_stale(index_path) is False


def test_index_stale_true_when_a_wiki_file_is_newer_than_index(tmp_path, monkeypatch):
    """A wiki page edited after the last /index build -> stale."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    page = wiki_dir / "entities" / "acme.md"
    page.write_text("# Acme\n", encoding="utf-8")

    index_path = tmp_path / ".kb" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(index_path, (now - 10, now - 10))
    os.utime(page, (now, now))  # edited after the index was built

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.index_stale(index_path) is True


def test_index_stale_checks_all_three_subdirs(tmp_path, monkeypatch):
    """A newer file in concepts/ or qa/ (not just entities/) trips staleness too."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "concepts").mkdir(parents=True)
    page = wiki_dir / "concepts" / "refunds.md"
    page.write_text("# Refunds\n", encoding="utf-8")

    index_path = tmp_path / ".kb" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(index_path, (now - 10, now - 10))
    os.utime(page, (now, now))

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.index_stale(index_path) is True


def test_index_stale_ignores_wiki_root_meta_files(tmp_path, monkeypatch):
    """A newer wiki/log.md (root meta file, not entities/concepts/qa) does not
    trip staleness — mirrors wiki_page_count's exclusion of root meta files."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    page = wiki_dir / "entities" / "acme.md"
    page.write_text("# Acme\n", encoding="utf-8")

    index_path = tmp_path / ".kb" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(page, (now - 10, now - 10))
    os.utime(index_path, (now - 5, now - 5))
    log_file = wiki_dir / "log.md"
    log_file.write_text("meta\n", encoding="utf-8")
    os.utime(log_file, (now, now))  # newer than index, but a root meta file

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    assert indexer_module.index_stale(index_path) is False


def test_index_stale_uses_module_index_path_by_default(tmp_path, monkeypatch):
    """Called with no argument, reads the module-level INDEX_PATH (mirrors
    indexed_sections_count's default-arg contract)."""
    import app.indexer as indexer_module

    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")

    assert indexer_module.index_stale() is False
