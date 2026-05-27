"""Shared fixtures for the lint test sub-package.

Provides:
  - tmp_wiki_dir:   a temporary wiki/ directory with entities/ + concepts/ sub-dirs
  - tmp_docs_dir:   a temporary docs/ directory (empty by default)
  - lint_env:       monkeypatches WIKI_DIR, DOCS_DIR, LOG_PATH, and INDEX_PATH to
                    the tmp directories so lint.run_lint() and friends never touch
                    the real filesystem.  Composes on top of the parent conftest's
                    autouse `_redirect_paths_to_tmp` (which patches indexer and logger);
                    this fixture additionally patches `app.lint` once that module exists.

Pattern for subsequent slices
------------------------------
1. Add a page fixture to this conftest (or inline in the test file) that writes a
   frontmatter YAML block to ``tmp_wiki_dir / "entities" / "<slug>.md"`` (or
   ``concepts/``).
2. Add a source fixture that touches a file under ``tmp_docs_dir`` so existence
   checks produce the desired result.
3. Call ``run_lint(wiki_dir=tmp_wiki_dir, docs_dir=tmp_docs_dir, log_path=<path>)``
   and assert on the returned ``LintReport``.

The ``monkeypatch.setattr(lint_module, "WIKI_DIR", tmp_wiki_dir)`` pattern mirrors
the parent conftest's ``_redirect_paths_to_tmp``.  Import ``app.lint`` lazily inside
the fixture body so that pre-implementation test collection does not fail (the module
may not exist yet at collection time).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_wiki_dir(tmp_path: Path) -> Path:
    """Return a temporary wiki/ directory with entities/ and concepts/ sub-dirs.

    Future slices may populate sub-dirs via additional fixtures or inline helpers.
    """
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    return wiki


@pytest.fixture()
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Return an empty temporary docs/ directory.

    Tests that need real Source files should create them here (e.g.
    ``(tmp_docs_dir / "refund_policy.md").write_text("# Refund Policy\\n...")``)
    so that C11 existence checks see a controlled filesystem state.
    """
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    return docs


@pytest.fixture()
def lint_log_path(tmp_path: Path) -> Path:
    """Return a temporary log path for lint tests.

    Separate from the parent conftest's LOG_PATH so lint-generated log entries
    don't bleed into other test assertions.
    """
    log_dir = tmp_path / "wiki"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "log.md"


@pytest.fixture()
def lint_env(tmp_wiki_dir: Path, tmp_docs_dir: Path, lint_log_path: Path, monkeypatch):
    """Wire WIKI_DIR, DOCS_DIR, and log_path into lint module defaults.

    Returns a dict with keys ``wiki_dir``, ``docs_dir``, ``log_path`` so tests
    can pass them as kwargs to ``run_lint(...)`` without reimporting path constants.

    Subsequent slices can extend this fixture by importing additional module
    attributes to monkeypatch (e.g. ``monkeypatch.setattr(lint_module, "DOCS_DIR", ...)``
    once that constant is introduced).
    """
    return {
        "wiki_dir": tmp_wiki_dir,
        "docs_dir": tmp_docs_dir,
        "log_path": lint_log_path,
    }
