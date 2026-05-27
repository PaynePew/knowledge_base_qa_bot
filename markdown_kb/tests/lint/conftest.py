"""Shared fixtures for the lint test sub-package.

Provides:
  - tmp_wiki_dir:   a temporary wiki/ directory with entities/ + concepts/ sub-dirs
  - tmp_docs_dir:   a temporary docs/ directory (empty by default)
  - lint_log_path:  a temporary path for lint-emitted log.md entries
  - lint_env:       bundles the three paths into a dict so tests can splat them as
                    kwargs to ``run_lint(...)`` without importing constants. The
                    parent conftest's autouse ``_redirect_paths_to_tmp`` already
                    isolates indexer/logger paths; this fixture is purely a
                    convenience aggregator and does NOT itself monkeypatch.

Pattern for subsequent slices
------------------------------
1. Add a page fixture to this conftest (or inline in the test file) that writes a
   frontmatter YAML block to ``tmp_wiki_dir / "entities" / "<slug>.md"`` (or
   ``concepts/``).
2. Add a source fixture that touches a file under ``tmp_docs_dir`` so existence
   checks produce the desired result.
3. Call ``run_lint(wiki_dir=tmp_wiki_dir, docs_dir=tmp_docs_dir, log_path=<path>)``
   and assert on the returned ``LintResponse``.

If a future check requires monkeypatching module-level constants on ``app.lint``,
add a dedicated fixture that takes ``monkeypatch`` and calls
``monkeypatch.setattr(lint_module, "<ATTR>", <value>)`` — mirror the parent
conftest's ``_redirect_paths_to_tmp`` pattern. Import ``app.lint`` lazily inside
the fixture body if the constant may not exist at collection time.
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
def lint_env(tmp_wiki_dir: Path, tmp_docs_dir: Path, lint_log_path: Path) -> dict[str, Path]:
    """Bundle the three lint path fixtures into a single kwargs dict.

    Returns a dict with keys ``wiki_dir``, ``docs_dir``, ``log_path`` so tests
    can splat it into ``run_lint(**lint_env)`` without reimporting path constants.
    Does NOT monkeypatch — the parent conftest's ``_redirect_paths_to_tmp``
    already isolates indexer/logger globals; lint accepts these paths as kwargs.
    """
    return {
        "wiki_dir": tmp_wiki_dir,
        "docs_dir": tmp_docs_dir,
        "log_path": lint_log_path,
    }
