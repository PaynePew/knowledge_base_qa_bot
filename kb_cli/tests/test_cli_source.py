"""Integration tests for the ``kb source`` command group (issue #606, ADR-0041).

``kb source retire|restore|rename|trash`` wrap ``markdown_kb.app.
source_lifecycle``'s ``retire``/``restore``/``rename``/``list_trash`` directly
(mirrors ``kb alias add``'s Direct-class, human-surfaces-only rationale — MCP
sees Source Trash state but writes none of it, ADR-0041 Invariant).

Tests use typer's CliRunner in-process against real docs/ / wiki/ / .trash
fixtures, with ``markdown_kb.app.source_lifecycle``'s module-level
``DOCS_DIR`` / ``WIKI_DIR`` / ``TRASH_DIR`` monkeypatched per test — this
suite's autouse conftest (``kb_cli/tests/conftest.py``) only redirects the
indexer/logger paths, not ``source_lifecycle``'s own (separate module-level
bindings imported from ``._paths``); mirrors ``gateway/tests/
test_source_lifecycle_routes.py``'s ``sources_env`` fixture. ``kb source
rename`` additionally mocks ``markdown_kb.app.indexer.build_index`` (mirrors
``test_cli_qa.py``'s ``test_qa_promote_flips_status_and_reindexes`` — indexer
``SOURCE_DIRS`` is bound at import time to the production wiki/ subdirs and
is not redirected by this suite's conftest).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()


def _page_text(slug: str, sources: list[str]) -> str:
    sources_block = "".join(f"  - {s}\n" for s in sources) if sources else "  []\n"
    return (
        "---\n"
        f"id: {slug}\n"
        "type: concept\n"
        "created: '2026-01-01T00:00:00Z'\n"
        "updated: '2026-01-01T00:00:00Z'\n"
        f"sources:\n{sources_block}"
        "status: live\n"
        "open_questions: []\n"
        "source_hashes: {}\n"
        "---\n\n"
        f"# {slug}\n\nSome content.\n"
    )


@pytest.fixture()
def source_env(tmp_path: Path, monkeypatch):
    """Wire source_lifecycle's DOCS_DIR/WIKI_DIR/TRASH_DIR to tmp dirs."""
    import markdown_kb.app.source_lifecycle as sl_module

    docs = tmp_path / "docs"
    wiki = tmp_path / "wiki"
    trash = tmp_path / ".trash"
    docs.mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)

    monkeypatch.setattr(sl_module, "DOCS_DIR", docs)
    monkeypatch.setattr(sl_module, "WIKI_DIR", wiki)
    monkeypatch.setattr(sl_module, "TRASH_DIR", trash)

    return {"docs": docs, "wiki": wiki, "trash": trash}


def _retired_timestamp(trash_dir: Path, relpath: str) -> str:
    matches = list(trash_dir.glob(f"*/docs/{relpath}"))
    assert len(matches) == 1, f"expected exactly one trash entry for {relpath}, found {matches}"
    return matches[0].parent.parent.name


# ---------------------------------------------------------------------------
# kb source retire
# ---------------------------------------------------------------------------


def test_retire_not_found_exits_nonzero(source_env):
    from kb_cli.main import app

    result = runner.invoke(app, ["source", "retire", "missing.md"], input="y\n")
    assert result.exit_code != 0
    assert "no source found" in result.output.lower()


def test_retire_shows_impact_and_moves_the_file_on_confirmation(source_env):
    from kb_cli.main import app

    (source_env["docs"] / "policy.md").write_text(
        "# Policy\n\nRefunds take 7 days.\n", encoding="utf-8"
    )
    (source_env["wiki"] / "concepts" / "full-orphan.md").write_text(
        _page_text("full-orphan", ["policy.md#refunds"]), encoding="utf-8"
    )

    result = runner.invoke(app, ["source", "retire", "policy.md"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "full-orphan" in result.output
    assert "Retired docs/policy.md" in result.output
    assert not (source_env["docs"] / "policy.md").exists()
    assert len(list(source_env["trash"].glob("*/docs/policy.md"))) == 1


def test_retire_declined_confirmation_leaves_source_in_place(source_env):
    from kb_cli.main import app

    (source_env["docs"] / "policy.md").write_text("# Policy\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(app, ["source", "retire", "policy.md"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "cancelled" in result.output.lower()
    assert (source_env["docs"] / "policy.md").exists()


# ---------------------------------------------------------------------------
# kb source restore
# ---------------------------------------------------------------------------


def test_restore_no_trash_entry_exits_nonzero(source_env):
    from kb_cli.main import app

    result = runner.invoke(app, ["source", "restore", "20260101T000000000000Z", "missing.md"])
    assert result.exit_code != 0
    assert "no trash entry" in result.output.lower()


def test_restore_round_trips_a_retired_source(source_env):
    from kb_cli.main import app

    (source_env["docs"] / "policy.md").write_text("# Policy\n\nBody.\n", encoding="utf-8")
    result = runner.invoke(app, ["source", "retire", "policy.md"], input="y\n")
    assert result.exit_code == 0, result.output
    timestamp = _retired_timestamp(source_env["trash"], "policy.md")

    result = runner.invoke(app, ["source", "restore", timestamp, "policy.md"])
    assert result.exit_code == 0, result.output
    assert "Restored docs/policy.md" in result.output
    assert (source_env["docs"] / "policy.md").exists()


def test_restore_occupied_target_exits_nonzero(source_env):
    from kb_cli.main import app

    (source_env["docs"] / "policy.md").write_text("# Policy\n\nBody.\n", encoding="utf-8")
    result = runner.invoke(app, ["source", "retire", "policy.md"], input="y\n")
    assert result.exit_code == 0, result.output
    timestamp = _retired_timestamp(source_env["trash"], "policy.md")

    # A new Source landed at the original relpath since the retire.
    (source_env["docs"] / "policy.md").write_text("# New Policy\n\nBody.\n", encoding="utf-8")

    result = runner.invoke(app, ["source", "restore", timestamp, "policy.md"])
    assert result.exit_code != 0
    assert "occupied" in result.output.lower()


# ---------------------------------------------------------------------------
# kb source rename
# ---------------------------------------------------------------------------


def test_rename_not_found_exits_nonzero(source_env, monkeypatch):
    import markdown_kb.app.indexer as wiki_indexer

    from kb_cli.main import app

    monkeypatch.setattr(wiki_indexer, "build_index", lambda: (0, 0))

    result = runner.invoke(app, ["source", "rename", "missing.md", "new.md"])
    assert result.exit_code != 0
    assert "no source found" in result.output.lower()


def test_rename_repoints_citations_and_reindexes(source_env, monkeypatch):
    import markdown_kb.app.indexer as wiki_indexer

    from kb_cli.main import app

    (source_env["docs"] / "policy.md").write_text("# Policy\n\nBody.\n", encoding="utf-8")
    (source_env["wiki"] / "concepts" / "citer.md").write_text(
        _page_text("citer", ["policy.md#body"]), encoding="utf-8"
    )
    monkeypatch.setattr(wiki_indexer, "build_index", lambda: (3, 9))

    result = runner.invoke(app, ["source", "rename", "policy.md", "renamed.md"])
    assert result.exit_code == 0, result.output
    assert "Renamed docs/policy.md -> docs/renamed.md" in result.output
    assert "Repointed 1 page(s): citer" in result.output
    assert "Reindexed 3 file(s), 9 section(s)." in result.output
    assert not (source_env["docs"] / "policy.md").exists()
    assert (source_env["docs"] / "renamed.md").exists()

    citer_text = (source_env["wiki"] / "concepts" / "citer.md").read_text(encoding="utf-8")
    fm = yaml.safe_load(citer_text.split("---", 2)[1])
    assert fm["sources"] == ["renamed.md#body"]


def test_rename_collision_exits_nonzero_naming_the_collision(source_env, monkeypatch):
    import markdown_kb.app.indexer as wiki_indexer

    from kb_cli.main import app

    (source_env["docs"] / "policy.md").write_text("# Policy\n\nBody.\n", encoding="utf-8")
    (source_env["docs"] / "other.md").write_text("# Other\n\nBody.\n", encoding="utf-8")
    monkeypatch.setattr(wiki_indexer, "build_index", lambda: (0, 0))

    result = runner.invoke(app, ["source", "rename", "policy.md", "other.md"])
    assert result.exit_code != 0
    assert "other.md" in result.output


# ---------------------------------------------------------------------------
# kb source trash
# ---------------------------------------------------------------------------


def test_trash_empty_reports_empty(source_env):
    from kb_cli.main import app

    result = runner.invoke(app, ["source", "trash"])
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower()


def test_trash_lists_retired_entries(source_env):
    from kb_cli.main import app

    (source_env["docs"] / "policy.md").write_text("# Policy\n\nBody.\n", encoding="utf-8")
    result = runner.invoke(app, ["source", "retire", "policy.md"], input="y\n")
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["source", "trash"])
    assert result.exit_code == 0, result.output
    assert "policy.md" in result.output
