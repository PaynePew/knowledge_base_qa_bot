"""Unit tests for kb_mcp.freshness.reload_if_stale.

Verifies the mtime-gating contract:
  - First call loads the index (cold start).
  - Second call with the same mtime does NOT reload.
  - Call after mtime advances does reload.
  - Missing file returns False without loading.

These tests use monkeypatch / tmp_path only — no real index I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_index(path: Path) -> None:
    """Write a minimal valid index.json so load_index_json succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [],
        "stats": {
            "files_indexed": 0,
            "sections_indexed": 0,
            "avg_doc_len": 0.0,
            "doc_freq": {},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture: redirect freshness module's INDEX_PATH + reset _last_mtime
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_freshness_state(tmp_path, monkeypatch):
    """Reset module-level state before each test so tests are independent."""
    import kb_mcp.freshness as freshness_mod

    monkeypatch.setattr(freshness_mod, "_last_mtime", None)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reload_if_stale_returns_false_when_file_missing(tmp_path):
    """Returns False without calling load_index_json when the file is absent."""
    missing = tmp_path / ".kb" / "index.json"
    import kb_mcp.freshness as freshness_mod

    # The file does not exist — reload_if_stale must return False immediately
    # without touching _last_mtime or calling load_index_json.
    result = freshness_mod.reload_if_stale(missing)
    assert result is False
    assert freshness_mod._last_mtime is None


def test_reload_if_stale_loads_on_cold_start(tmp_path, monkeypatch):
    """Returns True and updates _last_mtime on the first call (cold start)."""
    index_path = tmp_path / ".kb" / "index.json"
    _write_minimal_index(index_path)

    import kb_mcp.freshness as freshness_mod

    load_calls: list[Path | None] = []

    def fake_load(path=None):
        load_calls.append(path)
        return (0, 0)

    monkeypatch.setattr("markdown_kb.app.indexer.load_index_json", fake_load)

    result = freshness_mod.reload_if_stale(index_path)

    assert result is True
    assert len(load_calls) == 1
    assert freshness_mod._last_mtime == index_path.stat().st_mtime


def test_reload_if_stale_no_reload_same_mtime(tmp_path, monkeypatch):
    """Returns False and does NOT reload when called again with the same mtime."""
    index_path = tmp_path / ".kb" / "index.json"
    _write_minimal_index(index_path)

    import kb_mcp.freshness as freshness_mod

    load_calls: list[Path | None] = []

    def fake_load(path=None):
        load_calls.append(path)
        return (0, 0)

    monkeypatch.setattr("markdown_kb.app.indexer.load_index_json", fake_load)

    # First call — loads.
    freshness_mod.reload_if_stale(index_path)
    assert len(load_calls) == 1

    # Second call with same mtime — must NOT reload.
    result = freshness_mod.reload_if_stale(index_path)
    assert result is False
    assert len(load_calls) == 1  # still 1; no second load


def test_reload_if_stale_reloads_after_mtime_change(tmp_path, monkeypatch):
    """Returns True and reloads when the mtime advances."""
    index_path = tmp_path / ".kb" / "index.json"
    _write_minimal_index(index_path)

    import kb_mcp.freshness as freshness_mod

    load_calls: list[Path | None] = []

    def fake_load(path=None):
        load_calls.append(path)
        return (0, 0)

    monkeypatch.setattr("markdown_kb.app.indexer.load_index_json", fake_load)

    # Cold start.
    freshness_mod.reload_if_stale(index_path)
    assert len(load_calls) == 1

    # Simulate mtime change by writing the file again (updates mtime).
    _write_minimal_index(index_path)
    # Force mtime to differ even if filesystem has 1-second granularity.
    old_mtime = freshness_mod._last_mtime
    assert old_mtime is not None
    # Manually set _last_mtime to a different value to simulate the file changing.
    monkeypatch.setattr(freshness_mod, "_last_mtime", old_mtime - 1.0)

    result = freshness_mod.reload_if_stale(index_path)
    assert result is True
    assert len(load_calls) == 2
