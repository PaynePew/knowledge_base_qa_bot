"""Unit tests for the committed-invariant guard helper (``conftest_support``).

The repo-root ``conftest.py`` uses these to snapshot/restore git-tracked invariant
files (notably ``.kb/index.json``) across a test session, so a live test that leaks
a real write path cannot leave the committed file mutated (#204). The session-scoped
fixture itself is awkward to assert on directly, so the restore logic lives in a
plain importable module and is unit-tested here on tmp files.
"""

from __future__ import annotations

from pathlib import Path

import conftest_support as guard


def test_read_bytes_or_none_returns_none_for_missing(tmp_path):
    assert guard.read_bytes_or_none(tmp_path / "absent.json") is None


def test_restore_is_noop_when_unchanged(tmp_path):
    p = tmp_path / "f.json"
    p.write_bytes(b"committed bytes")
    snapshot = guard.read_bytes_or_none(p)

    assert guard.restore_if_changed(p, snapshot) is False
    assert p.read_bytes() == b"committed bytes"


def test_restore_reverts_a_mutation_byte_for_byte(tmp_path):
    p = tmp_path / "f.json"
    p.write_bytes(b"committed bytes")
    snapshot = guard.read_bytes_or_none(p)

    p.write_bytes(b"clobbered by a leaky live test")  # simulate the leak
    changed = guard.restore_if_changed(p, snapshot)

    assert changed is True
    assert p.read_bytes() == b"committed bytes"  # restored exactly


def test_restore_deletes_a_file_created_during_the_run(tmp_path):
    p = tmp_path / "f.json"
    snapshot = guard.read_bytes_or_none(p)  # None — absent at snapshot time
    assert snapshot is None

    p.write_bytes(b"created mid-session")
    changed = guard.restore_if_changed(p, snapshot)

    assert changed is True
    assert not p.exists()  # restored to the original (absent) state


# ---------------------------------------------------------------------------
# #303: the guard's _PROTECTED set must cover the committed wiki invariants.
#
# build_index() also writes wiki/index.md, and log_event() writes wiki/log.md.
# A gateway test that hit the real /wiki/index handler without redirecting
# WIKI_DIR / LOG_PATH leaked those committed files (the per-file gateway
# redirect fixtures were not applied to every file). These pins keep the
# defence-in-depth backstop (AC#3) and the conftest-level redirect (AC#2)
# from silently regressing.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATEWAY_CONFTEST = _REPO_ROOT / "gateway" / "tests" / "conftest.py"


def test_guard_protects_committed_wiki_invariants():
    """The repo-root session guard snapshots/restores wiki/index.md + wiki/log.md."""
    import conftest as repo_conftest

    protected = {p.as_posix() for p in repo_conftest._PROTECTED}
    assert any(s.endswith("wiki/index.md") for s in protected), (
        "session guard _PROTECTED must include wiki/index.md so the backstop "
        "restores it if a test leaks build_index() (#303)"
    )
    assert any(s.endswith("wiki/log.md") for s in protected), (
        "session guard _PROTECTED must include wiki/log.md so the backstop "
        "restores it if a test leaks log_event() (#303)"
    )


def test_gateway_conftest_redirects_wiki_paths_to_tmp():
    """gateway/tests/conftest.py redirects the wiki write paths for every test.

    Text assertion (mirrors gateway/tests/test_ui_citation_links.py, which
    asserts on source text): pins AC#2 — a conftest-level autouse fixture must
    redirect WIKI_DIR + INDEX_PATH + LOG_PATH to tmp so NO gateway test can
    write the committed wiki/, not just the files that remembered a per-file
    fixture.
    """
    src = _GATEWAY_CONFTEST.read_text(encoding="utf-8")
    assert "autouse=True" in src
    assert "WIKI_DIR" in src, "gateway conftest must redirect indexer.WIKI_DIR (#303)"
    assert "LOG_PATH" in src, "gateway conftest must redirect logger.LOG_PATH (#303)"
    assert "INDEX_PATH" in src, "gateway conftest must redirect indexer.INDEX_PATH (#303)"
    assert "tmp_path" in src, "gateway conftest must redirect to tmp_path (#303)"
