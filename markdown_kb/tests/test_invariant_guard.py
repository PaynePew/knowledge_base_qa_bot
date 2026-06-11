"""Unit tests for the committed-invariant guard helper (``conftest_support``).

The repo-root ``conftest.py`` uses these to snapshot/restore git-tracked invariant
files (notably ``.kb/index.json``) across a test session, so a live test that leaks
a real write path cannot leave the committed file mutated (#204). The session-scoped
fixture itself is awkward to assert on directly, so the restore logic lives in a
plain importable module and is unit-tested here on tmp files.
"""

from __future__ import annotations

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
