"""Tests for markdown_kb.app.atomic — the shared atomic-write helpers.

Mirrors the behaviour exercised by eval/paraphrase_comparison/tests/test_atomic_replace.py,
but targeting the canonical home in markdown_kb.app.atomic (CODING_STANDARD §2.6).

Round-trip coverage + the Windows-retry path (monkeypatched os.replace).
"""

from __future__ import annotations

import os

import pytest

from app.atomic import replace_atomic, write_bytes_atomic, write_text_atomic

# ---------------------------------------------------------------------------
# replace_atomic
# ---------------------------------------------------------------------------


def test_replace_atomic_recovers_after_one_transient_permission_error(tmp_path, monkeypatch):
    """One transient PermissionError on os.replace is retried successfully."""
    import app.atomic as atomic_module

    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.txt"
    src.write_text("payload", encoding="utf-8")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("WinError 32: file in use by another process")
        return real_replace(a, b)

    monkeypatch.setattr(atomic_module.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    replace_atomic(str(src), dst)

    assert calls["n"] == 2  # one failure, one success
    assert dst.read_text(encoding="utf-8") == "payload"


def test_replace_atomic_reraises_after_exhausting_attempts(tmp_path, monkeypatch):
    """All attempts fail → PermissionError propagates."""
    import app.atomic as atomic_module

    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.txt"
    src.write_text("payload", encoding="utf-8")

    def always_locked(_a, _b):
        raise PermissionError("persistently locked")

    monkeypatch.setattr(atomic_module.os, "replace", always_locked)
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        replace_atomic(str(src), dst, attempts=3)


# ---------------------------------------------------------------------------
# write_text_atomic
# ---------------------------------------------------------------------------


def test_write_text_atomic_round_trip(tmp_path):
    """write_text_atomic creates the file with the exact content."""
    dst = tmp_path / "out.md"
    write_text_atomic(dst, "hello\nworld\n")
    assert dst.read_text(encoding="utf-8") == "hello\nworld\n"


def test_write_text_atomic_forces_lf_line_endings(tmp_path):
    """write_text_atomic writes LF-only line endings (newline='\\n'), not CRLF."""
    dst = tmp_path / "out.txt"
    write_text_atomic(dst, "line1\nline2\n")
    raw = dst.read_bytes()
    assert b"\r\n" not in raw


def test_write_text_atomic_creates_parent_dirs(tmp_path):
    """write_text_atomic creates missing parent directories."""
    dst = tmp_path / "deep" / "nested" / "out.txt"
    write_text_atomic(dst, "content")
    assert dst.read_text(encoding="utf-8") == "content"


def test_write_text_atomic_recovers_through_replace_retry(tmp_path, monkeypatch):
    """write_text_atomic retries os.replace on transient PermissionError."""
    import app.atomic as atomic_module

    dst = tmp_path / "report.md"
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("transient AV lock")
        return real_replace(a, b)

    monkeypatch.setattr(atomic_module.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    write_text_atomic(dst, "hello\nworld\n")

    assert dst.read_text(encoding="utf-8") == "hello\nworld\n"
    assert calls["n"] == 2


def test_write_text_atomic_cleans_up_tmp_on_failure(tmp_path, monkeypatch):
    """write_text_atomic removes the .tmp file when os.replace fails permanently."""
    import app.atomic as atomic_module

    dst = tmp_path / "out.md"
    tmp_files_created: list[str] = []

    original_mkstemp = __import__("tempfile").mkstemp

    def recording_mkstemp(**kwargs):
        fd, name = original_mkstemp(**kwargs)
        tmp_files_created.append(name)
        return fd, name

    def always_locked(_a, _b):
        raise PermissionError("locked")

    monkeypatch.setattr(atomic_module.os, "replace", always_locked)
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        write_text_atomic(dst, "content")

    # All tmp files must have been cleaned up
    for tmp in tmp_files_created:
        assert not os.path.exists(tmp), f"tmp file lingered: {tmp}"


# ---------------------------------------------------------------------------
# write_bytes_atomic
# ---------------------------------------------------------------------------


def test_write_bytes_atomic_round_trip(tmp_path):
    """write_bytes_atomic creates the file with the exact bytes."""
    dst = tmp_path / "out.bin"
    payload = b"\x00\x01\x02\xff\xfe\r\n"
    write_bytes_atomic(dst, payload)
    assert dst.read_bytes() == payload


def test_write_bytes_atomic_no_newline_translation(tmp_path):
    """write_bytes_atomic must NOT translate newlines — bytes are verbatim."""
    dst = tmp_path / "out.txt"
    # Write content with both LF and CRLF
    payload = b"line1\r\nline2\nline3\r\n"
    write_bytes_atomic(dst, payload)
    assert dst.read_bytes() == payload, "Binary write must not translate newlines"


def test_write_bytes_atomic_creates_parent_dirs(tmp_path):
    """write_bytes_atomic creates missing parent directories."""
    dst = tmp_path / "deep" / "nested" / "out.bin"
    write_bytes_atomic(dst, b"payload")
    assert dst.read_bytes() == b"payload"


def test_write_bytes_atomic_recovers_through_replace_retry(tmp_path, monkeypatch):
    """write_bytes_atomic retries os.replace on transient PermissionError."""
    import app.atomic as atomic_module

    dst = tmp_path / "out.bin"
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("transient AV lock")
        return real_replace(a, b)

    monkeypatch.setattr(atomic_module.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    write_bytes_atomic(dst, b"hello bytes")

    assert dst.read_bytes() == b"hello bytes"
    assert calls["n"] == 2


def test_write_bytes_atomic_cleans_up_tmp_on_failure(tmp_path, monkeypatch):
    """write_bytes_atomic removes the .tmp file when os.replace fails permanently."""
    import app.atomic as atomic_module

    dst = tmp_path / "out.bin"
    tmp_files_created: list[str] = []

    original_mkstemp = __import__("tempfile").mkstemp

    def recording_mkstemp(**kwargs):
        fd, name = original_mkstemp(**kwargs)
        tmp_files_created.append(name)
        return fd, name

    def always_locked(_a, _b):
        raise PermissionError("locked")

    monkeypatch.setattr(atomic_module.os, "replace", always_locked)
    monkeypatch.setattr(atomic_module.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        write_bytes_atomic(dst, b"content")

    # All tmp files must have been cleaned up
    for tmp in tmp_files_created:
        assert not os.path.exists(tmp), f"tmp file lingered: {tmp}"
