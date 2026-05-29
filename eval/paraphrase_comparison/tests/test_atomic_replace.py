"""Atomic-write retry tests (external behaviour only, CODING_STANDARD §0.2).

``replace_atomic`` wraps ``os.replace`` with a bounded retry for the transient
Windows ``PermissionError`` an antivirus / Search indexer raises when it briefly
locks a just-written file (issue #156). These tests drive the retry behaviour
deterministically by stubbing ``os.replace`` — no real OS race needed.
"""

from __future__ import annotations

import os

import pytest

from eval.paraphrase_comparison import loader
from eval.paraphrase_comparison.loader import replace_atomic, write_text_atomic


def test_replace_atomic_recovers_after_one_transient_permission_error(
    tmp_path, monkeypatch
):
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

    monkeypatch.setattr(loader.os, "replace", flaky_replace)
    monkeypatch.setattr(
        loader.time, "sleep", lambda _s: None
    )  # no real backoff in test

    replace_atomic(str(src), dst)

    assert calls["n"] == 2  # one failure, one success
    assert dst.read_text(encoding="utf-8") == "payload"


def test_replace_atomic_reraises_after_exhausting_attempts(tmp_path, monkeypatch):
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.txt"
    src.write_text("payload", encoding="utf-8")

    def always_locked(_a, _b):
        raise PermissionError("persistently locked")

    monkeypatch.setattr(loader.os, "replace", always_locked)
    monkeypatch.setattr(loader.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        replace_atomic(str(src), dst, attempts=3)


def test_write_text_atomic_recovers_through_replace_retry(tmp_path, monkeypatch):
    dst = tmp_path / "report.md"
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("transient AV lock")
        return real_replace(a, b)

    monkeypatch.setattr(loader.os, "replace", flaky_replace)
    monkeypatch.setattr(loader.time, "sleep", lambda _s: None)

    write_text_atomic(dst, "hello\nworld\n")

    assert dst.read_text(encoding="utf-8") == "hello\nworld\n"
    assert calls["n"] == 2
