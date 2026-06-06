"""Shared atomic-write helpers (CODING_STANDARD §2.6).

Public surface: ``write_text_atomic``, ``replace_atomic``.

This is the canonical home for the atomic-write pattern used across
``markdown_kb`` and consumed via re-export by ``eval.paraphrase_comparison.loader``.
``kb_mcp.hot_cache`` imports from here directly (``kb_mcp → markdown-kb`` is a
declared dependency; the old ``kb_mcp → eval`` direction was undeclared).

Implementations moved verbatim from ``eval/paraphrase_comparison/loader.py``
(issue #211 — consolidate four copies into one).
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from pathlib import Path


def replace_atomic(src: str | Path, dst: Path, *, attempts: int = 5, backoff: float = 0.1) -> None:
    """``os.replace`` with a bounded retry for transient Windows file locks.

    On Windows an antivirus or the Search indexer can briefly open a just-written
    file to scan it, making ``os.replace`` raise a transient ``PermissionError``
    (WinError 5 / 32). A short retry clears the race; a persistent failure still
    raises (CODING_STANDARD §2.6). Zero overhead on the normal first-try success;
    on POSIX the race does not occur so the loop runs exactly once.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff)


def write_text_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (tmp + os.replace; §2.6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f"{path.stem}_")
    try:
        # newline="\n": force LF on every OS so committed artifacts honour the
        # repo's `* eol=lf` .gitattributes (CODING_STANDARD §1.1) — Windows text
        # mode would otherwise translate "\n" to CRLF and dirty the working tree.
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        replace_atomic(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
