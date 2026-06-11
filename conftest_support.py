"""Support helpers for the repo-root conftest's committed-invariant guard.

Kept in a separate importable module (the repo root is on ``pythonpath`` — see the
root ``pyproject.toml``) so the snapshot/restore logic can be unit-tested without
importing the session-scoped conftest fixture itself.
"""

from __future__ import annotations

from pathlib import Path


def read_bytes_or_none(path: Path) -> bytes | None:
    """Return the file's bytes, or ``None`` if it does not exist."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def restore_if_changed(path: Path, original: bytes | None) -> bool:
    """Restore ``path`` to ``original`` if its current content differs.

    Returns ``True`` if a change was detected and restored, ``False`` if the file
    is unchanged. ``original=None`` means the file did not exist at snapshot time;
    if a run created it, it is deleted so the original (absent) state is restored.
    """
    current = read_bytes_or_none(path)
    if current == original:
        return False
    if original is None:
        if path.exists():
            path.unlink()
        return True
    path.write_bytes(original)
    return True
