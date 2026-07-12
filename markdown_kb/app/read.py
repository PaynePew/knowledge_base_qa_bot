"""Deep module per Ousterhout. Public surface: ``list_tree``, ``read_file``, ``count_tree``, ``TreeEntry``.

Resource browser — read-only access to the whitelisted corpus roots.

Exposes ``list_tree(relpath)`` and ``read_file(relpath)`` constrained to a
whitelist of roots: ``docs/``, ``raw/``, ``wiki/``, ``.trash/`` (ADR-0041,
issue #604 — read-only pre-restore inspection of a retired Source; see
``source_lifecycle.py`` for the write side).  The ``.kb/`` directory is
explicitly excluded.  ``count_tree(relpath)`` (issue #559 A1) is a cheap
recursive file count over the same whitelist, for the Operator Console's
live artifact-node counts.

Security guarantees (path-traversal defence):
  - ``..`` components and absolute paths are rejected before any I/O.
  - After Path.resolve(), the resolved path must be ``is_relative_to`` one of
    the whitelisted roots; symlink escapes are caught at this step.
  - ``.kb/`` is not in the whitelist and cannot be reached.

Public surface:
    ``list_tree(relpath='')`` — list one level of the whitelist tree.
        ``relpath=''`` lists the roots themselves (docs, raw, wiki).
        ``relpath='docs'`` lists entries inside docs/.
        Returns ``list[TreeEntry]`` sorted: dirs first (alpha), then files (alpha).

    ``read_file(relpath)`` — read and return the UTF-8 text of a file inside
        one of the whitelisted roots.  Returns the raw text as a ``str``.

    ``count_tree(relpath)`` — recursively count files under a whitelisted
        root or sub-path (e.g. ``'raw'``, ``'docs'``).  Same security rules
        as list_tree; a root that does not exist on disk yet counts as 0.
        Returns an ``int``.  No file content is read.

    ``TreeEntry`` — named dataclass describing one directory entry.
        ``name``:  basename.
        ``relpath``: the relative path string to pass back to list_tree / read_file.
        ``is_dir``: True for directories, False for files.
        ``size``:  file size in bytes (0 for dirs).

Raises:
    ``PathRejected`` — any path that fails the security checks.
    ``FileNotFound`` — the resolved path does not exist.
    ``NotAFile`` — caller asked read_file on a directory.
    ``ReadError`` — OS-level I/O failure.

See GitHub issue #171 (Phase 15 S5) and PRD #168 for design rationale.
count_tree added by issue #559 A1 (Operator Console artifact-node counts).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from ._paths import _REPO_ROOT, DOCS_DIR, TRASH_DIR, WIKI_DIR

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RAW_DIR: Path = _REPO_ROOT / "raw"

# The whitelist maps a root name to its resolved absolute path.
# .kb/ is intentionally absent — it must never be reachable.
# .trash is read-only in practice: this module exposes no write path at all,
# so adding it here only ever grants pre-restore inspection (ADR-0041), never
# a way to write into the trash tree.
_WHITELIST_ROOTS: dict[str, Path] = {
    "docs": DOCS_DIR,
    "raw": RAW_DIR,
    "wiki": WIKI_DIR,
    ".trash": TRASH_DIR,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PathRejected(ValueError):
    """Raised when a path fails the security whitelist check."""


class FileNotFound(LookupError):
    """Raised when the requested path does not exist."""


class NotAFile(ValueError):
    """Raised when read_file is called on a directory."""


class ReadError(OSError):
    """Raised on an OS-level I/O failure inside read_file."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TreeEntry:
    """One entry in a directory listing.

    ``name``    — the basename of the entry.
    ``relpath`` — the relative path string (use as argument to list_tree /
                  read_file), e.g. ``'docs/policy.md'``.
    ``is_dir``  — True for directories, False for files.
    ``size``    — file size in bytes; 0 for directories.
    """

    name: str
    relpath: str
    is_dir: bool
    size: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_relpath(relpath: str) -> tuple[str, list[str]]:
    """Parse relpath and return (root_name, parts_after_root).

    Validates that:
      - The path is not absolute.
      - No ``..`` component is present.

    Returns (root_name, remaining_parts) where root_name is the first
    component (e.g. "docs") and remaining_parts is the rest (possibly []).

    Empty relpath ``""`` is valid — it means "list the roots".
    """
    # Reject absolute paths
    if relpath.startswith("/") or relpath.startswith("\\"):
        raise PathRejected(f"Absolute paths are not allowed: {relpath!r}")

    # Normalise separators for cross-platform safety
    normalised = relpath.replace("\\", "/").strip("/")

    if not normalised:
        return "", []

    parts = normalised.split("/")

    # Reject any '..' component
    if ".." in parts:
        raise PathRejected(f"Path traversal ('..') is not allowed: {relpath!r}")

    root_name = parts[0]
    remaining = parts[1:]
    return root_name, remaining


def _resolve_and_check(root_dir: Path, sub_parts: list[str]) -> Path:
    """Build an absolute path from root_dir + sub_parts and verify it stays inside root_dir.

    Uses Path.resolve() to dereference symlinks, then checks is_relative_to
    so that even a symlink pointing outside the root is rejected.

    Returns the resolved Path.

    Raises PathRejected if the resolved path escapes the root.
    """
    target = root_dir.joinpath(*sub_parts) if sub_parts else root_dir
    resolved_root = root_dir.resolve()

    # We must resolve the target for symlink-escape detection.  If the target
    # does not exist yet we still need to check its would-be resolved path:
    # Path.resolve() works on non-existent paths (Python 3.6+, strict=False).
    resolved_target = target.resolve()

    # is_relative_to is Python 3.9+; the project already requires >=3.9.
    # The message deliberately omits the resolved absolute paths so a symlink
    # escape cannot disclose the server's on-disk layout to the HTTP client.
    if not resolved_target.is_relative_to(resolved_root):
        raise PathRejected("Path escapes the allowed root (symlink escape rejected).")

    return resolved_target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_tree(
    relpath: str = "",
    *,
    roots: dict[str, Path] | None = None,
) -> list[TreeEntry]:
    """List one level of the whitelisted resource tree.

    Args:
        relpath: Relative path within the whitelist tree.  Empty string (the
            default) lists the roots themselves (docs, raw, wiki).
            Pass ``'docs'`` to list entries inside ``docs/``, etc.
        roots:   Override the module-level ``_WHITELIST_ROOTS`` (used by tests
            via injection; production callers omit this).

    Returns:
        ``list[TreeEntry]`` — directories first (alphabetical), then files
        (alphabetical).  Entries starting with ``.`` are hidden and excluded.

    Raises:
        PathRejected: path contains ``..``, is absolute, or resolves outside
            the whitelist root.
        FileNotFound: the resolved path does not exist.
        NotAFile: only raised from read_file; list_tree never raises this.
    """
    effective_roots = roots if roots is not None else _WHITELIST_ROOTS

    root_name, sub_parts = _parse_relpath(relpath)

    # Empty relpath → list the root entries (docs, raw, wiki) as virtual dirs.
    if not root_name:
        entries = []
        for name, _root_path in sorted(effective_roots.items()):
            entries.append(
                TreeEntry(
                    name=name,
                    relpath=name,
                    is_dir=True,
                    size=0,
                )
            )
        return entries

    # Validate root name is whitelisted
    if root_name not in effective_roots:
        raise PathRejected(
            f"Root {root_name!r} is not in the whitelist. Allowed: {sorted(effective_roots)}"
        )

    root_dir = effective_roots[root_name]
    resolved = _resolve_and_check(root_dir, sub_parts)

    if not resolved.exists():
        raise FileNotFound(f"Path does not exist: {relpath!r}")

    if not resolved.is_dir():
        raise NotAFile(f"Path is a file, not a directory (use read_file): {relpath!r}")

    dirs: list[TreeEntry] = []
    files: list[TreeEntry] = []

    for child in resolved.iterdir():
        # Skip hidden entries (dot-files)
        if child.name.startswith("."):
            continue

        # Compute the relpath for this child
        child_relpath = f"{relpath.rstrip('/')}/{child.name}" if relpath else child.name

        if child.is_dir():
            dirs.append(TreeEntry(name=child.name, relpath=child_relpath, is_dir=True, size=0))
        else:
            size = 0
            with contextlib.suppress(OSError):
                size = child.stat().st_size
            files.append(TreeEntry(name=child.name, relpath=child_relpath, is_dir=False, size=size))

    dirs.sort(key=lambda e: e.name)
    files.sort(key=lambda e: e.name)

    return dirs + files


def read_file(
    relpath: str,
    *,
    roots: dict[str, Path] | None = None,
) -> str:
    """Read and return the UTF-8 text of a file inside the whitelisted roots.

    Args:
        relpath: Relative path such as ``'docs/policy.md'`` or
            ``'wiki/log.md'``.
        roots:   Override the module-level ``_WHITELIST_ROOTS`` (tests only).

    Returns:
        The raw file content as a ``str``.

    Raises:
        PathRejected: path contains ``..``, is absolute, or resolves outside
            the whitelist root (including symlink escapes and ``.kb/`` attempts).
        FileNotFound: the resolved path does not exist.
        NotAFile: the resolved path is a directory.
        ReadError: OS-level I/O failure.
    """
    effective_roots = roots if roots is not None else _WHITELIST_ROOTS

    root_name, sub_parts = _parse_relpath(relpath)

    if not root_name:
        raise PathRejected(
            "read_file requires a path inside a whitelisted root, not the root listing."
        )

    if root_name not in effective_roots:
        raise PathRejected(
            f"Root {root_name!r} is not in the whitelist. Allowed: {sorted(effective_roots)}"
        )

    root_dir = effective_roots[root_name]
    resolved = _resolve_and_check(root_dir, sub_parts)

    if not resolved.exists():
        raise FileNotFound(f"Path does not exist: {relpath!r}")

    if resolved.is_dir():
        raise NotAFile(f"Path is a directory, not a file (use list_tree): {relpath!r}")

    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReadError(f"Failed to read {relpath!r}: {exc}") from exc


def count_tree(
    relpath: str,
    *,
    roots: dict[str, Path] | None = None,
) -> int:
    """Recursively count files under a whitelisted root or sub-path.

    A cheap directory-listing count for the Operator Console's live
    artifact-node counts (issue #559 A1) — no file content is read, no
    frontmatter is parsed. Same security whitelist and traversal rules as
    list_tree (root name validated, ``..``/absolute paths and symlink
    escapes rejected). Hidden entries (any path component starting with
    ``.``) are excluded, matching list_tree's visibility rule.

    Args:
        relpath: e.g. ``'raw'`` or ``'docs'`` — same shape as list_tree,
            minus the empty-string "list the roots" case.
        roots:   test-only override, same as list_tree.

    Returns:
        Total file count under the resolved directory (directories
        themselves are not counted). A root that does not exist on disk yet
        (a fresh pipeline instance with no raw/ or docs/ directory) counts
        as 0 rather than raising FileNotFound.

    Raises:
        PathRejected: path contains ``..``, is absolute, names an
            unwhitelisted root, or resolves outside the whitelist root.
        NotAFile: the resolved path is a file, not a directory.
    """
    effective_roots = roots if roots is not None else _WHITELIST_ROOTS

    root_name, sub_parts = _parse_relpath(relpath)
    if not root_name:
        raise PathRejected(
            "count_tree requires a path inside a whitelisted root, not the root listing."
        )
    if root_name not in effective_roots:
        raise PathRejected(
            f"Root {root_name!r} is not in the whitelist. Allowed: {sorted(effective_roots)}"
        )

    root_dir = effective_roots[root_name]
    resolved = _resolve_and_check(root_dir, sub_parts)

    if not resolved.exists():
        return 0
    if not resolved.is_dir():
        raise NotAFile(f"Path is a file, not a directory: {relpath!r}")

    return sum(
        1
        for p in resolved.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(resolved).parts)
    )
