"""Hermetic unit tests for ``markdown_kb.app.read.list_tree_recursive`` (issue #644).

Companion to ``test_read.py`` (kept untouched — a new file per this slice's
one-file-per-new-surface convention, matching #643's
``test_ui_console_field_manual_search.py`` sibling). Covers:
  - Nested files appear with correct relpaths, flattened, dirs AND files.
  - Whitelist/traversal guards hold in recursive mode — same PathRejected /
    FileNotFound / NotAFile behaviour as list_tree (issue #644 AC).
  - ``.kb/`` is never reachable even recursively.
  - The hard cap (``limit``) + ``truncated`` flag.
  - ``list_tree`` itself (non-recursive) is untouched by this addition.

No OPENAI_API_KEY required — pure filesystem I/O, same hermetic pattern as
test_read.py (tmp_path + root injection).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_roots(tmp_path: Path) -> dict[str, Path]:
    """Create minimal docs/, raw/, wiki/ dirs and return the roots dict."""
    roots = {
        "docs": tmp_path / "docs",
        "raw": tmp_path / "raw",
        "wiki": tmp_path / "wiki",
    }
    for p in roots.values():
        p.mkdir(parents=True, exist_ok=True)
    return roots


# ---------------------------------------------------------------------------
# Happy path — flat, nested, dirs + files
# ---------------------------------------------------------------------------


def test_recursive_lists_root_pseudo_dirs(tmp_path):
    """list_tree_recursive('') includes the whitelisted roots themselves,
    same as list_tree('') does at its own level."""
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    entries, truncated = list_tree_recursive("", roots=roots)

    names = {e.name for e in entries}
    assert {"docs", "raw", "wiki"}.issubset(names)
    assert truncated is False


def test_recursive_flattens_nested_files_with_correct_relpaths(tmp_path):
    """A file several folders deep appears in the flat list with its full
    relpath — the whole point of a cross-tree quick-find (issue #644 AC)."""
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    sub = roots["docs"] / "a" / "b"
    sub.mkdir(parents=True)
    (sub / "deep.md").write_text("# Deep\n", encoding="utf-8")

    entries, truncated = list_tree_recursive("", roots=roots)
    relpaths = {e.relpath for e in entries}
    assert "docs/a/b/deep.md" in relpaths
    assert "docs/a" in relpaths
    assert "docs/a/b" in relpaths
    assert truncated is False


def test_recursive_includes_both_dirs_and_files(tmp_path):
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    (roots["docs"] / "sub").mkdir()
    (roots["docs"] / "top.md").write_text("x", encoding="utf-8")

    entries, _truncated = list_tree_recursive("docs", roots=roots)
    is_dir_by_name = {e.name: e.is_dir for e in entries}
    assert is_dir_by_name == {"sub": True, "top.md": False}


def test_recursive_scoped_to_a_subpath(tmp_path):
    """list_tree_recursive('docs') only walks docs/, not the other roots."""
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    (roots["docs"] / "a.md").write_text("a", encoding="utf-8")
    (roots["raw"] / "b.txt").write_text("b", encoding="utf-8")

    entries, _truncated = list_tree_recursive("docs", roots=roots)
    relpaths = {e.relpath for e in entries}
    assert relpaths == {"docs/a.md"}


def test_recursive_entry_relpaths_are_navigable(tmp_path):
    """A relpath from list_tree_recursive works as input to read_file."""
    from app.read import list_tree_recursive, read_file

    roots = _make_roots(tmp_path)
    sub = roots["wiki"] / "entities"
    sub.mkdir()
    (sub / "acme.md").write_text("# Acme\n", encoding="utf-8")

    entries, _truncated = list_tree_recursive("wiki", roots=roots)
    file_entry = next(e for e in entries if not e.is_dir)
    assert "Acme" in read_file(file_entry.relpath, roots=roots)


def test_recursive_hides_dotfiles_and_dot_directories(tmp_path):
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    (roots["docs"] / ".hidden.md").write_text("secret", encoding="utf-8")
    archive = roots["wiki"] / ".archive"
    archive.mkdir()
    (archive / "old.md").write_text("old", encoding="utf-8")

    entries, _truncated = list_tree_recursive("", roots=roots)
    relpaths = {e.relpath for e in entries}
    assert not any(".hidden" in r or ".archive" in r for r in relpaths)


# ---------------------------------------------------------------------------
# SECURITY — same guards as list_tree, exercised recursively (issue #644 AC)
# ---------------------------------------------------------------------------


def test_recursive_dotdot_rejected(tmp_path):
    from app.read import PathRejected, list_tree_recursive

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree_recursive("docs/../wiki", roots=roots)


def test_recursive_absolute_path_rejected(tmp_path):
    from app.read import PathRejected, list_tree_recursive

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree_recursive("/etc/passwd", roots=roots)


def test_recursive_kb_root_not_in_whitelist_rejected(tmp_path):
    """.kb/ is never reachable, even recursively (issue #644 AC)."""
    from app.read import PathRejected, list_tree_recursive

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree_recursive(".kb", roots=roots)


def test_recursive_kb_never_appears_in_a_root_listing_even_if_present_on_disk(tmp_path):
    """.kb/ sitting on disk next to the whitelisted roots must never surface
    in a recursive '' walk — mirrors list_tree's own hidden-root exclusion
    (``.kb`` is dot-prefixed and not a _WHITELIST_ROOTS key)."""
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    kb = tmp_path / ".kb"
    kb.mkdir()
    (kb / "index.json").write_text("{}", encoding="utf-8")

    entries, _truncated = list_tree_recursive("", roots=roots)
    assert not any(".kb" in e.relpath for e in entries)


def test_recursive_symlink_escape_rejected(tmp_path):
    from app.read import PathRejected, list_tree_recursive

    roots = _make_roots(tmp_path)
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("SECRET", encoding="utf-8")
    link_dir = roots["docs"] / "escape_link"
    try:
        link_dir.symlink_to(tmp_path / "outside_dir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not available in this environment")

    with pytest.raises(PathRejected):
        list_tree_recursive("docs", roots=roots)


def test_recursive_within_root_symlink_cycle_terminates(tmp_path):
    """A dir symlink aliasing its own ancestor must not recurse forever —
    each resolved directory is walked exactly once (defensive gap flagged by
    the slice's adversarial verify)."""
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    sub = roots["docs"] / "sub"
    sub.mkdir()
    (sub / "leaf.md").write_text("# Leaf\n", encoding="utf-8")
    try:
        (sub / "loop").symlink_to(sub, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not available in this environment")

    entries, truncated = list_tree_recursive("docs", roots=roots)

    assert truncated is False
    relpaths = [e.relpath for e in entries]
    assert "docs/sub/leaf.md" in relpaths
    # the alias may be listed, but its contents are never walked a second time
    assert relpaths.count("docs/sub/loop/leaf.md") == 0


def test_recursive_nonexistent_subpath_raises_file_not_found(tmp_path):
    from app.read import FileNotFound, list_tree_recursive

    roots = _make_roots(tmp_path)
    with pytest.raises(FileNotFound):
        list_tree_recursive("docs/no-such-subdir", roots=roots)


def test_recursive_on_a_file_raises_not_a_file(tmp_path):
    from app.read import NotAFile, list_tree_recursive

    roots = _make_roots(tmp_path)
    (roots["docs"] / "leaf.md").write_text("# Leaf\n", encoding="utf-8")
    with pytest.raises(NotAFile):
        list_tree_recursive("docs/leaf.md", roots=roots)


# ---------------------------------------------------------------------------
# Hard cap + truncated flag (issue #644 AC — corpus-growth guard, not
# pagination)
# ---------------------------------------------------------------------------


def test_recursive_under_the_limit_is_not_truncated(tmp_path):
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    for i in range(5):
        (roots["docs"] / f"f{i}.md").write_text("x", encoding="utf-8")

    entries, truncated = list_tree_recursive("docs", roots=roots, limit=10_000)
    assert len(entries) == 5
    assert truncated is False


def test_recursive_hits_the_cap_and_reports_truncated(tmp_path):
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    for i in range(20):
        (roots["docs"] / f"f{i:02d}.md").write_text("x", encoding="utf-8")

    entries, truncated = list_tree_recursive("docs", roots=roots, limit=7)
    assert len(entries) == 7
    assert truncated is True


def test_recursive_cap_stops_the_whole_walk_not_just_one_directory(tmp_path):
    """Once the cap is hit inside one subdirectory, sibling directories are
    not walked at all — the cap bounds total work, not per-directory."""
    from app.read import list_tree_recursive

    roots = _make_roots(tmp_path)
    a = roots["docs"] / "a"
    b = roots["docs"] / "b"
    a.mkdir()
    b.mkdir()
    for i in range(5):
        (a / f"a{i}.md").write_text("x", encoding="utf-8")
        (b / f"b{i}.md").write_text("x", encoding="utf-8")

    entries, truncated = list_tree_recursive("docs", roots=roots, limit=3)
    assert len(entries) == 3
    assert truncated is True
    # Cap was hit before b/ was ever reached (dirs-first/alpha order: a before b).
    assert not any(e.relpath.startswith("docs/b/") for e in entries)


# ---------------------------------------------------------------------------
# list_tree itself stays byte-identical (non-recursive default unchanged)
# ---------------------------------------------------------------------------


def test_list_tree_non_recursive_default_is_unaffected_by_this_addition(tmp_path):
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / "sub").mkdir()
    (roots["docs"] / "sub" / "nested.md").write_text("x", encoding="utf-8")
    (roots["docs"] / "top.md").write_text("y", encoding="utf-8")

    entries = list_tree("docs", roots=roots)
    names = {e.name for e in entries}
    # Only ONE level — the nested file must NOT appear.
    assert names == {"sub", "top.md"}
