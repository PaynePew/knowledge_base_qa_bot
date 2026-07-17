"""Hermetic unit tests for the resource read deep module (markdown_kb.app.read).

AC coverage (issue #171 — Phase 15 S5):
  - list_tree('') returns the three whitelisted roots (docs, raw, wiki) as dirs.
  - list_tree('docs') lists entries inside docs/.
  - list_tree returns dirs first (alpha), then files (alpha).
  - read_file returns the raw UTF-8 content of a file.
  - SECURITY — path traversal ('..'): rejected by PathRejected.
  - SECURITY — absolute paths: rejected by PathRejected.
  - SECURITY — symlink escapes: rejected by PathRejected.
  - SECURITY — .kb/ root: rejected by PathRejected (not in whitelist).
  - FileNotFound raised for non-existent paths.
  - NotAFile raised when read_file is called on a directory.
  - list_tree on a file raises NotAFile.

No OPENAI_API_KEY required — the read module is pure filesystem I/O.
Uses tmp_path + root injection (same hermetic pattern as test_upload.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


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
# list_tree — root level
# ---------------------------------------------------------------------------


def test_list_tree_root_returns_three_roots(tmp_path):
    """list_tree('') returns exactly the three whitelisted roots as dirs."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    entries = list_tree("", roots=roots)

    names = {e.name for e in entries}
    assert names == {"docs", "raw", "wiki"}
    for e in entries:
        assert e.is_dir is True


def test_list_tree_root_entries_have_relpath(tmp_path):
    """Root entries have relpath equal to their name."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    entries = list_tree("", roots=roots)

    for e in entries:
        assert e.relpath == e.name


# ---------------------------------------------------------------------------
# list_tree — inside a root
# ---------------------------------------------------------------------------


def test_list_tree_empty_dir_returns_empty(tmp_path):
    """list_tree('raw') on an empty dir returns []."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    entries = list_tree("raw", roots=roots)
    assert entries == []


def test_list_tree_files_listed(tmp_path):
    """list_tree('docs') lists .md files present in docs/."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
    (roots["docs"] / "beta.md").write_text("# Beta\n", encoding="utf-8")

    entries = list_tree("docs", roots=roots)
    names = [e.name for e in entries]
    assert "alpha.md" in names
    assert "beta.md" in names


def test_list_tree_dirs_before_files(tmp_path):
    """Directories appear before files in list_tree results."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    docs = roots["docs"]
    (docs / "subdir").mkdir()
    (docs / "alpha.md").write_text("# A\n", encoding="utf-8")

    entries = list_tree("docs", roots=roots)
    assert entries[0].is_dir is True
    assert entries[-1].is_dir is False


def test_list_tree_alpha_sort_within_kind(tmp_path):
    """Dirs and files are each sorted alphabetically."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    docs = roots["docs"]
    (docs / "zzz").mkdir()
    (docs / "aaa").mkdir()
    (docs / "z-file.md").write_text("z", encoding="utf-8")
    (docs / "a-file.md").write_text("a", encoding="utf-8")

    entries = list_tree("docs", roots=roots)
    dir_names = [e.name for e in entries if e.is_dir]
    file_names = [e.name for e in entries if not e.is_dir]
    assert dir_names == sorted(dir_names)
    assert file_names == sorted(file_names)


def test_list_tree_subdirectory(tmp_path):
    """list_tree('docs/sub') lists files inside docs/sub/."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    sub = roots["docs"] / "sub"
    sub.mkdir()
    (sub / "note.md").write_text("# Note\n", encoding="utf-8")

    entries = list_tree("docs/sub", roots=roots)
    names = [e.name for e in entries]
    assert "note.md" in names


def test_list_tree_entry_relpath_is_navigable(tmp_path):
    """Entries returned from list_tree have relpaths usable in subsequent calls."""
    from app.read import list_tree, read_file

    roots = _make_roots(tmp_path)
    (roots["docs"] / "policy.md").write_text("# Policy\n\nContent.\n", encoding="utf-8")

    entries = list_tree("docs", roots=roots)
    assert entries  # sanity
    file_entry = next(e for e in entries if not e.is_dir)

    # The relpath from list_tree should work as input to read_file
    text = read_file(file_entry.relpath, roots=roots)
    assert "Policy" in text


def test_list_tree_hides_dotfiles(tmp_path):
    """Hidden entries (starting with '.') are excluded from list_tree."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / ".hidden").write_text("secret", encoding="utf-8")
    (roots["docs"] / "visible.md").write_text("public", encoding="utf-8")

    entries = list_tree("docs", roots=roots)
    names = [e.name for e in entries]
    assert ".hidden" not in names
    assert "visible.md" in names


def test_list_tree_file_size_reported(tmp_path):
    """File entries have a non-zero size when the file has content."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / "sized.md").write_text("hello", encoding="utf-8")

    entries = list_tree("docs", roots=roots)
    file_entry = next(e for e in entries if e.name == "sized.md")
    assert file_entry.size > 0


# ---------------------------------------------------------------------------
# read_file — happy path
# ---------------------------------------------------------------------------


def test_read_file_returns_text(tmp_path):
    """read_file returns the exact UTF-8 text of a file."""
    from app.read import read_file

    roots = _make_roots(tmp_path)
    content = "# Hello\n\nWorld.\n"
    (roots["docs"] / "hello.md").write_text(content, encoding="utf-8")

    result = read_file("docs/hello.md", roots=roots)
    assert result == content


def test_read_file_wiki_log(tmp_path):
    """read_file can read wiki/log.md when present."""
    from app.read import read_file

    roots = _make_roots(tmp_path)
    log_content = "## [2026-05-29T00:00:00Z] index_built | files=3\n"
    (roots["wiki"] / "log.md").write_text(log_content, encoding="utf-8")

    result = read_file("wiki/log.md", roots=roots)
    assert result == log_content


def test_read_file_lint_report(tmp_path):
    """read_file can read wiki/lint-report.md when present."""
    from app.read import read_file

    roots = _make_roots(tmp_path)
    report_content = "# Lint Report\n\nC1: 0 issues\n"
    (roots["wiki"] / "lint-report.md").write_text(report_content, encoding="utf-8")

    result = read_file("wiki/lint-report.md", roots=roots)
    assert result == report_content


def test_read_file_raw_dir(tmp_path):
    """read_file can read a file in raw/."""
    from app.read import read_file

    roots = _make_roots(tmp_path)
    (roots["raw"] / "source.txt").write_text("Raw content.\n", encoding="utf-8")

    result = read_file("raw/source.txt", roots=roots)
    assert result == "Raw content.\n"


# ---------------------------------------------------------------------------
# SECURITY — path traversal rejection (PRIORITY per AC)
# ---------------------------------------------------------------------------


def test_dotdot_in_relpath_rejected(tmp_path):
    """'..' in relpath raises PathRejected."""
    from app.read import PathRejected, list_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree("docs/../wiki", roots=roots)


def test_dotdot_in_read_file_rejected(tmp_path):
    """'..' in relpath for read_file raises PathRejected."""
    from app.read import PathRejected, read_file

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        read_file("docs/../../etc/passwd", roots=roots)


def test_dotdot_only_rejected(tmp_path):
    """A relpath of '..' raises PathRejected."""
    from app.read import PathRejected, list_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree("..", roots=roots)


def test_absolute_path_in_list_tree_rejected(tmp_path):
    """An absolute path in list_tree raises PathRejected."""
    from app.read import PathRejected, list_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree("/etc/passwd", roots=roots)


def test_absolute_path_in_read_file_rejected(tmp_path):
    """An absolute path in read_file raises PathRejected."""
    from app.read import PathRejected, read_file

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        read_file("/etc/shadow", roots=roots)


def test_windows_absolute_path_rejected(tmp_path):
    r"""A Windows-style absolute path (C:\...) raises PathRejected."""
    from app.read import PathRejected, read_file

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        read_file("C:\\Windows\\System32\\cmd.exe", roots=roots)


def test_symlink_escape_rejected(tmp_path):
    """A symlink that points outside the root raises PathRejected."""
    from app.read import PathRejected, read_file

    roots = _make_roots(tmp_path)

    # Create an outside file and a symlink to it inside docs/
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("SECRET", encoding="utf-8")

    link_path = roots["docs"] / "escape_link.txt"
    try:
        link_path.symlink_to(outside)
    except (OSError, NotImplementedError):
        # Symlink creation may require elevation on Windows — skip gracefully.
        pytest.skip("Symlink creation not available in this environment")

    with pytest.raises(PathRejected):
        read_file("docs/escape_link.txt", roots=roots)


def test_kb_root_not_in_whitelist_rejected(tmp_path):
    """.kb/ root is not in the whitelist — raises PathRejected."""
    from app.read import PathRejected, list_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree(".kb", roots=roots)


def test_kb_path_traversal_rejected(tmp_path):
    """A path that would reach .kb/ via traversal raises PathRejected."""
    from app.read import PathRejected, read_file

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        read_file("docs/../.kb/index.json", roots=roots)


def test_unknown_root_rejected(tmp_path):
    """A relpath with an unknown root name raises PathRejected."""
    from app.read import PathRejected, list_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        list_tree("secrets", roots=roots)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_list_tree_nonexistent_subpath_raises_file_not_found(tmp_path):
    """list_tree on a non-existent subpath raises FileNotFound."""
    from app.read import FileNotFound, list_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(FileNotFound):
        list_tree("docs/no-such-subdir", roots=roots)


def test_list_tree_missing_root_directory_is_empty_listing(tmp_path):
    """list_tree on a whitelisted root that has not been created on disk yet
    (e.g. .trash before the first retire, ADR-0041) returns [] instead of
    raising FileNotFound — an advertised root must always be listable
    (issue #629)."""
    from app.read import list_tree

    roots = _make_roots(tmp_path)
    roots[".trash"] = tmp_path / ".trash"  # deliberately not created

    assert list_tree(".trash", roots=roots) == []


def test_list_tree_missing_root_subpath_still_raises_file_not_found(tmp_path):
    """A nonexistent SUB-path under a not-yet-created root still raises
    FileNotFound — only the bare root itself gets the empty-listing
    treatment (issue #629)."""
    from app.read import FileNotFound, list_tree

    roots = _make_roots(tmp_path)
    roots[".trash"] = tmp_path / ".trash"  # deliberately not created

    with pytest.raises(FileNotFound):
        list_tree(".trash/nope", roots=roots)


def test_read_file_nonexistent_raises_file_not_found(tmp_path):
    """read_file on a non-existent file raises FileNotFound."""
    from app.read import FileNotFound, read_file

    roots = _make_roots(tmp_path)
    with pytest.raises(FileNotFound):
        read_file("docs/ghost.md", roots=roots)


def test_read_file_on_directory_raises_not_a_file(tmp_path):
    """read_file called on a directory raises NotAFile."""
    from app.read import NotAFile, read_file

    roots = _make_roots(tmp_path)
    (roots["docs"] / "subdir").mkdir()

    with pytest.raises(NotAFile):
        read_file("docs/subdir", roots=roots)


def test_list_tree_on_file_raises_not_a_file(tmp_path):
    """list_tree called on a file path raises NotAFile."""
    from app.read import NotAFile, list_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / "leaf.md").write_text("# Leaf\n", encoding="utf-8")

    with pytest.raises(NotAFile):
        list_tree("docs/leaf.md", roots=roots)


def test_read_file_empty_relpath_raises_path_rejected(tmp_path):
    """read_file with an empty relpath raises PathRejected (needs a real file path)."""
    from app.read import PathRejected, read_file

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        read_file("", roots=roots)


# ---------------------------------------------------------------------------
# count_tree (issue #559 A1 — Operator Console artifact-node live counts)
# ---------------------------------------------------------------------------


def test_count_tree_empty_dir_is_zero(tmp_path):
    """count_tree on an empty whitelisted dir returns 0."""
    from app.read import count_tree

    roots = _make_roots(tmp_path)
    assert count_tree("raw", roots=roots) == 0


def test_count_tree_counts_top_level_files(tmp_path):
    """count_tree counts files directly inside the root."""
    from app.read import count_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / "alpha.md").write_text("a", encoding="utf-8")
    (roots["docs"] / "beta.md").write_text("b", encoding="utf-8")

    assert count_tree("docs", roots=roots) == 2


def test_count_tree_counts_nested_files_recursively(tmp_path):
    """count_tree recurses into subdirectories."""
    from app.read import count_tree

    roots = _make_roots(tmp_path)
    sub = roots["wiki"] / "entities"
    sub.mkdir()
    (roots["wiki"] / "index.md").write_text("root", encoding="utf-8")
    (sub / "acme.md").write_text("nested", encoding="utf-8")

    assert count_tree("wiki", roots=roots) == 2


def test_count_tree_excludes_hidden_files(tmp_path):
    """count_tree excludes dot-prefixed files, matching list_tree's visibility rule."""
    from app.read import count_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / ".hidden").write_text("secret", encoding="utf-8")
    (roots["docs"] / "visible.md").write_text("public", encoding="utf-8")

    assert count_tree("docs", roots=roots) == 1


def test_count_tree_excludes_files_in_hidden_directories(tmp_path):
    """count_tree excludes files nested inside a dot-prefixed directory."""
    from app.read import count_tree

    roots = _make_roots(tmp_path)
    archive = roots["wiki"] / ".archive"
    archive.mkdir()
    (archive / "old.md").write_text("old", encoding="utf-8")
    (roots["wiki"] / "keep.md").write_text("keep", encoding="utf-8")

    assert count_tree("wiki", roots=roots) == 1


def test_count_tree_missing_root_directory_is_zero(tmp_path):
    """count_tree on a root that does not exist on disk yet returns 0, not FileNotFound."""
    from app.read import count_tree

    roots = {
        "docs": tmp_path / "docs",
        "raw": tmp_path / "raw",  # deliberately not created
        "wiki": tmp_path / "wiki",
    }
    assert count_tree("raw", roots=roots) == 0


def test_count_tree_on_file_raises_not_a_file(tmp_path):
    """count_tree called on a file path raises NotAFile."""
    from app.read import NotAFile, count_tree

    roots = _make_roots(tmp_path)
    (roots["docs"] / "leaf.md").write_text("# Leaf\n", encoding="utf-8")

    with pytest.raises(NotAFile):
        count_tree("docs/leaf.md", roots=roots)


def test_count_tree_empty_relpath_raises_path_rejected(tmp_path):
    """count_tree with an empty relpath raises PathRejected (no root-listing mode)."""
    from app.read import PathRejected, count_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        count_tree("", roots=roots)


def test_count_tree_unknown_root_rejected(tmp_path):
    """count_tree with an unknown root name raises PathRejected."""
    from app.read import PathRejected, count_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        count_tree("secrets", roots=roots)


def test_count_tree_dotdot_rejected(tmp_path):
    """count_tree rejects path traversal like list_tree does."""
    from app.read import PathRejected, count_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        count_tree("docs/../wiki", roots=roots)


def test_count_tree_kb_root_not_in_whitelist_rejected(tmp_path):
    """.kb/ is not in the whitelist — count_tree('.kb') raises PathRejected."""
    from app.read import PathRejected, count_tree

    roots = _make_roots(tmp_path)
    with pytest.raises(PathRejected):
        count_tree(".kb", roots=roots)
