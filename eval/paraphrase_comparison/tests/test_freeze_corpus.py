"""Tests for freeze_corpus() hermetic snapshot (AC3, issue #142).

freeze_corpus() copies docs/fake-docs/ into eval/paraphrase_comparison/corpus/
at eval-run time so Phase 8 always operates on a frozen snapshot, not mutable
demo content. All tests are offline and deterministic (pure filesystem I/O,
no LLM calls).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.paraphrase_comparison.generate_paraphrases import freeze_corpus


@pytest.fixture()
def tmp_source(tmp_path: Path) -> Path:
    """A small fake source directory with two markdown files."""
    src = tmp_path / "source"
    src.mkdir()
    (src / "alpha.md").write_text("# Alpha\n\nAlpha content.\n", encoding="utf-8")
    (src / "beta.md").write_text("# Beta\n\nBeta content.\n", encoding="utf-8")
    return src


@pytest.fixture()
def tmp_dest(tmp_path: Path) -> Path:
    """An initially empty destination directory."""
    dest = tmp_path / "corpus"
    dest.mkdir()
    return dest


def test_freeze_corpus_copies_all_md_files(tmp_source: Path, tmp_dest: Path) -> None:
    """freeze_corpus() copies every *.md from source_dir into dest_dir."""
    n = freeze_corpus(source_dir=tmp_source, dest_dir=tmp_dest)
    assert n == 2
    assert (tmp_dest / "alpha.md").exists()
    assert (tmp_dest / "beta.md").exists()


def test_freeze_corpus_overwrites_existing_file(
    tmp_source: Path, tmp_dest: Path
) -> None:
    """A pre-existing file in dest is overwritten with the source version."""
    (tmp_dest / "alpha.md").write_text("OLD CONTENT", encoding="utf-8")
    freeze_corpus(source_dir=tmp_source, dest_dir=tmp_dest)
    assert (tmp_dest / "alpha.md").read_text(encoding="utf-8") == (
        tmp_source / "alpha.md"
    ).read_text(encoding="utf-8")


def test_freeze_corpus_removes_stale_files(tmp_source: Path, tmp_dest: Path) -> None:
    """Files in dest that are absent from source are removed (exact mirror)."""
    (tmp_dest / "stale.md").write_text("STALE", encoding="utf-8")
    freeze_corpus(source_dir=tmp_source, dest_dir=tmp_dest)
    assert not (tmp_dest / "stale.md").exists()


def test_freeze_corpus_creates_dest_if_missing(
    tmp_path: Path, tmp_source: Path
) -> None:
    """freeze_corpus() creates dest_dir if it does not yet exist."""
    dest = tmp_path / "new_corpus"
    assert not dest.exists()
    n = freeze_corpus(source_dir=tmp_source, dest_dir=dest)
    assert dest.is_dir()
    assert n == 2


def test_freeze_corpus_raises_on_missing_source(tmp_dest: Path) -> None:
    """freeze_corpus() raises FileNotFoundError when source_dir is missing."""
    with pytest.raises(FileNotFoundError, match="source_dir not found"):
        freeze_corpus(source_dir=tmp_dest / "nonexistent", dest_dir=tmp_dest)


def test_freeze_corpus_returns_file_count(tmp_source: Path, tmp_dest: Path) -> None:
    """Return value equals the number of *.md files in source_dir."""
    # Add a third file to source.
    (tmp_source / "gamma.md").write_text(
        "# Gamma\n\nGamma content.\n", encoding="utf-8"
    )
    n = freeze_corpus(source_dir=tmp_source, dest_dir=tmp_dest)
    assert n == 3


def test_freeze_corpus_is_idempotent(tmp_source: Path, tmp_dest: Path) -> None:
    """Calling freeze_corpus() twice yields the same state (idempotent)."""
    freeze_corpus(source_dir=tmp_source, dest_dir=tmp_dest)
    freeze_corpus(source_dir=tmp_source, dest_dir=tmp_dest)
    assert (tmp_dest / "alpha.md").read_text(encoding="utf-8") == (
        tmp_source / "alpha.md"
    ).read_text(encoding="utf-8")
    # File count unchanged.
    assert len(list(tmp_dest.glob("*.md"))) == 2
