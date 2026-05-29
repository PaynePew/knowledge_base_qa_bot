"""Hermetic unit tests for the Upload deep module (markdown_kb.app.upload).

AC coverage (issue #169 — Phase 15 S1):
  - Extension routing: .html/.txt land in raw/, .md lands in docs/
  - Type rejection: unsupported extensions return status=rejected with reason
  - Size limit enforcement: files > MAX_UPLOAD_BYTES are rejected
  - Traversal-safe filename: '..' and absolute paths are rejected
  - Structured per-file UploadFileResult returned for each input
  - Wiki Log events emitted: upload_batch_started, upload_file,
    upload_rejected, upload_error, upload_batch_completed

No OPENAI_API_KEY required — the upload module is pure I/O, no LLM calls.
Uses tmp_path + monkeypatch to redirect raw/ and docs/ (same pattern as
test_import_html_happy_path.py / test_import_failure_modes.py).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_upload_file(filename: str, content: bytes) -> tuple[str, bytes]:
    """Return (filename, content) tuple as expected by upload_files."""
    return filename, content


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def upload_env(tmp_path, monkeypatch):
    """Wire RAW_DIR and DOCS_DIR into the upload module and logger for isolation."""
    from app import logger as logger_module
    from app import upload as upload_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(upload_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(upload_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")

    return {
        "raw_dir": raw_dir,
        "docs_dir": docs_dir,
        "log_path": tmp_path / "wiki" / "log.md",
    }


# ---------------------------------------------------------------------------
# Extension routing tests (RED → GREEN for each)
# ---------------------------------------------------------------------------


def test_html_file_lands_in_raw(upload_env):
    """.html files are staged to raw/."""
    from app.upload import upload_files

    raw_dir = upload_env["raw_dir"]

    files = [("page.html", b"<h1>Hello</h1><p>World.</p>")]
    batch = upload_files(files)

    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.status == "written"
    assert result.filename == "page.html"
    # target_dir must be the raw/ dir
    assert "raw" in result.target_dir
    # file actually exists on disk
    assert (raw_dir / "page.html").exists()


def test_txt_file_lands_in_raw(upload_env):
    """.txt files are staged to raw/."""
    from app.upload import upload_files

    raw_dir = upload_env["raw_dir"]

    files = [("notes.txt", b"Line one\nLine two\n")]
    batch = upload_files(files)

    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.status == "written"
    assert (raw_dir / "notes.txt").exists()


def test_md_file_lands_in_docs(upload_env):
    """.md files land directly in docs/ (skipping Import)."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]

    files = [("policy.md", b"# Policy\n\nContent here.\n")]
    batch = upload_files(files)

    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.status == "written"
    assert "docs" in result.target_dir
    assert (docs_dir / "policy.md").exists()


# ---------------------------------------------------------------------------
# Type rejection tests
# ---------------------------------------------------------------------------


def test_unsupported_extension_rejected(upload_env):
    """Files with extensions not in the allow-list are rejected."""
    from app.upload import upload_files

    files = [("report.pdf", b"%PDF-1.4")]
    batch = upload_files(files)

    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.status == "rejected"
    assert result.reason is not None
    assert len(result.reason) > 0


def test_docx_rejected(upload_env):
    """.docx files are rejected."""
    from app.upload import upload_files

    files = [("doc.docx", b"PK\x03\x04")]
    batch = upload_files(files)

    assert batch.results[0].status == "rejected"


def test_no_extension_rejected(upload_env):
    """Files with no extension are rejected."""
    from app.upload import upload_files

    files = [("README", b"Just some text")]
    batch = upload_files(files)

    assert batch.results[0].status == "rejected"


# ---------------------------------------------------------------------------
# Size limit tests
# ---------------------------------------------------------------------------


def test_oversized_file_rejected(upload_env):
    """Files exceeding the size limit are rejected without writing."""
    from app import upload as upload_module
    from app.upload import upload_files

    # Temporarily set a tiny limit to test without huge allocations
    original_limit = upload_module.MAX_UPLOAD_BYTES
    upload_module.MAX_UPLOAD_BYTES = 10  # 10 bytes

    try:
        files = [("big.html", b"<h1>This is a large file</h1>")]
        batch = upload_files(files)
    finally:
        upload_module.MAX_UPLOAD_BYTES = original_limit

    assert batch.results[0].status == "rejected"
    assert batch.results[0].reason is not None


def test_exactly_max_size_allowed(upload_env):
    """A file exactly at the size limit is accepted."""
    from app import upload as upload_module
    from app.upload import upload_files

    original_limit = upload_module.MAX_UPLOAD_BYTES
    content = b"x" * original_limit
    # Write an .html file of exactly MAX_UPLOAD_BYTES
    files = [("exact.html", content)]
    batch = upload_files(files)

    assert batch.results[0].status == "written"


# ---------------------------------------------------------------------------
# Traversal-safe filename tests
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(upload_env):
    """Filenames containing '..' are rejected."""
    from app.upload import upload_files

    files = [("../etc/passwd.html", b"<h1>Nope</h1>")]
    batch = upload_files(files)

    assert batch.results[0].status == "rejected"
    assert batch.results[0].reason is not None


def test_absolute_path_filename_rejected(upload_env):
    """Filenames that are absolute paths are rejected."""
    from app.upload import upload_files

    files = [("/etc/shadow.html", b"<h1>Nope</h1>")]
    batch = upload_files(files)

    assert batch.results[0].status == "rejected"


def test_path_separator_in_filename_rejected(upload_env):
    """Filenames with a slash in them (subdirectory notation) are rejected."""
    from app.upload import upload_files

    files = [("sub/dir.html", b"<h1>Content</h1>")]
    batch = upload_files(files)

    assert batch.results[0].status == "rejected"


def test_clean_filename_accepted(upload_env):
    """A clean filename with no path components is accepted."""
    from app.upload import upload_files

    files = [("valid-file_name.html", b"<h1>Valid</h1>")]
    batch = upload_files(files)

    assert batch.results[0].status == "written"


# ---------------------------------------------------------------------------
# Structured per-file result tests
# ---------------------------------------------------------------------------


def test_batch_result_has_per_file_entry(upload_env):
    """Each file in the batch has its own result entry."""
    from app.upload import upload_files

    files = [
        ("a.html", b"<h1>A</h1>"),
        ("b.txt", b"B content"),
        ("c.md", b"# C\n"),
        ("d.pdf", b"%PDF"),  # rejected
    ]
    batch = upload_files(files)

    assert len(batch.results) == 4

    statuses = {r.filename: r.status for r in batch.results}
    assert statuses["a.html"] == "written"
    assert statuses["b.txt"] == "written"
    assert statuses["c.md"] == "written"
    assert statuses["d.pdf"] == "rejected"


def test_rejected_result_has_reason(upload_env):
    """Rejected files have a non-empty reason string."""
    from app.upload import upload_files

    files = [("bad.exe", b"\x4d\x5a")]
    batch = upload_files(files)

    result = batch.results[0]
    assert result.status == "rejected"
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


def test_written_result_has_target_dir(upload_env):
    """Written files have a non-empty target_dir."""
    from app.upload import upload_files

    files = [("ok.txt", b"content")]
    batch = upload_files(files)

    result = batch.results[0]
    assert result.status == "written"
    assert isinstance(result.target_dir, str)
    assert len(result.target_dir) > 0


# ---------------------------------------------------------------------------
# Wiki Log event tests
# ---------------------------------------------------------------------------


def test_upload_emits_batch_started_and_completed(upload_env):
    """upload_batch_started and upload_batch_completed events are emitted."""
    from app.upload import upload_files

    log_path = upload_env["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    files = [("item.html", b"<p>Hello</p>")]
    upload_files(files)

    log_text = log_path.read_text(encoding="utf-8")
    assert "upload_batch_started" in log_text
    assert "upload_batch_completed" in log_text


def test_upload_emits_upload_file_for_written(upload_env):
    """upload_file event is emitted for each successfully written file."""
    from app.upload import upload_files

    log_path = upload_env["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    files = [("ok.html", b"<h1>OK</h1>")]
    upload_files(files)

    log_text = log_path.read_text(encoding="utf-8")
    assert "upload_file" in log_text


def test_upload_emits_upload_rejected_for_rejected(upload_env):
    """upload_rejected event is emitted for each rejected file."""
    from app.upload import upload_files

    log_path = upload_env["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    files = [("bad.zip", b"PK")]
    upload_files(files)

    log_text = log_path.read_text(encoding="utf-8")
    assert "upload_rejected" in log_text


def test_atomic_write_no_tmp_lingers(upload_env):
    """No .tmp file remains in raw/ or docs/ after a successful upload."""
    from app.upload import upload_files

    raw_dir = upload_env["raw_dir"]
    docs_dir = upload_env["docs_dir"]

    files = [("page.html", b"<p>Clean</p>"), ("note.md", b"# Note\n")]
    upload_files(files)

    raw_tmps = list(raw_dir.glob("*.tmp"))
    docs_tmps = list(docs_dir.glob("*.tmp"))
    assert raw_tmps == [], f"Stale .tmp in raw/: {raw_tmps}"
    assert docs_tmps == [], f"Stale .tmp in docs/: {docs_tmps}"


# ---------------------------------------------------------------------------
# Empty batch
# ---------------------------------------------------------------------------


def test_empty_batch_returns_empty_results(upload_env):
    """Uploading an empty batch returns an UploadBatchResult with no results."""
    from app.upload import upload_files

    batch = upload_files([])
    assert batch.results == []
