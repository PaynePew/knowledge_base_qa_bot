"""Hermetic unit tests for the Upload deep module (markdown_kb.app.upload).

AC coverage (issue #169 — Phase 15 S1; issue #417 — PDF added to the allow-list):
  - Extension routing: .html/.txt/.pdf land in raw/, .md lands in docs/
  - Type rejection: unsupported extensions return status=rejected with reason
  - Size limit enforcement: files > MAX_UPLOAD_BYTES are rejected
  - Traversal-safe filename: '..' and absolute paths are rejected
  - Structured per-file UploadFileResult returned for each input
  - Wiki Log events emitted: upload_batch_started, upload_file,
    upload_rejected, upload_error, upload_batch_completed

AC coverage (issue #533 — destination-aware in-place Source overwrite,
ADR-0036 §6):
  - overwrite_relpath overwrites an existing subdirectory Source in place;
    no second copy appears at docs/ root
  - the overwritten Source resolves cleanly (not ambiguous_source) afterward
  - ambiguous origin (basename in 2+ subdirs) is refused; nothing written
  - missing origin is refused; nothing written
  - traversal / absolute / outside-docs/ relpaths are rejected; nothing written
  - a relpath naming a different basename than the upload is rejected
  - omitting overwrite_relpath leaves the existing root-write behavior unchanged
  - the overwrite is logged (op=overwrite on the upload_file event)

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


def test_pdf_file_lands_in_raw(upload_env):
    """.pdf files are staged to raw/ as an Import candidate (issue #417, ADR-0011).

    Upload only stages the bytes; Import (not exercised here) does the
    PDF→Markdown conversion (ADR-0031).
    """
    from app.upload import upload_files

    raw_dir = upload_env["raw_dir"]

    files = [("policy.pdf", b"%PDF-1.4 fake content")]
    batch = upload_files(files)

    assert len(batch.results) == 1
    result = batch.results[0]
    assert result.status == "written"
    assert "raw" in result.target_dir
    assert (raw_dir / "policy.pdf").exists()
    # Bytes are staged verbatim — Upload never converts (ADR-0011).
    assert (raw_dir / "policy.pdf").read_bytes() == b"%PDF-1.4 fake content"


# ---------------------------------------------------------------------------
# Type rejection tests
# ---------------------------------------------------------------------------


def test_unsupported_extension_rejected(upload_env):
    """Files with extensions not in the allow-list are rejected.

    Note: .pdf was the example extension here prior to issue #417 (ADR-0011),
    which added .pdf to the Upload allow-list. .pptx now exercises the same
    AC — an extension genuinely outside the supported set.
    """
    from app.upload import upload_files

    files = [("report.pptx", b"PK\x03\x04")]
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
    """Each file in the batch has its own result entry.

    Note: the 4th (rejected) file was .pdf prior to issue #417 (ADR-0011),
    which added .pdf to the allow-list; .pptx now exercises the same
    unsupported-extension AC.
    """
    from app.upload import upload_files

    files = [
        ("a.html", b"<h1>A</h1>"),
        ("b.txt", b"B content"),
        ("c.md", b"# C\n"),
        ("d.pptx", b"PK\x03\x04"),  # rejected
    ]
    batch = upload_files(files)

    assert len(batch.results) == 4

    statuses = {r.filename: r.status for r in batch.results}
    assert statuses["a.html"] == "written"
    assert statuses["b.txt"] == "written"
    assert statuses["c.md"] == "written"
    assert statuses["d.pptx"] == "rejected"


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
# Destination-aware overwrite (issue #533, ADR-0036 §6)
# ---------------------------------------------------------------------------


def test_overwrite_relpath_writes_in_place(upload_env):
    """overwrite_relpath overwrites an existing subdirectory Source in place.

    No second copy appears at docs/ root (the C5/C3 fix-source loop-breaking
    gap ADR-0036 §6 closes).
    """
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    subdir = docs_dir / "planted-zh"
    subdir.mkdir()
    original = subdir / "退貨期限提醒.md"
    original.write_text("# 退貨期限提醒\n\n30 天\n", encoding="utf-8")

    corrected = "# 退貨期限提醒\n\n14 天\n".encode()
    files = [("退貨期限提醒.md", corrected)]
    batch = upload_files(files, overwrite_relpath="docs/planted-zh/退貨期限提醒.md")

    result = batch.results[0]
    assert result.status == "written"
    assert original.read_bytes() == corrected
    assert not (docs_dir / "退貨期限提醒.md").exists()


def test_overwrite_relpath_resolves_cleanly_afterward(upload_env):
    """After an overwrite, the Source resolves to exactly one match.

    Reuses the same basename-glob rule C3's fix-source / re-ingest use
    (mirrored locally in upload.py) to prove the write doesn't leave a state
    that would raise ``ambiguous_source`` on the next re-ingest.
    """
    from app.upload import _resolve_overwrite_target, upload_files

    docs_dir = upload_env["docs_dir"]
    subdir = docs_dir / "demo-zh"
    subdir.mkdir()
    (subdir / "退款與退貨.md").write_text("# 退款與退貨\n\n14 天\n", encoding="utf-8")

    files = [("退款與退貨.md", "# 退款與退貨\n\n14 天 non-perishable\n".encode())]
    batch = upload_files(files, overwrite_relpath="docs/demo-zh/退款與退貨.md")
    assert batch.results[0].status == "written"

    matches = sorted(docs_dir.glob("**/退款與退貨.md"))
    assert len(matches) == 1
    # A second overwrite (the natural "fix again" case) still resolves
    # cleanly — the guard's own resolver reports no ambiguity.
    target_path, refusal = _resolve_overwrite_target(
        "退款與退貨.md", ".md", docs_dir, "docs/demo-zh/退款與退貨.md"
    )
    assert refusal == ""
    assert target_path == matches[0]


def test_overwrite_relpath_missing_origin_refused(upload_env):
    """A relpath naming a Source that doesn't exist anywhere is refused."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    files = [("nope.md", b"# Nope\n")]
    batch = upload_files(files, overwrite_relpath="docs/nope.md")

    result = batch.results[0]
    assert result.status == "rejected"
    assert result.reason
    assert not (docs_dir / "nope.md").exists()


def test_overwrite_relpath_ambiguous_origin_refused(upload_env):
    """A basename matching 2+ existing files is refused; neither is touched."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    (docs_dir / "a").mkdir()
    (docs_dir / "b").mkdir()
    (docs_dir / "a" / "dup.md").write_text("# A\n", encoding="utf-8")
    (docs_dir / "b" / "dup.md").write_text("# B\n", encoding="utf-8")

    files = [("dup.md", b"# Corrected\n")]
    batch = upload_files(files, overwrite_relpath="docs/a/dup.md")

    result = batch.results[0]
    assert result.status == "rejected"
    assert result.reason
    assert (docs_dir / "a" / "dup.md").read_text(encoding="utf-8") == "# A\n"
    assert (docs_dir / "b" / "dup.md").read_text(encoding="utf-8") == "# B\n"


def test_overwrite_relpath_traversal_rejected(upload_env):
    """A relpath containing '..' is rejected; the existing file is untouched."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    subdir = docs_dir / "sub"
    subdir.mkdir()
    (subdir / "x.md").write_text("# X\n", encoding="utf-8")

    files = [("x.md", b"# Y\n")]
    batch = upload_files(files, overwrite_relpath="docs/sub/../../etc/x.md")

    assert batch.results[0].status == "rejected"
    assert (subdir / "x.md").read_text(encoding="utf-8") == "# X\n"


def test_overwrite_relpath_absolute_rejected(upload_env):
    """An absolute relpath is rejected; the existing file is untouched."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    subdir = docs_dir / "sub"
    subdir.mkdir()
    (subdir / "x.md").write_text("# X\n", encoding="utf-8")

    files = [("x.md", b"# Y\n")]
    batch = upload_files(files, overwrite_relpath="/docs/sub/x.md")

    assert batch.results[0].status == "rejected"
    assert (subdir / "x.md").read_text(encoding="utf-8") == "# X\n"


def test_overwrite_relpath_outside_docs_rejected(upload_env):
    """A relpath outside docs/ is rejected; nothing written."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    files = [("x.md", b"# Y\n")]
    batch = upload_files(files, overwrite_relpath="raw/x.md")

    result = batch.results[0]
    assert result.status == "rejected"
    assert not (docs_dir / "x.md").exists()


def test_overwrite_relpath_filename_mismatch_refused(upload_env):
    """A relpath naming a different basename than the uploaded file is refused."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    subdir = docs_dir / "demo-zh"
    subdir.mkdir()
    (subdir / "a.md").write_text("# A\n", encoding="utf-8")

    files = [("a.md", b"# Corrected\n")]
    batch = upload_files(files, overwrite_relpath="docs/demo-zh/b.md")

    result = batch.results[0]
    assert result.status == "rejected"
    assert (subdir / "a.md").read_text(encoding="utf-8") == "# A\n"


def test_overwrite_relpath_rejected_for_non_md(upload_env):
    """overwrite_relpath only applies to .md Source uploads."""
    from app.upload import upload_files

    raw_dir = upload_env["raw_dir"]
    files = [("note.txt", b"hello")]
    batch = upload_files(files, overwrite_relpath="docs/note.txt")

    assert batch.results[0].status == "rejected"
    assert not (raw_dir / "note.txt").exists()


def test_overwrite_relpath_logs_op_overwrite(upload_env):
    """A successful overwrite tags the upload_file log line with op=overwrite."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    log_path = upload_env["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    subdir = docs_dir / "demo-zh"
    subdir.mkdir()
    (subdir / "note.md").write_text("# Note\n\nOld\n", encoding="utf-8")

    files = [("note.md", b"# Note\n\nNew\n")]
    upload_files(files, overwrite_relpath="docs/demo-zh/note.md")

    log_text = log_path.read_text(encoding="utf-8")
    assert "op=overwrite" in log_text


def test_omitting_overwrite_relpath_keeps_root_write_unchanged(upload_env):
    """Omitting overwrite_relpath still writes to docs/ root, even when a
    same-named Source already exists in a subdirectory — the default
    root-write path is byte-for-byte unchanged by this feature."""
    from app.upload import upload_files

    docs_dir = upload_env["docs_dir"]
    subdir = docs_dir / "demo-zh"
    subdir.mkdir()
    original = subdir / "note.md"
    original.write_text("# Note\n\nOriginal\n", encoding="utf-8")

    files = [("note.md", b"# Note\n\nRoot copy\n")]
    batch = upload_files(files)

    result = batch.results[0]
    assert result.status == "written"
    assert (docs_dir / "note.md").read_bytes() == b"# Note\n\nRoot copy\n"
    # The subdirectory Source is untouched by the unrelated root write.
    assert original.read_text(encoding="utf-8") == "# Note\n\nOriginal\n"


# ---------------------------------------------------------------------------
# Empty batch
# ---------------------------------------------------------------------------


def test_empty_batch_returns_empty_results(upload_env):
    """Uploading an empty batch returns an UploadBatchResult with no results."""
    from app.upload import upload_files

    batch = upload_files([])
    assert batch.results == []
