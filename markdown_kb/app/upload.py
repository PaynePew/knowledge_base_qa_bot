"""Deep module per Ousterhout. Public surface: ``upload_files``, ``UploadBatchResult``, ``UploadFileResult``.

Upload staging — browser bytes → raw/ or docs/.

Provides ``upload_files(files) -> UploadBatchResult`` which validates each file
(type allow-list ``.html``/``.txt``/``.md``/``.pdf``, size limit, traversal-safe
filename), routes by extension (``.html``/``.txt``/``.pdf`` → ``raw/``, ``.md``
→ ``docs/``), writes atomically, emits ``upload_*`` Wiki Log events, and returns
a structured per-file result.

``.pdf`` is staged as an Import candidate exactly like ``.html``/``.txt`` (issue
#417, PRD #414): Upload never converts, it only stages the bytes — Import
(``POST /import``) does the PDF→Markdown extraction (ADR-0031).

Per ADR-0011, Upload is a system boundary: all untrusted-input validation lives
here. ``/import`` is UNCHANGED — Upload only stages bytes; Import converts.

Upload is completely disjoint from Import (no LLM calls, no format conversion).

Destination-aware overwrite (issue #533, ADR-0036 §6): an optional
``overwrite_relpath`` routes a ``.md`` upload to overwrite an EXISTING Source
at that resolved ``docs/`` path in place, instead of the default root write.
This closes the fix-source loop for a Source living in a subdirectory
(``docs/demo-zh/…``, ``docs/planted-zh/…``) — re-uploading the corrected file
no longer lands a second copy at ``docs/`` root that would make the next
``/ingest`` raise ``ambiguous_source``. Guard: overwrite-only of an existing
Source; the origin is independently re-resolved from the uploaded filename's
basename (mirrors ``lint._resolve_c3_source_path``'s basename-glob rule) and
must match the caller-supplied ``overwrite_relpath`` — an ambiguous/missing
origin, or a malformed (traversal/absolute/outside-``docs/``) relpath, is
refused with a clear reason and NOTHING is written; there is no root fallback.

Public surface:
    ``upload_files(files)`` — accepts a list of ``(filename, content_bytes)`` pairs,
    returns ``UploadBatchResult`` with one ``UploadFileResult`` per input file.

Result status values for UploadFileResult:
    - ``'written'``:  file staged successfully to ``raw/`` or ``docs/``.
    - ``'rejected'``: file failed validation (type, size, filename safety).
    - ``'error'``:    unexpected OS-level write failure.

Wiki Log events emitted (per ``project-docs/log-kinds.md`` — Phase 15):
    ``upload_batch_started`` / ``upload_file`` / ``upload_rejected`` /
    ``upload_error`` / ``upload_batch_completed``.

See ADR-0011 and GitHub issue #168 (Phase 15 PRD) for design rationale.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from ._paths import _REPO_ROOT, DOCS_DIR
from .atomic import write_bytes_atomic
from .logger import log_event

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RAW_DIR: Path = _REPO_ROOT / "raw"

# Maximum upload size in bytes.  10 MB matches the Import OversizedSource limit.
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Allowed file extensions and their target directories (ADR-0011).
# ``.pdf`` routes to raw/ like ``.html``/``.txt`` — an Import candidate, not a
# final Source (issue #417, PRD #414); Upload stages bytes only and never
# converts.
_ALLOWED_EXTENSIONS: dict[str, str] = {
    ".html": "raw",
    ".txt": "raw",
    ".pdf": "raw",
    ".md": "docs",
}

# Bidi control characters (CVE-2021-42574 Trojan Source) — same set as importer.py.
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")
_CONTROL_RE = re.compile(r"[\x00-\x1f]")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class UploadFileResult:
    """Per-file outcome from upload_files.

    ``filename``    — the original filename supplied by the caller.
    ``status``      — ``'written'`` | ``'rejected'`` | ``'error'``.
    ``target_dir``  — the target directory path string (set when status=written).
    ``reason``      — rejection reason string (set when status=rejected or error).
    """

    filename: str
    status: Literal["written", "rejected", "error"]
    target_dir: str = ""
    reason: str = ""


@dataclass
class UploadBatchResult:
    """Aggregated outcome of upload_files.

    ``results`` lists one ``UploadFileResult`` per input file, in the same order
    as the input.
    """

    results: list[UploadFileResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_safe_basename(filename: str) -> tuple[bool, str]:
    """Return (is_safe, rejection_reason) for a filename.

    Safe filenames are path-traversal-resistant: no directory separators,
    no '..', no control chars, no bidi control chars.
    Only the basename (no directory component) is accepted.

    Returns ``(True, '')`` if safe, ``(False, reason)`` if not.
    """
    # Reject any path separators — client must supply a plain filename.
    if "/" in filename or "\\" in filename:
        return False, f"Filename must not contain path separators: {filename!r}"

    # Reject absolute paths
    p = Path(filename)
    if p.is_absolute():
        return False, f"Absolute paths are not allowed: {filename!r}"

    # Reject '..' traversal
    if ".." in p.parts:
        return False, f"Path traversal ('..') is not allowed: {filename!r}"

    # Reject empty name
    if not filename.strip():
        return False, "Filename must not be empty."

    # Reject control characters
    if _CONTROL_RE.search(filename):
        return False, f"Filename contains control character: {filename!r}"

    # Reject bidi control characters (CVE-2021-42574)
    for bidi in _BIDI_CONTROLS:
        if bidi in filename:
            return (
                False,
                f"Filename contains bidi control character U+{ord(bidi):04X}: {filename!r}",
            )

    # Reject '#' (breaks Section.id contract {filename}#{heading-slug})
    if "#" in filename:
        return False, f"Filename must not contain '#': {filename!r}"

    return True, ""


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_write_bytes(content: bytes, target: Path) -> None:
    """Write bytes to target atomically via the shared write_bytes_atomic helper (§2.6).

    Thin wrapper that delegates to ``write_bytes_atomic`` and re-wraps any
    exception as ``OSError`` so callers continue to receive the same error contract
    (``OSError("Atomic write failed for ...")``) regardless of the underlying
    failure (e.g. ``PermissionError`` from the Windows-retry path).
    """
    try:
        write_bytes_atomic(target, content)
    except Exception as exc:
        raise OSError(f"Atomic write failed for {target}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upload_files(
    files: list[tuple[str, bytes]],
    *,
    raw_dir: Path | None = None,
    docs_dir: Path | None = None,
    overwrite_relpath: str | None = None,
) -> UploadBatchResult:
    """Stage a batch of uploaded files onto the server.

    Validates each file (filename safety, extension allow-list, size limit),
    routes by extension (``.html``/``.txt`` → ``raw/``, ``.md`` → ``docs/``),
    writes atomically, and emits Wiki Log events for audit parity with
    ``/import``.

    Args:
        files:    List of ``(filename, content_bytes)`` pairs.
        raw_dir:  Override ``RAW_DIR`` (used by tests via monkeypatch; production
                  callers may omit).
        docs_dir: Override ``DOCS_DIR`` (same).
        overwrite_relpath: When given (issue #533, ADR-0036 §6), route the
                  ``.md`` upload to overwrite that resolved, existing Source
                  path under ``docs/`` in place, instead of the default root
                  write. Refused (nothing written) when the origin can't be
                  uniquely resolved from the upload's own filename, or when
                  ``overwrite_relpath`` doesn't match that resolution — see
                  ``_resolve_overwrite_target``.

    Returns:
        ``UploadBatchResult`` with one ``UploadFileResult`` per input file.
    """
    # Resolve directories: use module-level globals by default so monkeypatch
    # works the same way as in importer.py tests.
    effective_raw_dir = raw_dir if raw_dir is not None else RAW_DIR
    effective_docs_dir = docs_dir if docs_dir is not None else DOCS_DIR

    batch_start = time.monotonic()
    result = UploadBatchResult()

    log_event(
        "upload_batch_started",
        f"files={len(files)}",
    )

    for filename, content in files:
        file_result = _upload_one_file(
            filename,
            content,
            effective_raw_dir,
            effective_docs_dir,
            overwrite_relpath=overwrite_relpath,
        )
        result.results.append(file_result)

    duration_ms = int((time.monotonic() - batch_start) * 1000)
    written = sum(1 for r in result.results if r.status == "written")
    rejected = sum(1 for r in result.results if r.status == "rejected")
    errors = sum(1 for r in result.results if r.status == "error")
    log_event(
        "upload_batch_completed",
        f"written={written} rejected={rejected} errors={errors} duration_ms={duration_ms}",
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _upload_one_file(
    filename: str,
    content: bytes,
    raw_dir: Path,
    docs_dir: Path,
    *,
    overwrite_relpath: str | None = None,
) -> UploadFileResult:
    """Process one file: validate, route, write atomically.

    Returns an ``UploadFileResult`` for the file.
    """
    # 1. Filename safety check
    is_safe, reason = _is_safe_basename(filename)
    if not is_safe:
        log_event("upload_rejected", f"filename={filename!r} reason={reason!r}")
        return UploadFileResult(filename=filename, status="rejected", reason=reason)

    # 2. Extension allow-list check
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        reason = f"Unsupported file type {ext!r}. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        log_event("upload_rejected", f"filename={filename!r} reason={reason!r}")
        return UploadFileResult(filename=filename, status="rejected", reason=reason)

    # 3. Extension must be non-empty (bare filename without dot)
    # (Already handled above — ext='' not in _ALLOWED_EXTENSIONS)

    # 4. Size limit check
    if len(content) > MAX_UPLOAD_BYTES:
        reason = (
            f"File exceeds size limit ({len(content)} bytes > {MAX_UPLOAD_BYTES} bytes): "
            f"{filename!r}"
        )
        log_event("upload_rejected", f"filename={filename!r} reason={reason!r}")
        return UploadFileResult(filename=filename, status="rejected", reason=reason)

    # 5. Destination-aware overwrite (issue #533, ADR-0036 §6) — resolved and
    # validated before any filesystem write; never falls back to a root write
    # when the origin can't be uniquely determined.
    if overwrite_relpath is not None:
        target_path, refusal = _resolve_overwrite_target(filename, ext, docs_dir, overwrite_relpath)
        if target_path is None:
            log_event("upload_rejected", f"filename={filename!r} reason={refusal!r}")
            return UploadFileResult(filename=filename, status="rejected", reason=refusal)

        try:
            _atomic_write_bytes(content, target_path)
        except OSError as exc:
            reason = str(exc)[:200]
            log_event("upload_error", f"filename={filename!r} reason={reason!r}")
            return UploadFileResult(filename=filename, status="error", reason=reason)

        log_event(
            "upload_file",
            f"filename={filename!r} target={str(target_path.parent)!r} op=overwrite",
        )
        return UploadFileResult(
            filename=filename,
            status="written",
            target_dir=str(target_path.parent),
        )

    # 6. Route to target directory (default root write, unchanged)
    target_subdir = _ALLOWED_EXTENSIONS[ext]
    target_dir = raw_dir if target_subdir == "raw" else docs_dir

    # 7. Atomic write
    target_path = target_dir / filename
    try:
        _atomic_write_bytes(content, target_path)
    except OSError as exc:
        reason = str(exc)[:200]
        log_event("upload_error", f"filename={filename!r} reason={reason!r}")
        return UploadFileResult(filename=filename, status="error", reason=reason)

    log_event("upload_file", f"filename={filename!r} target={str(target_dir)!r}")
    return UploadFileResult(
        filename=filename,
        status="written",
        target_dir=str(target_dir),
    )


def _resolve_overwrite_target(
    filename: str,
    ext: str,
    docs_dir: Path,
    overwrite_relpath: str,
) -> tuple[Path | None, str]:
    """Resolve and validate an ``overwrite_relpath`` request (issue #533).

    Guard, per ADR-0036 §6: overwrite-only of an EXISTING Source under
    ``docs_dir`` — never create new nesting, never traverse, never fall back
    to a root write. The origin is resolved from the uploaded filename's own
    basename via the same basename-glob rule C3's fix-source already uses
    (mirrors ``lint._resolve_c3_source_path`` — module-private helpers stay
    module-private per CODING_STANDARD §2.4, so the small resolution rule is
    duplicated here rather than imported, matching the existing
    ``ingest._resolve_single_source_pairs`` precedent). The caller-supplied
    ``overwrite_relpath`` must name the SAME file as the upload (same
    basename) and match that independently-resolved origin exactly — a
    mismatch, an ambiguous/missing origin, or a malformed relpath (empty,
    backslashes, absolute, ``..``, outside ``docs/``) is refused.

    Returns ``(target_path, "")`` on success, or ``(None, reason)`` when the
    request must be refused with a clear, non-empty ``reason``.
    """
    if ext != ".md":
        return None, "overwrite_relpath is only supported for .md Source uploads."

    stripped = overwrite_relpath.strip()
    if not stripped:
        return None, "overwrite_relpath must not be empty."
    if "\\" in stripped:
        return None, f"overwrite_relpath must use forward slashes: {stripped!r}"

    relpath = PurePosixPath(stripped)
    if relpath.is_absolute():
        return None, f"overwrite_relpath must not be an absolute path: {stripped!r}"
    if ".." in relpath.parts:
        return None, f"overwrite_relpath must not contain '..': {stripped!r}"
    if relpath.parts[:1] != ("docs",) or len(relpath.parts) < 2:
        return None, f"overwrite_relpath must name an existing file under 'docs/': {stripped!r}"
    if relpath.name != filename:
        return None, (
            f"overwrite_relpath basename {relpath.name!r} does not match the "
            f"uploaded filename {filename!r}."
        )

    matches = sorted(docs_dir.glob(f"**/{filename}"))
    if not matches:
        return None, f"No existing Source named {filename!r} under docs/; refusing to write."
    if len(matches) > 1:
        return (
            None,
            f"{filename!r} matches multiple Sources under docs/; ambiguous overwrite target.",
        )

    resolved = matches[0]
    resolved_label = f"docs/{resolved.relative_to(docs_dir).as_posix()}"
    if resolved_label != stripped:
        return None, (
            f"overwrite_relpath {stripped!r} does not match the resolved origin "
            f"{resolved_label!r}; refusing."
        )

    return resolved, ""
