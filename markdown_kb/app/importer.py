"""Deep module per Ousterhout. Public surface: ``import_sources``, ``ImportBatchResult``, ``ImportSourceResult``, ``ImportFailure``.

Multi-Format Import coordinator — raw/ → docs/ format-conversion pipeline.

Provides ``import_sources(source_filter)`` which converts raw ``.html`` and
``.txt`` files into normalized Markdown files in ``docs/`` with provenance
frontmatter.  This is a mechanical conversion (no LLM calls) — the import
path is completely disjoint from ``/ingest``.

Pipeline per source file:
    1. Resolve source path(s): batch = glob ``raw/**/*.{html,txt}``; single =
       ``raw_dir / source_filter``.
    2. Validate: NFC-normalize filename, reject invalid filenames and source
       paths, check extension, existence, size, UTF-8 encoding.
    3. Convert: ``.html`` → markdownify with semantic whitelist + strip list;
       ``.txt`` → passthrough (no heading inference).
    4. Render output: YAML frontmatter (imported_from, original_format,
       imported_at) + converted body.
    5. Atomic write: tempfile in same dir + ``os.replace`` + cleanup on
       exception (CODING_STANDARD §2.6).
    6. Emit Wiki Log events: ``import_batch_started``, ``import_source``
       (per success), ``import_error`` (per failure),
       ``import_batch_completed``.

Error handling (slice 7-2 — full 12 typed failure modes):
    - ``InvalidFilename``        — basename contains rejected character class
    - ``InvalidSourcePath``      — single-mode source format violation
    - ``FileNotFoundError``      — single-mode source not found in raw/
    - ``UnsupportedExtension``   — batch silently skips; single-mode fails
    - ``HandAuthoredCollision``  — docs target exists without imported_from
    - ``EmptySource``            — 0-byte raw file
    - ``OversizedSource``        — raw file > 10 MB hard limit
    - ``UnicodeDecodeError``     — raw file not UTF-8
    - ``MarkdownifyError``       — markdownify internal exception
    - ``FilenameCollision``      — two batch files map to same docs basename
    - ``IOError``                — atomic-write OS failure

Continue-on-error: one failing source does not abort the batch.

Concurrency: inherits Phase 3 Q7 single-writer assumption — no new lock.
``docs/`` is the only write target; reads from ``raw/`` only.

See PRD #89 (Phase 7) and issue #91 (Slice 7-2) for the full design.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup, Comment
from markdownify import markdownify

from ._paths import _REPO_ROOT, DOCS_DIR
from .logger import log_event

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RAW_DIR: Path = _REPO_ROOT / "raw"

# Maximum raw file size in bytes — protects markdownify from in-memory OOM.
_MAX_SOURCE_BYTES: int = 10 * 1024 * 1024  # 10 MB

# HTML elements to preserve (semantic whitelist per PRD #89 §7)
_HTML_CONVERT_TAGS = [
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "li",
    "strong",
    "b",
    "em",
    "i",
    "a",
    "code",
    "pre",
    "blockquote",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "img",
    "hr",
]

# HTML elements to strip (strip list per PRD #89 §7)
_HTML_STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    "input",
    "button",
    "meta",
    "link",
]

_SUPPORTED_EXTENSIONS = {".html", ".txt"}

# Rejected characters in basenames per PRD #89 §"Filename validation":
#   - '#'  breaks Section.id contract {filename}#{heading_slug}
#   - '/'  path separator (defensive — basename should not contain it)
#   - '\\' Windows path separator
#   - ':'  Windows reserved; macOS resource-fork notation
#   - control chars \x00–\x1f
#   - Bidi control chars (CVE-2021-42574 Trojan Source):
#       U+202A LRE, U+202B RLE, U+202C PDF, U+202D LRO, U+202E RLO,
#       U+2066 LRI, U+2067 RLI, U+2068 FSI, U+2069 PDI
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩")
_CONTROL_RE = re.compile(r"[\x00-\x1f]")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ImportSourceResult:
    """Per-source successful import outcome.

    ``raw_path`` is the path to the raw source (relative string for the API).
    ``docs_path`` is the output Markdown file path (relative string).
    ``original_format`` is ``'html'`` or ``'txt'``.
    ``content_sha256`` is empty in slice 7-1/7-2 (populated in slice 7-3).
    ``status`` is ``'created'`` in slice 7-1.
    """

    raw_path: str
    docs_path: str
    original_format: Literal["html", "txt"]
    content_sha256: str = ""
    status: Literal["created", "updated", "skipped"] = "created"


@dataclass
class ImportFailure:
    """Per-source failure record.

    ``raw_path`` is the path string (best-effort; may be the requested source name).
    ``error_type`` is one of the 12 typed failure mode strings (PRD #89 §7-2).
    ``error_message`` is truncated to 200 characters, no stack trace.
    """

    raw_path: str
    error_type: str
    error_message: str


@dataclass
class ImportBatchResult:
    """Aggregated outcome of import_sources.

    ``imported_sources`` lists one ImportSourceResult per successfully processed file.
    ``skipped_sources`` is empty in slice 7-2 (populated in slice 7-3 for hash-match).
    ``failed_sources`` lists ImportFailure records for files that could not be processed.
    """

    imported_sources: list[ImportSourceResult] = field(default_factory=list)
    skipped_sources: list[ImportSourceResult] = field(default_factory=list)
    failed_sources: list[ImportFailure] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_sources(source_filter: str | None) -> ImportBatchResult:
    """Convert raw sources to Markdown docs.

    ``source_filter=None``: batch mode — globs ``raw/**/*.{html,txt}`` recursively.
    ``source_filter='foo.html'``: single mode — processes one named file.

    NFC normalization is applied to ``source_filter`` (single-mode) and to
    each glob basename (batch mode) on entry.

    Returns an ``ImportBatchResult`` summarising all outcomes.
    Wiki Log events are emitted: ``import_batch_started``, ``import_source``
    (per success), ``import_error`` (per failure), ``import_batch_completed``.
    """
    batch_start = time.monotonic()
    result = ImportBatchResult()

    log_event(
        "import_batch_started",
        f"mode={'single' if source_filter else 'batch'} source={source_filter or '*'}",
    )

    if source_filter is None:
        # Batch mode: glob all supported extensions
        raw_paths = _collect_batch_sources()
    else:
        # Single mode: NFC-normalize, validate path, resolve to absolute
        source_filter = unicodedata.normalize("NFC", source_filter)
        raw_path, failure = _resolve_single_source(source_filter)
        if failure is not None:
            result.failed_sources.append(failure)
            _emit_import_error(failure)
            _emit_batch_completed(result, batch_start)
            return result
        raw_paths = [raw_path]

    # Track docs basenames seen in this batch for FilenameCollision detection.
    seen_docs_basenames: dict[str, str] = {}

    for raw_path in raw_paths:
        # NFC-normalize the stem used for output filename (batch glob basenames).
        stem = unicodedata.normalize("NFC", raw_path.stem)
        _process_one_source(raw_path, stem, result, seen_docs_basenames)

    _emit_batch_completed(result, batch_start)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_batch_sources() -> list[Path]:
    """Glob raw/**/*.{html,txt} recursively.

    Returns all matching absolute Paths. Unsupported extensions are silently
    skipped (per PRD #89 §23 batch mode semantics).
    """
    sources: list[Path] = []
    if not RAW_DIR.exists():
        return sources
    for ext in (".html", ".txt"):
        sources.extend(RAW_DIR.glob(f"**/*{ext}"))
    return sources


def _validate_filename(basename: str, raw_path_str: str) -> ImportFailure | None:
    """Return an ImportFailure if the NFC-normalized basename contains a rejected character.

    Rejected character classes (PRD #89 §"Filename validation"):
      1. '#'  — breaks Section.id contract {filename}#{heading_slug}
      2. '/'  — path separator (defensive)
      3. '\\'  — Windows path separator
      4. ':'  — Windows reserved; macOS resource-fork notation
      5. control chars \\x00–\\x1f
      6. bidi control chars U+202A-E, U+2066-9 (CVE-2021-42574 Trojan Source)
    """
    for ch in ("#", "/", "\\", ":"):
        if ch in basename:
            return ImportFailure(
                raw_path=raw_path_str,
                error_type="InvalidFilename",
                error_message=f"Basename contains rejected character {repr(ch)}: {basename}"[:200],
            )
    if _CONTROL_RE.search(basename):
        return ImportFailure(
            raw_path=raw_path_str,
            error_type="InvalidFilename",
            error_message=f"Basename contains control character: {repr(basename)}"[:200],
        )
    for bidi in _BIDI_CONTROLS:
        if bidi in basename:
            return ImportFailure(
                raw_path=raw_path_str,
                error_type="InvalidFilename",
                error_message=(
                    f"Basename contains bidi control character U+{ord(bidi):04X} "
                    f"(CVE-2021-42574): {repr(basename)}"
                )[:200],
            )
    return None


def _validate_source_path(source_filter: str) -> ImportFailure | None:
    """Validate single-mode source_filter format.

    Rejects (PRD #89 §"Single-mode source path validation"):
      1. Absolute paths (starts with '/' or matches Windows drive letter like 'C:\\')
      2. '..' traversal segments
      3. 'raw/' prefix (require 'foo.html' not 'raw/foo.html')

    Additionally resolves ``RAW_DIR / source_filter`` and verifies the result
    does not escape ``RAW_DIR`` (symlink traversal defence).
    """
    # Rule 1: reject absolute paths
    p = Path(source_filter)
    if p.is_absolute():
        return ImportFailure(
            raw_path=source_filter,
            error_type="InvalidSourcePath",
            error_message=f"Absolute paths are not allowed: {source_filter}"[:200],
        )
    # Rule 2: reject '..' traversal
    if ".." in p.parts:
        return ImportFailure(
            raw_path=source_filter,
            error_type="InvalidSourcePath",
            error_message=f"Path traversal ('..') is not allowed: {source_filter}"[:200],
        )
    # Rule 3: reject 'raw/' prefix
    parts = p.parts
    if parts and parts[0].lower() == "raw":
        return ImportFailure(
            raw_path=source_filter,
            error_type="InvalidSourcePath",
            error_message=(
                f"Source must be relative to raw/ directory, not include 'raw/' prefix: "
                f"{source_filter}"
            )[:200],
        )
    # Rule 4: resolved path must stay within RAW_DIR
    resolved = (RAW_DIR / source_filter).resolve()
    raw_dir_resolved = RAW_DIR.resolve()
    try:
        resolved.relative_to(raw_dir_resolved)
    except ValueError:
        return ImportFailure(
            raw_path=source_filter,
            error_type="InvalidSourcePath",
            error_message=f"Path escapes raw/ directory after resolution: {source_filter}"[:200],
        )
    return None


def _resolve_single_source(source_filter: str) -> tuple[Path, ImportFailure | None]:
    """Resolve a single-mode source_filter to an absolute Path.

    Applies validation in this order:
      1. Source path format validation (InvalidSourcePath)
      2. Filename validation on the basename (InvalidFilename)
      3. Supported extension check (UnsupportedExtension)
      4. File existence check (FileNotFoundError)

    Returns ``(path, None)`` on success or ``(Path(''), failure)`` on error.
    """
    # Rule 1: source path validation
    path_failure = _validate_source_path(source_filter)
    if path_failure is not None:
        return Path(""), path_failure

    raw_path = RAW_DIR / source_filter
    basename = raw_path.name

    # Rule 2: filename validation
    filename_failure = _validate_filename(basename, str(raw_path))
    if filename_failure is not None:
        return Path(""), filename_failure

    # Rule 3: extension check (single-mode fails with typed error; batch silently skips)
    ext = raw_path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        return Path(""), ImportFailure(
            raw_path=str(raw_path),
            error_type="UnsupportedExtension",
            error_message=f"Unsupported file extension '{ext}': {source_filter}"[:200],
        )

    # Rule 4: existence check
    if not raw_path.exists():
        return Path(""), ImportFailure(
            raw_path=str(raw_path),
            error_type="FileNotFoundError",
            error_message=f"File not found: {source_filter}"[:200],
        )

    return raw_path, None


def _process_one_source(
    raw_path: Path,
    stem: str,
    result: ImportBatchResult,
    seen_docs_basenames: dict[str, str],
) -> None:
    """Convert one raw source and write its Markdown output to docs/.

    Updates ``result`` in-place: appends to ``imported_sources`` on success or
    to ``failed_sources`` on any typed failure.

    ``stem`` is the NFC-normalized stem to use for the output filename.
    ``seen_docs_basenames`` tracks docs basenames already claimed in this batch
    for FilenameCollision detection.
    """
    ext = raw_path.suffix.lower()
    basename = raw_path.name
    docs_filename = f"{stem}.md"
    docs_path = DOCS_DIR / docs_filename

    # Filename validation for batch-mode files.
    filename_failure = _validate_filename(basename, str(raw_path))
    if filename_failure is not None:
        result.failed_sources.append(filename_failure)
        _emit_import_error(filename_failure)
        return

    # Determine format — unsupported extensions are silently skipped in batch mode.
    if ext == ".html":
        fmt: Literal["html", "txt"] = "html"
    elif ext == ".txt":
        fmt = "txt"
    else:
        # Batch mode: silent skip (no failure entry, no log)
        return

    # FilenameCollision: two batch sources produce the same docs/<stem>.md
    docs_basename = docs_filename.lower()
    if docs_basename in seen_docs_basenames:
        first_raw = seen_docs_basenames[docs_basename]
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="FilenameCollision",
            error_message=(
                f"Docs basename collision: '{docs_filename}' already claimed by '{first_raw}'"
            )[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return
    seen_docs_basenames[docs_basename] = str(raw_path)

    # HandAuthoredCollision: docs target exists without imported_from frontmatter
    hand_collision = _check_hand_authored_collision(docs_path)
    if hand_collision is not None:
        result.failed_sources.append(hand_collision)
        _emit_import_error(hand_collision)
        return

    # EmptySource: 0-byte file
    try:
        file_size = raw_path.stat().st_size
    except OSError as exc:
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="IOError",
            error_message=str(exc)[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return

    if file_size == 0:
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="EmptySource",
            error_message=f"File is empty (0 bytes): {basename}"[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return

    # OversizedSource: > 10 MB — protect markdownify from in-memory OOM
    if file_size > _MAX_SOURCE_BYTES:
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="OversizedSource",
            error_message=(f"File exceeds 10 MB limit ({file_size} bytes): {basename}")[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return

    # Read raw content — UnicodeDecodeError for non-UTF-8
    try:
        raw_text = raw_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="UnicodeDecodeError",
            error_message=str(exc)[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return
    except OSError as exc:
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="IOError",
            error_message=str(exc)[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return

    # Convert to Markdown — MarkdownifyError for markdownify internal exception
    try:
        md_body = _convert_to_markdown(raw_text, fmt)
    except Exception as exc:
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="MarkdownifyError",
            error_message=str(exc)[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return

    # Build full output with frontmatter
    output = _render_output(md_body, raw_path, fmt)

    # Atomic write — IOError for os.replace failure
    try:
        _atomic_write(output, docs_path)
    except OSError as exc:
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="IOError",
            error_message=str(exc)[:200],
        )
        result.failed_sources.append(failure)
        _emit_import_error(failure)
        return

    log_event(
        "import_source",
        f"source={basename} docs={docs_filename} format={fmt}",
    )

    result.imported_sources.append(
        ImportSourceResult(
            raw_path=str(raw_path),
            docs_path=str(docs_path),
            original_format=fmt,
            content_sha256="",  # populated in slice 7-3
            status="created",
        )
    )


def _check_hand_authored_collision(docs_path: Path) -> ImportFailure | None:
    """Return ImportFailure if docs_path exists without ``imported_from`` frontmatter.

    A docs file that lacks ``imported_from`` is assumed to be hand-authored.
    Overwriting it would destroy curator work, so the import is refused.
    """
    if not docs_path.exists():
        return None
    try:
        content = docs_path.read_text(encoding="utf-8")
    except OSError:
        # If we can't read it, play it safe and refuse to overwrite.
        return ImportFailure(
            raw_path=str(docs_path),
            error_type="HandAuthoredCollision",
            error_message=f"Could not read existing docs file to check provenance: {docs_path.name}"[
                :200
            ],
        )
    # Check for imported_from in the YAML frontmatter block
    if content.startswith("---\n") and "imported_from:" in content:
        # Has provenance — safe to overwrite (existing import, not hand-authored)
        return None
    return ImportFailure(
        raw_path=str(docs_path),
        error_type="HandAuthoredCollision",
        error_message=(
            f"Docs file exists without 'imported_from' frontmatter (hand-authored?): "
            f"{docs_path.name}"
        )[:200],
    )


def _convert_to_markdown(raw_text: str, fmt: Literal["html", "txt"]) -> str:
    """Convert raw source text to Markdown body.

    For ``html``: apply markdownify with semantic whitelist; strip the strip-list tags.
    For ``txt``: passthrough — return the raw text unchanged.
    """
    if fmt == "txt":
        return raw_text

    # Build BeautifulSoup parse and strip unwanted tags before markdownify
    soup = BeautifulSoup(raw_text, "html.parser")

    # Strip unwanted tags and their content
    for tag_name in _HTML_STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Strip HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Strip data-* attributes from all remaining tags
    for tag in soup.find_all(True):
        data_attrs = [k for k in list(tag.attrs.keys()) if k.startswith("data-")]
        for attr in data_attrs:
            del tag.attrs[attr]

    # Convert remaining HTML to Markdown using markdownify
    md = markdownify(str(soup), heading_style="ATX", strip=_HTML_STRIP_TAGS)

    # Clean up excessive blank lines (markdownify can leave many)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    return md


def _render_output(
    md_body: str,
    raw_path: Path,
    fmt: Literal["html", "txt"],
) -> str:
    """Build the full docs/*.md content: YAML frontmatter + converted body.

    Frontmatter fields (slice 7-1/7-2): imported_from, original_format, imported_at.
    ``content_sha256`` is intentionally omitted until slice 7-3.
    """
    # Compute relative raw path for the frontmatter (relative to REPO_ROOT)
    try:
        rel_raw = raw_path.relative_to(_REPO_ROOT)
    except ValueError:
        rel_raw = raw_path  # absolute fallback for tests with tmp_path

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Quote imported_at as a string so PyYAML round-trips it as str, not datetime.
    frontmatter = (
        f"---\nimported_from: {rel_raw}\noriginal_format: {fmt}\nimported_at: '{ts}'\n---\n"
    )
    return frontmatter + "\n" + md_body + "\n"


def _atomic_write(content: str, target: Path) -> None:
    """Write content to target atomically via tempfile + os.replace.

    Reuses the wiki_writer.py pattern (CODING_STANDARD §2.6):
    - Create tempfile in same directory (ensures same filesystem for os.replace).
    - Write content.
    - os.replace to atomically overwrite target.
    - On any exception: clean up tempfile and re-raise as IOError.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=target.parent,
        suffix=".tmp",
        prefix=f"{target.stem}_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path_str, target)
    except Exception as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path_str)
        raise OSError(f"Atomic write failed for {target}: {exc}") from exc


def _emit_import_error(failure: ImportFailure) -> None:
    """Emit an ``import_error`` Wiki Log event for a failed source.

    Per log-kinds.md §'/import' route: payload is
    ``raw=<raw_path> error_type=<type> error_message=<truncated≤200>``.
    """
    log_event(
        "import_error",
        f"raw={failure.raw_path} "
        f"error_type={failure.error_type} "
        f"error_message={failure.error_message[:200]!r}",
    )


def _emit_batch_completed(result: ImportBatchResult, batch_start: float) -> None:
    """Emit the import_batch_completed Wiki Log event."""
    duration_ms = int((time.monotonic() - batch_start) * 1000)
    log_event(
        "import_batch_completed",
        f"imported={len(result.imported_sources)} "
        f"skipped={len(result.skipped_sources)} "
        f"failed={len(result.failed_sources)} "
        f"duration_ms={duration_ms}",
    )
