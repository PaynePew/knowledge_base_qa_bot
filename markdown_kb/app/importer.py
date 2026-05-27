"""Deep module per Ousterhout. Public surface: ``import_sources``, ``ImportBatchResult``, ``ImportSourceResult``, ``ImportFailure``.

Multi-Format Import coordinator — raw/ → docs/ format-conversion pipeline.

Provides ``import_sources(source_filter)`` which converts raw ``.html`` and
``.txt`` files into normalized Markdown files in ``docs/`` with provenance
frontmatter.  This is a mechanical conversion (no LLM calls) — the import
path is completely disjoint from ``/ingest``.

Pipeline per source file:
    1. Resolve source path(s): batch = glob ``raw/**/*.{html,txt}``; single =
       ``raw_dir / source_filter``.
    2. Validate: extension supported, file exists, not empty, not oversized.
    3. Convert: ``.html`` → markdownify with semantic whitelist + strip list;
       ``.txt`` → passthrough (no heading inference).
    4. Render output: YAML frontmatter (imported_from, original_format,
       imported_at) + converted body.
    5. Atomic write: tempfile in same dir + ``os.replace`` + cleanup on
       exception (CODING_STANDARD §2.6).
    6. Emit Wiki Log events: ``import_batch_started``, ``import_source``
       (per success), ``import_batch_completed``.

Error handling (slice 7-1 — minimal):
    - ``FileNotFoundError`` (single-mode missing source) → ``failed_sources``
    - ``IOError`` (atomic write failure) → ``failed_sources``

Continue-on-error: one failing source does not abort the batch.

Concurrency: inherits Phase 3 Q7 single-writer assumption — no new lock.
``docs/`` is the only write target; reads from ``raw/`` only.

See PRD #89 (Phase 7) and issue #90 (Slice 7-1) for the full design.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import re
import tempfile
import time
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


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ImportSourceResult:
    """Per-source successful import outcome.

    ``raw_path`` is the path to the raw source (relative string for the API).
    ``docs_path`` is the output Markdown file path (relative string).
    ``original_format`` is ``'html'`` or ``'txt'``.
    ``content_sha256`` is empty in slice 7-1 (populated in slice 7-3).
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
    ``error_type`` is the exception class name.
    ``error_message`` is truncated to 200 characters.
    """

    raw_path: str
    error_type: str
    error_message: str


@dataclass
class ImportBatchResult:
    """Aggregated outcome of import_sources.

    ``imported_sources`` lists one ImportSourceResult per successfully processed file.
    ``skipped_sources`` is empty in slice 7-1 (populated in slice 7-3 for hash-match).
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

    Returns an ``ImportBatchResult`` summarising all outcomes.
    Wiki Log events are emitted: ``import_batch_started``, ``import_source``
    (per success), ``import_batch_completed``.
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
        # Single mode: validate path and resolve to absolute
        raw_path, failure = _resolve_single_source(source_filter)
        if failure is not None:
            result.failed_sources.append(failure)
            _emit_batch_completed(result, batch_start)
            return result
        raw_paths = [raw_path]

    for raw_path in raw_paths:
        _process_one_source(raw_path, result)

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


def _resolve_single_source(source_filter: str) -> tuple[Path, ImportFailure | None]:
    """Resolve a single-mode source_filter to an absolute Path.

    Returns ``(path, None)`` on success or ``(Path(''), failure)`` on error.
    Only FileNotFoundError is handled here (slice 7-1 minimal error handling).
    """
    raw_path = RAW_DIR / source_filter
    if not raw_path.exists():
        failure = ImportFailure(
            raw_path=str(raw_path),
            error_type="FileNotFoundError",
            error_message=f"File not found: {source_filter}"[:200],
        )
        return Path(""), failure
    return raw_path, None


def _process_one_source(raw_path: Path, result: ImportBatchResult) -> None:
    """Convert one raw source and write its Markdown output to docs/.

    Updates ``result`` in-place: appends to ``imported_sources`` on success or
    to ``failed_sources`` on ``IOError``.
    """
    ext = raw_path.suffix.lower()
    basename = raw_path.name
    stem = raw_path.stem
    docs_filename = f"{stem}.md"
    docs_path = DOCS_DIR / docs_filename

    # Determine format
    if ext == ".html":
        fmt: Literal["html", "txt"] = "html"
    else:
        fmt = "txt"

    try:
        # Read raw content
        raw_text = raw_path.read_text(encoding="utf-8")

        # Convert to Markdown
        md_body = _convert_to_markdown(raw_text, fmt)

        # Build full output with frontmatter
        output = _render_output(md_body, raw_path, fmt)

        # Atomic write
        _atomic_write(output, docs_path)

    except OSError as exc:
        result.failed_sources.append(
            ImportFailure(
                raw_path=str(raw_path),
                error_type="IOError",
                error_message=str(exc)[:200],
            )
        )
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

    Frontmatter fields (slice 7-1): imported_from, original_format, imported_at.
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
