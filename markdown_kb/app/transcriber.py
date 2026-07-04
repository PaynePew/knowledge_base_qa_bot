"""Deep module per Ousterhout. Public surface: ``transcribe_available``, ``probe_has_text_layer``, ``transcribe_pdf_bytes``, ``transcribe_source``, ``transcribe_path``, ``get_transcribe_llm``, ``set_page_budget_hook``, ``get_page_budget_hook``, ``TranscribeSourceResult``, ``TranscribePathError``, ``TranscribeUnavailable``, ``TranscribePageLimitExceeded``, ``TranscribeBudgetExceeded``, ``TranscribeError``.

Transcribe — model-assisted PDF conversion (issue #426, ADR-0032). Sits
beside ``importer.py`` as the second ``raw/`` -> ``docs/`` converter:
Import is mechanical (no LLM calls); Transcribe is model-assisted, reserved
for PDFs a cheap text-layer probe finds no text in (or that a curator
force-converts per-file).

Pipeline (force entry — ``transcribe_source`` / ``transcribe_path``):
    1. Resolve + validate the source path (mirrors ``importer``'s filename /
       extension / existence checks; ``.pdf`` only — Transcribe has no other
       format).
    2. Read raw bytes, compute SHA-256, hash-skip check against the existing
       ``docs/<stem>.md`` frontmatter (same envelope ``importer`` reads/writes
       — Transcribe and Import share one hash-skip contract because they
       write the same docs target).
    3. Availability check (``transcribe_available``): missing ``OPENAI_API_KEY``
       or ``KB_TRANSCRIBE_ENABLED`` not set -> typed ``TranscribeUnavailable``,
       checked before any model call.
    4. Page-count guard (``KB_TRANSCRIBE_MAX_PAGES``, default 50) -> typed
       ``TranscribePageLimitExceeded``, checked before any model call.
    5. Rasterize and transcribe pages one at a time, in order (pypdfium2 render
       -> vision model under a faithful-transcription prompt — form only, no
       summarization, no synthesis, no completion). Each page's rendered
       image is released before the next page is rasterized, so peak memory
       is O(one page) rather than O(page count) (issue #456). A page that
       still fails after the ``ChatOpenAI`` wrapper's bounded retry fails the
       WHOLE file (typed ``TranscribeError``) with no partial write
       (atomicity, mirrors Import's atomic-write convention).
    6. Assemble page bodies, pass through ``kangxi_normalize`` (issue #425)
       as defense in depth, and render the standard provenance envelope
       (``imported_from``, ``original_format: pdf``, ``imported_at``,
       ``content_sha256``) plus ``origin: transcribed`` + ``transcribe_model``.
    7. Atomic write to ``docs/<stem>.md``; emit Wiki Log events
       (``transcribe_batch_started`` / ``transcribe_source`` /
       ``transcribe_skipped`` / ``transcribe_error`` /
       ``transcribe_batch_completed`` — see ``log-kinds.md``).

The probe (``probe_has_text_layer``) and availability check
(``transcribe_available``) are also called directly by ``importer.py``'s
shared conversion entry (``_process_one_source``) to auto-route a text-less
PDF to ``transcribe_pdf_bytes`` in place of the old post-hoc ``NoTextLayer``
detector (ADR-0032) — that integration lives in ``importer.py``, not here.

``KB_TRANSCRIBE_ENABLED`` is a deliberate SEPARATE gate from
``OPENAI_API_KEY`` presence (ADR-0032's "TranscribeUnavailable (missing
key/feature off)"): Transcribe is opt-in even with a valid key configured,
so a bare ``git pull`` never starts billing a curator's OpenAI account, and
the default test suite / CI's dummy-key session (see ``.github/workflows/ci.yml``)
never attempts a live network call from the probe-routing path
(CODING_STANDARD §6.3 — the LLM getter is not the only live network seam).

Budget hook (issue #460): ``transcribe_pdf_bytes`` calls an optional
module-level hook with the PDF's page count immediately after the
``KB_TRANSCRIBE_MAX_PAGES`` check and BEFORE any page is rasterized or sent
to the vision model. markdown_kb has no dependency on ``gateway`` (ADR-0002's
one-way Stack boundary), so it cannot charge the Gateway's per-UTC-day USD
budget ledger itself; ``set_page_budget_hook`` is the inversion point the
Gateway composition root (``gateway/app/main.py``) wires at startup so both
Transcribe entries — the ``/wiki/import`` auto-route and the forced
``/wiki/transcribe`` — charge the SAME shared ledger per page, and a hook
that raises ``TranscribeBudgetExceeded`` rejects the file before any vision
call spends real money (reserve-before-spend). Standalone callers (kb_cli,
kb_mcp, bare markdown_kb) never install a hook, so the default ``None``
means unmetered, uncapped behaviour there — unchanged from before this hook
existed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal

import pdfplumber
import pypdfium2 as pdfium
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import importer as importer_module
from ._paths import _REPO_ROOT, DOCS_DIR
from .atomic import write_bytes_atomic, write_text_atomic
from .kangxi_normalize import normalize_kangxi_radicals
from .logger import log_event

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RAW_DIR: Path = _REPO_ROOT / "raw"

_DEFAULT_MAX_PAGES = 50

# Rasterization scale: pypdfium2's ``render(scale=...)`` is a multiplier on
# the PDF's native 72 DPI page units. 2.0 -> ~144 DPI, legible for a vision
# model without producing an unreasonably large base64 payload per page.
_RENDER_SCALE = 2.0

_TRANSCRIBE_SYSTEM_PROMPT = """\
You are transcribing one page of a scanned/image-only PDF into Markdown.

Rules — this is a FAITHFUL TRANSCRIPTION, not a summary:
- Convert the visible content on this page into Markdown form only. Do not \
summarize, synthesize, add commentary, or complete/continue anything not \
visibly on the page.
- Preserve headings as literal ATX Markdown (`#`, `##`, ...) matching the \
visual heading level on the page.
- Preserve the original language exactly — do not translate.
- Preserve tables as Markdown tables when a table is visible.
- If the page is blank or contains no legible text, respond with an empty \
string.
- Output ONLY the transcribed Markdown for this page — no preamble, no \
explanation, no code fences.
"""


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class TranscribeUnavailable(Exception):
    """Transcribe is not configured: missing ``OPENAI_API_KEY`` or
    ``KB_TRANSCRIBE_ENABLED`` is not set (ADR-0032)."""


class TranscribePageLimitExceeded(Exception):
    """The PDF's page count exceeds ``KB_TRANSCRIBE_MAX_PAGES``. Raised
    before any model call (ADR-0032 fail-closed guard)."""


class TranscribeBudgetExceeded(Exception):
    """The registered page-budget hook rejected this page count (issue #460).

    Raised before any model call, same as ``TranscribePageLimitExceeded`` —
    the Gateway's daily USD cost cap is already at/over the ceiling, so this
    file's vision calls must not start. Never raised when no hook is
    installed (standalone markdown_kb, kb_cli, kb_mcp)."""


class TranscribeError(Exception):
    """A page failed to transcribe after the LLM wrapper's bounded retry, or
    assembly failed. The whole file fails — no partial ``docs/`` write."""


class TranscribePathError(Exception):
    """Raised by ``transcribe_source`` / ``transcribe_path`` when a forced
    transcription cannot proceed.

    Carries a human-readable ``message`` (safe to surface to the caller) and
    an ``error_type`` string — reuses ``importer.ImportFailure``'s validation
    type names (``FileNotFoundError``, ``InvalidFilename``,
    ``InvalidSourcePath``, ``UnsupportedExtension``, ``IOError``) plus the
    three Transcribe-specific ones above, so both converters' typed-failure
    vocabularies stay one family (ADR-0032).
    """

    def __init__(self, message: str, *, error_type: str = "TranscribeError") -> None:
        self.message = message
        self.error_type = error_type
        super().__init__(message)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TranscribeSourceResult:
    """Successful (or hash-skipped) forced-transcription outcome.

    Mirrors ``importer.ImportSourceResult`` in shape; ``origin`` and
    ``transcribe_model`` are the Transcribe-specific provenance fields
    written to the docs/ frontmatter (ADR-0032).
    """

    raw_path: str
    docs_path: str
    content_sha256: str
    transcribe_model: str
    original_format: Literal["pdf"] = "pdf"
    status: Literal["created", "updated", "skipped"] = "created"
    origin: Literal["transcribed"] = "transcribed"


# ---------------------------------------------------------------------------
# Availability + probe
# ---------------------------------------------------------------------------


def _transcribe_model_name() -> str:
    """Resolve the configured Transcribe model — single source of truth.

    Resolution (mirrors ``templates._ingest_model_name``'s two-layer fallback):
        OPENAI_TRANSCRIBE_MODEL  ->  OPENAI_MODEL  ->  gpt-5-mini
    Read at call time so a restart-free env change takes effect on the next call.
    """
    return os.getenv("OPENAI_TRANSCRIBE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini"))


def transcribe_available() -> bool:
    """True iff Transcribe is both configured and explicitly enabled.

    Two independent gates (ADR-0032's ``TranscribeUnavailable (missing
    key/feature off)``):
      1. ``OPENAI_API_KEY`` must be set (non-empty).
      2. ``KB_TRANSCRIBE_ENABLED`` must be ``"true"``/``"1"``/``"yes"``
         (case-insensitive) — Transcribe is opt-in even with a valid key
         present. See the module docstring for why this second gate exists.
    Read at call time (no restart needed).
    """
    if not os.getenv("OPENAI_API_KEY"):
        return False
    return os.getenv("KB_TRANSCRIBE_ENABLED", "").strip().lower() in ("1", "true", "yes")


def probe_has_text_layer(raw_bytes: bytes) -> bool:
    """Cheap, deterministic check: does this PDF have an extractable text layer?

    ADR-0032: used at the shared conversion entry (``importer._process_one_source``)
    to route text-less PDFs to Transcribe and keep digital-native PDFs on the
    free mechanical path. No model call, no full extraction — just a
    page-by-page ``pdfplumber`` text probe, so it is always safe and instant.

    Returns True iff any page has non-whitespace extractable text.

    Raises whatever ``pdfplumber.open`` / ``page.extract_text`` raises for a
    PDF it cannot open at all (encrypted, corrupt) — callers should treat a
    raised exception as "cannot determine from the probe" and defer to the
    existing mechanical extractor's own error classification, which
    independently raises the same underlying failure class.
    """
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
    return False


# ---------------------------------------------------------------------------
# Lazy singleton LLM (CODING_STANDARD §2.7)
# ---------------------------------------------------------------------------

_transcribe_llm: ChatOpenAI | None = None


def get_transcribe_llm() -> ChatOpenAI:
    """Return a lazy singleton ``ChatOpenAI`` for page transcription.

    Model resolution delegates to ``_transcribe_model_name``. Pinned to
    ``temperature=0`` for faithful, reproducible transcription (ADR-0032:
    "convert form only"). ``max_retries`` gives the bounded-retry-then-fail
    behaviour the AC requires without a hand-rolled retry loop (ADR-0005
    borrowing rationale).
    """
    global _transcribe_llm
    if _transcribe_llm is None:
        _transcribe_llm = ChatOpenAI(
            model=_transcribe_model_name(),
            temperature=0,
            timeout=60,
            max_retries=int(os.getenv("KB_TRANSCRIBE_MAX_RETRIES", "3")),
        )
    return _transcribe_llm


# ---------------------------------------------------------------------------
# Budget hook (issue #460) — see module docstring "Budget hook" section.
# ---------------------------------------------------------------------------

_page_budget_hook: Callable[[int], None] | None = None


def set_page_budget_hook(hook: Callable[[int], None] | None) -> None:
    """Register (or clear, with ``None``) the per-page budget callback.

    Called by ``transcribe_pdf_bytes`` with a PDF's page count, after the
    ``KB_TRANSCRIBE_MAX_PAGES`` guard and before any page is rasterized or
    sent to the vision model. The hook may raise ``TranscribeBudgetExceeded``
    to reject the file before any spend. See the module docstring's "Budget
    hook" section for why this indirection exists instead of markdown_kb
    importing the Gateway's budget ledger directly.
    """
    global _page_budget_hook
    _page_budget_hook = hook


def get_page_budget_hook() -> Callable[[int], None] | None:
    """Return the currently-installed page-budget hook, or ``None`` if unset.

    Symmetric public counterpart to ``set_page_budget_hook`` (CODING_STANDARD
    §2.4 — no reaching into another module's private ``_page_budget_hook``
    state from outside this module; a caller that needs to introspect or
    drive the installed hook, e.g. the Gateway's own tests, uses this getter
    instead).
    """
    return _page_budget_hook


# ---------------------------------------------------------------------------
# Public API — page rendering + transcription
# ---------------------------------------------------------------------------


def transcribe_pdf_bytes(raw_bytes: bytes) -> tuple[str, str]:
    """Transcribe PDF bytes into Markdown via the vision model (ADR-0032).

    Assumes the caller has already confirmed ``transcribe_available()``.
    Checks the page count against ``KB_TRANSCRIBE_MAX_PAGES`` (default 50)
    BEFORE any model call, then — if a page-budget hook is installed (issue
    #460) — gives it the page count, also before any model call, then
    rasterizes and transcribes each page page-at-a-time (issue #456) — page
    i's rendered image is released before page i+1 is rasterized, so peak
    memory is O(one page) instead of O(page count). Page bodies are joined
    with a blank line, in order.

    Returns ``(assembled_markdown, model_name)``. ``assembled_markdown`` has
    already passed through ``normalize_kangxi_radicals`` (issue #425, defense
    in depth — model output can carry the same font-adjacent contamination
    class as mechanical extraction).

    Raises:
        TranscribePageLimitExceeded: page count > ``KB_TRANSCRIBE_MAX_PAGES``.
        TranscribeBudgetExceeded: the installed page-budget hook rejected
            this page count (issue #460); never raised when no hook is set.
        TranscribeError: a page failed after the LLM wrapper's bounded retry,
            or the model transcribed every page as empty (assembly failure —
            mirrors Import's "never write a silent empty Source" invariant).
    """
    max_pages = int(os.getenv("KB_TRANSCRIBE_MAX_PAGES", str(_DEFAULT_MAX_PAGES)))

    pdf = pdfium.PdfDocument(raw_bytes)
    try:
        page_count = len(pdf)
        if page_count > max_pages:
            raise TranscribePageLimitExceeded(
                f"PDF has {page_count} page(s), exceeding KB_TRANSCRIBE_MAX_PAGES={max_pages}"
            )
        if _page_budget_hook is not None:
            _page_budget_hook(page_count)
        page_bodies = [_render_and_transcribe_page(pdf, i) for i in range(page_count)]
    finally:
        pdf.close()

    assembled = "\n\n".join(body for body in page_bodies if body)
    if not assembled.strip():
        raise TranscribeError(
            f"Transcription produced no content across {page_count} page(s); assembly failed."
        )
    return normalize_kangxi_radicals(assembled), _transcribe_model_name()


def _render_and_transcribe_page(pdf: pdfium.PdfDocument, page_index: int) -> str:
    """Render one page to PNG and transcribe it, then let the image go out of scope.

    The rendered PNG bytes live only inside this function's frame — they are
    dropped when this call returns, before the caller renders the next page
    (issue #456). This is what keeps ``transcribe_pdf_bytes`` from holding
    more than one page image in memory at a time.
    """
    page_png = _render_page_png(pdf[page_index])
    return _transcribe_one_page(page_png, page_index + 1)


def _render_page_png(page: pdfium.PdfPage) -> bytes:
    """Rasterize one pypdfium2 page to PNG bytes at ``_RENDER_SCALE``."""
    bitmap = page.render(scale=_RENDER_SCALE)
    pil_image = bitmap.to_pil()
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _transcribe_one_page(page_png: bytes, page_num: int) -> str:
    """Send one page image to the vision model under the faithful-transcription prompt.

    Raises ``TranscribeError`` if the underlying call fails after
    ``ChatOpenAI``'s bounded retry is exhausted — this is what makes a
    mid-file page failure abort the whole file (no partial write).
    """
    b64 = base64.b64encode(page_png).decode("ascii")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"Transcribe page {page_num} of this PDF into Markdown.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
    )
    try:
        llm = get_transcribe_llm()
        response = llm.invoke([SystemMessage(content=_TRANSCRIBE_SYSTEM_PROMPT), message])
    except Exception as exc:
        raise TranscribeError(f"page {page_num} transcription failed: {exc}"[:200]) from exc
    return response.content


# ---------------------------------------------------------------------------
# Public API — force entry (single-source, bypasses the probe)
# ---------------------------------------------------------------------------


def transcribe_source(source_filter: str) -> TranscribeSourceResult:
    """Force-transcribe one named file already staged under ``raw/``.

    Mirrors ``importer._resolve_single_source``'s validation chain (path
    format, filename, extension, existence) but accepts only ``.pdf`` — used
    by ``POST /transcribe`` (ADR-0032 designed-PDF escape hatch: bypasses the
    probe, always transcribes).

    Raises ``TranscribePathError`` for any validation, availability, or
    conversion failure. Emits ``transcribe_batch_started`` /
    ``transcribe_source`` or ``transcribe_skipped`` / ``transcribe_error`` /
    ``transcribe_batch_completed`` Wiki Log events (mirrors the ``import_*``
    family — issue #426).
    """
    batch_start = time.monotonic()
    log_event("transcribe_batch_started", f"mode=single source={source_filter}")

    try:
        raw_path = _resolve_pdf_source(source_filter)
        result = _force_transcribe(raw_path)
    except TranscribePathError as exc:
        log_event(
            "transcribe_error",
            f"raw={source_filter} error_type={exc.error_type} error_message={exc.message[:200]!r}",
        )
        log_event(
            "transcribe_batch_completed",
            f"transcribed=0 skipped=0 failed=1 duration_ms={_elapsed_ms(batch_start)}",
        )
        raise

    if result.status == "skipped":
        log_event(
            "transcribe_batch_completed",
            f"transcribed=0 skipped=1 failed=0 duration_ms={_elapsed_ms(batch_start)}",
        )
    else:
        log_event(
            "transcribe_batch_completed",
            f"transcribed=1 skipped=0 failed=0 duration_ms={_elapsed_ms(batch_start)}",
        )
    return result


def transcribe_path(path: Path) -> TranscribeSourceResult:
    """Stage a local file into ``raw/`` and force-transcribe it.

    Path-accepting entry mirroring ``importer.import_path`` — used by
    ``kb transcribe <path>`` (CLI). Conversion logic delegates to
    ``transcribe_source`` after staging so the CLI cannot bypass the
    programmatic validation chain.

    Raises ``TranscribePathError`` for any validation, availability, or
    conversion failure.
    """
    try:
        if not path.exists():
            raise TranscribePathError(f"File not found: {path}", error_type="FileNotFoundError")
        if not path.is_file():
            raise TranscribePathError(
                f"Path is not a regular file: {path}", error_type="InvalidSourcePath"
            )
    except (OSError, PermissionError) as exc:
        raise TranscribePathError(str(exc)[:200], error_type="IOError") from exc

    basename = unicodedata.normalize("NFC", path.name)
    filename_failure = importer_module.validate_filename(basename, str(path))
    if filename_failure is not None:
        raise TranscribePathError(
            filename_failure.error_message, error_type=filename_failure.error_type
        )

    ext = Path(basename).suffix.lower()
    if ext != ".pdf":
        raise TranscribePathError(
            f"Unsupported file extension '{ext}': {basename}. Transcribe only handles .pdf.",
            error_type="UnsupportedExtension",
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise TranscribePathError(str(exc)[:200], error_type="IOError") from exc

    staged_path = RAW_DIR / basename
    try:
        write_bytes_atomic(staged_path, raw_bytes)
    except OSError as exc:
        raise TranscribePathError(
            f"Failed to stage {basename} into raw/: {exc}"[:200], error_type="IOError"
        ) from exc

    return transcribe_source(basename)


# ---------------------------------------------------------------------------
# Internal helpers — force-entry validation, hash-skip, render, write
# ---------------------------------------------------------------------------


def _resolve_pdf_source(source_filter: str) -> Path:
    """Resolve + validate a raw/-relative source filter to an absolute Path.

    Mirrors ``importer._resolve_single_source``'s four validation rules
    (source-path format, filename, extension, existence), restricted to
    ``.pdf`` (Transcribe's only supported format). Raises
    ``TranscribePathError`` on any failure.
    """
    source_filter = unicodedata.normalize("NFC", source_filter)
    p = Path(source_filter)

    if p.is_absolute() or PureWindowsPath(source_filter).is_absolute():
        raise TranscribePathError(
            f"Absolute paths are not allowed: {source_filter}"[:200],
            error_type="InvalidSourcePath",
        )
    if ".." in p.parts:
        raise TranscribePathError(
            f"Path traversal ('..') is not allowed: {source_filter}"[:200],
            error_type="InvalidSourcePath",
        )
    if p.parts and p.parts[0].lower() == "raw":
        raise TranscribePathError(
            (
                f"Source must be relative to raw/ directory, not include 'raw/' prefix: "
                f"{source_filter}"
            )[:200],
            error_type="InvalidSourcePath",
        )

    raw_path = RAW_DIR / source_filter
    raw_dir_resolved = RAW_DIR.resolve()
    try:
        raw_path.resolve().relative_to(raw_dir_resolved)
    except ValueError:
        raise TranscribePathError(
            f"Path escapes raw/ directory after resolution: {source_filter}"[:200],
            error_type="InvalidSourcePath",
        ) from None

    basename = raw_path.name
    filename_failure = importer_module.validate_filename(basename, str(raw_path))
    if filename_failure is not None:
        raise TranscribePathError(
            filename_failure.error_message, error_type=filename_failure.error_type
        )

    ext = raw_path.suffix.lower()
    if ext != ".pdf":
        raise TranscribePathError(
            f"Unsupported file extension '{ext}': {source_filter}. Transcribe only handles .pdf."[
                :200
            ],
            error_type="UnsupportedExtension",
        )

    if not raw_path.exists():
        raise TranscribePathError(
            f"File not found: {source_filter}"[:200], error_type="FileNotFoundError"
        )

    return raw_path


def _force_transcribe(raw_path: Path) -> TranscribeSourceResult:
    """Run the hash-skip + transcribe + write pipeline for a resolved raw/ PDF path.

    Raises ``TranscribePathError`` for availability / page-limit / model
    failures. No partial ``docs/`` write on any failure path.
    """
    basename = raw_path.name
    stem = unicodedata.normalize("NFC", raw_path.stem)
    docs_path = DOCS_DIR / f"{stem}.md"

    try:
        raw_bytes = raw_path.read_bytes()
    except OSError as exc:
        raise TranscribePathError(str(exc)[:200], error_type="IOError") from exc

    content_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    if docs_path.exists():
        existing_sha = importer_module.read_frontmatter_sha256(docs_path)
        if existing_sha is not None and existing_sha == content_sha256:
            log_event(
                "transcribe_skipped",
                f"raw={raw_path} docs={docs_path} content_sha256={content_sha256}",
            )
            return TranscribeSourceResult(
                raw_path=str(raw_path),
                docs_path=str(docs_path),
                content_sha256=content_sha256,
                transcribe_model=_transcribe_model_name(),
                status="skipped",
            )

    status: Literal["created", "updated"] = "updated" if docs_path.exists() else "created"

    if not transcribe_available():
        raise TranscribePathError(
            (
                f"Transcribe is unavailable for {basename}: missing OPENAI_API_KEY or "
                "KB_TRANSCRIBE_ENABLED is not set. Configure both to force-transcribe."
            )[:200],
            error_type="TranscribeUnavailable",
        )

    try:
        md_body, model_name = transcribe_pdf_bytes(raw_bytes)
    except TranscribePageLimitExceeded as exc:
        raise TranscribePathError(str(exc)[:200], error_type="TranscribePageLimitExceeded") from exc
    except TranscribeBudgetExceeded as exc:
        raise TranscribePathError(str(exc)[:200], error_type="TranscribeBudgetExceeded") from exc
    except TranscribeError as exc:
        raise TranscribePathError(str(exc)[:200], error_type="TranscribeError") from exc

    output = importer_module.render_output(
        md_body,
        raw_path,
        "pdf",
        content_sha256,
        origin="transcribed",
        transcribe_model=model_name,
    )

    try:
        write_text_atomic(docs_path, output)
    except OSError as exc:
        raise TranscribePathError(str(exc)[:200], error_type="IOError") from exc

    log_event(
        "transcribe_source",
        f"source={basename} docs={docs_path.name} model={model_name} status={status}",
    )

    return TranscribeSourceResult(
        raw_path=str(raw_path),
        docs_path=str(docs_path),
        content_sha256=content_sha256,
        transcribe_model=model_name,
        status=status,
    )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
