"""Deep module per Ousterhout. Public surface: ``transcribe_available``, ``probe_has_text_layer``, ``transcribe_pdf_bytes``, ``transcribe_pdf_bytes_concurrent``, ``transcribe_source``, ``transcribe_path``, ``get_transcribe_llm``, ``set_page_budget_hook``, ``get_page_budget_hook``, ``TranscribeSourceResult``, ``TranscribePathError``, ``TranscribeUnavailable``, ``TranscribePageLimitExceeded``, ``TranscribeBudgetExceeded``, ``TranscribeError``.

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

Budget hook (issue #460): both ``transcribe_pdf_bytes`` and
``transcribe_pdf_bytes_concurrent`` call an optional module-level hook with
the PDF's page count immediately after the ``KB_TRANSCRIBE_MAX_PAGES`` check
and BEFORE any page is rasterized or sent to the vision model. markdown_kb
has no dependency on ``gateway`` (ADR-0002's one-way Stack boundary), so it
cannot charge the Gateway's per-UTC-day USD budget ledger itself;
``set_page_budget_hook`` is the inversion point the Gateway composition root
(``gateway/app/main.py``) wires at startup so both Transcribe entries — the
``/wiki/import`` auto-route and the forced ``/wiki/transcribe`` — charge the
SAME shared ledger per page, and a hook that raises
``TranscribeBudgetExceeded`` rejects the file before any vision call spends
real money (reserve-before-spend). Standalone callers (kb_cli, kb_mcp, bare
markdown_kb) never install a hook, so the default ``None`` means unmetered,
uncapped behaviour there — unchanged from before this hook existed.

Responsiveness (issue #459): a large scanned PDF measured 30+ minutes on the
strictly-sequential path (OOM'd the demo container before finishing). Two
independent fixes:

  * ``get_transcribe_llm`` pins ``reasoning_effort="minimal"`` — gpt-5-mini is
    a reasoning model; the OCR-shaped transcription task needs none of that
    "thinking" (measured ~5x faster, identical output).
  * ``transcribe_pdf_bytes_concurrent`` fans page rendering+transcription out
    across a PROCESS-WIDE bounded worker pool (``KB_TRANSCRIBE_CONCURRENCY``,
    default 16) instead of one page at a time. It is a separate function from
    ``transcribe_pdf_bytes`` (kept sequential, unchanged) rather than a
    behavioural change to it, because the two have different, deliberately
    incompatible memory shapes: strictly-sequential holds at most ONE
    rendered page image at a time (issue #456's guarantee, still exactly
    true for ``transcribe_pdf_bytes``); the concurrent pool holds at most
    ``KB_TRANSCRIBE_CONCURRENCY`` images at a time (bounded and tunable, but
    necessarily > 1 whenever concurrency > 1). ``_force_transcribe`` (hence
    ``transcribe_source`` / ``transcribe_path`` / ``POST /transcribe`` / ``kb
    transcribe``) and ``importer._process_one_source``'s auto-route both call
    the concurrent version — that is where the real 63-page-scan pain lives.
    Both entries into the concurrent pool honour the SAME budget hook as the
    sequential path (issue #460 invariant preserved across the speedup).

Thread safety (issue #468): giving each worker thread its OWN
``pdfium.PdfDocument`` (previous design) turned out to be INSUFFICIENT —
pypdfium2 / PDFium keeps process-wide global state, so concurrent
open/render/close from separate threads races and intermittently segfaults
(Linux CI, exit 139) regardless of per-document isolation. The fix keeps the
concurrency win but moves the serialization point: ``_pdfium_lock`` (module
level) guards EVERY pypdfium2 call in this module — document open, page
access, render, and close — so no two threads are ever inside PDFium at the
same time. The (slow, I/O-bound, thread-safe) vision-model call always runs
OUTSIDE that lock, so pages still transcribe in parallel; only the
microsecond-scale render serializes.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import os
import threading
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
    "convert form only"), and ``reasoning_effort="minimal"`` (issue #459) —
    the default model (gpt-5-mini) is a reasoning model that otherwise
    "thinks" before transcribing, pure overhead for a faithful-copy task
    (measured ~5x slower at default reasoning, byte-identical output at
    minimal). ``max_retries`` gives the bounded-retry-then-fail behaviour the
    AC requires without a hand-rolled retry loop (ADR-0005 borrowing
    rationale).

    ``temperature=0`` + ``reasoning_effort="minimal"`` together are safe:
    langchain-openai's own ``validate_temperature`` model validator already
    drops ``temperature`` for gpt-5 (non-chat) models whenever
    ``reasoning_effort`` is anything other than ``"none"`` (verified against
    the installed langchain-openai — see ``ChatOpenAI.validate_temperature``),
    so there is no ctor-time or call-time conflict to guard against here; for
    any non-gpt-5 model swapped in via ``OPENAI_TRANSCRIBE_MODEL`` the
    ``temperature=0`` pin still applies normally.
    """
    global _transcribe_llm
    if _transcribe_llm is None:
        _transcribe_llm = ChatOpenAI(
            model=_transcribe_model_name(),
            temperature=0,
            reasoning_effort="minimal",
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

# Process-wide PDFium lock (issue #468). pypdfium2 / PDFium is NOT
# thread-safe at the library level — it keeps global state, so concurrent
# open/render/close calls from separate threads race and segfault even when
# each thread uses its OWN ``PdfDocument`` instance. Every pypdfium2 touch in
# this module (document open, page access, render, close) acquires this ONE
# lock first; the vision-model call is deliberately kept OUTSIDE it (see
# ``_render_and_transcribe_page`` / ``_render_and_transcribe_page_isolated``)
# so the issue #459 concurrency win survives on the part that actually
# benefits from parallelism.
_pdfium_lock = threading.Lock()


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

    Every pypdfium2 call (open, render, close) goes through ``_pdfium_lock``
    (issue #468) so this sequential path can never race with another thread
    concurrently inside PDFium — e.g. ``transcribe_pdf_bytes_concurrent``'s
    worker pool running on another file at the same time.

    Raises:
        TranscribePageLimitExceeded: page count > ``KB_TRANSCRIBE_MAX_PAGES``.
        TranscribeBudgetExceeded: the installed page-budget hook rejected
            this page count (issue #460); never raised when no hook is set.
        TranscribeError: a page failed after the LLM wrapper's bounded retry,
            or the model transcribed every page as empty (assembly failure —
            mirrors Import's "never write a silent empty Source" invariant).
    """
    max_pages = int(os.getenv("KB_TRANSCRIBE_MAX_PAGES", str(_DEFAULT_MAX_PAGES)))

    with _pdfium_lock:
        pdf = pdfium.PdfDocument(raw_bytes)
        page_count = len(pdf)
    try:
        if page_count > max_pages:
            raise TranscribePageLimitExceeded(
                f"PDF has {page_count} page(s), exceeding KB_TRANSCRIBE_MAX_PAGES={max_pages}"
            )
        if _page_budget_hook is not None:
            _page_budget_hook(page_count)
        page_bodies = [_render_and_transcribe_page(pdf, i) for i in range(page_count)]
    finally:
        with _pdfium_lock:
            pdf.close()

    assembled = "\n\n".join(body for body in page_bodies if body)
    if not assembled.strip():
        raise TranscribeError(
            f"Transcription produced no content across {page_count} page(s); assembly failed."
        )
    return normalize_kangxi_radicals(assembled), _transcribe_model_name()


# ---------------------------------------------------------------------------
# Process-wide bounded page-worker pool (issue #459)
# ---------------------------------------------------------------------------

_DEFAULT_PAGE_CONCURRENCY = 16


def _page_concurrency() -> int:
    """Max concurrent in-flight page-transcription calls, process-wide.

    Override with ``KB_TRANSCRIBE_CONCURRENCY``; default
    ``_DEFAULT_PAGE_CONCURRENCY`` (16). Read once, at module import, into the
    module-level ``_page_semaphore`` below — mirrors
    ``gateway/app/middleware.py``'s ``admin_sem``/``read_sem`` idiom ("read
    once at import so tests can drain/restore them") rather than
    ``ingest.py``'s per-call semaphore, deliberately: two concurrent
    ``/transcribe`` or ``/import`` requests must share ONE process-wide cap,
    not each get their own N-worker pool (2 requests x N workers would
    compound past the cap and OOM the box — the failure mode this issue
    exists to avoid). A restart is required for an env change to take effect,
    same limitation as ``get_transcribe_llm``'s singleton.
    """
    raw = os.getenv("KB_TRANSCRIBE_CONCURRENCY")
    if raw is None:
        return _DEFAULT_PAGE_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_PAGE_CONCURRENCY
    return value if value > 0 else _DEFAULT_PAGE_CONCURRENCY


# Process-wide (not per-call) semaphore — see _page_concurrency docstring.
# BoundedSemaphore catches an accidental over-release (a programming error).
_page_semaphore = threading.BoundedSemaphore(_page_concurrency())


def transcribe_pdf_bytes_concurrent(
    raw_bytes: bytes,
    *,
    on_page_done: Callable[[int, int], None] | None = None,
) -> tuple[str, str]:
    """Transcribe PDF bytes via a PROCESS-WIDE bounded concurrent page pool (issue #459).

    Same contract as ``transcribe_pdf_bytes`` (page-count guard before any
    model call, bounded-retry-then-fail per page, all-blank assembly failure,
    Kangxi normalization, page order preserved in the assembled output) —
    the difference is pages are rendered and transcribed CONCURRENTLY across
    ``_page_semaphore`` (``KB_TRANSCRIBE_CONCURRENCY``, default 16) instead of
    strictly one at a time. A 63-page scan that took 30+ minutes sequentially
    (and OOM'd the demo container before finishing) completes in well under a
    minute at concurrency 16+ (measured).

    Each page opens its OWN ``pdfium.PdfDocument(raw_bytes)`` instead of
    sharing one across worker threads — a shared ``PdfDocument`` raises
    "Failed to load page" under concurrent access. Per-document isolation
    alone is NOT sufficient, though: pypdfium2 / PDFium keeps process-wide
    global state and is not thread-safe at the library level, so two threads
    each rendering from their OWN independent document can still race and
    segfault (issue #468, measured — intermittent SIGSEGV on Linux CI). Every
    pypdfium2 call therefore also goes through the module-level
    ``_pdfium_lock`` (see ``_render_and_transcribe_page_isolated``) so no two
    threads are ever inside PDFium at the same time; only the (slow,
    thread-safe) vision-model call runs outside that lock, which is what
    keeps this concurrent. The document (and its rendered image) is released
    before returning, so peak memory is O(concurrency) — bounded and tunable
    — rather than O(page count), which is what keeps this a genuine fix and
    not a reversion to the pre-#456 bug.

    ``on_page_done``, if given, is called exactly once per completed page —
    ``on_page_done(pages_done, page_count)`` — regardless of completion
    order. This is the progress data source for a pollable job status field
    (issue #459 AC4; see ``transcribe_jobs.py``).

    Raises:
        TranscribePageLimitExceeded: page count > ``KB_TRANSCRIBE_MAX_PAGES``.
        TranscribeBudgetExceeded: the installed page-budget hook rejected
            this page count (issue #460), same as the sequential
            ``transcribe_pdf_bytes``; never raised when no hook is set.
        TranscribeError: a page failed after the LLM wrapper's bounded retry,
            or every page transcribed empty (assembly failure).
    """
    max_pages = int(os.getenv("KB_TRANSCRIBE_MAX_PAGES", str(_DEFAULT_MAX_PAGES)))

    with _pdfium_lock:
        pdf = pdfium.PdfDocument(raw_bytes)
        try:
            page_count = len(pdf)
        finally:
            pdf.close()
    if page_count > max_pages:
        raise TranscribePageLimitExceeded(
            f"PDF has {page_count} page(s), exceeding KB_TRANSCRIBE_MAX_PAGES={max_pages}"
        )
    if _page_budget_hook is not None:
        _page_budget_hook(page_count)

    page_bodies: list[str | None] = [None] * page_count
    pages_done = 0
    progress_lock = threading.Lock()

    def _worker(page_index: int) -> None:
        nonlocal pages_done
        with _page_semaphore:
            page_bodies[page_index] = _render_and_transcribe_page_isolated(raw_bytes, page_index)
        if on_page_done is not None:
            with progress_lock:
                pages_done += 1
                done_snapshot = pages_done
            on_page_done(done_snapshot, page_count)

    if page_count > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=page_count) as executor:
            futures = [executor.submit(_worker, i) for i in range(page_count)]
            # Checked in page-index order (not completion order): re-raises
            # whichever exception belongs to the LOWEST-index failed page —
            # the executor's own __exit__ still blocks for every other
            # in-flight page before this call returns (no leaked threads).
            for future in futures:
                future.result()

    assembled = "\n\n".join(body for body in page_bodies if body)
    if not assembled.strip():
        raise TranscribeError(
            f"Transcription produced no content across {page_count} page(s); assembly failed."
        )
    return normalize_kangxi_radicals(assembled), _transcribe_model_name()


def _render_and_transcribe_page_isolated(raw_bytes: bytes, page_index: int) -> str:
    """Render page ``page_index`` from an INDEPENDENT ``PdfDocument`` and transcribe it.

    Each call here opens its own document rather than sharing one across
    worker threads — a shared ``PdfDocument`` raises "Failed to load page"
    under concurrent access (issue #459). Per-document isolation alone is
    NOT enough, though: pypdfium2 / PDFium keeps process-wide global state
    and is not thread-safe at the library level, so open/render/close is
    additionally guarded by the module-level ``_pdfium_lock`` (issue #468) —
    no two threads are ever inside PDFium at the same time, regardless of
    which document they hold. The (slow, thread-safe) vision-model call
    happens OUTSIDE the lock, so pages still transcribe concurrently — only
    the microsecond-scale render serializes. The document is closed (and the
    rendered PNG goes out of scope after transcription) before this function
    returns, so nothing outlives one page's worth of work.
    """
    with _pdfium_lock:
        pdf = pdfium.PdfDocument(raw_bytes)
        try:
            page_png = _render_page_png(pdf[page_index])
        finally:
            pdf.close()
    return _transcribe_one_page(page_png, page_index + 1)


def _render_and_transcribe_page(pdf: pdfium.PdfDocument, page_index: int) -> str:
    """Render one page to PNG and transcribe it, then let the image go out of scope.

    The rendered PNG bytes live only inside this function's frame — they are
    dropped when this call returns, before the caller renders the next page
    (issue #456). This is what keeps ``transcribe_pdf_bytes`` from holding
    more than one page image in memory at a time. The render itself runs
    under ``_pdfium_lock`` (issue #468 — no two threads ever inside PDFium at
    once); the vision-model call happens after the lock is released.
    """
    with _pdfium_lock:
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


def transcribe_source(
    source_filter: str,
    *,
    on_page_done: Callable[[int, int], None] | None = None,
) -> TranscribeSourceResult:
    """Force-transcribe one named file already staged under ``raw/``.

    Mirrors ``importer._resolve_single_source``'s validation chain (path
    format, filename, extension, existence) but accepts only ``.pdf`` — used
    by ``POST /transcribe`` (ADR-0032 designed-PDF escape hatch: bypasses the
    probe, always transcribes).

    ``on_page_done``, if given, is passed straight through to
    ``transcribe_pdf_bytes_concurrent`` (issue #459 AC4) — used by
    ``transcribe_jobs.py`` to update a pollable job's progress as pages land.

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
        result = _force_transcribe(raw_path, on_page_done=on_page_done)
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


def _force_transcribe(
    raw_path: Path,
    *,
    on_page_done: Callable[[int, int], None] | None = None,
) -> TranscribeSourceResult:
    """Run the hash-skip + transcribe + write pipeline for a resolved raw/ PDF path.

    Uses ``transcribe_pdf_bytes_concurrent`` (issue #459) — the process-wide
    bounded pool, not the strictly-sequential ``transcribe_pdf_bytes`` — since
    this is the force-entry path a curator or the async batch job would use
    for exactly the large scans that need the speedup.

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
        md_body, model_name = transcribe_pdf_bytes_concurrent(raw_bytes, on_page_done=on_page_done)
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
