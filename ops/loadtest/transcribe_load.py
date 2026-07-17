"""``POST /wiki/transcribe/batch`` (raw -> docs, model-assisted) load driver
(issue #627, S5 — the transcribe-inclusive gap #600 deliberately left
unmeasured; see ``scenarios.py``'s module docstring for why every scenario,
including this one, still needs the fake upstream + a dummy key to boot).

Fixture-inventory note (issue #627 triage): the issue's own scope narrowing
says fixtures already exist (``markdown_kb/tests/fixtures/raw_import/`` has
scanned/image-only PDFs) and forbids committing a NEW one. Every committed
scanned fixture there is single-page, though, and page-level concurrency
(``KB_TRANSCRIBE_CONCURRENCY``, the #456/#459 OOM knob) needs multiple pages
in ONE source to actually saturate — a batch of several 1-page sources stays
sequential-over-sources (``transcribe_jobs._run_batch``, by design). This
module assembles a multi-page PDF **at runtime, in memory**, by repeating a
committed fixture's page 0 via ``pypdfium2``'s own page-import API — never
written to a committed path, always staged under a run-unique ``raw/``
filename and removed in ``finally``, mirroring ``import_load.py``'s
plant/submit/poll/cleanup shape.

Stub-answer sizing: the fake upstream's plain-text stub (~71 chars) assembled
across even 16 pages stays at ~1.2KB, comfortably under
``KB_LONGFORM_MIN_CHARS`` (2000) — same "keep the Structure Enrichment gate
closed" discipline ``import_load.py`` documents for its own fixtures, so this
scenario's only LLM traffic is the per-page transcription calls it exists to
measure.
"""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import pypdfium2 as pdfium

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "raw"
DOCS_DIR = REPO_ROOT / "docs"

# Single-page committed scanned fixture whose page 0 is repeated to build the
# in-memory multi-page source — never written back, never committed.
_SOURCE_FIXTURE = (
    REPO_ROOT / "markdown_kb" / "tests" / "fixtures" / "raw_import" / "image_only.pdf"
)


@dataclass
class TranscribeLoadResult:
    status: str
    files_total: int
    files_done: int
    pages_total: int
    pages_done: int
    wall_clock_sec: float
    error: str | None = None


def _build_multipage_pdf(num_pages: int) -> bytes:
    """Return in-memory PDF bytes with *num_pages* copies of the fixture's page 0.

    ``pdfium.PdfDocument.import_pages`` accepts a repeated-index list directly
    (verified during implementation — no per-page loop needed).
    """
    src = pdfium.PdfDocument(str(_SOURCE_FIXTURE))
    try:
        dst = pdfium.PdfDocument.new()
        try:
            dst.import_pages(src, pages=[0] * num_pages)
            buf = io.BytesIO()
            dst.save(buf)
            return buf.getvalue()
        finally:
            dst.close()
    finally:
        src.close()


def run_transcribe_load(
    base_url: str, num_pages: int = 16, poll_timeout: float = 90.0
) -> TranscribeLoadResult:
    """Stage one synthetic multi-page scanned PDF under ``raw/``, submit it as
    a single-source Transcribe batch job (``POST /wiki/transcribe/batch``),
    poll ``GET /wiki/transcribe/jobs/{job_id}`` to a terminal state, then
    remove every file this run planted or produced — always, even on
    failure. One source with *num_pages* pages exercises page-level
    concurrency within ``transcribe_jobs``' sequential-over-sources batch
    loop (see module docstring)."""
    run_id = uuid.uuid4().hex[:8]
    stem = f"_loadtest_transcribe_{run_id}"
    filename = f"{stem}.pdf"
    raw_path = RAW_DIR / filename
    docs_path = DOCS_DIR / f"{stem}.md"
    start = time.monotonic()

    try:
        RAW_DIR.mkdir(exist_ok=True)
        raw_path.write_bytes(_build_multipage_pdf(num_pages))

        with httpx.Client(timeout=10.0) as client:
            submit_resp = client.post(
                f"{base_url}/wiki/transcribe/batch", json={"sources": [filename]}
            )
            submit_resp.raise_for_status()
            job_id = submit_resp.json()["job_id"]

            deadline = time.monotonic() + poll_timeout
            status = "submitted"
            pages_total = 0
            pages_done = 0
            files_done = 0
            while time.monotonic() < deadline:
                poll_resp = client.get(f"{base_url}/wiki/transcribe/jobs/{job_id}")
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data["status"]
                pages_total = data.get("pages_total", 0)
                pages_done = data.get("pages_done", 0)
                files_done = len(data.get("results", []))
                if status in ("completed", "failed"):
                    break
                time.sleep(0.3)

        return TranscribeLoadResult(
            status=status,
            files_total=1,
            files_done=files_done,
            pages_total=pages_total,
            pages_done=pages_done,
            wall_clock_sec=round(time.monotonic() - start, 3),
        )
    except httpx.HTTPError as exc:
        return TranscribeLoadResult(
            status="error",
            files_total=1,
            files_done=0,
            pages_total=0,
            pages_done=0,
            wall_clock_sec=round(time.monotonic() - start, 3),
            error=str(exc),
        )
    finally:
        raw_path.unlink(missing_ok=True)
        docs_path.unlink(missing_ok=True)
