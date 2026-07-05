"""Deep module per Ousterhout. Public surface: ``submit_batch``, ``status``.

In-process job registry for background Transcribe batches (issue #459 AC5).

``POST /transcribe`` force-transcribes exactly ONE named file, synchronously
(ADR-0032: single-source only, no batch mode — unchanged by this module).
A BATCH of large scans is a different problem: ``transcribe_source`` calls
run one file after another (mirrors ``import_sources``'s sequential-over-
sources design), so N large PDFs in a row can take minutes — long enough to
blow an HTTP client's connection window even with per-PDF page concurrency
(issue #459 item 5). This module offloads a batch onto a background
``asyncio.Task`` and returns a job id immediately; the Console (or any
client) polls ``status(job_id)`` for progress and, eventually, results.
Each named file still lands in ``docs/`` (origin: transcribed) exactly as it
would have via a synchronous ``POST /transcribe`` call — nothing about the
per-file contract changes, only when the caller finds out.

Mirrors ``kb_mcp/kb_mcp/ingest_jobs.py``'s submit/poll pattern (Fix 1b) —
same lifecycle, same GC-safety concern, adapted for a list-of-sources batch
with per-page progress instead of a single Source.

Job lifecycle: submitted -> working -> completed | failed

GC safety note (CRITICAL, same as ingest_jobs.py):
  asyncio.create_task() keeps only a WEAK reference to the Task — if no
  other reference exists, the GC can cancel the task mid-run silently. We
  keep a strong reference in the module-level ``_TASKS`` set and remove each
  task from it via add_done_callback, so the Task is never GC-collected
  before it reaches a terminal state.

Progress (issue #459 AC4): ``job.pages_done`` / ``job.pages_total`` track
pages across the WHOLE batch, updated as ``transcribe_source``'s
``on_page_done`` callback fires from page-worker THREADS (see
``transcriber.transcribe_pdf_bytes_concurrent``) — a ``threading.Lock``
guards the increment since those callbacks do not run on the event loop
thread. ``pages_total`` grows incrementally as each file's own page count
becomes known (its first ``on_page_done`` call), since a batch's total page
count isn't known until each PDF has been opened.

Concurrent-job cap (issue #474 sub-issue A): ``ProdMiddleware.admin_sem``
(``gateway/app/middleware.py``) only holds ``KB_MAX_ADMIN`` for the
``await self.app(...)`` call — for this route that is just ``submit_batch``
scheduling a Task and returning, not the multi-minute batch it starts. The
admin semaphore therefore never bounds how many ``_run_batch`` coroutines
run at once. ``submit_batch`` enforces its own ceiling
(``KB_TRANSCRIBE_MAX_CONCURRENT_JOBS``, default
``_DEFAULT_MAX_CONCURRENT_JOBS``) by counting jobs still in
``submitted``/``working`` state and raising a typed ``TranscribePathError``
(``error_type="TranscribeJobCapacityExceeded"``) when the ceiling is already
reached — a clear rejection at submit time rather than a silently-queued
invisible job.

Job-store eviction (issue #474 sub-issue B): ``_JOBS`` retained every job
forever (only the test-only ``_reset_jobs`` cleared it), so an anonymous
submit loop grows it without bound. ``_evict_terminal_jobs`` sweeps
``completed``/``failed`` jobs whose ``completed_at`` is older than
``KB_TRANSCRIBE_JOB_TTL_SECONDS`` (default ``_DEFAULT_JOB_TTL_SECONDS``) —
mirrors ``gateway/app/conversation_store.py``'s TTL-sweep-over-a-snapshot
idiom (CODING_STANDARD §2.6). Run opportunistically at the top of
``submit_batch`` (the same call site that grows the registry) rather than a
background task loop.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from .transcriber import TranscribePathError
from .transcriber import transcribe_source as _transcribe_source

# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------


@dataclass
class TranscribeJobResult:
    """Per-source outcome recorded in ``TranscribeJob.results``.

    Mirrors ``transcriber.TranscribeSourceResult`` in shape for the
    successful cases; ``error_type`` / ``error_message`` are populated only
    when ``status == "failed"`` (a ``TranscribePathError`` or any other
    exception raised while processing this one source — the batch continues
    with the remaining sources, mirroring ``import_sources``' continue-on-
    error policy).
    """

    source: str
    status: Literal["created", "updated", "skipped", "failed"]
    docs_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class TranscribeJob:
    """State for one background Transcribe batch.

    Attributes:
        job_id:       Unique identifier (uuid4().hex).
        status:       One of "submitted" | "working" | "completed" | "failed".
        pages_done:   Pages completed so far, across the WHOLE batch.
        pages_total:  Pages known so far, across the whole batch (grows as
                      each file's page count is discovered — see module docstring).
        results:      One TranscribeJobResult per source, appended as each
                      source finishes (success or failure).
        error:        Set only if the batch coroutine itself raised
                      unexpectedly (should not happen in normal operation —
                      per-source failures land in ``results`` instead).
        completed_at: ``time.monotonic()`` value when ``status`` last became
                      "completed" or "failed"; ``None`` while still
                      submitted/working. Read by ``_evict_terminal_jobs``
                      (issue #474 sub-issue B) — never set back to ``None``.
        _task:        Strong reference to the asyncio.Task (prevents GC).
        _lock:        Guards pages_done/pages_total against concurrent
                      updates from page-worker threads.
    """

    job_id: str
    status: Literal["submitted", "working", "completed", "failed"] = "submitted"
    pages_done: int = 0
    pages_total: int = 0
    results: list[TranscribeJobResult] = field(default_factory=list)
    error: str | None = None
    completed_at: float | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


# ---------------------------------------------------------------------------
# Registry (module-level globals)
# ---------------------------------------------------------------------------

_JOBS: dict[str, TranscribeJob] = {}

# Strong-reference set: prevents GC from cancelling Tasks before they finish.
# Each task removes itself via add_done_callback.
_TASKS: set[asyncio.Task] = set()

# ---------------------------------------------------------------------------
# Concurrent-job cap (issue #474 sub-issue A)
# ---------------------------------------------------------------------------

# Same order of magnitude as the Gateway's KB_MAX_ADMIN default (2) — see the
# module docstring for why admin_sem cannot cover this instead.
_DEFAULT_MAX_CONCURRENT_JOBS = 2


def _max_concurrent_jobs() -> int:
    """Max number of batch jobs allowed in submitted/working state at once.

    Override with ``KB_TRANSCRIBE_MAX_CONCURRENT_JOBS``; default
    ``_DEFAULT_MAX_CONCURRENT_JOBS``. Read at call time (no restart needed),
    mirroring ``transcriber._page_concurrency``'s env-parsing shape.
    """
    raw = os.getenv("KB_TRANSCRIBE_MAX_CONCURRENT_JOBS")
    if raw is None:
        return _DEFAULT_MAX_CONCURRENT_JOBS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_CONCURRENT_JOBS
    return value if value > 0 else _DEFAULT_MAX_CONCURRENT_JOBS


# ---------------------------------------------------------------------------
# Terminal-job eviction (issue #474 sub-issue B)
# ---------------------------------------------------------------------------

_DEFAULT_JOB_TTL_SECONDS = 15 * 60  # 15 minutes


def _job_ttl_seconds() -> int:
    """Idle TTL, in seconds, before a completed/failed job is evicted from ``_JOBS``.

    Override with ``KB_TRANSCRIBE_JOB_TTL_SECONDS``; default
    ``_DEFAULT_JOB_TTL_SECONDS``. Read at call time (no restart needed).
    """
    raw = os.getenv("KB_TRANSCRIBE_JOB_TTL_SECONDS")
    if raw is None:
        return _DEFAULT_JOB_TTL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_JOB_TTL_SECONDS
    return value if value > 0 else _DEFAULT_JOB_TTL_SECONDS


def _now() -> float:
    """Monotonic clock for job-TTL bookkeeping (``completed_at`` + eviction).

    A single indirection so a test can freeze the TTL clock WITHOUT
    monkeypatching the global ``time.monotonic`` — patching that also freezes
    the asyncio event loop's own clock (``BaseEventLoop.time``), which stalls
    ``asyncio.sleep`` and deadlocks any coroutine that polls (the #474 ubuntu
    CI hang: a frozen loop clock never fired ``_poll_until_terminal``'s sleep
    OR its own timeout). Tests patch ``transcribe_jobs._now`` instead.
    """
    return time.monotonic()


def _evict_terminal_jobs() -> None:
    """Delete completed/failed jobs whose TTL has elapsed.

    Mirrors ``gateway/app/conversation_store.py``'s ``evict_expired`` idiom
    (CODING_STANDARD §2.6): sweeps over a **snapshot** of keys so deleting
    entries mid-sweep never raises ``RuntimeError: dictionary changed size
    during iteration``. Called at the top of ``submit_batch`` — the same
    call site that grows ``_JOBS`` — rather than a background task loop.
    """
    ttl = _job_ttl_seconds()
    now = _now()
    expired = [
        job_id
        for job_id, job in list(_JOBS.items())
        if job.completed_at is not None and now - job.completed_at > ttl
    ]
    for job_id in expired:
        del _JOBS[job_id]


# ---------------------------------------------------------------------------
# Progress plumbing
# ---------------------------------------------------------------------------


def _progress_callback_for_source(job: TranscribeJob) -> Callable[[int, int], None]:
    """Return an ``on_page_done`` callback that folds one source's progress into ``job``.

    A fresh closure per source (not a shared one reused across the batch
    loop) so ``added_total`` tracks "have I already added THIS source's page
    count to job.pages_total" independently per file — avoids the classic
    for-loop late-binding closure bug.
    """
    added_total = False

    def _on_page_done(_done_for_source: int, total_for_source: int) -> None:
        nonlocal added_total
        with job._lock:
            if not added_total:
                job.pages_total += total_for_source
                added_total = True
            job.pages_done += 1

    return _on_page_done


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _run_batch(job: TranscribeJob, sources: list[str]) -> None:
    """Background coroutine: set working -> transcribe each source in turn -> completed.

    Sequential over sources (mirrors ``import_sources``'s design — see issue
    #459 item 5's own framing: the fix is making the WHOLE batch
    non-blocking, not parallelising across files); per-file speed comes from
    ``transcribe_pdf_bytes_concurrent``'s page-level pool. Never raises —
    a per-source failure is recorded in ``job.results`` and the batch
    continues; only a genuinely unexpected exception (a bug, not a source
    failure) sets ``job.status = "failed"`` with ``job.error`` populated.
    """
    job.status = "working"
    try:
        for source in sources:
            callback = _progress_callback_for_source(job)
            try:
                result = await asyncio.to_thread(_transcribe_source, source, on_page_done=callback)
                job.results.append(
                    TranscribeJobResult(
                        source=source,
                        status=result.status,
                        docs_path=result.docs_path,
                    )
                )
            except TranscribePathError as exc:
                job.results.append(
                    TranscribeJobResult(
                        source=source,
                        status="failed",
                        error_type=exc.error_type,
                        error_message=exc.message,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - per-source guard, batch continues
                job.results.append(
                    TranscribeJobResult(
                        source=source,
                        status="failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:200],
                    )
                )
    except Exception as exc:  # noqa: BLE001 - last-resort guard against a bug in the loop itself
        job.status = "failed"
        job.error = str(exc)[:200]
        job.completed_at = _now()
        return

    job.status = "completed"
    job.completed_at = _now()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_batch(sources: list[str]) -> TranscribeJob:
    """Schedule a background Transcribe batch and return a Job immediately.

    Creates a Job in status='submitted', schedules an asyncio.Task that runs
    ``_run_batch(job, sources)``, keeps a strong reference to prevent GC
    cancellation, registers the Job in ``_JOBS``, and returns it.

    First sweeps expired terminal jobs (``_evict_terminal_jobs``, issue #474
    sub-issue B), then rejects with a typed ``TranscribePathError``
    (``error_type="TranscribeJobCapacityExceeded"``) if the number of jobs
    still in submitted/working state is already at ``_max_concurrent_jobs()``
    (issue #474 sub-issue A) — a clear rejection rather than a silently
    over-subscribed background queue.

    MUST be called from inside a running event loop (i.e. from an ``async
    def`` route handler).
    """
    _evict_terminal_jobs()

    active = sum(1 for job in _JOBS.values() if job.status in ("submitted", "working"))
    cap = _max_concurrent_jobs()
    if active >= cap:
        raise TranscribePathError(
            f"{active} transcribe batch job(s) already running (cap={cap}); retry later",
            error_type="TranscribeJobCapacityExceeded",
        )

    job_id = uuid4().hex
    job = TranscribeJob(job_id=job_id)
    _JOBS[job_id] = job

    task = asyncio.create_task(_run_batch(job, sources))
    job._task = task  # keep strong ref on the Job itself

    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)

    return job


def status(job_id: str) -> TranscribeJob | None:
    """Return the Job for job_id, or None when not found."""
    return _JOBS.get(job_id)


def _reset_jobs() -> None:
    """Clear all registry state. Called by tests for isolation."""
    _JOBS.clear()
    _TASKS.clear()
