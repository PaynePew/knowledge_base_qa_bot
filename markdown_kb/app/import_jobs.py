"""Deep module per Ousterhout. Public surface: ``submit``, ``status``,
``ImportJobCapacityExceeded``.

In-process job registry for background Import runs (issue #497).

``POST /import`` converts every staged raw source synchronously. That was
fine while conversion was mechanical (MarkItDown / markdownify — sub-second
per file), but the ADR-0032 auto-route sends a text-less PDF through
Transcribe INSIDE the same request: one vision-model call per page, minutes
for a real scan — long past the edge proxy's window. The proxy then answers
502 with an empty body while the server keeps working, and the Console has
nothing honest to render (issue #497's prod repro). This module offloads an
Import run onto a background ``asyncio.Task`` and returns a job id
immediately; the Console polls ``status(job_id)`` for progress and,
eventually, the same ``ImportBatchResult`` the synchronous route would have
returned. Nothing about the per-source contract changes — only when the
caller finds out.

Mirrors ``transcribe_jobs.py``'s submit/poll pattern (issue #459) — same
lifecycle, same GC-safety concern, adapted for a whole Import run (glob +
convert + auto-route) instead of a named list of PDFs.

Job lifecycle: submitted -> working -> completed | failed

GC safety note (CRITICAL, same as transcribe_jobs.py / ingest_jobs.py):
  asyncio.create_task() keeps only a WEAK reference to the Task — if no
  other reference exists, the GC can cancel the task mid-run silently. We
  keep a strong reference in the module-level ``_TASKS`` set and remove each
  task from it via add_done_callback, so the Task is never GC-collected
  before it reaches a terminal state.

Progress: ``files_done``/``files_total`` advance as ``import_sources``'s
``on_source_done`` fires after each file; ``pages_done``/``pages_total``
advance as the auto-route's per-page ``on_transcribe_page`` callback fires
from page-worker THREADS (see ``transcriber.transcribe_pdf_bytes_concurrent``)
— a ``threading.Lock`` guards the increments since those callbacks do not
run on the event loop thread. ``pages_total`` grows incrementally as each
scanned file's own page count becomes known (its first page callback), since
a batch's total page count isn't known until each PDF has been opened —
mirrors ``transcribe_jobs``'s incremental-total contract exactly.

Concurrent-job cap (mirrors issue #474 sub-issue A): the Gateway's admin
semaphore only covers the submit route's own request — scheduling a Task and
returning — not the multi-minute run it starts. ``submit`` enforces its own
ceiling (``KB_IMPORT_MAX_CONCURRENT_JOBS``, default
``_DEFAULT_MAX_CONCURRENT_JOBS``) by counting jobs still in
``submitted``/``working`` state and raising ``ImportJobCapacityExceeded``
when the ceiling is already reached — a clear rejection at submit time
rather than a silently over-subscribed background queue.

Job-store eviction (mirrors issue #474 sub-issue B): ``_evict_terminal_jobs``
sweeps ``completed``/``failed`` jobs whose ``completed_at`` is older than
``KB_IMPORT_JOB_TTL_SECONDS`` (default ``_DEFAULT_JOB_TTL_SECONDS``), over a
snapshot of keys (CODING_STANDARD §2.6), at the top of ``submit`` — the same
call site that grows the registry.
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

from .importer import ImportBatchResult
from .importer import import_sources as _import_sources

# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------


class ImportJobCapacityExceeded(Exception):
    """Raised by ``submit`` when the concurrent-job ceiling is already reached."""


@dataclass
class ImportJob:
    """State for one background Import run.

    Attributes:
        job_id:       Unique identifier (uuid4().hex).
        status:       One of "submitted" | "working" | "completed" | "failed".
        files_done:   Sources processed so far (success or failure).
        files_total:  Total sources in this run (known after the first
                      ``on_source_done`` callback; 0 until then).
        pages_done:   Transcribed pages completed so far, across the run.
        pages_total:  Transcribed pages known so far (grows as each scanned
                      file's page count is discovered — see module docstring).
        result:       The completed run's ``ImportBatchResult``; ``None``
                      while still submitted/working or after a ``failed``.
        error:        Set only if the run coroutine itself raised unexpectedly
                      (per-source failures land in ``result.failed_sources``
                      instead — ``import_sources`` is continue-on-error).
        completed_at: ``time.monotonic()`` value when ``status`` last became
                      terminal; read by ``_evict_terminal_jobs``.
        _task:        Strong reference to the asyncio.Task (prevents GC).
        _lock:        Guards the progress counters against concurrent updates
                      from page-worker threads.
    """

    job_id: str
    status: Literal["submitted", "working", "completed", "failed"] = "submitted"
    files_done: int = 0
    files_total: int = 0
    pages_done: int = 0
    pages_total: int = 0
    result: ImportBatchResult | None = None
    error: str | None = None
    completed_at: float | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


# ---------------------------------------------------------------------------
# Registry (module-level globals)
# ---------------------------------------------------------------------------

_JOBS: dict[str, ImportJob] = {}

# Strong-reference set: prevents GC from cancelling Tasks before they finish.
# Each task removes itself via add_done_callback.
_TASKS: set[asyncio.Task] = set()

# ---------------------------------------------------------------------------
# Concurrent-job cap (mirrors issue #474 sub-issue A)
# ---------------------------------------------------------------------------

# Same order of magnitude as KB_TRANSCRIBE_MAX_CONCURRENT_JOBS' default — see
# the module docstring for why admin_sem cannot cover this instead.
_DEFAULT_MAX_CONCURRENT_JOBS = 2


def _max_concurrent_jobs() -> int:
    """Max number of Import jobs allowed in submitted/working state at once.

    Override with ``KB_IMPORT_MAX_CONCURRENT_JOBS``; default
    ``_DEFAULT_MAX_CONCURRENT_JOBS``. Read at call time (no restart needed),
    mirroring ``transcribe_jobs._max_concurrent_jobs``'s env-parsing shape.
    """
    raw = os.getenv("KB_IMPORT_MAX_CONCURRENT_JOBS")
    if raw is None:
        return _DEFAULT_MAX_CONCURRENT_JOBS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_CONCURRENT_JOBS
    return value if value > 0 else _DEFAULT_MAX_CONCURRENT_JOBS


# ---------------------------------------------------------------------------
# Terminal-job eviction (mirrors issue #474 sub-issue B)
# ---------------------------------------------------------------------------

_DEFAULT_JOB_TTL_SECONDS = 15 * 60  # 15 minutes


def _job_ttl_seconds() -> int:
    """Idle TTL, in seconds, before a completed/failed job is evicted from ``_JOBS``.

    Override with ``KB_IMPORT_JOB_TTL_SECONDS``; default
    ``_DEFAULT_JOB_TTL_SECONDS``. Read at call time (no restart needed).
    """
    raw = os.getenv("KB_IMPORT_JOB_TTL_SECONDS")
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
    the asyncio event loop's own clock, which stalls ``asyncio.sleep`` and
    deadlocks any polling coroutine (the #474 ubuntu CI hang). Tests patch
    ``import_jobs._now`` instead.
    """
    return time.monotonic()


def _evict_terminal_jobs() -> None:
    """Delete completed/failed jobs whose TTL has elapsed.

    Sweeps over a **snapshot** of keys (CODING_STANDARD §2.6) so deleting
    entries mid-sweep never raises. Called at the top of ``submit`` — the
    same call site that grows ``_JOBS`` — rather than a background loop.
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


def _progress_callbacks_for(
    job: ImportJob,
) -> tuple[Callable[[int, int], None], Callable[[str, int, int], None]]:
    """Return ``(on_source_done, on_transcribe_page)`` closures folding into ``job``.

    ``seen_sources`` tracks which files have already contributed their page
    count to ``pages_total`` — the per-page callback reports
    ``total_for_source`` on every call, so only the first call per source may
    add it (the same have-I-added-this-file's-total bookkeeping
    ``transcribe_jobs._progress_callback_for_source`` keeps per closure,
    keyed by basename here because one Import run shares one callback across
    every auto-routed source).
    """
    seen_sources: set[str] = set()

    def _on_source_done(done: int, total: int) -> None:
        with job._lock:
            job.files_done = done
            job.files_total = total

    def _on_transcribe_page(source: str, _done_for_source: int, total_for_source: int) -> None:
        with job._lock:
            if source not in seen_sources:
                seen_sources.add(source)
                job.pages_total += total_for_source
            job.pages_done += 1

    return _on_source_done, _on_transcribe_page


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _run_import(job: ImportJob, source_filter: str | None) -> None:
    """Background coroutine: set working -> run import_sources -> completed.

    ``import_sources`` is continue-on-error by contract (per-source failures
    land in ``failed_sources``), so only a genuinely unexpected exception (a
    bug, not a source failure) sets ``job.status = "failed"`` with
    ``job.error`` populated.
    """
    job.status = "working"
    on_source_done, on_transcribe_page = _progress_callbacks_for(job)
    try:
        batch = await asyncio.to_thread(
            _import_sources,
            source_filter,
            on_source_done=on_source_done,
            on_transcribe_page=on_transcribe_page,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard against a bug in the run itself
        job.status = "failed"
        job.error = str(exc)[:200]
        job.completed_at = _now()
        return

    job.result = batch
    job.status = "completed"
    job.completed_at = _now()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit(source_filter: str | None) -> ImportJob:
    """Schedule a background Import run and return a Job immediately.

    Creates a Job in status='submitted', schedules an asyncio.Task that runs
    ``_run_import(job, source_filter)``, keeps a strong reference to prevent
    GC cancellation, registers the Job in ``_JOBS``, and returns it.

    First sweeps expired terminal jobs (``_evict_terminal_jobs``), then
    raises ``ImportJobCapacityExceeded`` if the number of jobs still in
    submitted/working state is already at ``_max_concurrent_jobs()`` — a
    clear rejection rather than a silently over-subscribed background queue.

    MUST be called from inside a running event loop (i.e. from an ``async
    def`` route handler).
    """
    _evict_terminal_jobs()

    active = sum(1 for job in _JOBS.values() if job.status in ("submitted", "working"))
    cap = _max_concurrent_jobs()
    if active >= cap:
        raise ImportJobCapacityExceeded(
            f"{active} import job(s) already running (cap={cap}); retry later"
        )

    job_id = uuid4().hex
    job = ImportJob(job_id=job_id)
    _JOBS[job_id] = job

    task = asyncio.create_task(_run_import(job, source_filter))
    job._task = task  # keep strong ref on the Job itself

    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)

    return job


def status(job_id: str) -> ImportJob | None:
    """Return the Job for job_id, or None when not found."""
    return _JOBS.get(job_id)


def _reset_jobs() -> None:
    """Clear all registry state. Called by tests for isolation."""
    _JOBS.clear()
    _TASKS.clear()
