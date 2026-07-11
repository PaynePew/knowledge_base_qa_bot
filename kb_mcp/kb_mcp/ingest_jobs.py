"""Deep module per Ousterhout. Public surface: ``submit``, ``status``.

MCP-agnostic in-memory job registry for background ingest tasks (Fix 1b).

Provides a submit/poll pattern so large Sources that would exceed the MCP host
tool-call timeout (-32001) can be offloaded to a background asyncio.Task and
polled for completion.

Public surface:
  submit(source)     -> Job  (schedules work, returns immediately)
  status(job_id)     -> Job | None
  _reset_jobs()      -> None  (test isolation)

Job lifecycle:  submitted → working → completed | failed

GC safety note (CRITICAL):
  asyncio.create_task() keeps only a WEAK reference to the Task — if no
  other reference exists, the GC can cancel the task mid-run silently.
  We keep a strong reference in the module-level ``_TASKS`` set and remove
  each task from it via add_done_callback, so the Task is never GC-collected
  before it reaches a terminal state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """State for one background ingest task.

    Attributes:
        job_id:   Unique identifier (uuid4().hex).
        status:   One of "submitted" | "working" | "completed" | "failed".
        progress: (done, total) tuple.  (0, 1) at submission; (1, 1) when done.
        result:   On completed: dict with the same shape kb_ingest_v1 returns.
        error:    On failed: dict with {code, message}.
        _task:    Strong reference to the asyncio.Task (prevents GC).
    """

    job_id: str
    status: str = "submitted"
    progress: tuple[int, int] = field(default_factory=lambda: (0, 1))
    result: dict | None = None
    error: dict | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Registry  (module-level globals)
# ---------------------------------------------------------------------------

_JOBS: dict[str, Job] = {}

# Strong-reference set: prevents GC from cancelling Tasks before they finish.
# Each task removes itself via add_done_callback.
_TASKS: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _run(job: Job, source: str) -> None:
    """Background coroutine: set working → call aingest_sources → completed/failed.

    Never raises — all exceptions are caught and mapped to job.status='failed'.
    """
    from markdown_kb.app.errors import LLMError
    from markdown_kb.app.ingest import aingest_sources

    job.status = "working"
    try:
        batch = await aingest_sources([source])
    except LLMError as exc:
        code = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        job.status = "failed"
        job.error = {"code": code, "message": exc.message}
        return
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = {"code": "INGEST_ERROR", "message": str(exc)}
        return

    # Map IngestBatchResult → neutral result dict (same shape as kb_ingest_v1).
    if batch.failed_sources and source in batch.failed_sources:
        result: dict = {
            "source": source,
            "pages_created": [],
            "pages_overwritten": [],
            "grounding_failed_pages": [],
            "failed": True,
            "status": "failed",
        }
        reason = batch.failed_reasons.get(source)
        if reason:
            result["reason"] = reason
        error_type = batch.failed_error_types.get(source)
        if error_type:
            result["error_type"] = error_type
    elif batch.results:
        src_result = batch.results[0]
        result = {
            "source": source,
            "pages_created": src_result.pages_created,
            "pages_overwritten": src_result.pages_updated,
            "grounding_failed_pages": batch.pages_with_failed_grounding,
            "failed": False,
            "status": src_result.status,
            "sections_count": src_result.sections_count,
            "uncarried_chars": src_result.uncarried_chars,
            "enriched_chars": src_result.enriched_chars,
        }
    else:
        # Skipped (hash-match no-op)
        skipped = batch.skipped_sources[0] if batch.skipped_sources else None
        result = {
            "source": source,
            "pages_created": [],
            "pages_overwritten": [],
            "grounding_failed_pages": [],
            "failed": False,
            "status": skipped.status if skipped else "skipped",
            "sections_count": skipped.sections_count if skipped else 0,
            "uncarried_chars": skipped.uncarried_chars if skipped else 0,
            "enriched_chars": skipped.enriched_chars if skipped else 0,
        }

    job.result = result
    job.progress = (1, 1)
    job.status = "completed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit(source: str) -> Job:
    """Schedule a background ingest and return a Job immediately.

    Creates a Job in status='submitted', schedules an asyncio.Task that runs
    _run(job, source), keeps a strong reference to prevent GC cancellation,
    registers the Job in _JOBS, and returns it.

    MUST be called from inside a running event loop (i.e. from an async
    context, e.g. an async FastMCP tool handler).
    """
    job_id = uuid4().hex
    job = Job(job_id=job_id)
    _JOBS[job_id] = job

    # Schedule the background work.  create_task requires a running event loop —
    # this is satisfied when called from an async MCP tool handler.
    task = asyncio.create_task(_run(job, source))
    job._task = task  # keep strong ref on the Job itself

    # Also register in the module-level set; remove on completion so the set
    # doesn't grow unbounded.
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)

    return job


def status(job_id: str) -> Job | None:
    """Return the Job for job_id, or None when not found."""
    return _JOBS.get(job_id)


def _reset_jobs() -> None:
    """Clear all registry state.  Called by tests for isolation."""
    _JOBS.clear()
    _TASKS.clear()
