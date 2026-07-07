"""Shallow module per Ousterhout. Public surface: ``router``.

HTTP wiring for /health, /index, /chat, /qa/{slug}/promote,
/qa/{slug}/demote, /qa/promote-batch, /qa/{slug} (DELETE, PUT), /qa/{slug}/refile, /ingest,
/lint, /import, /transcribe, /transcribe/batch, /transcribe/jobs/{job_id},
/transcribe/page-count, /pages/reconcile, /pages/reconcile/apply, /pages/collision/merge,
/pages/collision/merge/apply, /pages/collision/differentiate,
/pages/collision/differentiate/apply, /pages/{slug} (DELETE),
/pages/{slug}/aliases (POST), /pages/{slug}/aliases/{alias} (DELETE),
/pages/resolution-map (GET). No domain logic."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import import_jobs, transcribe_jobs
from . import indexer as _indexer
from . import pages as pages_module
from . import qa as qa_module
from . import reconcile as reconcile_module
from .errors import LLMError
from .importer import ImportBatchResult
from .importer import import_sources as run_import
from .indexer import build_index
from .ingest import ingest_sources
from .lint import run_lint
from .retrieval import query
from .schemas import (
    AliasAssignRequest,
    ChatRequest,
    ChatResponse,
    CollisionDifferentiateApplyRequest,
    CollisionDifferentiateApplyResponse,
    CollisionDifferentiateGenerateRequest,
    CollisionDifferentiateGenerateResponse,
    CollisionMergeApplyRequest,
    CollisionMergeApplyResponse,
    CollisionMergeGenerateRequest,
    CollisionMergeGenerateResponse,
    FiledStatus,
    GroundingClaim,
    GroundingInfo,
    ImportFailureSchema,
    ImportJobStatusResponse,
    ImportJobSubmitResponse,
    ImportRequest,
    ImportResponse,
    ImportSourceResultSchema,
    IndexResponse,
    IngestRequest,
    IngestResponse,
    LintResponse,
    QaEditRequest,
    QaPromoteBatchRequest,
    QaPromoteBatchResponse,
    QaRefileResponse,
    ReconcileApplyRequest,
    ReconcileApplyResponse,
    ReconcileGenerateRequest,
    ReconcileGenerateResponse,
    ResolutionMapResponse,
    TranscribeBatchRequest,
    TranscribeBatchSubmitResponse,
    TranscribeJobResultSchema,
    TranscribeJobStatusResponse,
    TranscribePageCountResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from .transcriber import TranscribePathError
from .transcriber import page_count_for_source_async as get_page_count_for_source_async
from .transcriber import transcribe_source as run_transcribe

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs() -> IndexResponse:
    files_count, sections_count = build_index()
    wiki_written, wiki_path, wiki_error = _indexer.last_wiki_index_outcome
    return IndexResponse(
        files_indexed=files_count,
        sections_indexed=sections_count,
        wiki_index_written=wiki_written,
        wiki_index_path=str(wiki_path) if wiki_path is not None else None,
        wiki_index_error=wiki_error,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a query and return a fully-structured ChatResponse with grounding.

    query() returns {answer, sources, grounding_outcome}.  This handler maps
    the GroundingOutcome to the API-exposed GroundingInfo (ADR-0004 Q8
    selective expose): passes through passed/reason/claims/unsupported_claims;
    suppresses reasoning, error_type, retries_attempted (server logs only).

    Phase 6 Slice 6-2: when the Grounding Check passes, dispatch one line to
    ``qa.maybe_file_answer(...)`` to create or touch ``wiki/qa/<slug>.md`` as
    a side-effect (CODING_STANDARD §2.3 — all complexity lives in ``qa.py``).
    Cannot-Confirm paths (``outcome.passed == False``) skip filing entirely so
    failed queries never pollute the wiki. The filing result populates
    ``ChatResponse.filed`` for caller audit; ``None`` covers Cannot-Confirm,
    IOError fail-soft, and orphan-status touch refusal.
    """
    try:
        result = query(req.query)
    except LLMError as e:
        raise HTTPException(
            status_code=503 if e.retryable else 500,
            detail=e.message,
        ) from e
    outcome = result["grounding_outcome"]

    # Build claims list if the verifier ran and produced structured claims.
    claims = None
    if outcome.result is not None and outcome.result.claims:
        claims = [
            GroundingClaim(
                text=c.text,
                supported=c.supported,
                citing_section_ids=c.citing_section_ids,
            )
            for c in outcome.result.claims
        ]

    # unsupported_claims only populated when reason=claim_unsupported (ADR-0004 Q8).
    unsupported_claims = None
    if outcome.reason == "claim_unsupported" and outcome.result is not None:
        unsupported_claims = outcome.result.unsupported_claims or None

    grounding = GroundingInfo(
        passed=outcome.passed,
        reason=outcome.reason,
        claims=claims,
        unsupported_claims=unsupported_claims,
    )

    # Phase 6 Slice 6-3: pass ``derived_from`` through to the API boundary.
    # retrieval.query populates it as a list of {source, heading} dicts (matches
    # the CitationRef schema) or None. Pydantic validates at the ChatResponse
    # boundary. Older retrieval responses without the key still validate
    # because SourceInfo.derived_from defaults to None (Slice 6-1 schema).
    sources = [
        {
            "source": s["source"],
            "heading": s["heading"],
            "score": s["score"],
            "content": s["content"],
            "derived_from": s.get("derived_from"),
        }
        for s in result["sources"]
    ]

    # Phase 6 Slice 6-2 / Phase 9 Slice 4: gated side-effect — file Grounded
    # Answers only. Cannot Confirm paths (passed=False) do NOT file.
    # Delegation to ``qa.dispatch_filing`` keeps the gating + SectionRef adapter
    # in one place shared with the Wiki stream_query path (issue #121 AC1).
    filed = qa_module.dispatch_filing(req.query, result)

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        grounding=grounding,
        filed=filed,
    )


@router.post("/qa/{slug}/promote", response_model=FiledStatus)
def promote_qa(slug: str) -> FiledStatus:
    """Curator endpoint: flip ``wiki/qa/<slug>.md`` ``status: draft -> live``.

    Phase 6 Slice 6-4. Closes the two-stage curation loop: filing auto-creates
    drafts (Slice 6-2), ``/lint`` surfaces promotion candidates (Slice 6-5),
    and this endpoint is the explicit promote action that admits the page into
    the BM25 corpus for ``/chat`` retrieval.

    ADR-0020 Consequence 1: after the status flip, ``build_index`` is called
    automatically so the promoted page enters the BM25 corpus immediately
    without requiring a separate ``POST /index`` call. Full rebuild is
    deliberate at the current corpus size (sub-second; guaranteed correct;
    incremental patching deferred — see ADR-0020 Considered Options).

    Exception mapping (all-or-nothing — build_index is NOT called when promote
    itself raises):

    - ``QaPageNotFound`` → ``404`` (slug has never been filed)
    - ``QaPageCorrupt``  → ``500`` (orphan-visibility — surface broken state
      to the curator rather than silently rewriting it)

    Idempotent on already-live pages: re-promote returns the existing
    ``FiledStatus`` with ``200 OK`` (no second log entry, no file write);
    build_index is still called so the corpus is guaranteed consistent.
    """
    try:
        result = qa_module.promote(slug)
    except qa_module.QaPageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki/qa/{slug}.md not found",
        ) from exc
    except qa_module.QaPageCorrupt as exc:
        raise HTTPException(
            status_code=500,
            detail=f"wiki/qa/{slug}.md has corrupt frontmatter: {exc}",
        ) from exc
    # ADR-0020: auto-reindex so the live page is retrievable immediately.
    build_index()
    return result


@router.post("/qa/{slug}/demote", response_model=FiledStatus)
def demote_qa(slug: str) -> FiledStatus:
    """Curator endpoint: flip ``wiki/qa/<slug>.md`` ``status: live -> draft`` in place.

    Issue #535 / ADR-0037 — the C10 remediation for a schema-invalid
    ``status: live`` page: ``qa.delete`` refuses any live page (ADR-0012),
    so a live-but-defective Filed Answer could previously neither be
    discarded nor fixed. Demote is the reversible inverse of ``promote`` —
    a lifecycle bit flip, no LLM, no synthesis — so the page leaves the BM25
    corpus and re-enters the Promote/Edit/Discard Curation Queue loop, where
    the curator either fixes the schema and re-promotes, or discards it
    (draft delete is already allowed).

    Shallow wrapper around ``qa.demote`` (CODING_STANDARD §2.3 — all domain
    logic lives in ``qa.py``). ``build_index()`` is called here, once, after
    a successful demote — mirrors ``POST /qa/{slug}/promote``'s auto-reindex
    convention (reindex is a route-layer concern, not a domain-layer one).

    Exception mapping (build_index is NOT called when demote itself raises):

    - ``QaPageNotFound`` → ``404`` (slug has never been filed)
    - ``QaPageCorrupt``  → ``500`` (orphan-visibility — surface broken state
      rather than silently rewriting it)

    Idempotent on already-draft pages: re-demote returns the existing
    ``FiledStatus`` with ``200 OK`` (no second log entry, no file write);
    build_index is still called so the corpus is guaranteed consistent.
    """
    try:
        result = qa_module.demote(slug)
    except qa_module.QaPageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki/qa/{slug}.md not found",
        ) from exc
    except qa_module.QaPageCorrupt as exc:
        raise HTTPException(
            status_code=500,
            detail=f"wiki/qa/{slug}.md has corrupt frontmatter: {exc}",
        ) from exc
    # ADR-0037: auto-reindex so the demoted page leaves the BM25 corpus immediately.
    build_index()
    return result


@router.post("/qa/promote-batch", response_model=QaPromoteBatchResponse)
def promote_batch_qa(req: QaPromoteBatchRequest) -> QaPromoteBatchResponse:
    """Curator endpoint: batch-promote an explicit slugs list, one reindex.

    tier-B S6 (issue #382) / ADR-0023 Consequences — the one pre-authorized
    Direct-tier batch endpoint ("a batch-promote endpoint, deferred" — this
    slice ships it). ``req.slugs`` must be exactly what the operator saw
    rendered in the Curation Queue, never "all drafts" resolved server-side,
    so a draft filed after the operator looked is never approved
    sight-unseen.

    Shallow wrapper around ``qa.promote_batch`` (CODING_STANDARD §2.3 — all
    domain logic, including per-slug validation, lives in ``qa.py``).
    ``build_index()`` is called here exactly once after the loop, regardless
    of how many slugs were promoted (issue #382 AC) — mirrors ``POST
    /qa/{slug}/promote``'s auto-reindex convention, just once for the whole
    batch instead of once per slug.

    Always returns HTTP 200 — an individual slug's validation failure is
    reported honestly in ``response.skipped``, not raised as an exception
    (ADR-0023: "Non-transactional, honestly reported"); there is nothing to
    map to an error status here.
    """
    result = qa_module.promote_batch(req.slugs)
    # ADR-0023: exactly one reindex for the whole batch, regardless of N.
    build_index()
    return result


@router.delete("/qa/{slug}", status_code=204)
def delete_qa(slug: str) -> None:
    """Curator endpoint: delete an inert ``wiki/qa/<slug>.md`` page.

    Phase 15 Slice 6 (issue #174) / ADR-0012. Complements the Promote
    endpoint: discards a Filed Answer that has never entered the BM25 corpus
    (``status: draft`` or schema-invalid / unparseable frontmatter) and
    **refuses** to delete a ``status: live`` page (the precious corpus state).

    Shallow wrapper around ``qa.delete(slug)`` (CODING_STANDARD §2.3 — all
    business logic lives in ``qa.py``). Exception mapping:

    - ``QaPageNotFound`` → ``404`` (slug has never been filed)
    - ``QaPageLive``     → ``409`` (live page delete refused — ADR-0012)

    Returns HTTP 204 No Content on success (the resource is gone; no body).
    """
    try:
        qa_module.delete(slug)
    except qa_module.QaPageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki/qa/{slug}.md not found",
        ) from exc
    except qa_module.QaPageLive as exc:
        raise HTTPException(
            status_code=409,
            detail=f"wiki/qa/{slug}.md has status=live; delete refused (ADR-0012)",
        ) from exc


@router.put("/qa/{slug}", response_model=FiledStatus)
def edit_qa(slug: str, req: QaEditRequest) -> FiledStatus:
    """Curator endpoint: edit a draft ``wiki/qa/<slug>.md``'s question/body in place.

    tier-B S3 (issue #379) / ADR-0026 decision 2. Completes the Curation
    Queue gate's verb set — approve (``/promote``) / edit-then-approve
    (this endpoint, then ``/promote``) / discard (``DELETE``). Draft-only:
    refuses a ``status: live`` page (live hand-edits keep the documented
    file-level path). Re-runs the LLM-free grounding check against the
    page's cited Sections on the submitted ``body``; a failing check writes
    nothing (ADR-0026: "the re-check is LLM-free and instant").

    Shallow wrapper around ``qa.edit`` (CODING_STANDARD §2.3 — all domain
    logic lives in ``qa.py``). No reindex: an edited page stays ``status:
    draft`` and drafts never enter the BM25 corpus (``promote`` reindexes).

    Exception mapping:

    - ``QaPageNotFound`` → ``404`` (slug has never been filed)
    - ``QaPageCorrupt``  → ``500`` (orphan-visibility — surface broken state
      rather than silently rewriting it)
    - ``QaPageLive``     → ``409`` (edit refused — draft-only, ADR-0026)
    - ``QaEditRejected`` → ``422`` (grounding re-check failed;
      ``detail.failures`` lists every problem)
    """
    try:
        return qa_module.edit(slug, req.question, req.body)
    except qa_module.QaPageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki/qa/{slug}.md not found",
        ) from exc
    except qa_module.QaPageCorrupt as exc:
        raise HTTPException(
            status_code=500,
            detail=f"wiki/qa/{slug}.md has corrupt frontmatter: {exc}",
        ) from exc
    except qa_module.QaPageLive as exc:
        raise HTTPException(
            status_code=409,
            detail=f"wiki/qa/{slug}.md has status=live; edit refused (ADR-0026)",
        ) from exc
    except qa_module.QaEditRejected as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "qa edit content failed grounding re-check",
                "failures": exc.failures,
            },
        ) from exc


@router.post("/qa/{slug}/refile", response_model=QaRefileResponse)
def refile_qa(slug: str) -> QaRefileResponse:
    """Curator endpoint: C9 remediation — chained re-file of a stale Filed Answer.

    tier-B S4 (issue #380) / ADR-0026 decision 1. Fixed internal order (see
    ``qa.refile`` docstring): re-synthesize the page's question via the chat
    pipeline with ``wiki/qa/`` excluded from retrieval, grounding-check the
    fresh answer BEFORE any write. On a passing re-ground it overwrites the
    same slug in place as ``status: draft`` (fresh content). On a CONTENT
    re-ground failure (the KB can no longer ground the answer) it RETIRES a
    LIVE page — demotes it to draft in place with its OLD content
    (``retired: true``, ADR-0035) — so a stale answer never keeps serving
    un-groundable content. Only a TRANSIENT re-ground failure (verifier/index
    unavailable), or a non-live page, writes nothing (422).

    Shallow wrapper around ``qa.refile`` (CODING_STANDARD §2.3 — all domain
    logic lives in ``qa.py``). ``build_index()`` is called here, once, after a
    refile OR a retire (both remove the answer from the live BM25 corpus) —
    mirrors ``POST /qa/{slug}/promote``'s auto-reindex convention (reindex is a
    route-layer concern, not a domain-layer one).

    Exception mapping (build_index is NOT called on any exception path):

    - ``QaPageNotFound``     → ``404`` (slug has never been filed, or was
      deleted by a concurrent operation during re-synthesis)
    - ``QaPageCorrupt``      → ``500`` (orphan-visibility — surface broken
      state rather than silently rewriting it)
    - ``QaRefileRejected``   → ``422`` (the re-synthesis failed for a TRANSIENT
      reason — verifier/index unavailable — or on a non-live page;
      ``detail.reason`` / ``detail.unsupported_claims`` report why; nothing was
      written. A CONTENT failure on a live page RETIRES instead and returns
      200 with ``retired: true``.)
    """
    try:
        result = qa_module.refile(slug)
    except qa_module.QaPageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki/qa/{slug}.md not found",
        ) from exc
    except qa_module.QaPageCorrupt as exc:
        raise HTTPException(
            status_code=500,
            detail=f"wiki/qa/{slug}.md has corrupt frontmatter: {exc}",
        ) from exc
    except qa_module.QaRefileRejected as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "qa refile failed to re-ground",
                "reason": exc.grounding.reason,
                "unsupported_claims": exc.grounding.unsupported_claims or [],
            },
        ) from exc

    # ADR-0026/0035: auto-reindex so the answer (fresh-refiled OR retired) leaves
    # the live corpus immediately.
    _files_indexed, sections_indexed = build_index()
    return QaRefileResponse(
        filed=result.filed,
        grounding=result.grounding,
        sections_indexed=sections_indexed,
        retired=result.retired,
    )


@router.post("/lint", response_model=LintResponse)
def lint(include_c5: bool = True) -> LintResponse:
    """Run the wiki lint pass and return a structured findings report.

    Executes the lint checks and writes ``wiki/lint-report.md``.  Returns 200
    with a ``LintResponse`` in all cases, including when one or more checks
    raised (errors recorded in ``LintResponse.check_errors``).

    Query params:
        include_c5: When ``true`` (default) run the full audit including C5
            (page-pair contradiction detection — the only LLM-backed check,
            one call per candidate pair). Pass ``?include_c5=false`` for the
            fast, LLM-free path used to populate the Console's Curation Queue
            (which needs only the local C8/C9/C10 checks).

    Shallow wrapper around ``lint.run_lint()``.  All domain logic lives in
    ``lint.py`` (CODING_STANDARD §2.3).
    """
    return run_lint(include_c5=include_c5)


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest | None = None) -> IngestResponse:
    """Ingest one or all Sources and write wiki synthesis pages.

    - No body (or body with ``source=null`` and no ``sources``): batch mode —
      processes all Sources discovered under docs/ via ``glob("**/*.md")``.
    - Body with ``sources=["a.md", "b.md"]``: explicit multi-source batch
      mode — ingests exactly those named Sources in one call, sharing the
      cross-source ``used_slugs`` set for slug collision detection (#54).
      Used by the Operator Console drop-batch path (Phase 15 S3, issue #172).
    - Body with ``source="<filename>"``: single-source mode (back-compat).

    Priority: ``sources`` list (non-empty) → ``source`` single-string →
    all-docs batch mode.

    Shallow wrapper around `ingest_sources(...)`.  All domain logic
    (parse → classify → synthesise → write) lives in `ingest.py`
    (CODING_STANDARD §2.3).

    Returns 200 with the IngestResponse in all cases, including when a Source
    is not found (reflected in ``failed_sources``).
    """
    force = req.force if req is not None else False
    if req is not None and req.sources:
        # Explicit multi-source batch (Phase 15 S3): one call, shared used_slugs.
        batch = ingest_sources(req.sources, force=force)
    elif req is None or req.source is None:
        # All-docs batch mode: ingest everything under docs/
        batch = ingest_sources(None, force=force)
    else:
        # Single-source mode (back-compat)
        batch = ingest_sources([req.source], force=force)
    return IngestResponse(
        results=batch.results,
        failed_sources=batch.failed_sources,
        pages_with_failed_grounding=batch.pages_with_failed_grounding,
        skipped_sources=batch.skipped_sources,
    )


def _to_import_response(batch: ImportBatchResult) -> ImportResponse:
    """Map an ``ImportBatchResult`` onto the API-boundary ``ImportResponse``.

    Shared by the synchronous ``POST /import`` and the async
    ``GET /import/jobs/{job_id}`` (issue #497) so both surfaces return the
    byte-identical shape for the same run.
    """
    return ImportResponse(
        imported_sources=[
            ImportSourceResultSchema(
                raw_path=r.raw_path,
                docs_path=r.docs_path,
                original_format=r.original_format,
                content_sha256=r.content_sha256,
                status=r.status,
            )
            for r in batch.imported_sources
        ],
        skipped_sources=[
            ImportSourceResultSchema(
                raw_path=r.raw_path,
                docs_path=r.docs_path,
                original_format=r.original_format,
                content_sha256=r.content_sha256,
                status=r.status,
            )
            for r in batch.skipped_sources
        ],
        failed_sources=[
            ImportFailureSchema(
                raw_path=f.raw_path,
                error_type=f.error_type,
                error_message=f.error_message,
            )
            for f in batch.failed_sources
        ],
    )


@router.post("/import", response_model=ImportResponse)
def import_raw(req: ImportRequest | None = None) -> ImportResponse:
    """Convert raw sources to Markdown docs.

    - No body (or body with ``source=null``): batch mode — globs
      ``raw/**/*.{html,txt}`` recursively and writes each to ``docs/``.
    - Body with ``source="<filename>"``: single-source mode — converts one
      specified file.

    Shallow wrapper around ``importer.import_sources(...)`` (CODING_STANDARD
    §2.3 — all domain logic lives in ``importer.py``).

    Always returns HTTP 200 with an ``ImportResponse``.  Per-source failures
    are recorded in ``failed_sources`` without aborting the batch.
    """
    source_filter = req.source if req is not None else None
    batch: ImportBatchResult = run_import(source_filter)

    return _to_import_response(batch)


# ---------------------------------------------------------------------------
# /import/jobs — async submit/poll for a whole Import run (issue #497)
# ---------------------------------------------------------------------------


@router.post("/import/jobs", response_model=ImportJobSubmitResponse)
async def import_jobs_submit(req: ImportRequest | None = None) -> ImportJobSubmitResponse:
    """Submit a background Import run (same request contract as ``POST /import``).

    issue #497: the synchronous route auto-transcribes a text-less PDF
    INSIDE the request (ADR-0032) — one vision call per page, minutes for a
    real scan, long past the edge proxy's window; the proxy then 502s with
    an empty body while the server keeps working. This returns a job id
    IMMEDIATELY; the background task keeps running independent of the client
    connection. Poll ``GET /import/jobs/{job_id}`` for progress
    (files + transcribed pages) and, eventually, the same ``ImportResponse``
    the synchronous route would have returned.

    Shallow wrapper around ``import_jobs.submit`` (CODING_STANDARD §2.3 —
    all domain logic lives in ``import_jobs.py`` / ``importer.py``).

    Raises:
        HTTP 503: the concurrent-job cap (``KB_IMPORT_MAX_CONCURRENT_JOBS``,
            mirrors issue #474 sub-issue A) is already reached — a clear
            rejection instead of a silently over-subscribed background queue.
    """
    try:
        job = import_jobs.submit(req.source if req is not None else None)
    except import_jobs.ImportJobCapacityExceeded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ImportJobSubmitResponse(job_id=job.job_id)


@router.get("/import/jobs/{job_id}", response_model=ImportJobStatusResponse)
async def get_import_job(job_id: str) -> ImportJobStatusResponse:
    """Poll a background Import run's status, progress, and (once done) result.

    ``pages_done`` / ``pages_total`` are the real, server-owned Transcribe
    page counts (the Console progress bar's data source — §12.8 bans
    client-guessed percentages, not real counts); ``files_done`` /
    ``files_total`` track whole-run file progress.

    Shallow wrapper around ``import_jobs.status`` (CODING_STANDARD §2.3).

    Raises:
        HTTP 404: no job with this id — never submitted, the process
            restarted (the registry is in-memory only), or the job was
            evicted after ``KB_IMPORT_JOB_TTL_SECONDS`` past completion.
    """
    job = import_jobs.status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"import job not found: {job_id}")

    return ImportJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        files_done=job.files_done,
        files_total=job.files_total,
        pages_done=job.pages_done,
        pages_total=job.pages_total,
        result=_to_import_response(job.result) if job.result is not None else None,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# /transcribe — Transcribe force entry (issue #426, ADR-0032)
# ---------------------------------------------------------------------------


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe_raw(req: TranscribeRequest) -> TranscribeResponse:
    """Force-transcribe one named file already staged under ``raw/``.

    issue #426 / ADR-0032: the designed-PDF escape hatch — bypasses the
    text-layer probe that ``POST /import`` applies automatically, always
    running the model-assisted conversion. Single-source only, no batch mode
    (mirrors the CLI's ``kb transcribe <path>``, which stages a local file
    into ``raw/`` first and then calls the same underlying deep-module
    function this route calls).

    Shallow wrapper around ``transcriber.transcribe_source`` (CODING_STANDARD
    §2.3 — all domain logic lives in ``transcriber.py``).

    Raises:
        HTTP 400: invalid filename / source path / unsupported extension
            (Transcribe only handles ``.pdf``).
        HTTP 404: the named file does not exist under ``raw/``.
        HTTP 413: the PDF's page count exceeds ``KB_TRANSCRIBE_MAX_PAGES``.
        HTTP 500: a page failed transcription after bounded retry, or an
            atomic-write ``IOError``.
        HTTP 503: Transcribe is unavailable (missing ``OPENAI_API_KEY`` or
            ``KB_TRANSCRIBE_ENABLED`` not set), or the Gateway's daily USD
            budget hook rejected this file's page count before any model
            call (issue #460 — same 503 family as budget-exhaustion
            elsewhere in the Gateway).
    """
    try:
        result = run_transcribe(req.source)
    except TranscribePathError as exc:
        status_map = {
            "FileNotFoundError": 404,
            "InvalidFilename": 400,
            "InvalidSourcePath": 400,
            "UnsupportedExtension": 400,
            "TranscribeUnavailable": 503,
            "TranscribePageLimitExceeded": 413,
            "TranscribeBudgetExceeded": 503,
        }
        status_code = status_map.get(exc.error_type, 500)
        raise HTTPException(status_code=status_code, detail=exc.message) from exc

    return TranscribeResponse(
        raw_path=result.raw_path,
        docs_path=result.docs_path,
        content_sha256=result.content_sha256,
        transcribe_model=result.transcribe_model,
        status=result.status,
    )


# ---------------------------------------------------------------------------
# /transcribe/batch + /transcribe/jobs/{job_id} — async submit/poll (issue #459 AC5)
# ---------------------------------------------------------------------------


@router.post("/transcribe/batch", response_model=TranscribeBatchSubmitResponse)
async def transcribe_batch(req: TranscribeBatchRequest) -> TranscribeBatchSubmitResponse:
    """Submit a batch of named raw/ PDFs for background force-transcription.

    issue #459 item 5: a batch of large scans run one after another (mirrors
    ``import_sources``) can take minutes — long enough to blow an HTTP
    client's connection window even with per-PDF page concurrency. This
    returns a job id IMMEDIATELY; the background task keeps running
    independent of the client connection, and each source lands in
    ``docs/`` (origin: transcribed) exactly as a synchronous ``POST
    /transcribe`` call would. Poll ``GET /transcribe/jobs/{job_id}`` for
    progress and, eventually, per-source results.

    Shallow wrapper around ``transcribe_jobs.submit_batch`` (CODING_STANDARD
    §2.3 — all domain logic lives in ``transcribe_jobs.py`` /
    ``transcriber.py``). Per-source validation and conversion failures are
    recorded in the job's ``results`` once it completes, not raised here
    (mirrors ``POST /import``'s always-200 contract for batch failures).

    Raises:
        HTTP 422: ``sources`` exceeds ``MAX_BATCH_SOURCES`` (Pydantic
            boundary validation, issue #474 sub-issue B).
        HTTP 503: the concurrent-batch-job cap (``KB_TRANSCRIBE_MAX_CONCURRENT_JOBS``,
            issue #474 sub-issue A) is already reached — a clear rejection
            instead of a silently over-subscribed background queue.
    """
    try:
        job = transcribe_jobs.submit_batch(req.sources)
    except TranscribePathError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    return TranscribeBatchSubmitResponse(job_id=job.job_id)


@router.get("/transcribe/jobs/{job_id}", response_model=TranscribeJobStatusResponse)
async def get_transcribe_job(job_id: str) -> TranscribeJobStatusResponse:
    """Poll a background Transcribe batch's status, progress, and results.

    issue #459 AC4: ``pages_done`` / ``pages_total`` track pages across the
    WHOLE batch, updated as each PDF's pages complete — the data source for
    a Console progress bar (#447).

    Shallow wrapper around ``transcribe_jobs.status`` (CODING_STANDARD §2.3).

    Raises:
        HTTP 404: no job with this id — never submitted, the process
            restarted (the registry is in-memory only), or the job was
            evicted after ``KB_TRANSCRIBE_JOB_TTL_SECONDS`` past completion
            (issue #474 sub-issue B).
    """
    job = transcribe_jobs.status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"transcribe job not found: {job_id}")

    return TranscribeJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        pages_done=job.pages_done,
        pages_total=job.pages_total,
        results=[
            TranscribeJobResultSchema(
                source=r.source,
                status=r.status,
                docs_path=r.docs_path,
                error_type=r.error_type,
                error_message=r.error_message,
            )
            for r in job.results
        ],
        error=job.error,
    )


# ---------------------------------------------------------------------------
# /transcribe/page-count — mechanical preflight, no model call (issue #447)
# ---------------------------------------------------------------------------


@router.get("/transcribe/page-count", response_model=TranscribePageCountResponse)
async def transcribe_page_count(source: str) -> TranscribePageCountResponse:
    """Return a staged raw/ PDF's page count and the configured page cap.

    Mechanical preflight (no model call) the Console's guarded Transcribe
    action calls before showing its confirm step — names the real page count
    rather than guessing a bound client-side (CODING_STANDARD §12.5). Shares
    ``/transcribe``'s validation chain, so a source that would 404/400 there
    404s/400s here too, before any confirm dialog is shown.

    ``async def`` + ``transcriber.page_count_for_source_async`` rather than a
    plain sync route + ``page_count_for_source`` (issue #482) — see that
    function's docstring for why a sync route here could starve other
    endpoints' shared threadpool capacity.

    Shallow wrapper around ``transcriber.page_count_for_source_async``
    (CODING_STANDARD §2.3 — all domain logic lives in ``transcriber.py``).

    Raises:
        HTTP 400: invalid filename / source path / unsupported extension.
        HTTP 404: the named file does not exist under ``raw/``.
    """
    try:
        page_count, max_pages = await get_page_count_for_source_async(source)
    except TranscribePathError as exc:
        status_map = {
            "FileNotFoundError": 404,
            "InvalidFilename": 400,
            "InvalidSourcePath": 400,
            "UnsupportedExtension": 400,
        }
        status_code = status_map.get(exc.error_type, 500)
        raise HTTPException(status_code=status_code, detail=exc.message) from exc

    return TranscribePageCountResponse(source=source, page_count=page_count, max_pages=max_pages)


# ---------------------------------------------------------------------------
# /pages/reconcile — C5 Reconcile two-phase flow (tier-B S1, ADR-0028)
# ---------------------------------------------------------------------------


@router.post("/pages/reconcile", response_model=ReconcileGenerateResponse)
def reconcile_generate(req: ReconcileGenerateRequest) -> ReconcileGenerateResponse:
    """Draft a reconciled version of two contradicting wiki pages.

    ADR-0028 phase 1 of the stateless two-phase Reconcile flow: the LLM
    drafts from the union of both pages' Sources, the draft is grounding-
    checked, and the response carries the draft + grounding report + each
    page's content hash. **Writes nothing to disk** (Invariant).

    Shallow wrapper around ``reconcile.generate_reconcile`` (CODING_STANDARD
    §2.3 — all domain logic lives in ``reconcile.py``).

    Raises:
        HTTP 400: ``page_a`` and ``page_b`` name the same slug.
        HTTP 404: either slug does not resolve to an existing
            ``wiki/entities/`` or ``wiki/concepts/`` page.
        HTTP 500: either page exists but its frontmatter is corrupt
            (orphan-visibility — surface broken state rather than silently
            rewriting it, mirrors ``POST /qa/{slug}/promote``).
    """
    try:
        return reconcile_module.generate_reconcile(req.page_a, req.page_b)
    except reconcile_module.ReconcileInvalidPair as exc:
        raise HTTPException(
            status_code=400,
            detail=f"page_a and page_b must name different pages, got {exc}",
        ) from exc
    except reconcile_module.PageNotFound as exc:
        raise HTTPException(status_code=404, detail=f"wiki page not found: {exc}") from exc
    except reconcile_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500, detail=f"wiki page has corrupt frontmatter: {exc}"
        ) from exc


@router.post("/pages/reconcile/apply", response_model=ReconcileApplyResponse)
def reconcile_apply(req: ReconcileApplyRequest) -> ReconcileApplyResponse:
    """Re-verify and commit the final (possibly human-edited) reconcile content.

    ADR-0028 phase 2: re-runs the grounding check on the EXACT submitted
    content and refuses (409) when either page's content hash no longer
    matches the value returned by ``POST /pages/reconcile`` (the finding may
    no longer hold). On pass, rewrites both pages in place (both slugs
    survive — Invariant) and the caller triggers exactly one BM25 reindex.

    Shallow wrapper around ``reconcile.apply_reconcile`` (CODING_STANDARD
    §2.3 — all domain logic lives in ``reconcile.py``). ``build_index()`` is
    called here, once, after a successful apply — mirrors
    ``POST /qa/{slug}/promote``'s auto-reindex convention (reindex is a
    route-layer concern, not a domain-layer one).

    Raises:
        HTTP 400: ``page_a`` and ``page_b`` name the same slug.
        HTTP 404: either slug does not resolve to an existing page.
        HTTP 409: either page's on-disk content changed since generate time.
        HTTP 422: the apply-time grounding re-check failed for either page's
            submitted content; ``detail.unsupported_claims`` lists the
            offending claims.
        HTTP 500: either page exists but its frontmatter is corrupt.
    """
    try:
        result = reconcile_module.apply_reconcile(req)
    except reconcile_module.ReconcileInvalidPair as exc:
        raise HTTPException(
            status_code=400,
            detail=f"page_a and page_b must name different pages, got {exc}",
        ) from exc
    except reconcile_module.PageNotFound as exc:
        raise HTTPException(status_code=404, detail=f"wiki page not found: {exc}") from exc
    except reconcile_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500, detail=f"wiki page has corrupt frontmatter: {exc}"
        ) from exc
    except reconcile_module.ReconcileHashMismatch as exc:
        raise HTTPException(
            status_code=409,
            detail=f"one or both pages changed since generate — reconcile refused: {exc}",
        ) from exc
    except reconcile_module.ReconcileGroundingFailed as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "reconcile content failed grounding re-check",
                "reason": exc.grounding.reason,
                "unsupported_claims": exc.grounding.unsupported_claims or [],
            },
        ) from exc

    # ADR-0028: exactly one reindex after both pages are written.
    _files_indexed, sections_indexed = build_index()
    return ReconcileApplyResponse(
        page_a=result.page_a,
        page_b=result.page_b,
        grounding=result.grounding,
        sections_indexed=sections_indexed,
    )


# ---------------------------------------------------------------------------
# /pages/collision — C4 dual resolution (tier-B S2, ADR-0028, issue #378)
# ---------------------------------------------------------------------------


@router.post("/pages/collision/merge", response_model=CollisionMergeGenerateResponse)
def collision_merge_generate(req: CollisionMergeGenerateRequest) -> CollisionMergeGenerateResponse:
    """Draft a merged version of a C4 slug-collision group's base page.

    ADR-0028 phase 1 of the merge-into-base resolution: the LLM drafts one
    merged page from the union of every group member's Sources, the draft is
    grounding-checked, and the response carries the draft + grounding report
    + every member's content hash. **Writes nothing to disk** (Invariant).

    Shallow wrapper around ``reconcile.generate_collision_merge``
    (CODING_STANDARD §2.3 — all domain logic lives in ``reconcile.py``).

    Raises:
        HTTP 400: ``variant_slugs`` is empty, contains ``base_slug``, or has
            duplicates.
        HTTP 404: ``base_slug`` or any variant slug does not resolve to an
            existing ``wiki/entities/`` or ``wiki/concepts/`` page.
        HTTP 500: any named page exists but its frontmatter is corrupt.
    """
    try:
        return reconcile_module.generate_collision_merge(req.base_slug, req.variant_slugs)
    except reconcile_module.CollisionInvalidGroup as exc:
        raise HTTPException(status_code=400, detail=f"invalid collision group: {exc}") from exc
    except reconcile_module.PageNotFound as exc:
        raise HTTPException(status_code=404, detail=f"wiki page not found: {exc}") from exc
    except reconcile_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500, detail=f"wiki page has corrupt frontmatter: {exc}"
        ) from exc


@router.post("/pages/collision/merge/apply", response_model=CollisionMergeApplyResponse)
def collision_merge_apply(req: CollisionMergeApplyRequest) -> CollisionMergeApplyResponse:
    """Re-verify and commit the final (possibly human-edited) merge content,
    then delete the reference-free variants.

    ADR-0028 phase 2 of merge-into-base: re-runs the grounding check on the
    EXACT submitted base content and refuses (409) when any page's content
    hash no longer matches the value returned by
    ``POST /pages/collision/merge`` (the finding may no longer hold). Before
    writing, the server also refuses (409) when any variant slated for
    deletion still has an inbound ``[[link]]`` or qa citation — the
    **inbound-reference guard** (ADR-0028 Invariant), listing the referrers
    so the Console can render the refusal honestly. On pass: the base page
    is rewritten in place and the reference-free variants are deleted; the
    caller triggers exactly one BM25 reindex.

    Shallow wrapper around ``reconcile.apply_collision_merge``
    (CODING_STANDARD §2.3 — all domain logic lives in ``reconcile.py``).
    ``build_index()`` is called here, once, after a successful apply —
    mirrors ``POST /pages/reconcile/apply``'s auto-reindex convention.

    Raises:
        HTTP 400: malformed group shape (``CollisionInvalidGroup``).
        HTTP 404: ``base_slug`` or any variant slug does not resolve.
        HTTP 409: hash mismatch (plain-string detail) OR the inbound-
            reference guard refused (``detail`` is a dict listing
            ``referrers``) — the client distinguishes the two by the shape
            of ``detail``.
        HTTP 422: the apply-time grounding re-check failed for the
            submitted base content; ``detail.unsupported_claims`` lists the
            offending claims.
        HTTP 500: any named page exists but its frontmatter is corrupt.
    """
    try:
        result = reconcile_module.apply_collision_merge(req)
    except reconcile_module.CollisionInvalidGroup as exc:
        raise HTTPException(status_code=400, detail=f"invalid collision group: {exc}") from exc
    except reconcile_module.PageNotFound as exc:
        raise HTTPException(status_code=404, detail=f"wiki page not found: {exc}") from exc
    except reconcile_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500, detail=f"wiki page has corrupt frontmatter: {exc}"
        ) from exc
    except reconcile_module.CollisionHashMismatch as exc:
        raise HTTPException(
            status_code=409,
            detail=f"one or more pages changed since generate — merge refused: {exc}",
        ) from exc
    except reconcile_module.CollisionReferenceGuardFailed as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "merge refused: one or more variants are still referenced",
                "referrers": [
                    {
                        "variant_slug": r.variant_slug,
                        "wiki_referrers": r.wiki_referrers,
                        "qa_referrers": r.qa_referrers,
                    }
                    for r in exc.referrers
                ],
            },
        ) from exc
    except reconcile_module.CollisionGroundingFailed as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "merge content failed grounding re-check",
                "reason": exc.grounding.reason,
                "unsupported_claims": exc.grounding.unsupported_claims or [],
            },
        ) from exc

    # ADR-0028: exactly one reindex after the base rewrite + variant deletions.
    _files_indexed, sections_indexed = build_index()
    return CollisionMergeApplyResponse(
        base_slug=result.base_slug,
        deleted_variants=result.deleted_variants,
        grounding=result.grounding,
        sections_indexed=sections_indexed,
    )


@router.post(
    "/pages/collision/differentiate", response_model=CollisionDifferentiateGenerateResponse
)
def collision_differentiate_generate(
    req: CollisionDifferentiateGenerateRequest,
) -> CollisionDifferentiateGenerateResponse:
    """Draft complementary content for every page in a C4 collision group.

    ADR-0028 phase 1 of the differentiate resolution: the LLM rewrites every
    group member from the union of every member's Sources so each keeps its
    own distinct angle; the draft is grounding-checked per page, and the
    response carries the drafts + grounding report + every member's content
    hash. **Writes nothing to disk** (Invariant).

    Shallow wrapper around ``reconcile.generate_collision_differentiate``
    (CODING_STANDARD §2.3 — all domain logic lives in ``reconcile.py``).

    Raises:
        HTTP 400: ``slugs`` has fewer than 2 members or duplicates.
        HTTP 404: any slug does not resolve to an existing page.
        HTTP 500: any named page exists but its frontmatter is corrupt.
    """
    try:
        return reconcile_module.generate_collision_differentiate(req.slugs)
    except reconcile_module.CollisionInvalidGroup as exc:
        raise HTTPException(status_code=400, detail=f"invalid collision group: {exc}") from exc
    except reconcile_module.PageNotFound as exc:
        raise HTTPException(status_code=404, detail=f"wiki page not found: {exc}") from exc
    except reconcile_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500, detail=f"wiki page has corrupt frontmatter: {exc}"
        ) from exc


@router.post(
    "/pages/collision/differentiate/apply", response_model=CollisionDifferentiateApplyResponse
)
def collision_differentiate_apply(
    req: CollisionDifferentiateApplyRequest,
) -> CollisionDifferentiateApplyResponse:
    """Re-verify and commit the final (possibly human-edited) differentiate
    content for every group member.

    ADR-0028 phase 2 of differentiate: re-runs the grounding check on the
    EXACT submitted content for every page and refuses (409) when any page's
    content hash no longer matches the value returned by
    ``POST /pages/collision/differentiate``. No deletion happens on this
    path — every slug in the group survives, rewritten in place — so there
    is no inbound-reference guard to run. On pass, the caller triggers
    exactly one BM25 reindex.

    Shallow wrapper around ``reconcile.apply_collision_differentiate``
    (CODING_STANDARD §2.3 — all domain logic lives in ``reconcile.py``).

    Raises:
        HTTP 400: malformed group shape (``CollisionInvalidGroup``).
        HTTP 404: any slug does not resolve to an existing page.
        HTTP 409: any page's on-disk content changed since generate time.
        HTTP 422: the apply-time grounding re-check failed for any page's
            submitted content; ``detail.unsupported_claims`` lists the
            offending claims.
        HTTP 500: any named page exists but its frontmatter is corrupt.
    """
    try:
        result = reconcile_module.apply_collision_differentiate(req)
    except reconcile_module.CollisionInvalidGroup as exc:
        raise HTTPException(status_code=400, detail=f"invalid collision group: {exc}") from exc
    except reconcile_module.PageNotFound as exc:
        raise HTTPException(status_code=404, detail=f"wiki page not found: {exc}") from exc
    except reconcile_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500, detail=f"wiki page has corrupt frontmatter: {exc}"
        ) from exc
    except reconcile_module.CollisionHashMismatch as exc:
        raise HTTPException(
            status_code=409,
            detail=f"one or more pages changed since generate — differentiate refused: {exc}",
        ) from exc
    except reconcile_module.CollisionGroundingFailed as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "differentiate content failed grounding re-check",
                "reason": exc.grounding.reason,
                "unsupported_claims": exc.grounding.unsupported_claims or [],
            },
        ) from exc

    # ADR-0028: exactly one reindex after every group member is written.
    _files_indexed, sections_indexed = build_index()
    return CollisionDifferentiateApplyResponse(
        slugs=result.slugs,
        grounding=result.grounding,
        sections_indexed=sections_indexed,
    )


# ---------------------------------------------------------------------------
# /pages/{slug} — C11 Confirmed orphan-delete (tier-B S5, issue #381, ADR-0025)
# ---------------------------------------------------------------------------


@router.delete("/pages/{slug}", status_code=204)
def delete_page(slug: str) -> None:
    """Curator endpoint: delete an entities/concepts page — full orphans only.

    tier-B S5 (issue #381) / ADR-0025. Confirmed Remediation (ADR-0024): the
    human confirms the named irreversible operation; no LLM is involved; not
    a general page delete — the server recomputes the full-orphan predicate
    (``sources`` non-empty and every citation's file missing under docs/**)
    at delete time and refuses otherwise. Slug resolved server-side across
    ``entities/`` / ``concepts/`` (slugs are corpus-unique).

    Shallow wrapper around ``pages.delete_full_orphan`` (CODING_STANDARD
    §2.3 — all domain logic lives in ``pages.py``). ``build_index()`` is
    called here, once, after a successful delete — mirrors ``POST
    /qa/{slug}/promote``'s auto-reindex convention (reindex is a route-layer
    concern, not a domain-layer one).

    Exception mapping (build_index is NOT called on any exception path):

    - ``PageNotFound``      → ``404`` (slug has never been written, or was
      already deleted by a concurrent operation)
    - ``PageCorrupt``       → ``500`` (orphan-visibility — surface broken
      state rather than silently acting on it)
    - ``PageNotFullOrphan`` → ``409`` (predicate does not hold NOW — stale
      lint report, restored/re-imported Source, partial orphan, or a page
      with no sources at all; ADR-0025 Invariant)

    Returns HTTP 204 No Content on success (the resource is gone; no body).
    """
    try:
        pages_module.delete_full_orphan(slug)
    except pages_module.PageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki page '{slug}' not found under entities/ or concepts/",
        ) from exc
    except pages_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500,
            detail=f"wiki page '{slug}' has corrupt frontmatter: {exc}",
        ) from exc
    except pages_module.PageNotFullOrphan as exc:
        raise HTTPException(
            status_code=409,
            detail=(f"wiki page '{slug}' is not a full orphan (ADR-0025); delete refused: {exc}"),
        ) from exc

    # ADR-0025 Invariant: a successful delete triggers exactly one BM25 reindex.
    build_index()


# ---------------------------------------------------------------------------
# /pages/{slug}/aliases — assign-alias (issue #409, ADR-0030 decision 3)
# ---------------------------------------------------------------------------


@router.post("/pages/{slug}/aliases", status_code=204)
def assign_alias(slug: str, request: AliasAssignRequest) -> None:
    """Curator endpoint: assign an alias to an existing entities/concepts page.

    issue #409 / ADR-0030 decision 3. Direct class (no LLM, no batch — ADR-0030
    Invariant): the human authors the mapping in the very gesture. Two Console
    entry points share this one endpoint (the C2 red-link "Assign Alias"
    picker and the post-fill honest-miss offer) plus ``kb alias add`` (CLI).
    No MCP write tool exists for this operation (ADR-0030 Invariant).

    Shallow wrapper around ``pages.add_alias`` (CODING_STANDARD §2.3 — all
    domain logic lives in ``pages.py``). No reindex here — aliases never
    enter the BM25 corpus (ADR-0030 Invariant), so there is nothing to
    reindex; the Console's own re-lint (client-side, ``include_c5=false``)
    is what refreshes the C2/C12 findings.

    Exception mapping:

    - ``PageNotFound``    → ``404`` (slug has never been written, or was
      deleted by a concurrent operation)
    - ``PageCorrupt``     → ``500`` (orphan-visibility — surface broken
      state rather than silently acting on it)
    - ``InvalidAlias``    → ``422`` (blank/whitespace-only alias)
    - ``AliasCollision``  → ``409`` naming the conflicting owner (ADR-0030
      decision 3: "409 with the conflicting owner named — consistent with
      C12 semantics"). Nothing is written.

    Returns HTTP 204 No Content on success (mirrors ``DELETE /pages/{slug}``).
    """
    try:
        pages_module.add_alias(slug, request.alias)
    except pages_module.PageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki page '{slug}' not found under entities/ or concepts/",
        ) from exc
    except pages_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500,
            detail=f"wiki page '{slug}' has corrupt frontmatter: {exc}",
        ) from exc
    except pages_module.InvalidAlias as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except pages_module.AliasCollision as exc:
        raise HTTPException(
            status_code=409,
            detail=f"alias '{exc.alias}' already resolves to page '{exc.owner}'",
        ) from exc


# ---------------------------------------------------------------------------
# /pages/{slug}/aliases/{alias} — remove-alias (issue #491, ADR-0030 extension)
# ---------------------------------------------------------------------------


@router.delete("/pages/{slug}/aliases/{alias}", status_code=204)
def remove_alias(slug: str, alias: str) -> None:
    """Curator endpoint: remove an alias from an existing entities/concepts page.

    issue #491 (ADR-0030 extension) — the executable fix the C12
    alias-collision Remediation names. Direct class (no LLM, no batch,
    mirrors ``POST /pages/{slug}/aliases``'s own Invariant): one call clears
    one page's claim on one alias. The Console's C12 row calls this once per
    claimant page it wants cleared — never a server-side batch.

    Shallow wrapper around ``pages.remove_alias`` (CODING_STANDARD §2.3 — all
    domain logic lives in ``pages.py``). No reindex here — aliases never
    enter the BM25 corpus (ADR-0030 Invariant); the Console's own re-lint
    (client-side, ``include_c5=false``) is what refreshes the C12 finding.

    Exception mapping:

    - ``PageNotFound``   → ``404`` (slug has never been written, or was
      deleted by a concurrent operation)
    - ``PageCorrupt``    → ``500`` (orphan-visibility — surface broken
      state rather than silently acting on it)
    - ``AliasNotFound``  → ``404`` (alias is not currently assigned to this
      page — refused honestly rather than a fake-success no-op, e.g. a
      stale Console row after a concurrent removal)

    Returns HTTP 204 No Content on success (mirrors ``POST
    /pages/{slug}/aliases``).
    """
    try:
        pages_module.remove_alias(slug, alias)
    except pages_module.PageNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"wiki page '{slug}' not found under entities/ or concepts/",
        ) from exc
    except pages_module.PageCorrupt as exc:
        raise HTTPException(
            status_code=500,
            detail=f"wiki page '{slug}' has corrupt frontmatter: {exc}",
        ) from exc
    except pages_module.AliasNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"alias '{alias}' is not assigned to page '{slug}'",
        ) from exc


# ---------------------------------------------------------------------------
# /pages/resolution-map — linkify resolution map (issue #410, ADR-0030 decision 5)
# ---------------------------------------------------------------------------


@router.get("/pages/resolution-map", response_model=ResolutionMapResponse)
def get_resolution_map() -> ResolutionMapResponse:
    """Read-only linkify resolution map, consumed by clients.

    issue #410 / ADR-0030 decision 5. Every ``[[wikilink]]``-rendering
    surface (Console ``/read/file`` viewer, reader chat answer bodies, the
    chat-side citation viewer) consults this ONE endpoint instead of
    building its own slug set (ADR-0030 Invariant) — a wikilink resolves
    here iff C2 would NOT flag it as red for the same corpus.

    Cache-friendly: no query params, no side effects, a pure read of the
    current corpus state.

    Shallow wrapper around ``pages.get_resolution_map`` (CODING_STANDARD
    §2.3 — all domain logic lives in ``pages.py``).

    Returns:
        ``ResolutionMapResponse`` — see its docstring for field shapes.
    """
    return pages_module.get_resolution_map()
