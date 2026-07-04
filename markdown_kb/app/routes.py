"""Shallow module per Ousterhout. Public surface: ``router``.

HTTP wiring for /health, /index, /chat, /qa/{slug}/promote,
/qa/promote-batch, /qa/{slug} (DELETE, PUT), /qa/{slug}/refile, /ingest,
/lint, /import, /pages/reconcile, /pages/reconcile/apply,
/pages/collision/merge, /pages/collision/merge/apply,
/pages/collision/differentiate, /pages/collision/differentiate/apply,
/pages/{slug} (DELETE), /pages/{slug}/aliases (POST). No domain logic."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
)

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
    fresh answer BEFORE any write, and only on pass overwrite the same slug
    in place as ``status: draft``. A failed re-ground writes nothing — the
    old live page keeps serving and the C9 finding stays (Invariant).

    Shallow wrapper around ``qa.refile`` (CODING_STANDARD §2.3 — all domain
    logic lives in ``qa.py``). ``build_index()`` is called here, once, after
    a successful refile — mirrors ``POST /qa/{slug}/promote``'s auto-reindex
    convention (reindex is a route-layer concern, not a domain-layer one);
    this is what actually removes the stale answer from the BM25 corpus.

    Exception mapping (build_index is NOT called on any exception path):

    - ``QaPageNotFound``     → ``404`` (slug has never been filed, or was
      deleted by a concurrent operation during re-synthesis)
    - ``QaPageCorrupt``      → ``500`` (orphan-visibility — surface broken
      state rather than silently rewriting it)
    - ``QaRefileRejected``   → ``422`` (the fresh re-synthesis failed the
      Grounding Check; ``detail.reason`` / ``detail.unsupported_claims``
      report why; nothing was written)
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

    # ADR-0026: auto-reindex so the stale live answer leaves the corpus immediately.
    _files_indexed, sections_indexed = build_index()
    return QaRefileResponse(
        filed=result.filed,
        grounding=result.grounding,
        sections_indexed=sections_indexed,
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
