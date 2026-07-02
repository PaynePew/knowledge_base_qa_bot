"""Shallow module per Ousterhout. Public surface: ``router``.

HTTP wiring for /health, /index, /chat, /ingest, /lint, /import,
/pages/reconcile, /pages/reconcile/apply. No domain logic."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import indexer as _indexer
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
    ChatRequest,
    ChatResponse,
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
