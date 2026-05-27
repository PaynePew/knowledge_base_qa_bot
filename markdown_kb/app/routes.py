"""Shallow module per Ousterhout. Public surface: ``router``.

HTTP wiring for /health, /index, /chat, /ingest, /lint. No domain logic."""

from __future__ import annotations

from fastapi import APIRouter

import app.indexer as _indexer

from . import qa as qa_module
from .indexer import build_index
from .ingest import ingest_sources
from .lint import run_lint
from .retrieval import query
from .schemas import (
    ChatRequest,
    ChatResponse,
    GroundingClaim,
    GroundingInfo,
    IndexResponse,
    IngestRequest,
    IngestResponse,
    LintResponse,
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
    result = query(req.query)
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

    # Phase 6 Slice 6-2: gated side-effect — file Grounded Answers only.
    # Cannot Confirm paths (passed=False) do NOT file: skip junk into the wiki.
    # Adapter SectionRef satisfies the CitableContent Protocol with just the
    # ``id`` field (qa.maybe_file_answer reads ``.id`` only); avoids leaking
    # the indexer's full Section dataclass into route territory.
    filed = None
    if outcome.passed:
        cited_refs = [_SectionRef(id=s["source"]) for s in result["sources"]]
        filed = qa_module.maybe_file_answer(req.query, result["answer"], cited_refs)

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        grounding=grounding,
        filed=filed,
    )


class _SectionRef:
    """Minimal CitableContent adapter for ``qa.maybe_file_answer``.

    The qa module's public ``maybe_file_answer`` accepts any
    ``CitableContent`` (Protocol — requires ``id``, ``heading_path``,
    ``content``). The handler holds only ``result["sources"]`` (list of dicts
    with ``source`` = the bare section id); reconstructing a full
    ``indexer.Section`` here would force the route to import indexer's
    dataclass for one field. This shim is the smallest viable adapter — only
    ``id`` is set because the filing path reads no other field.
    """

    __slots__ = ("content", "heading_path", "id")

    def __init__(self, id: str) -> None:
        self.id = id
        self.heading_path = [id]
        self.content = ""


@router.post("/lint", response_model=LintResponse)
def lint() -> LintResponse:
    """Run the wiki lint pass and return a structured findings report.

    Executes all lint checks (Slice 5-1: C11 orphan detection only) and writes
    ``wiki/lint-report.md``.  Returns 200 with a ``LintResponse`` in all cases,
    including when one or more checks raised (errors recorded in
    ``LintResponse.check_errors``).

    Shallow wrapper around ``lint.run_lint()``.  All domain logic lives in
    ``lint.py`` (CODING_STANDARD §2.3).
    """
    return run_lint()


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest | None = None) -> IngestResponse:
    """Ingest one or all Sources and write wiki synthesis pages.

    - No body (or body with ``source=null``): batch mode — processes all
      Sources discovered under docs/ via ``glob("**/*.md")``.
    - Body with ``source="<filename>"``: single-source mode.

    Shallow wrapper around `ingest_sources(...)`.  All domain logic
    (parse → classify → synthesise → write) lives in `ingest.py`
    (CODING_STANDARD §2.3).

    Returns 200 with the IngestResponse in all cases, including when a Source
    is not found (reflected in ``failed_sources``).
    """
    if req is None or req.source is None:
        # Batch mode: ingest all docs/
        batch = ingest_sources(None)
    else:
        batch = ingest_sources([req.source])
    return IngestResponse(
        results=batch.results,
        failed_sources=batch.failed_sources,
        pages_with_failed_grounding=batch.pages_with_failed_grounding,
    )
