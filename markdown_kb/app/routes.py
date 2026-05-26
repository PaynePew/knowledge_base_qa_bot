"""HTTP wiring for /health, /index, /chat, /ingest. No domain logic."""

from __future__ import annotations

from fastapi import APIRouter

import app.indexer as _indexer

from .indexer import build_index
from .ingest import ingest_sources
from .retrieval import query
from .schemas import (
    ChatRequest,
    ChatResponse,
    GroundingClaim,
    GroundingInfo,
    IndexResponse,
    IngestRequest,
    IngestResponse,
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

    sources = [
        {
            "source": s["source"],
            "heading": s["heading"],
            "score": s["score"],
            "content": s["content"],
        }
        for s in result["sources"]
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        grounding=grounding,
    )


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
    )
