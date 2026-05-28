"""Shallow module per Ousterhout. Public surface: ``router``.

FastAPI routes for Stack B. Wires the deep retrieval/indexer modules to the
HTTP surface; carries no business logic (CODING_STANDARD §2.3). OpenAI
exception → HTTP status mapping lives in the deep LLM-facing modules
(``indexer.build_index`` for embeddings, ``retrieval.query`` for /chat), so
this layer only maps the outcome to the response schema.
"""

from __future__ import annotations

from fastapi import APIRouter

from .indexer import build_index
from .retrieval import query
from .schemas import (
    ChatRequest,
    ChatResponse,
    GroundingClaim,
    GroundingInfo,
    IndexResponse,
)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs() -> IndexResponse:
    files_count, chunks_count = build_index()
    return IndexResponse(files_indexed=files_count, chunks_indexed=chunks_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a query and return a fully-structured ChatResponse with grounding.

    query() returns {answer, sources, grounding_outcome}. This handler maps the
    GroundingOutcome to the API-exposed GroundingInfo (ADR-0004 Q8 selective
    expose): passes through passed/reason/claims/unsupported_claims; suppresses
    reasoning, error_type, retries_attempted (server logs only).
    """
    result = query(req.query)
    outcome = result["grounding_outcome"]

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

    unsupported_claims = None
    if outcome.reason == "claim_unsupported" and outcome.result is not None:
        unsupported_claims = outcome.result.unsupported_claims or None

    grounding = GroundingInfo(
        passed=outcome.passed,
        reason=outcome.reason,
        claims=claims,
        unsupported_claims=unsupported_claims,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        grounding=grounding,
    )
