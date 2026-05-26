"""Pydantic request/response models for the FastAPI routes. No domain logic."""

from typing import Literal

from pydantic import BaseModel


class IndexResponse(BaseModel):
    files_indexed: int
    sections_indexed: int
    wiki_index_written: bool = False
    wiki_index_path: str | None = None  # absolute path string when written
    wiki_index_error: str | None = None  # "<ErrorClass>: <message>" when not written


class ChatRequest(BaseModel):
    query: str


class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str


class GroundingClaim(BaseModel):
    """A single atomic claim extracted from the draft answer.

    Re-exported from grounding.py so routes and schemas share the same type
    without letting LangChain types leak past retrieval.py (CODING_STANDARD §2.4).
    Only the claim-level data the client needs is included; citing_section_ids
    is included so clients can render per-claim provenance (PRD User Story 2).
    Internal fields (reasoning, error_type, retries_attempted) are kept server-side.
    """

    text: str
    supported: bool
    citing_section_ids: list[str]


class GroundingInfo(BaseModel):
    """API-exposed subset of GroundingOutcome (ADR-0004 Q8 selective expose).

    Always populated on ChatResponse — never Optional. Covers both pre-LLM
    gate outcomes (retrieval_empty, below_threshold, index_missing) and
    post-LLM verifier outcomes (claim_supported, claim_unsupported,
    verifier_unavailable).

    NOT exposed: reasoning (CoT scratchpad), error_type, retries_attempted.
    Those stay in server logs only (log_event).
    """

    passed: bool
    reason: Literal[
        # post-LLM (verifier ran)
        "claim_supported",
        "claim_unsupported",
        "verifier_unavailable",
        # pre-LLM (verifier did not run)
        "below_threshold",
        "retrieval_empty",
        "index_missing",
    ]
    claims: list[GroundingClaim] | None = None
    unsupported_claims: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    grounding: GroundingInfo
