"""Shallow module per Ousterhout. Public surface: the Pydantic request/response models.

Pydantic boundary schemas for Stack B's FastAPI surface. Primitives only — no
LangChain types cross this boundary (CODING_STANDARD §2.4). ``GroundingInfo``
mirrors markdown_kb's API-exposed grounding subset (ADR-0004 Q8) so the two
apps present a symmetric /chat contract (issue #103).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class IndexResponse(BaseModel):
    files_indexed: int
    chunks_indexed: int


class ChatRequest(BaseModel):
    query: str


class SourceInfo(BaseModel):
    source: str
    heading: str
    content: str


class GroundingClaim(BaseModel):
    """A single atomic claim extracted from the draft answer.

    Carries only the client-facing data from ``grounding.GroundingClaim`` so no
    LangChain type leaks past the LLM-facing module (CODING_STANDARD §2.4).
    """

    text: str
    supported: bool
    citing_section_ids: list[str]


class GroundingInfo(BaseModel):
    """API-exposed subset of ``grounding.GroundingOutcome`` (ADR-0004 Q8).

    Always populated on ChatResponse — never Optional. Covers both pre-LLM gate
    outcomes (retrieval_empty, index_missing) and post-LLM verifier outcomes
    (claim_supported, claim_unsupported, verifier_unavailable). Internal fields
    (reasoning, error_type, retries_attempted) stay in server logs only.
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
