"""Grounding Check module — schemas, CitableContent Protocol, and verify() stub.

The public interface is a single function:

    verify(draft: str, sections: list[CitableContent]) -> GroundingOutcome

All complexity (verifier prompt, with_structured_output binding, retry policy,
error classification) will be encapsulated here in Slice #3. This module
currently provides the schema layer only; verify() raises NotImplementedError
until Slice #3 fills in the implementation.

See ADR-0004 for the full design rationale.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# CitableContent Protocol (Q9 — retrieval-agnostic input contract)
# ---------------------------------------------------------------------------


@runtime_checkable
class CitableContent(Protocol):
    """Structural type satisfied by any retrieval unit that has the three
    required fields.  markdown_kb's Section satisfies this Protocol at
    runtime; vector_rag's chunk type will do the same without any changes to
    this module.
    """

    id: str
    heading_path: list[str]
    content: str


# ---------------------------------------------------------------------------
# Verifier structured output schemas (Q3 / Q7)
# ---------------------------------------------------------------------------


class GroundingClaim(BaseModel):
    """A single atomic claim extracted from the draft answer."""

    text: str
    supported: bool
    citing_section_ids: list[str]  # empty when supported=False


class GroundingResult(BaseModel):
    """Structured output returned by the verifier LLM call.

    reasoning is the first field so with_structured_output fills it first,
    acting as a chain-of-thought scratchpad before the model commits to
    structured judgments (ADR-0004 Q7 / CoT scratchpad pattern).
    """

    reasoning: str  # CoT scratchpad — kept internal, not exposed in ChatResponse
    claims: list[GroundingClaim]
    unsupported_claims: list[str]
    passed: bool


# ---------------------------------------------------------------------------
# Error classification enum (Q5)
# ---------------------------------------------------------------------------


class VerifierErrorType(StrEnum):
    """Granular error types for failed verifier calls.

    Logged server-side only; not exposed in ChatResponse (ADR-0004 Q8).
    """

    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    MALFORMED_JSON = "malformed_json"
    REFUSAL = "refusal"
    AUTH = "auth"


# ---------------------------------------------------------------------------
# Caller-facing outcome (Q8)
# ---------------------------------------------------------------------------


class GroundingOutcome(BaseModel):
    """Unified caller-facing outcome of the grounding check.

    Covers both post-LLM verifier outcomes and pre-LLM gate outcomes so the
    /chat route has a single type for all grounding results (ADR-0004 Q8).
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
    result: GroundingResult | None = None
    error_type: VerifierErrorType | None = None
    retries_attempted: int = 0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def verify(draft: str, sections: list[CitableContent]) -> GroundingOutcome:
    """Verify that every claim in *draft* is supported by *sections*.

    Returns a GroundingOutcome describing the overall result and per-claim
    evidence.  Slice #3 replaces this stub with the full implementation using
    ChatOpenAI.with_structured_output(GroundingResult).
    """
    raise NotImplementedError("Slice #3")
