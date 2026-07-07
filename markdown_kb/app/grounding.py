"""Deep module per Ousterhout. Public surface: ``verify``, ``GroundingOutcome``, ``CitableContent`` (Protocol).

Grounding Check module — schemas, CitableContent Protocol, and verify().

The public interface is a single function:

    verify(draft: str, sections: list[CitableContent]) -> GroundingOutcome

All complexity (verifier prompt template, with_structured_output binding,
retry policy, error classification, fail-mode mapping) lives here.

See ADR-0004 for the full design rationale.
"""

from __future__ import annotations

import os
import time
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import openai
import pydantic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .logger import log_event

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


# Reasons where the KB itself could not ground an answer (a *content* failure),
# as opposed to a transient/operational failure (``verifier_unavailable`` /
# ``index_missing``). Single source of truth for the content-vs-transient split
# of ``GroundingOutcome.reason``, co-located with the reason enum so the two
# can never drift. Consumed by C1 coverage-gap aggregation (``lint``) and the
# C9 re-file retire gate (``qa``, ADR-0035) — both act only on content failures.
# Adding a new content-failure reason to the enum above must add it here too.
CONTENT_FAILURE_REASONS = frozenset(
    {"retrieval_empty", "below_threshold", "claim_unsupported"}
)


# ---------------------------------------------------------------------------
# Verifier system prompt (AC #1)
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = """\
You are a factual grounding verifier. Your task is to determine whether \
every atomic claim in DRAFT_ANSWER is explicitly supported by CITED_SECTIONS.

Rules:
- Identify ALL atomic factual claims in DRAFT_ANSWER.
- For each claim, judge whether it is **explicitly** supported by the \
content of CITED_SECTIONS. Paraphrasing is acceptable; logical inference \
beyond what is written is NOT acceptable; world knowledge is NOT acceptable.
- First fill the `reasoning` field as a chain-of-thought scratchpad: \
walk through each claim, identify which section (if any) supports it, and \
explain your judgment before committing to structured fields.
- Set `passed` to false if ANY single claim is unsupported. \
A `passed=false` result is a successful, expected outcome — it means \
the draft goes beyond the cited sources, not that the verification failed.
- `unsupported_claims` should list the text of every claim you judged \
as NOT supported.
- `citing_section_ids` for a supported claim should list the section IDs \
that support it; leave empty for unsupported claims.

The user message will provide CITED_SECTIONS formatted as:

[Source: <id>]
Heading: <heading path joined with " > ">
<content>

followed by:

DRAFT_ANSWER:
<draft text>
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_message(draft: str, sections: list[CitableContent]) -> str:
    """Format sections and draft into the verifier user message."""
    parts: list[str] = []
    for sec in sections:
        heading = " > ".join(sec.heading_path)
        parts.append(f"[Source: {sec.id}]\nHeading: {heading}\n{sec.content}")

    sections_text = "\n\n".join(parts) if parts else "(no sections provided)"
    return f"{sections_text}\n\nDRAFT_ANSWER:\n{draft}"


def _classify_error(exc: Exception) -> VerifierErrorType:
    """Map an exception to a VerifierErrorType for logging and policy selection.

    Note: AuthenticationError is a subclass of APIStatusError, so it must be
    checked first to prevent the APIStatusError branch from matching it.
    """
    if isinstance(exc, openai.APITimeoutError):
        return VerifierErrorType.TIMEOUT
    if isinstance(exc, openai.RateLimitError):
        return VerifierErrorType.TIMEOUT  # 429 is treated as transient
    if isinstance(exc, openai.AuthenticationError):
        # Must come before APIStatusError (auth is a subclass of it)
        return VerifierErrorType.AUTH
    if isinstance(exc, openai.APIStatusError):
        # Any remaining 4xx or 5xx
        return VerifierErrorType.SERVER_ERROR
    if isinstance(exc, (pydantic.ValidationError, ValueError)):
        return VerifierErrorType.MALFORMED_JSON
    return VerifierErrorType.SERVER_ERROR


_TRANSIENT_TYPES = {VerifierErrorType.TIMEOUT, VerifierErrorType.SERVER_ERROR}
_MALFORMED_TYPES = {VerifierErrorType.MALFORMED_JSON}
_NO_RETRY_TYPES = {VerifierErrorType.REFUSAL, VerifierErrorType.AUTH}

# Backoff delays (seconds) between attempts: attempt 1 uses [0], attempt 2 uses [1]
_BACKOFF_DELAYS = [0.2, 0.8]

# Total verifier-side latency budget (seconds)
_LATENCY_BUDGET = 5.0


def _call_verifier_once(chain: object, user_message: str) -> GroundingResult:
    """Invoke the structured-output chain once and return a GroundingResult.

    Raises whatever exception the underlying LLM call raises — the caller is
    responsible for classification and retry decisions.
    """
    result = chain.invoke(user_message)  # type: ignore[union-attr]

    # Detect refusal: empty claims list with no reasoning is treated as refusal
    if not isinstance(result, GroundingResult):
        raise ValueError(f"Unexpected result type: {type(result)}")

    return result


def _is_refusal(result: GroundingResult) -> bool:
    """Return True if the result looks like a model refusal.

    A refusal is indicated by: empty claims list AND empty reasoning — the
    model produced a technically valid schema but with no content at all.
    """
    return not result.claims and not result.reasoning.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def verify(draft: str, sections: list[CitableContent]) -> GroundingOutcome:
    """Verify that every claim in *draft* is supported by *sections*.

    Uses ChatOpenAI.with_structured_output(GroundingResult) per ADR-0005's
    pre-blessed trigger for this verifier call.

    Returns a GroundingOutcome describing the overall result and per-claim
    evidence.  Never raises on verifier-side failure (only on programmer
    error such as wrong input types).
    """
    if not isinstance(draft, str):
        raise TypeError(f"draft must be str, got {type(draft)}")
    if not isinstance(sections, list):
        raise TypeError(f"sections must be list, got {type(sections)}")

    # Empty-sections short-circuit: zero cited sections cannot support any
    # claim (ADR-0001 strict grounding; ADR-0004 verifier must fail-closed).
    # Return deterministic passed=False without constructing the LLM client.
    # reason="claim_unsupported" — the claim is ungrounded by the cited set
    # (consistent with the non-empty unsupported path; "retrieval_empty" is
    # reserved for the pre-LLM gate in retrieval.query, not verify()).
    if not sections:
        log_event(
            "grounding_verify", "reason=claim_unsupported retries=0 latency=0s empty_sections=True"
        )
        return GroundingOutcome(
            passed=False,
            reason="claim_unsupported",
            result=GroundingResult(
                reasoning="No sections were cited; any claim is ungrounded by definition.",
                claims=[
                    GroundingClaim(
                        text=draft,
                        supported=False,
                        citing_section_ids=[],
                    )
                ],
                unsupported_claims=[draft],
                passed=False,
            ),
            error_type=None,
            retries_attempted=0,
        )

    model_name = os.getenv("OPENAI_VERIFIER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    # temperature=0: a non-deterministic verifier intermittently marks a
    # supported claim as unsupported, flipping a correct grounded answer to
    # Cannot Confirm when the same question is re-asked.
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = llm.with_structured_output(GroundingResult)

    user_message = _build_user_message(draft, sections)

    start_time = time.monotonic()
    retries_attempted = 0
    last_error_type: VerifierErrorType | None = None
    last_exc: Exception | None = None

    # Retry loop: up to 3 total attempts (initial + 2 retries for transient;
    # initial + 1 retry for malformed; just initial for refusal/auth).
    for _attempt in range(3):
        elapsed = time.monotonic() - start_time
        if elapsed >= _LATENCY_BUDGET:
            break

        try:
            result = _call_verifier_once(chain, user_message)

            if _is_refusal(result):
                # Refusal — no retry
                log_event(
                    "grounding_verifier_error",
                    f"error_type={VerifierErrorType.REFUSAL} retries={retries_attempted}",
                )
                return GroundingOutcome(
                    passed=False,
                    reason="verifier_unavailable",
                    result=None,
                    error_type=VerifierErrorType.REFUSAL,
                    retries_attempted=retries_attempted,
                )

            # Success — map result to outcome
            latency = round(time.monotonic() - start_time, 3)
            if result.passed:
                reason: Literal[
                    "claim_supported",
                    "claim_unsupported",
                    "verifier_unavailable",
                    "below_threshold",
                    "retrieval_empty",
                    "index_missing",
                ] = "claim_supported"
            else:
                reason = "claim_unsupported"

            log_event(
                "grounding_verify",
                f"reason={reason} retries={retries_attempted} latency={latency}s",
            )
            return GroundingOutcome(
                passed=result.passed,
                reason=reason,
                result=result,
                error_type=None,
                retries_attempted=retries_attempted,
            )

        except Exception as exc:  # noqa: BLE001
            error_type = _classify_error(exc)
            last_error_type = error_type
            last_exc = exc

            if error_type == VerifierErrorType.AUTH:
                # Hard error — no retry, emit prominent server log
                log_event(
                    "grounding_verifier_error",
                    f"error_type={error_type} retries={retries_attempted} "
                    f"exc={type(exc).__name__} HARD_ERROR",
                )
                return GroundingOutcome(
                    passed=False,
                    reason="verifier_unavailable",
                    result=None,
                    error_type=error_type,
                    retries_attempted=retries_attempted,
                )

            if error_type in _NO_RETRY_TYPES:
                # Refusal / other no-retry
                log_event(
                    "grounding_verifier_error",
                    f"error_type={error_type} retries={retries_attempted} exc={type(exc).__name__}",
                )
                return GroundingOutcome(
                    passed=False,
                    reason="verifier_unavailable",
                    result=None,
                    error_type=error_type,
                    retries_attempted=retries_attempted,
                )

            # Decide whether to retry
            max_retries = 2 if error_type in _TRANSIENT_TYPES else 1
            if retries_attempted >= max_retries:
                # Exhausted retries for this error type
                break

            # Apply backoff if budget allows
            backoff = (
                _BACKOFF_DELAYS[retries_attempted]
                if retries_attempted < len(_BACKOFF_DELAYS)
                else 0.8
            )
            remaining = _LATENCY_BUDGET - (time.monotonic() - start_time)
            if remaining <= backoff:
                break

            time.sleep(backoff)
            retries_attempted += 1

    # All retries exhausted or budget exceeded
    latency = round(time.monotonic() - start_time, 3)
    log_event(
        "grounding_verifier_error",
        f"reason=verifier_unavailable error_type={last_error_type} "
        f"retries={retries_attempted} latency={latency}s "
        f"exc={type(last_exc).__name__ if last_exc else 'budget_exceeded'}",
    )
    return GroundingOutcome(
        passed=False,
        reason="verifier_unavailable",
        result=None,
        error_type=last_error_type,
        retries_attempted=retries_attempted,
    )
