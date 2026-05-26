"""Pydantic request/response models for the FastAPI routes. No domain logic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Grounding failure block (Slice #4 — fail-soft grounding check on ingest)
# ---------------------------------------------------------------------------


class GroundingFailure(BaseModel):
    """Frontmatter sub-block written when a wiki page fails grounding check.

    Written as a nested YAML block under `grounding_failure:` in the page
    frontmatter.  Only present when `status == "failed_grounding"`.

    `reason` mirrors GroundingOutcome.reason from grounding.py.
    `unsupported_claims` is empty for `verifier_unavailable` (no claims were
    extracted when the verifier itself failed).
    """

    reason: Literal["claim_unsupported", "verifier_unavailable"]
    unsupported_claims: list[str] = []


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


# ---------------------------------------------------------------------------
# /ingest schemas (Phase 3 Slice #1)
# ---------------------------------------------------------------------------


SourceType = Literal["entity", "concept"]


class WikiPageFrontmatter(BaseModel):
    """Frontmatter schema for a wiki synthesis page.

    All base fields are required. `grounding_failure` is only populated when
    `status == "failed_grounding"` (Slice #4 fail-soft grounding check).
    `confidence` is intentionally deferred to Phase 5 /lint (no defensible
    algorithm at ingest time per PRD #28 Q10).
    """

    id: str
    type: SourceType
    created: str  # ISO-8601 UTC string, e.g. "2026-05-26T14:30:00Z"
    updated: str  # ISO-8601 UTC string, matches created on first write
    sources: list[str]  # list of "filename#slug" citation strings
    status: Literal["live", "failed_grounding"]
    open_questions: list[str]
    grounding_failure: GroundingFailure | None = None


class WikiPageDraft(BaseModel):
    """A single wiki page produced by the LLM synthesis step.

    Carries the full rendered body (LLM content) and the structured
    frontmatter separately so wiki_writer.py can serialize them without
    re-parsing the body text.
    """

    frontmatter: WikiPageFrontmatter
    body: str  # LLM-generated prose, may contain [[wikilinks]]
    citation_line: str  # e.g. "[Source: refund_policy.md#cancellation-window]"
    slug: str  # e.g. "cancellation-window" — used as the output filename stem


class IngestSourceResult(BaseModel):
    """Per-Source outcome within an IngestResponse.

    `pages_written` lists relative paths under wiki/ for ALL pages that exist
    after the ingest (created + updated). Empty on failure.
    `pages_created` lists paths for pages written for the first time.
    `pages_updated` lists paths for pages that already existed and were overwritten.
    `pages_deleted` lists paths of orphan pages removed during re-ingest.
    `error` is set when the source could not be processed.

    Meaningful population added in Slice #3 (orphan handling + created preservation).
    """

    source: str  # bare filename, e.g. "refund_policy.md"
    pages_written: list[str]
    pages_created: list[str] = []
    pages_updated: list[str] = []
    pages_deleted: list[str] = []
    error: str | None = None


class IngestRequest(BaseModel):
    """Request body for POST /ingest.

    `source` is the bare filename of a single Source to ingest
    (e.g. "refund_policy.md"). When omitted (or body omitted entirely),
    batch mode ingests all Sources under docs/ (Slice #2).
    """

    source: str | None = None


class IngestResponse(BaseModel):
    """Response body for POST /ingest.

    `results` lists one IngestSourceResult per successfully processed Source.
    `failed_sources` lists bare filenames that could not be processed (Source
    not found, parse error, etc.).
    `pages_with_failed_grounding` lists page ids (slugs) of pages that were
    written but failed the grounding check (status=failed_grounding).  Added
    in Slice #4 — empty on all prior slices.
    """

    results: list[IngestSourceResult]
    failed_sources: list[str]
    pages_with_failed_grounding: list[str] = []
