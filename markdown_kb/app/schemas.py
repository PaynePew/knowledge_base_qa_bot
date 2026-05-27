"""Shallow module per Ousterhout. Public surface: all Pydantic request/response models (``ChatRequest``, ``ChatResponse``, ``IndexResponse``, ``IngestRequest``, ``IngestResponse``, ``WikiPageDraft``, ``WikiPageFrontmatter``, ``GroundingFailure``, ``IngestSourceResult``, ``SourceType``, ``GroundingClaim``, ``GroundingInfo``, ``LintResponse``, ``LintSummary``, ``LintFindings``, ``OrphanPageFinding``, ``FailedGroundingFinding``, ``SlugCollisionFinding``, ``StalePageFinding``, ``RedLinkFinding``).

Pydantic request/response models for the FastAPI routes. No domain logic."""

from __future__ import annotations

import datetime
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

    ``heading`` is the human-readable H1 to render at the top of the page.
    It is captured by the templates layer (where the original Section
    heading or Source-stem form is known) instead of being reconstructed
    from the slug in wiki_writer — the previous ``slug.replace("-", " ").title()``
    approach lost acronyms ("HTTP" -> "Http") and non-ASCII characters.
    """

    frontmatter: WikiPageFrontmatter
    body: str  # LLM-generated prose, may contain [[wikilinks]]
    citation_line: str  # e.g. "[Source: refund_policy.md#cancellation-window]"
    slug: str  # e.g. "cancellation-window" — used as the output filename stem
    heading: str  # e.g. "Cancellation Window" — used as the page's H1


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


# ---------------------------------------------------------------------------
# /lint schemas (Phase 5 Slice 5-1)
# ---------------------------------------------------------------------------


class OrphanPageFinding(BaseModel):
    """C11 orphan page finding: wiki page whose sources no longer exist under docs/.

    A page is flagged as orphan when at least one entry in ``frontmatter.sources``
    references a file (the portion before ``#``) that does not exist under ``docs/``
    (including nested subdirectories).

    ``suggested_action`` mentions both rename and deletion paths because C11 cannot
    distinguish between a deleted Source and a renamed one — curator must judge.
    """

    page_slug: str
    missing_sources: list[str]
    suggested_action: str


class FailedGroundingFinding(BaseModel):
    """C3 failed-grounding finding: wiki page written with status=failed_grounding.

    Phase 3 fail-soft ingest writes these pages when the grounding verifier fails.
    Phase 4 W1 silently filters them from /chat retrieval; C3 surfaces them for
    curator action.

    ``source`` is ``frontmatter.sources[0]`` — the primary source that was being
    synthesised when grounding failed.
    ``reason`` mirrors ``GroundingFailure.reason``: either ``claim_unsupported``
    (verifier ran and rejected one or more claims) or ``verifier_unavailable``
    (LLM call itself failed).
    ``unsupported_claims`` is populated only for ``claim_unsupported``; empty for
    ``verifier_unavailable`` (no claims were extracted when the verifier failed).
    ``suggested_action`` suggests Source review and re-ingest OR page deletion.
    """

    page_slug: str
    source: str
    reason: Literal["claim_unsupported", "verifier_unavailable"]
    unsupported_claims: list[str] = []
    suggested_action: str


class SlugCollisionFinding(BaseModel):
    """C4-a slug-collision finding: multiple wiki pages with the same base slug.

    Phase 3 ingest appends ``-2``, ``-3``, ... suffixes when a slug already exists.
    These collisions indicate that two pages cover the same concept and should be
    merged, or that their headings should be renamed to be more specific.

    ``base_slug`` is the common root (e.g. ``"pricing"`` for ``pricing``,
    ``pricing-2``, ``pricing-3``).
    ``pages_in_group`` lists all slugs in the collision group, including the
    unsuffixed original and all suffixed variants (when present on disk).
    ``suggested_action`` suggests review and merge or heading rename.
    """

    base_slug: str
    pages_in_group: list[str]
    suggested_action: str


# ---------------------------------------------------------------------------
# /lint schemas (Phase 5 Slice 5-3 — C6 + C2)
# ---------------------------------------------------------------------------


class StalePageFinding(BaseModel):
    """C6 stale page finding: wiki page whose Source file has been modified after
    the page's ``updated`` frontmatter timestamp.

    ``source`` is the bare Source filename (e.g. ``refund_policy.md``).
    ``source_mtime`` is the filesystem mtime of the Source file.
    ``page_updated`` is the ``updated`` timestamp parsed from the page's frontmatter.
    ``drift_days`` is ``(source_mtime - page_updated).total_seconds() / 86400`` —
    positive means the Source is newer (stale page).
    ``suggested_action`` recommends re-ingesting the page from the Source.

    C6 only checks pages whose Source file exists; pages with missing sources are
    handled exclusively by C11 (orphan check) to avoid double-reporting.
    """

    page_slug: str
    source: str
    source_mtime: datetime.datetime
    page_updated: datetime.datetime
    drift_days: float
    suggested_action: str


class RedLinkFinding(BaseModel):
    """C2 red link finding: a ``[[wikilink]]`` target slug that does not resolve
    to any existing wiki page.

    ``slug`` is the unresolved target slug (anchor stripped; e.g. ``[[foo#bar]]``
    contributes slug ``foo``).
    ``mention_count`` is the total number of occurrences across ALL scanned wiki pages
    (multiple occurrences in the same page each count).
    ``referenced_by`` is the sorted list of page slugs that contain at least one mention.
    ``sample_context`` is up to 50 characters of surrounding text from the first occurrence,
    or ``None`` if context extraction failed.

    C2 scans only ``wiki/entities/`` and ``wiki/concepts/`` (matching ADR-0006 SOURCE_DIRS).
    The following files are explicitly excluded to prevent feedback loops / noise:
    ``wiki/index.md``, ``wiki/log.md``, ``wiki/hot.md``, ``wiki/lint-report.md``,
    ``wiki/README.md``, ``wiki/.archive/*``.
    """

    slug: str
    mention_count: int
    referenced_by: list[str]
    sample_context: str | None = None


class LintFindings(BaseModel):
    """Container for all check findings.

    Slice 5-1 populates only ``orphans`` (C11).
    Slice 5-2 adds ``failed_grounding`` (C3) and ``slug_collisions`` (C4-a).
    Slice 5-3 adds ``stale_pages`` (C6) and ``red_links`` (C2).
    Later slices add the remaining check fields without changing existing field names or types.
    """

    orphans: list[OrphanPageFinding] = []
    failed_grounding: list[FailedGroundingFinding] = []
    slug_collisions: list[SlugCollisionFinding] = []
    stale_pages: list[StalePageFinding] = []
    red_links: list[RedLinkFinding] = []


class LintSummary(BaseModel):
    """Aggregate metrics for a lint run.

    ``findings_by_check`` maps check identifier (``"c11"``, ``"c3"``, …) to
    finding count.  Slice 5-1 only includes ``"c11"``; later slices extend.
    ``llm_calls`` and ``cost_usd`` are 0 in Slice 5-1 (no LLM used by C11).
    """

    total_findings: int
    findings_by_check: dict[str, int]
    llm_calls: int = 0
    cost_usd: float = 0.0
    generated_at: str  # ISO-8601 UTC string


class LintResponse(BaseModel):
    """Response body for POST /lint.

    ``report_path`` is the relative path (from repo root) of the written
    markdown report.  ``check_errors`` maps check id to error string when a
    check raised under continue-on-error semantics.
    """

    report_path: str
    findings: LintFindings
    summary: LintSummary
    check_errors: dict[str, str] = {}
