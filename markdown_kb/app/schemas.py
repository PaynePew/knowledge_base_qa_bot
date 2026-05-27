"""Shallow module per Ousterhout. Public surface: all Pydantic request/response models (``ChatRequest``, ``ChatResponse``, ``IndexResponse``, ``IngestRequest``, ``IngestResponse``, ``WikiPageDraft``, ``WikiPageFrontmatter``, ``GroundingFailure``, ``IngestSourceResult``, ``SourceType``, ``GroundingClaim``, ``GroundingInfo``, ``CitationRef``, ``FiledStatus``, ``LintResponse``, ``LintSummary``, ``LintFindings``, ``OrphanPageFinding``, ``FailedGroundingFinding``, ``SlugCollisionFinding``, ``StalePageFinding``, ``RedLinkFinding``, ``CoverageGapFinding``, ``PagePairFinding``, ``PromotionCandidateFinding``, ``QaStalenessFinding``, ``InvalidQaSchemaFinding``, ``ImportRequest``, ``ImportSourceResultSchema``, ``ImportFailureSchema``, ``ImportResponse``).

Pydantic request/response models for the FastAPI routes. No domain logic."""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, Field

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


class CitationRef(BaseModel):
    """One wiki -> docs citation pointer.

    Mirrors the shape of a ``SourceInfo`` citation (``source``, ``heading``)
    but omits ``score`` and ``content`` because the docs/ chain is audit-only
    (not retrieved, not scored). Populated by Phase 6 Slice 6-3 from each
    retrieved wiki page's ``frontmatter.sources``. Defined here in Slice 6-1
    so the contract is locked before downstream slices land.
    """

    source: str
    heading: str


class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str
    # Phase 6: one-layer wiki -> docs citation chain (closes ADR-0006 deferred
    # item "PROMPT.md citation contract evolution"). Populated by Slice 6-3 in
    # ``retrieval.query``; defaults to ``None`` here so existing Slice 6-1
    # callers (no retrieval changes yet) keep working unchanged.
    # ``None`` is semantically distinct from ``[]`` — None means the wiki page
    # had no ``frontmatter.sources``, [] would mean an empty-but-present chain.
    derived_from: list[CitationRef] | None = None


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


class FiledStatus(BaseModel):
    """Outcome of a Phase 6 Answer Filing side-effect on ``POST /chat``.

    Surfaced on ``ChatResponse.filed`` so the caller has an audit trail back
    to the ``wiki/qa/<slug>.md`` page. ``None`` on Cannot-Confirm, on filing
    IOError (F3 fail-soft), and on touch attempts against an invalid-status
    orphan page (three-layer defence per PRD #78 Q8d).

    Populated by Slice 6-2 ``qa.maybe_file_answer`` (``op="created"`` or
    ``op="touched"``) and Slice 6-4 ``POST /qa/{slug}/promote``
    (``op="promoted"`` is *not* surfaced here — promote returns its own
    FiledStatus; this field is /chat-side only). Defined in Slice 6-1 so the
    type contract is locked early.
    """

    slug: str
    status: Literal["draft", "live"]
    op: Literal["created", "touched"]
    count: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    grounding: GroundingInfo
    # Phase 6 Answer Filing: populated by Slice 6-2's ``qa.maybe_file_answer``
    # side-effect inside the /chat handler whenever the Grounded Answer passed.
    # Defaults to ``None`` so existing tests and the Slice 6-1 codepath (no
    # filing yet) keep working unchanged. None encodes Cannot-Confirm, filing
    # IOError, and orphan-touch refusal.
    filed: FiledStatus | None = None


# ---------------------------------------------------------------------------
# /ingest schemas (Phase 3 Slice #1)
# ---------------------------------------------------------------------------


SourceType = Literal["entity", "concept", "qa"]


class WikiPageFrontmatter(BaseModel):
    """Frontmatter schema for a wiki synthesis page.

    All base fields are required. `grounding_failure` is only populated when
    `status == "failed_grounding"` (Slice #4 fail-soft grounding check).
    `confidence` is intentionally deferred to Phase 5 /lint (no defensible
    algorithm at ingest time per PRD #28 Q10).

    Phase 6 extensions (PRD #78 Q2):
    - ``type`` literal extends to include ``"qa"`` for Filed Answers.
    - ``status`` literal extends to forward-compat values: ``"draft"`` (filed
      Q&A awaiting curator promotion), ``"stale"`` and ``"superseded"``
      (reserved; recognised so the C10 lint check has names to validate
      against). Slice 6-1 only acts on the ``live``/``draft`` pair in the
      indexer filter; other values are still legal in the model so the
      filing-layer and lint-layer defences (PRD #78 Q8d) can introspect them.
    - ``question`` carries the verbatim user query for ``type == "qa"`` pages.
      Optional / None on entity and concept pages.
    - ``count`` is the re-ask counter for Filed Answers. Defaults to ``1`` so
      entity/concept construction does not need to know about the field.

    Phase 3 amendment (issue #93):
    - ``source_hashes`` is the 8th field, carrying per-source hash pairs.
      Shape: ``{<source_filename>: {"raw": <content_sha256 | null>, "docs_body": <hex>}}``.
      ``raw`` is the content_sha256 written by importer.py (Phase 7-3); null
      for hand-authored docs that were never imported.
      ``docs_body`` is SHA-256 of the source file as UTF-8 text; used by
      ``/ingest`` for hash-skip idempotency.
      Empty dict (default) means "drift state unknown" — do NOT skip on
      empty source_hashes; this is the legacy state for Phase 6 pages.
    """

    id: str
    type: SourceType
    created: str  # ISO-8601 UTC string, e.g. "2026-05-26T14:30:00Z"
    updated: str  # ISO-8601 UTC string, matches created on first write
    sources: list[str]  # list of "filename#slug" citation strings
    status: Literal["live", "draft", "failed_grounding", "stale", "superseded"]
    open_questions: list[str]
    grounding_failure: GroundingFailure | None = None
    # Phase 6 (Slice 6-1): qa-page-only fields. Optional/defaulted so all
    # existing entity/concept construction sites in templates.py and tests
    # keep working without modification (forward-compat schema change).
    question: str | None = None
    count: int = 1
    # Phase 3 amendment (issue #93): 8th field — per-source hash chain.
    # Default factory returns empty dict = "drift state unknown" (legacy Phase 6
    # pages have no source_hashes; /ingest must NOT skip on empty source_hashes).
    source_hashes: dict[str, dict[str, str | None]] = Field(default_factory=dict)


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
    `status` is one of ``'created'`` (fresh write), ``'updated'`` (hash-drift
    overwrite), or ``'skipped'`` (hash-match no-op — Phase 3 amendment #93).

    Meaningful population added in Slice #3 (orphan handling + created preservation).
    Phase 3 amendment (#93) adds `status` for skip/created/updated discrimination.
    """

    source: str  # bare filename, e.g. "refund_policy.md"
    pages_written: list[str]
    pages_created: list[str] = []
    pages_updated: list[str] = []
    pages_deleted: list[str] = []
    error: str | None = None
    # Phase 3 amendment (#93): status discriminates skip from write outcomes.
    # Default "created" for backward compat with callers that don't set it.
    status: Literal["created", "updated", "skipped"] = "created"


class IngestRequest(BaseModel):
    """Request body for POST /ingest.

    `source` is the bare filename of a single Source to ingest
    (e.g. "refund_policy.md"). When omitted (or body omitted entirely),
    batch mode ingests all Sources under docs/ (Slice #2).

    `force` bypasses hash-skip idempotency (Phase 3 amendment #93). When
    True, the source is re-ingested even if its docs_body_hash matches the
    existing wiki frontmatter. Defaults to False.
    """

    source: str | None = None
    force: bool = False


class IngestResponse(BaseModel):
    """Response body for POST /ingest.

    `results` lists one IngestSourceResult per successfully processed Source.
    `failed_sources` lists bare filenames that could not be processed (Source
    not found, parse error, etc.).
    `pages_with_failed_grounding` lists page ids (slugs) of pages that were
    written but failed the grounding check (status=failed_grounding).  Added
    in Slice #4 — empty on all prior slices.
    `skipped_sources` lists IngestSourceResult entries for hash-match no-ops
    (Phase 3 amendment #93). Empty when no hash matches were detected.
    """

    results: list[IngestSourceResult]
    failed_sources: list[str]
    pages_with_failed_grounding: list[str] = []
    skipped_sources: list[IngestSourceResult] = []


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


# ---------------------------------------------------------------------------
# /lint schemas (Phase 5 Slice 5-4 — C1)
# ---------------------------------------------------------------------------


class CoverageGapFinding(BaseModel):
    """C1 coverage gap finding: repeated Cannot Confirm queries that signal wiki gaps.

    Aggregated from ``chat_fallback`` (``retrieval_empty``, ``below_threshold``) and
    ``chat_grounding_fallback`` (``claim_unsupported``) log entries.

    ``query_canonical`` is the cluster key produced by ``_canonicalise()``.
    ``sample_raw_queries`` holds up to the first 3 unique raw query strings seen
    in the cluster.  ``hit_count`` is the total number of log entries in the cluster.
    ``first_seen`` / ``last_seen`` are ISO-8601 UTC strings parsed from log timestamps.
    ``top_section`` is only populated for ``below_threshold`` findings.
    ``cited_pages`` is only populated for ``claim_unsupported`` findings.
    ``suggested_action`` is the curator-actionable text derived from the reason.
    """

    reason: str  # "retrieval_empty" | "below_threshold" | "claim_unsupported"
    query_canonical: str
    sample_raw_queries: list[str]  # up to 3, deduplicated
    hit_count: int
    first_seen: str  # ISO-8601 UTC string
    last_seen: str  # ISO-8601 UTC string
    top_section: str | None = None  # populated for below_threshold only
    cited_pages: list[str] | None = None  # populated for claim_unsupported only
    suggested_action: str


class PagePairFinding(BaseModel):
    """C5 page-pair contradiction finding: two wiki pages that may contradict each other.

    The LLM emits a 4-value severity via ``with_structured_output``:
    - ``direct``    — explicit factual disagreement (different numbers, different policies).
                      Curator must fix.
    - ``tension``   — same topic, scope/wording differences raising reader confusion.
                      Curator reviews; may dismiss.
    - ``duplicate`` — same concept covered in two pages without contradiction.
                      Absorbs C4-b semantic-duplicate detection from Phase 3 Q5a.
                      Curator considers merging.
    - ``none``      — false positive surfaced by candidate filter; not a real overlap.

    ``page_a`` and ``page_b`` are always in canonical sorted order (sorted slug names
    so that ``(A, B)`` and ``(B, A)`` produce identical findings — symmetric pair
    short-circuit invariant).

    ``page_a_claim`` / ``page_b_claim`` are direct quotes from the respective page bodies.
    ``summary`` and ``suggested_action`` are LLM-generated prose.

    ``severity == "none"`` findings are filtered before returning from ``_check_c5_page_pair``
    so only actionable findings appear in the report.
    """

    severity: Literal["direct", "tension", "duplicate", "none"]
    page_a: str  # slug (always sorted ≤ page_b)
    page_b: str  # slug (always sorted ≥ page_a)
    page_a_claim: str  # direct quote from page_a body
    page_b_claim: str  # direct quote from page_b body
    summary: str  # LLM prose summary of the overlap/contradiction
    suggested_action: str  # LLM-generated curator action


# ---------------------------------------------------------------------------
# /lint schemas (Phase 6 Slice 6-5 — C8 / C9 / C10 Phase 5 amendment)
# ---------------------------------------------------------------------------


class PromotionCandidateFinding(BaseModel):
    """C8 promotion-candidate finding: a draft Filed Answer worth curator review.

    Surfaced to ``lint-report.md`` §``## Promotion Candidates``. Read-only —
    promotion itself is performed by Phase 6 ``POST /qa/{slug}/promote`` (Slice
    6-4), preserving PRD #65 Q3 invariant (`/lint` never mutates frontmatter).

    Ranking: ``count`` desc, then ``updated`` desc (tiebreak). The top
    ``KB_LINT_PROMOTION_TOP_N`` (env var, default 10) candidates are surfaced.

    ``slug``           — the qa page slug (filename stem under ``wiki/qa/``).
    ``question``       — the verbatim user query from ``frontmatter.question``,
                          truncated to a curator-friendly length for the report.
    ``count``          — the re-ask counter from ``frontmatter.count`` (popularity
                          signal).
    ``age_days``       — days since the page was filed
                          (``now() - frontmatter.created`` in UTC).
    ``cited_count``    — number of citation entries in ``frontmatter.sources``
                          (rough proxy for breadth of grounding).
    """

    slug: str
    question: str
    count: int
    age_days: float
    cited_count: int


class QaStalenessFinding(BaseModel):
    """C9 qa-staleness finding: a live Filed Answer whose cited entity is newer.

    Each finding represents a single ``wiki/qa/<slug>.md`` page that has
    ``status: live`` and whose ``frontmatter.sources`` references at least one
    entity page whose filesystem mtime is newer than the qa page's
    ``frontmatter.updated`` timestamp. The cited entities that drifted are
    listed in ``stale_citations``; the worst (largest) drift in days is in
    ``max_drift_days``.

    Closes the "entity re-ingested, qa stranded" failure mode (PRD #78 Q6b).

    Read-only — surfaced to ``lint-report.md`` §``## Stale Filed Answers``.

    ``page_slug``        — the qa page slug.
    ``stale_citations``  — the subset of ``frontmatter.sources`` entries whose
                            entity file mtime exceeded ``frontmatter.updated``.
    ``max_drift_days``   — the largest entity-mtime-minus-page-updated drift in
                            days across ``stale_citations`` (positive number).
    """

    page_slug: str
    stale_citations: list[str]
    max_drift_days: float


class InvalidQaSchemaFinding(BaseModel):
    """C10 qa-schema-validity finding: a structurally invalid qa page.

    One finding per (qa page, broken property) pair. Four invalidity classes
    are checked:

    - ``invalid_status``  — ``frontmatter.status`` is not in
                              ``{live, draft, stale, superseded}`` (e.g. typo
                              ``Live`` capital L).
    - ``wrong_type``      — ``frontmatter.type`` is not ``"qa"`` even though the
                              page lives under ``wiki/qa/``.
    - ``missing_question`` — ``frontmatter.question`` is absent or empty.
    - ``missing_count``   — ``frontmatter.count`` is absent or not a positive
                              integer.

    Closes the curator-typo orphan zombie failure mode (PRD #78 Q8d). Read-only
    — surfaced to ``lint-report.md`` §``## Invalid qa Schema``.

    ``page_slug``        — the qa page slug.
    ``property_name``    — the invalid frontmatter property name (one of the four
                            classes listed above; the field name itself, e.g.
                            ``"status"``, ``"type"``, ``"question"``, ``"count"``).
    ``offending_value``  — the value as found in the page (stringified), or a
                            short marker like ``"<missing>"`` when the field is
                            absent entirely.
    """

    page_slug: str
    property_name: str
    offending_value: str


class LintFindings(BaseModel):
    """Container for all check findings.

    Slice 5-1 populates only ``orphans`` (C11).
    Slice 5-2 adds ``failed_grounding`` (C3) and ``slug_collisions`` (C4-a).
    Slice 5-3 adds ``stale_pages`` (C6) and ``red_links`` (C2).
    Slice 5-4 adds ``coverage_gaps`` (C1).
    Slice 5-5 adds ``page_pairs`` (C5).
    Slice 6-5 (Phase 5 amendment) adds ``promotion_candidates`` (C8),
    ``stale_filed_answers`` (C9), and ``invalid_qa_schemas`` (C10).
    """

    orphans: list[OrphanPageFinding] = []
    failed_grounding: list[FailedGroundingFinding] = []
    slug_collisions: list[SlugCollisionFinding] = []
    stale_pages: list[StalePageFinding] = []
    red_links: list[RedLinkFinding] = []
    coverage_gaps: list[CoverageGapFinding] = []
    page_pairs: list[PagePairFinding] = []
    promotion_candidates: list[PromotionCandidateFinding] = []
    stale_filed_answers: list[QaStalenessFinding] = []
    invalid_qa_schemas: list[InvalidQaSchemaFinding] = []


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


# ---------------------------------------------------------------------------
# /import schemas (Phase 7 Slice 7-1)
# ---------------------------------------------------------------------------


class ImportRequest(BaseModel):
    """Request body for POST /import.

    ``source`` is the bare filename (or relative sub-path) within ``raw/``
    (e.g. ``"customer_handbook.html"``).  When omitted (or body omitted
    entirely), batch mode processes all ``raw/**/*.{html,txt}`` files.
    """

    source: str | None = None


class ImportSourceResultSchema(BaseModel):
    """Per-source successful import outcome.

    Mirrors ``importer.ImportSourceResult`` as a Pydantic model for the API
    boundary.  ``content_sha256`` is the hex SHA-256 of the raw bytes
    (populated in slice 7-3).  ``status`` is one of ``'created'`` (fresh
    write), ``'updated'`` (hash-drift overwrite), or ``'skipped'`` (hash-match
    no-op).
    """

    raw_path: str
    docs_path: str
    original_format: Literal["html", "txt"]
    content_sha256: str = ""
    status: Literal["created", "updated", "skipped"] = "created"


class ImportFailureSchema(BaseModel):
    """Per-source failure record for the API response.

    ``error_type`` is one of the typed error class names (slice 7-1: only
    ``FileNotFoundError`` and ``IOError`` are populated; the full enumeration
    lands in slice 7-2).  ``error_message`` is truncated to 200 characters.
    """

    raw_path: str
    error_type: str
    error_message: str


class ImportResponse(BaseModel):
    """Response body for POST /import.

    Always returns HTTP 200 regardless of per-source failures, matching the
    ``/ingest`` and ``/lint`` always-200 contract (PRD #89 §11).

    ``imported_sources`` lists outcomes for successfully converted sources
    (status ``'created'`` or ``'updated'``).
    ``skipped_sources`` lists per-source hash-match no-ops (status
    ``'skipped'``); populated in slice 7-3.
    ``failed_sources`` lists failures; slice 7-1 populates only
    ``FileNotFoundError`` and ``IOError`` entries (full enumeration in 7-2).
    """

    imported_sources: list[ImportSourceResultSchema]
    skipped_sources: list[ImportSourceResultSchema] = []
    failed_sources: list[ImportFailureSchema] = []
