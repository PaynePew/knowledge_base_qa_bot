"""Shallow module per Ousterhout. Public surface: all Pydantic request/response models (``ChatRequest``, ``ChatResponse``, ``IndexResponse``, ``IngestRequest``, ``IngestResponse``, ``WikiPageDraft``, ``WikiPageFrontmatter``, ``GroundingFailure``, ``IngestSourceResult``, ``SourceType``, ``GroundingClaim``, ``GroundingInfo``, ``CitationRef``, ``FiledStatus``, ``LintResponse``, ``LintSummary``, ``LintFindings``, ``OrphanPageFinding``, ``FailedGroundingFinding``, ``SlugCollisionFinding``, ``StalePageFinding``, ``RedLinkFinding``, ``CoverageGapFinding``, ``PagePairFinding``, ``PromotionCandidateFinding``, ``QaStalenessFinding``, ``InvalidQaSchemaFinding``, ``AliasCollisionFinding``, ``AliasAssignRequest``, ``QaEditRequest``, ``QaRefileResponse``, ``SkippedSlug``, ``QaPromoteBatchRequest``, ``QaPromoteBatchResponse``, ``ImportRequest``, ``ImportSourceResultSchema``, ``ImportFailureSchema``, ``ImportResponse``, ``ReconcileDraft``, ``ReconcileGenerateRequest``, ``CitedSourceSection``, ``ReconcileGenerateResponse``, ``ReconcileApplyRequest``, ``ReconcileApplyResponse``, ``CollisionMergeDraft``, ``CollisionPageDraft``, ``CollisionDifferentiateDraft``, ``CollisionMergeGenerateRequest``, ``CollisionMergeGenerateResponse``, ``CollisionMergeApplyRequest``, ``InboundReference``, ``CollisionMergeApplyResponse``, ``CollisionDifferentiateGenerateRequest``, ``CollisionDifferentiateGenerateResponse``, ``CollisionDifferentiateApplyRequest``, ``CollisionDifferentiateApplyResponse``, ``TranscribeRequest``, ``TranscribeResponse``, ``TranscribeBatchRequest``, ``TranscribeBatchSubmitResponse``, ``TranscribeJobResultSchema``, ``TranscribeJobStatusResponse``).

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


class QaEditRequest(BaseModel):
    """Request body for PUT /qa/{slug} (tier-B S3, issue #379, ADR-0026 decision 2).

    Draft-only edit of a Filed Answer's question/body — the server refuses a
    ``status: live`` page and re-runs the Grounding Check against the page's
    existing cited Sections on ``body`` before writing anything.
    """

    question: str
    body: str


class QaRefileResponse(BaseModel):
    """Response body for a ``POST /qa/{slug}/refile`` that changed the page
    (tier-B S4, issue #380, ADR-0026 decision 1; ``retired`` added ADR-0035).

    ``filed`` is a ``FiledStatus`` with ``status="draft"`` — the demoted corpus
    state the curator reviews via the existing Promote/Edit/Discard Curation
    Queue loop. ``sections_indexed`` is the count from the one BM25 reindex the
    route triggers after the write (mirrors ``POST /qa/{slug}/promote``'s
    auto-reindex convention).

    ``retired`` discriminates the two 200 outcomes:
    - ``retired == False`` — a fresh answer re-grounded and overwrote the page;
      ``grounding.passed == True``.
    - ``retired == True`` (ADR-0035) — the re-synthesis could not be grounded
      (a content failure) and the LIVE page was demoted in place with its OLD
      content, so it stops serving un-groundable content; ``grounding`` here is
      the FAILING outcome that justified the retire (``passed == False``,
      ``reason`` / ``unsupported_claims`` populated). A TRANSIENT re-ground
      failure instead returns HTTP 422 (nothing written), see the route docstring.
    """

    filed: FiledStatus
    grounding: GroundingInfo
    sections_indexed: int
    retired: bool = False


class SkippedSlug(BaseModel):
    """One slug submitted to ``POST /qa/promote-batch`` that was NOT promoted,
    with a machine-parseable reason (tier-B S6, issue #382, ADR-0023
    Consequences: "batch remediation ... surfaces partial failure").

    ``reason`` is one of ``"not_found"`` (no such slug on disk),
    ``"corrupt_frontmatter"`` (frontmatter unparseable), ``"invalid_status:
    <value>"`` (status outside ``{"draft", "live"}``), or ``"already_live"``
    (the slug is not a draft — nothing to promote). Rendered per-item by the
    Console (issue #382 AC).
    """

    slug: str
    reason: str


class QaPromoteBatchRequest(BaseModel):
    """Request body for POST /qa/promote-batch (tier-B S6, issue #382, ADR-0023).

    ``slugs`` is the explicit list of drafts the operator actually saw
    rendered in the Curation Queue — never "all drafts" resolved server-side
    — so a draft filed after the operator looked is never approved
    sight-unseen (ADR-0023 Consequences).
    """

    slugs: list[str]


class QaPromoteBatchResponse(BaseModel):
    """Response body for POST /qa/promote-batch (tier-B S6, issue #382).

    ``promoted`` lists the slugs actually flipped ``draft -> live``, in
    submission order. ``skipped`` lists every submitted slug that failed
    per-slug validation, each with a reason — non-transactional, honestly
    reported (ADR-0023 Consequences), never a silent drop.
    """

    promoted: list[str]
    skipped: list[SkippedSlug]


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

    Issue #406 (ADR-0030):
    - ``aliases`` is the 9th field, a curator-owned list of alternate slugs
      that resolve a ``[[wikilink]]`` to this page (entities/concepts pages
      only — never populated by the LLM synthesis draft). ``/ingest``'s
      page-overwrite preserves it across re-ingest, exactly like ``created``
      (ADR-0030 Invariant: preserve list is ``{created, aliases}``). Aliases
      are link-layer only — they never enter the Section Index (BM25) or the
      dense arm (ADR-0030 Invariant).
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
    # Issue #406 (ADR-0030): 9th field — curator-owned alias list, entities/
    # concepts pages only. Default empty list so every existing construction
    # site (templates.py, tests) keeps working unmodified.
    aliases: list[str] = Field(default_factory=list)


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

    Issue #511 (ADR-0033 "Ingest observability" decision) adds three fields so
    a degenerate parse is visible at the API surface instead of silently
    reporting plain success:
    `sections_count` is the number of Sections `parse_markdown` produced for
    this Source. `uncarried_chars` is the non-whitespace body character count
    the parse did not carry into any Section (see
    `indexer.count_uncarried_chars`) — normally 0 after issue #509 (preamble
    becomes a Section); non-zero flags a new parse/Section gap.
    `enriched_chars` is the character count of heading structure added by
    Structure Enrichment (ADR-0033 decision 2, issue #512) — the summed
    length of the `## title` heading lines it materialized into the Source,
    persisted in the Source's frontmatter (`enriched_chars:`, next to
    `structure: enriched`) at Import/Transcribe time and read back at ingest
    (issue #513). 0 for any Source that was never enriched.
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
    # Issue #511: additive observability fields, all default 0 for backward
    # compat with callers that don't set them.
    sections_count: int = 0
    uncarried_chars: int = 0
    enriched_chars: int = 0


class IngestRequest(BaseModel):
    """Request body for POST /ingest.

    `source` is the bare filename of a single Source to ingest
    (e.g. "refund_policy.md"). When omitted (or body omitted entirely),
    batch mode ingests all Sources under docs/ (Slice #2).

    `sources` is an explicit list of bare filenames for multi-source batch
    calls (Phase 15 S3, issue #172). Used by the Operator Console to send
    a drop batch as a single call so the cross-source `used_slugs` set is
    shared across all named Sources — essential for the "a Section is never
    silently overwritten" guarantee (#54 / ADR-0011).

    Back-compat priority:
      1. `sources` list (explicit multi-source batch) — wins if non-empty.
      2. `source` single-string — used when `sources` is absent/empty.
      3. Neither → all-docs batch mode (glob "**/*.md").

    `force` bypasses hash-skip idempotency (Phase 3 amendment #93). When
    True, the source is re-ingested even if its docs_body_hash matches the
    existing wiki frontmatter. Defaults to False.
    """

    source: str | None = None
    sources: list[str] | None = None
    force: bool = False


class IngestFailureSchema(BaseModel):
    """Per-source failure detail for the API response (issue #507).

    Mirrors ``ImportFailureSchema`` / ``TranscribeJobResultSchema``'s
    ``error_type`` + ``error_message`` shape. ``source`` is the bare filename
    (same string that also appears in ``IngestResponse.failed_sources``).
    ``error_type`` is a short machine-readable category (e.g.
    ``"SectionTooLarge"``, ``"SourceNotFound"``); ``error_message`` is a
    human-readable explanation, truncated to 200 characters, no stack trace.
    """

    source: str
    error_type: str
    error_message: str


class IngestResponse(BaseModel):
    """Response body for POST /ingest.

    `results` lists one IngestSourceResult per successfully processed Source.
    `failed_sources` lists bare filenames that could not be processed (Source
    not found, parse error, etc.).
    `failed_source_details` lists one `IngestFailureSchema` per
    `failed_sources` entry, in the same order, carrying WHY each source
    failed (issue #507) — additive alongside `failed_sources` so existing
    consumers that treat `failed_sources` as bare filenames are unaffected.
    `pages_with_failed_grounding` lists page ids (slugs) of pages that were
    written but failed the grounding check (status=failed_grounding).  Added
    in Slice #4 — empty on all prior slices.
    `skipped_sources` lists IngestSourceResult entries for hash-match no-ops
    (Phase 3 amendment #93). Empty when no hash matches were detected.
    """

    results: list[IngestSourceResult]
    failed_sources: list[str]
    failed_source_details: list[IngestFailureSchema] = []
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

    ``full`` (tier-B S5, issue #381, ADR-0025) distinguishes the CONTEXT.md
    "Orphan Page" full/partial split: ``True`` when ``sources`` is non-empty
    and EVERY citation's file is missing (nothing can ground the page —
    eligible for the Confirmed ``DELETE /pages/{slug}``); ``False`` when at
    least one citation still resolves (partial — advisory only, never
    delete). Defaults to ``False`` for pre-existing callers that construct
    this model directly without the field.
    """

    page_slug: str
    missing_sources: list[str]
    suggested_action: str
    full: bool = False


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
    ``suggested_action`` splits by ``reason`` (issue #407, ADR-0029 decision 3):
    ``claim_unsupported`` names the unsupported claims and points at amending the
    Source — never a bare Re-ingest, since the same unchanged Source feeds the
    same verifier and fails identically; ``verifier_unavailable`` recommends
    Re-ingest (a transient failure, not a Source problem).

    ``source_path`` / ``source_resolution`` (issue #445) resolve ``source``'s
    bare basename against the real docs tree at lint time, replacing the
    Console's former ``"docs/" + basename`` guess — which 404'd whenever the
    cited Source lived in a nested ``docs/`` subdirectory (ingest discovers
    nested Sources but records only the filename):
    - ``"resolved"``: exactly one file under ``docs/`` matches the basename;
      ``source_path`` is the repo-relative path (e.g.
      ``"docs/fake-docs/product_care.md"``).
    - ``"missing"``: no file matches (or the citation has no basename at
      all); ``source_path`` is ``None``.
    - ``"ambiguous"``: two or more files share the basename in different
      subdirectories; ``source_path`` is ``None`` — never guessed, so the
      Console can render this state distinctly instead of silently picking
      one of the candidates.
    """

    page_slug: str
    source: str
    source_path: str | None = None
    source_resolution: Literal["resolved", "missing", "ambiguous"] = "missing"
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
    """C5 page-pair contradiction finding: two wiki pages that give a reader
    incompatible answers to the SAME question.

    The LLM emits a 3-value severity via ``with_structured_output``. ADR-0034
    narrowed C5 from a similarity check to contradiction-only and **retired the
    former ``duplicate`` value**: consistent redundant coverage is not a
    contradiction (if two pages ever state a fact *differently*, that is
    ``direct``), and slug-collision duplicates are C4's job (Merge /
    Differentiate). The candidate filter (F1 ∪ F3) is still similarity-based —
    that is only a cheap cost gate; precision lives entirely in this judge,
    whose default verdict is ``none``.

    - ``direct``   — the two pages make incompatible factual claims about the
                     same question (a different number / amount / date / fee /
                     limit / deadline, or a policy that flips allowed↔not-allowed).
                     A reader following one page would be wrong per the other.
                     Reconcile converges it (fix → re-judge → ``none``).
    - ``tension``  — same question, no different fact stated, but one page
                     materially omits a condition/exception the other states, so
                     reading only that page misleads about that same question
                     (each cherry-picks part of one underlying rule). Used
                     sparingly; Reconcile converges it too.
    - ``none``     — NOT a finding. Adjacent-but-distinct topics that merely
                     share vocabulary (e.g. cancel vs pause — different actions,
                     both valid), broader-vs-specific pages that do not disagree,
                     one page merely carrying more detail/scope, and consistent
                     redundant coverage all resolve here.

    ``page_a`` and ``page_b`` are always in canonical sorted order (sorted slug names
    so that ``(A, B)`` and ``(B, A)`` produce identical findings — symmetric pair
    short-circuit invariant).

    ``page_a_claim`` / ``page_b_claim`` are direct quotes from the respective page bodies.
    ``summary`` and ``suggested_action`` are LLM-generated prose.

    ``severity == "none"`` findings are filtered before returning from ``_check_c5_page_pair``
    so only contradictions Reconcile can converge appear in the report (ADR-0034).
    """

    severity: Literal["direct", "tension", "none"]
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
    ``status``           — the page's raw ``frontmatter.status`` value verbatim
                            (``"<missing>"`` / ``"<unparseable>"`` when absent or
                            the page could not be parsed at all). Added by
                            ADR-0037 so a caller (the Console C10 card) can pick
                            demote-to-draft for a ``status: live`` page vs. the
                            existing one-click discard for everything else,
                            without a second read of the page.
    """

    page_slug: str
    property_name: str
    offending_value: str
    status: str


def qa_schema_lint_code(frontmatter: dict) -> str | None:
    """Return ``"C10"`` if ``frontmatter`` is a schema-invalid Filed Answer, else ``None``.

    A pure, I/O-free mirror of the C10 rule that :class:`InvalidQaSchemaFinding`
    documents and ``lint._check_c10_qa_schema_validity`` enforces on disk (``status``
    in ``{live, draft, stale, superseded}``, ``type == "qa"``, non-empty
    ``question``, positive-int ``count``). It exists so the chat source-builders can
    tag a schema-invalid Filed Answer with the SAME ``"C10"`` coordinate the Operator
    Console surfaces — WITHOUT importing ``lint`` (whose C5 path would pull the LLM
    client into the hot retrieval path). Non-qa frontmatter returns ``None``. lint
    keeps its file-oriented detector; a later pass may delegate it here to de-dup.
    """
    if frontmatter.get("type") != "qa":
        return None
    status = frontmatter.get("status")
    question = frontmatter.get("question")
    count = frontmatter.get("count")
    valid = (
        status in {"live", "draft", "stale", "superseded"}
        and isinstance(question, str)
        and question.strip() != ""
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    )
    return None if valid else "C10"


class AliasCollisionFinding(BaseModel):
    """C12 alias-collision finding (issue #406, ADR-0030, Coherence axis).

    Two collision shapes, both keyed on a single alias string:

    - ``kind == "alias_vs_slug"`` — ``alias`` matches an existing page's real
      slug. ``slug_owner`` is that real page (equal to ``alias``); resolution
      is unambiguous (a real page slug always wins over an alias per the
      shared resolver, ``slugs.build_alias_resolution_map``).
    - ``kind == "alias_vs_alias"`` — two or more pages independently claim
      the SAME alias and no real page owns that slug. ``slug_owner`` is
      ``None``; the shared resolver's tie-break (lexicographically-first
      canonical slug) decides ``resolved_to``.

    ``claimed_by`` lists every page (sorted, deduplicated) whose frontmatter
    declares this alias — usually one entry for ``alias_vs_slug``, at
    least two for ``alias_vs_alias``. ``resolved_to`` mirrors what the shared
    resolver actually returns for ``alias`` right now, so a curator can see
    the current (deterministic) resolution while deciding whether to edit
    frontmatter to remove the collision (Direct remediation — no endpoint
    exists yet in this foundation slice; see ``_REMEDIATION_TAXONOMY["C12"]``).
    """

    kind: Literal["alias_vs_slug", "alias_vs_alias"]
    alias: str
    claimed_by: list[str]
    slug_owner: str | None = None
    resolved_to: str
    suggested_action: str


class AliasAssignRequest(BaseModel):
    """Request body for ``POST /pages/{slug}/aliases`` (issue #409, ADR-0030
    decision 3).

    ``alias`` is the single alternate slug to assign to the ``{slug}`` path
    parameter's target page. Direct-class, never a batch (ADR-0030 Invariant
    "assign-alias never batches") — the shape carries exactly one string,
    never a list.
    """

    alias: str


class ResolutionMapResponse(BaseModel):
    """Response body for ``GET /pages/resolution-map`` (issue #410, ADR-0030
    decision 5).

    The read-only view every linkify client (Console ``/read/file`` viewer,
    reader chat answer bodies, chat-side citation viewer) consults to render
    ``[[wikilinks]]`` as navigation. Both fields derive from the SAME shared
    resolver (``slugs.build_alias_resolution_map`` — ADR-0030 Invariant), so
    "does this resolve" can never disagree with what C2 reports for the same
    corpus.

    ``slugs``: every real entities/concepts page slug -> its wiki-relative
    path (e.g. ``"wiki/entities/acme-shop.md"``). Carrying the relpath (not
    just the bare slug) lets a client navigate to a resolved page WITHOUT
    ever constructing a wiki path from a bare slug itself — the client-side
    convention CODING_STANDARD §12.5 already establishes for citation links
    (``source.path`` is server-supplied), extended here to wikilinks.

    ``aliases``: every alias -> its canonical (real) slug — a key into
    ``slugs`` for the target's relpath. ``[[slug|display]]`` pipe syntax
    resolves on the slug part; the client renders the display part
    regardless of whether resolution succeeds.
    """

    slugs: dict[str, str]
    aliases: dict[str, str]


class LintFindings(BaseModel):
    """Container for all check findings.

    Slice 5-1 populates only ``orphans`` (C11).
    Slice 5-2 adds ``failed_grounding`` (C3) and ``slug_collisions`` (C4-a).
    Slice 5-3 adds ``stale_pages`` (C6) and ``red_links`` (C2).
    Slice 5-4 adds ``coverage_gaps`` (C1).
    Slice 5-5 adds ``page_pairs`` (C5).
    Slice 6-5 (Phase 5 amendment) adds ``promotion_candidates`` (C8),
    ``stale_filed_answers`` (C9), and ``invalid_qa_schemas`` (C10).
    Issue #406 (ADR-0030) adds ``alias_collisions`` (C12; C7 is skipped and
    stays unassigned).
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
    alias_collisions: list[AliasCollisionFinding] = []


class LintSummary(BaseModel):
    """Aggregate metrics for a lint run.

    ``findings_by_check`` maps check identifier (``"c11"``, ``"c3"``, …) to
    finding count.  Slice 5-1 only includes ``"c11"``; later slices extend.
    ``llm_calls`` and ``cost_usd`` are 0 in Slice 5-1 (no LLM used by C11).

    ``c5_pairs_capped`` (issue #194) is the number of C5 candidate page-pairs
    that were NOT sent to the LLM judge because they fell below the
    ``KB_LINT_C5_MAX_PAIRS`` similarity cap. ``llm_calls`` is the judged count;
    ``llm_calls + c5_pairs_capped`` is the total candidate count. Surfaced so a
    capped audit is honest rather than silently partial; 0 when nothing was
    capped (small wiki) or C5 was skipped (``include_c5=False``).
    """

    total_findings: int
    findings_by_check: dict[str, int]
    llm_calls: int = 0
    cost_usd: float = 0.0
    c5_pairs_capped: int = 0
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
# /pages/reconcile schemas (tier-B S1 — issue #376, ADR-0028)
# ---------------------------------------------------------------------------


class ReconcileDraft(BaseModel):
    """LLM structured-output schema for the C5 Reconcile drafting call (ADR-0028).

    ``content_a``/``content_b`` are the full revised page content *after* the
    frontmatter fence (heading + prose + trailing citation line, structurally
    identical in shape to the original — mirrors the single opaque blob
    ``lint._load_wiki_pages`` already judges pages by for C5). Drafted from
    the union of both pages' Sources so the two pages state mutually
    consistent facts.
    """

    content_a: str
    content_b: str


class ReconcileGenerateRequest(BaseModel):
    """Request body for POST /pages/reconcile (ADR-0028)."""

    page_a: str
    page_b: str


class CitedSourceSection(BaseModel):
    """One Source Section a C5 Reconcile page cites, for the two-view
    Reconcile modal's Source comparison (issue #534, ADR-0036 decision 3).

    Narrower than the grounding union, which stays whole-file by design
    (ADR-0036 decision 7, unaffected by this model) — this is presentation
    data for ONE page's OWN citations, not the pair's combined grounding
    context. ``id`` is the anchored citation exactly as ``Section.id``
    produces it (e.g. ``"退款與退貨.md#退款申請窗口"``), so it round-trips
    against the page's own ``frontmatter.sources`` entries.

    ``heading``/``content`` are ``None`` when the citation could not be
    resolved to actual Section content — a missing/ambiguous Source file, or
    a stale anchor that no longer matches any parsed heading. The raw
    citation is still surfaced (never silently dropped), matching C3's
    ``source_resolution`` honesty convention. ``source_path`` mirrors
    ``FailedGroundingFinding.source_path``'s ``"docs/<relative>"`` display
    convention for a ``/read/file`` view link; ``None`` when
    ``source_resolution`` is not ``"resolved"``.
    """

    id: str
    heading: str | None = None
    content: str | None = None
    source_path: str | None = None
    source_resolution: Literal["resolved", "missing", "ambiguous"] = "missing"


class ReconcileGenerateResponse(BaseModel):
    """Response body for POST /pages/reconcile.

    **Invariant (ADR-0028)** — writes nothing to disk. ``old_content_a`` /
    ``old_content_b`` are each page's CURRENT on-disk content (post-
    frontmatter) so the Console can render an old-vs-draft side-by-side
    comparison without a second round trip. ``hash_a`` / ``hash_b`` are the
    SHA-256 of each page's full on-disk file text at generate time — the
    client must return them unchanged to POST /pages/reconcile/apply so the
    server can detect whether either page changed underneath the preview.

    ``cited_sections_a`` / ``cited_sections_b`` (issue #534, ADR-0036
    decision 3) carry each page's OWN cited Source sections — the two-view
    modal's Source comparison payload. This is presentation data only — the
    detection/grounding logic itself (the union stays whole-file, decision 7)
    is unchanged.

    ``converged`` (issue #545, ADR-0038) is the real source-rooted signal:
    the C5 contradiction oracle re-run on the two DRAFTS returned ``none``
    (the pages now agree). The client keys BOTH the default view and the Apply
    gate on it — ``converged`` → Wiki comparison + Apply enabled; not
    converged → Source comparison + Apply disabled (source-rooted: fix a
    Source). This REPLACES the earlier ``grounding.passed``-based routing:
    grounding is an existence check that a self-contradicting Source union
    passes for each faithful draft, so it cannot signal source-rooted.
    ``convergence_summary`` carries the oracle's prose (what still conflicts)
    for the Source-view note when not converged. Defaults to ``False`` so a
    response that omits it fails safe to Apply-disabled.
    """

    page_a: str
    page_b: str
    old_content_a: str
    old_content_b: str
    content_a: str
    content_b: str
    grounding: GroundingInfo
    converged: bool = False
    convergence_summary: str | None = None
    hash_a: str
    hash_b: str
    cited_sections_a: list[CitedSourceSection] = []
    cited_sections_b: list[CitedSourceSection] = []


class ReconcileApplyRequest(BaseModel):
    """Request body for POST /pages/reconcile/apply (ADR-0028).

    ``content_a`` / ``content_b`` are the final (possibly human-edited) page
    content submitted for write-back. ``hash_a`` / ``hash_b`` must be the
    values returned by the matching POST /pages/reconcile call — the server
    refuses (409) when either page's current on-disk hash no longer matches
    (the finding may no longer hold).
    """

    page_a: str
    page_b: str
    content_a: str
    content_b: str
    hash_a: str
    hash_b: str


class ReconcileApplyResponse(BaseModel):
    """Response body for a successful POST /pages/reconcile/apply.

    ``sections_indexed`` is the count from the one BM25 reindex the route
    triggers after both pages are written (ADR-0028 Invariant — a Gated
    Remediation never batches, but the reindex itself is a single full
    rebuild covering both rewritten pages, matching the existing
    ``POST /qa/{slug}/promote`` auto-reindex convention).
    """

    page_a: str
    page_b: str
    grounding: GroundingInfo
    sections_indexed: int


# ---------------------------------------------------------------------------
# /pages/collision/{merge,differentiate} schemas (tier-B S2 — issue #378, ADR-0028)
#
# C4 slug-collision groups gain both documented resolutions on top of S1's
# two-phase machinery (generate -> preview/edit -> apply-with-revalidation).
# ---------------------------------------------------------------------------


class CollisionMergeDraft(BaseModel):
    """LLM structured-output schema for the C4 merge-into-base drafting call
    (ADR-0028). ``content_base`` is the full revised content (post-frontmatter)
    for the group's unsuffixed base slug, drafted from the union of every
    group member's Sources."""

    content_base: str


class CollisionPageDraft(BaseModel):
    """One page's drafted content in a C4 differentiate response — a fixed
    ``{slug, content}`` shape (rather than a ``dict[str, str]``) so the LLM
    structured-output schema stays a plain JSON-schema object list regardless
    of how many pages are in the collision group."""

    slug: str
    content: str


class CollisionDifferentiateDraft(BaseModel):
    """LLM structured-output schema for the C4 differentiate drafting call
    (ADR-0028). ``pages`` carries one drafted entry per group member, each
    rewritten to be complementary and more specific — nobody is deleted."""

    pages: list[CollisionPageDraft]


class CollisionMergeGenerateRequest(BaseModel):
    """Request body for POST /pages/collision/merge (ADR-0028)."""

    base_slug: str
    variant_slugs: list[str]


class CollisionMergeGenerateResponse(BaseModel):
    """Response body for POST /pages/collision/merge.

    **Invariant (ADR-0028)** — writes nothing to disk. ``old_content_base`` is
    the base page's CURRENT on-disk content (post-frontmatter). ``hash_base`` /
    ``hash_variants`` are SHA-256 of each page's full on-disk file text at
    generate time — the client returns them unchanged to
    POST /pages/collision/merge/apply so the server can detect whether any
    page changed underneath the preview.
    """

    base_slug: str
    variant_slugs: list[str]
    old_content_base: str
    content_base: str
    grounding: GroundingInfo
    hash_base: str
    hash_variants: dict[str, str]


class CollisionMergeApplyRequest(BaseModel):
    """Request body for POST /pages/collision/merge/apply (ADR-0028).

    ``content_base`` is the final (possibly human-edited) content submitted
    for write-back onto ``base_slug``. ``hash_base`` / ``hash_variants`` must
    be the values returned by the matching POST /pages/collision/merge call —
    the server refuses (409) when any page's current on-disk hash no longer
    matches.
    """

    base_slug: str
    variant_slugs: list[str]
    content_base: str
    hash_base: str
    hash_variants: dict[str, str]


class InboundReference(BaseModel):
    """Inbound-reference guard detail for one variant slated for deletion by
    a C4 merge (ADR-0028 Invariant). Populated only for variants that
    actually have at least one referrer — an empty ``wiki_referrers`` and
    ``qa_referrers`` pair never appears in a guard-failure response."""

    variant_slug: str
    wiki_referrers: list[str]
    qa_referrers: list[str]


class CollisionMergeApplyResponse(BaseModel):
    """Response body for a successful POST /pages/collision/merge/apply.

    ``deleted_variants`` lists the slugs actually removed (reference-free —
    the guard already refused the call had any variant been referenced).
    ``sections_indexed`` is the count from the one BM25 reindex triggered
    after the base rewrite + variant deletions.
    """

    base_slug: str
    deleted_variants: list[str]
    grounding: GroundingInfo
    sections_indexed: int


class CollisionDifferentiateGenerateRequest(BaseModel):
    """Request body for POST /pages/collision/differentiate (ADR-0028)."""

    slugs: list[str]


class CollisionDifferentiateGenerateResponse(BaseModel):
    """Response body for POST /pages/collision/differentiate.

    **Invariant (ADR-0028)** — writes nothing to disk. ``old_content`` /
    ``content`` / ``hashes`` are keyed by slug, one entry per group member.
    """

    slugs: list[str]
    old_content: dict[str, str]
    content: dict[str, str]
    grounding: GroundingInfo
    hashes: dict[str, str]


class CollisionDifferentiateApplyRequest(BaseModel):
    """Request body for POST /pages/collision/differentiate/apply (ADR-0028).

    ``content`` is the final (possibly human-edited) content submitted for
    write-back, keyed by slug. ``hashes`` must be the values returned by the
    matching POST /pages/collision/differentiate call — the server refuses
    (409) when any page's current on-disk hash no longer matches.
    """

    slugs: list[str]
    content: dict[str, str]
    hashes: dict[str, str]


class CollisionDifferentiateApplyResponse(BaseModel):
    """Response body for a successful POST /pages/collision/differentiate/apply.

    All group members survive, rewritten in place. ``sections_indexed`` is
    the count from the one BM25 reindex triggered after every page is
    written.
    """

    slugs: list[str]
    grounding: GroundingInfo
    sections_indexed: int


# ---------------------------------------------------------------------------
# /import schemas (Phase 7 Slice 7-1)
# ---------------------------------------------------------------------------


class ImportRequest(BaseModel):
    """Request body for POST /import.

    ``source`` is the bare filename (or relative sub-path) within ``raw/``
    (e.g. ``"customer_handbook.html"``).  When omitted (or body omitted
    entirely), batch mode processes all ``raw/**/*.{html,txt,md,pdf}`` files.
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
    original_format: Literal["html", "txt", "md", "pdf"]
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


class ImportJobSubmitResponse(BaseModel):
    """Response body for POST /import/jobs — the submitted job id (issue #497)."""

    job_id: str


class ImportJobStatusResponse(BaseModel):
    """Response body for GET /import/jobs/{job_id} (issue #497).

    ``pages_done`` / ``pages_total`` track auto-routed Transcribe pages across
    the WHOLE run — the same server-owned numbers TranscribeJobStatusResponse
    exposes (the Console Import progress bar's data source; §12.8 bans
    client-guessed percentages, not real counts). ``pages_total`` grows
    incrementally as each scanned file's page count is discovered; a run with
    no scans keeps both at 0. ``files_done`` / ``files_total`` track
    whole-run file progress (``files_total`` is 0 until the first file
    finishes). ``result`` is the SAME ``ImportResponse`` the synchronous
    ``POST /import`` returns, present only once ``status == "completed"``.
    """

    job_id: str
    status: Literal["submitted", "working", "completed", "failed"]
    files_done: int
    files_total: int
    pages_done: int
    pages_total: int
    result: ImportResponse | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# /transcribe schemas (issue #426, ADR-0032)
# ---------------------------------------------------------------------------


class TranscribeRequest(BaseModel):
    """Request body for POST /transcribe.

    ``source`` is the bare filename (or relative sub-path) within ``raw/``
    (e.g. ``"scanned_manual.pdf"``) — single-source only, no batch mode.
    Force-transcribes exactly this PDF, bypassing the text-layer probe
    (ADR-0032 designed-PDF escape hatch: a curator-forced re-conversion of a
    digital-native PDF that extracted degraded through the mechanical path).
    """

    source: str


class TranscribePageCountResponse(BaseModel):
    """Response body for GET /transcribe/page-count (issue #447 preflight).

    A mechanical page count for a raw/ PDF, computed with NO model call — the
    Console's guarded Transcribe action names this real number (plus the
    configured page cap) in its confirm step before the operator commits to a
    forced transcription, rather than guessing a bound client-side.
    """

    source: str
    page_count: int
    max_pages: int


class TranscribeResponse(BaseModel):
    """Response body for POST /transcribe.

    Mirrors ``ImportSourceResultSchema`` in shape plus the Transcribe-specific
    ``origin`` / ``transcribe_model`` provenance fields (ADR-0032). On
    ``status="skipped"`` (hash-match no-op), ``transcribe_model`` reports the
    currently configured model, not necessarily the model that produced the
    existing file.
    """

    raw_path: str
    docs_path: str
    content_sha256: str
    transcribe_model: str
    status: Literal["created", "updated", "skipped"]
    origin: Literal["transcribed"] = "transcribed"


# ---------------------------------------------------------------------------
# /transcribe/batch + /transcribe/jobs schemas (issue #459 AC5)
# ---------------------------------------------------------------------------


# Cap on the number of sources in one batch submission (issue #474
# sub-issue B) — an unbounded list lets a single anonymous POST grow one
# job's ``results`` (and therefore process memory) without limit. Orthogonal
# to ``KB_TRANSCRIBE_MAX_CONCURRENT_JOBS`` (``transcribe_jobs.py``), which
# bounds how many BATCHES may run at once, not how big one batch is.
MAX_BATCH_SOURCES: int = 50


class TranscribeBatchRequest(BaseModel):
    """Request body for POST /transcribe/batch.

    ``sources`` names one or more files already staged under ``raw/`` (same
    per-file contract as ``TranscribeRequest.source``). Unlike ``POST
    /transcribe`` this returns immediately with a job id — see
    ``TranscribeBatchSubmitResponse`` — instead of blocking for the whole
    batch's duration (issue #459 item 5). Capped at ``MAX_BATCH_SOURCES``
    entries — FastAPI/Pydantic reject an over-long list with HTTP 422
    before ``submit_batch`` ever sees it (issue #474 sub-issue B).
    """

    sources: list[str] = Field(max_length=MAX_BATCH_SOURCES)


class TranscribeBatchSubmitResponse(BaseModel):
    """Response body for POST /transcribe/batch — the submitted job id."""

    job_id: str


class TranscribeJobResultSchema(BaseModel):
    """Per-source outcome inside a TranscribeJobStatusResponse.

    Mirrors ``transcribe_jobs.TranscribeJobResult``. ``error_type`` /
    ``error_message`` are populated only when ``status == "failed"``.
    """

    source: str
    status: Literal["created", "updated", "skipped", "failed"]
    docs_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class TranscribeJobStatusResponse(BaseModel):
    """Response body for GET /transcribe/jobs/{job_id}.

    ``pages_done`` / ``pages_total`` track pages across the WHOLE batch
    (issue #459 AC4) — the data source for a Console progress bar (#447).
    ``pages_total`` grows incrementally as each source's own page count is
    discovered; a freshly submitted job (before any source has started
    reporting) may show ``pages_total == 0``.
    """

    job_id: str
    status: Literal["submitted", "working", "completed", "failed"]
    pages_done: int
    pages_total: int
    results: list[TranscribeJobResultSchema] = []
    error: str | None = None
