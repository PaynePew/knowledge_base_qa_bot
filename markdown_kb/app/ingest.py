"""Deep module per Ousterhout. Public surface: ``ingest_sources``, ``aingest_sources``.

Ingest coordinator — Source → wiki synthesis page pipeline.

Provides `ingest_sources(source_filenames)` which orchestrates the ingest
pipeline for one or more Sources:

    1. Resolve Source path(s) under docs/ (batch: glob("**/*.md"))
    2. Parse each Source into Sections via indexer.parse_markdown
    3. Hash-skip check (Phase 3 amendment #93): compute docs_body_hash; if
       existing wiki page has matching hash and force=False → skip, emit
       ingest_skipped, add to batch.skipped_sources.
    4. Classify Source type via templates.classify_source
    5a. concept → one WikiPageDraft per Section (1:N expansion)
    5b. entity  → one WikiPageDraft collapsing the whole Source
    6. Resolve slug collisions across Sources (-2, -3 suffix)
    7. Preserve `created` timestamp for pages that already exist on disk
    8. Populate source_hashes frontmatter (Phase 3 amendment #93)
    9. Run grounding verifier on each generated page (Slice #4 — fail-soft)
    10. Delete orphan pages (per-Source scoped) via wiki_writer.delete_orphans
    11. Write pages via wiki_writer.write_pages_for_source
    12. Return an IngestBatchResult summarising outcomes with meaningful
        pages_created / pages_updated / pages_deleted per Source

Also provides `aingest_sources(source_filenames)` — async sibling that fans
out the per-section concept synthesis concurrently via ``asyncio.to_thread``
with a bounded semaphore (``KB_INGEST_CONCURRENCY``, default 8).  Slug
resolution and the write tail run sequentially after gather so slug
determinism and index-lock safety are preserved.

Continue-on-error: a Source that throws at any stage is recorded in
`failed_sources` but does not stop the batch (Q3 grill decision).

Concurrency: ingest holds ``indexer._index_lock`` for the write + orphan-delete
step so it is mutually exclusive with ``build_index()`` (Q7 grill decision).
No per-page locking is needed for the prototype.

Grounding Check (Slice #4 — ADR-0004 fail-soft):
- Verifier uses OPENAI_VERIFIER_MODEL (independent of OPENAI_INGEST_MODEL).
- On claim_supported: page written with status=live.
- On claim_unsupported or verifier_unavailable: page written with
  status=failed_grounding + grounding_failure frontmatter block.
  Page id added to IngestBatchResult.pages_with_failed_grounding.

Hash-Skip (Phase 3 amendment #93):
- docs_body_hash = sha256(source_path.read_text('utf-8').encode()).hexdigest()
  Deterministic convention: hash the UTF-8 text content (not raw bytes) to
  match docs/ ingest semantics (Sources are always read as UTF-8 text here).
- Skip decision matrix:
    | wiki page exists? | source_hashes? | hash match? | force? | Behavior |
    |---|---|---|---|---|
    | No                | —              | —           | —      | Ingest (fresh write) |
    | Yes               | empty/missing  | —           | —      | Ingest (unknown drift) |
    | Yes               | present        | YES         | False  | Skip |
    | Yes               | present        | YES         | True   | Ingest anyway |
    | Yes               | present        | NO          | —      | Ingest (overwrite) |
- Empty/missing source_hashes = "drift state unknown" — do NOT skip.
  Phase 6 legacy pages have no source_hashes; empty dict is the Pydantic default.
  NOTE: Phase 5 lint must also treat empty source_hashes as "unknown" to avoid
  false-negative drift reports on Phase 6 legacy pages (Phase 5's responsibility).

Wiki Log (Slice #4 — five new kind values):
- ingest_batch_started / ingest_batch_completed bracket the whole batch.
- ingest_source emitted per successful Source.
- ingest_grounding_failed emitted per page with failed grounding.
- ingest_error emitted per Source-level failure (replaces prior ad-hoc kinds).
- ingest_skipped emitted per hash-match no-op (Phase 3 amendment #93).

See PRD #28 for the full pipeline design.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from ._paths import DOCS_DIR
from .errors import LLMError
from .grounding import verify
from .indexer import _index_lock, parse_markdown, slugify, split_frontmatter
from .logger import log_event
from .schemas import GroundingFailure, IngestSourceResult
from .templates import classify_source, generate_entity_page, generate_page
from .wiki_writer import (
    delete_orphans,
    read_existing_frontmatter,
    resolve_slug_collision,
    write_pages_for_source,
)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestBatchResult:
    """Aggregated outcome of ingest_sources.

    `results` lists one IngestSourceResult per successfully-processed Source.
    `failed_sources` lists bare filenames that failed (Source not found, parse
    error, LLM call failure, or write failure).
    `pages_with_failed_grounding` lists page ids (slugs) that were written but
    failed the grounding check (status=failed_grounding).  Added in Slice #4.
    `skipped_sources` lists IngestSourceResult entries for hash-match no-ops
    (Phase 3 amendment #93). Empty when no hash matches were detected.
    `_llm_call_count` tracks total LLM calls for the batch_completed log entry.
    """

    results: list[IngestSourceResult] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)
    failed_reasons: dict[str, str] = field(default_factory=dict)
    pages_with_failed_grounding: list[str] = field(default_factory=list)
    skipped_sources: list[IngestSourceResult] = field(default_factory=list)
    _llm_call_count: int = 0


# ---------------------------------------------------------------------------
# Token guard
# ---------------------------------------------------------------------------
# KB_INGEST_MAX_BYTES has been retired.  The byte guard was CJK-unsafe (a CJK
# character is 3 bytes UTF-8 but typically 1-2 tokens, so the byte count
# over-counted English and under-counted dense CJK equally).  The token guard
# below uses a uniform //3 estimate (CJK-pessimistic: over-counts ASCII, which
# is the safe direction) and enforces two independent limits:
#
#   KB_INGEST_MAX_SECTION_TOKENS — per-section HARD cap (fail-fast)
#   KB_INGEST_MAX_TOKENS         — per-Source SOFT cap (routing hint)
#
# See project-docs/large-file-ingest-size-limit-findings.md.


def _estimate_tokens(content: str) -> int:
    """CJK-pessimistic token estimate: len(content) // 3.

    Uses integer floor-division by 3 uniformly regardless of script.
    This over-counts ASCII tokens (typically 1 char ≈ 0.25 tokens) and
    under-counts CJK (typically 1 char ≈ 0.5–1 token), which is the
    safe (conservative) direction for a guard: it may reject a borderline
    ASCII doc but will never silently pass a large CJK doc.
    """
    return len(content) // 3


def _max_ingest_tokens() -> int:
    """Per-Source SOFT token cap, read at call time (KB_SCORE_THRESHOLD pattern).

    Override with ``KB_INGEST_MAX_TOKENS``; a restart-free change takes effect
    on the next ingest.  Default 64 000 tokens (~192 KB UTF-8 ASCII, ~64 KB CJK).
    Sources that exceed this cap are routed to the async job path (Fix 1b).
    """
    return int(os.getenv("KB_INGEST_MAX_TOKENS", "64000"))


def _max_section_tokens() -> int:
    """Per-section HARD token cap, read at call time.

    Override with ``KB_INGEST_MAX_SECTION_TOKENS``; default 6 000 tokens.
    A Source with ANY section exceeding this cap is rejected with a clear
    reason before any LLM call is made.
    """
    return int(os.getenv("KB_INGEST_MAX_SECTION_TOKENS", "6000"))


def _max_concurrency() -> int:
    """Maximum concurrent in-flight LLM calls for aingest_sources.

    Override with ``KB_INGEST_CONCURRENCY``; default 8.  A restart-free change
    takes effect on the next aingest_sources call.
    """
    return int(os.getenv("KB_INGEST_CONCURRENCY", "8"))


def _should_route_async(content: str) -> bool:
    """Return True when the Source text exceeds the per-Source SOFT token cap.

    In Fix 2 this is a routing seam only; actual async routing is wired in
    Fix 1b.  The SOFT cap does NOT reject synchronous ingest — it is only a
    hint for the scheduler.

    Args:
        content: Full Source text (after frontmatter strip).

    Returns:
        True if _estimate_tokens(content) > _max_ingest_tokens().
    """
    return _estimate_tokens(content) > _max_ingest_tokens()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_docs_files(docs_dir: Path) -> list[tuple[str, Path]]:
    """Return (bare_filename, absolute_path) pairs for all Markdown Sources.

    Uses ``glob("**/*.md")`` so nested sub-folders are picked up.
    The bare_filename is ``p.name`` (e.g. ``"nested.md"``); Citation format
    stays ``{stem}.md#{slug}`` (CONTEXT.md).  Results are sorted by bare
    filename for deterministic ordering.
    """
    return sorted(
        ((p.name, p) for p in docs_dir.glob("**/*.md")),
        key=lambda t: t[0],
    )


def _compute_docs_body_hash(source_path: Path) -> str:
    """Compute SHA-256 of the source file's UTF-8 text content.

    Deterministic convention (Phase 3 amendment #93): hash the UTF-8 text
    (source_path.read_text('utf-8').encode()) so the hash is stable across
    platforms regardless of OS line-ending differences from binary reads.
    This is the ``docs_body_hash`` stored in ``wiki frontmatter.source_hashes``.
    """
    return hashlib.sha256(source_path.read_text(encoding="utf-8").encode()).hexdigest()


def _should_skip_source(
    source_name: str,
    docs_body_hash: str,
    wiki_dir: Path,
    force: bool,
) -> tuple[bool, list[str]]:
    """Check whether a Source should be skipped based on hash comparison.

    Decision matrix (Phase 3 amendment #93):
    - wiki page does not exist → False (ingest fresh)
    - wiki page exists, no source_hashes or empty → False (unknown drift, ingest)
    - wiki page exists, hash matches, force=False → True (skip)
    - wiki page exists, hash matches, force=True → False (force reprocess)
    - wiki page exists, hash differs → False (overwrite)

    Returns:
        (should_skip, slugs_checked) where slugs_checked is the list of wiki
        slugs that were inspected (used for the ingest_skipped log payload).

    NOTE on Phase 5 lint: empty source_hashes is treated as "drift unknown" here
    (do NOT skip). Phase 5 lint must apply the same logic to avoid false-negative
    drift reports on Phase 6 legacy pages that have no source_hashes.
    """
    if force:
        return False, []

    slugs_checked: list[str] = []
    # Scan both concepts/ and entities/ subdirs for pages derived from this source
    for subdir_name in ("concepts", "entities"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            fm = read_existing_frontmatter(page_path)
            if fm is None:
                continue
            # Only consider pages derived from this source
            page_sources: list = fm.get("sources", [])
            citation_prefix = f"{source_name}#"
            if not any(str(s).startswith(citation_prefix) for s in page_sources):
                continue
            slug = page_path.stem
            slugs_checked.append(slug)
            # Read source_hashes; empty/missing = unknown drift → do NOT skip
            source_hashes = fm.get("source_hashes", {})
            if not source_hashes:
                return False, slugs_checked
            # Check if there's a hash entry for this source
            entry = source_hashes.get(source_name)
            if entry is None:
                return False, slugs_checked
            existing_docs_body = entry.get("docs_body")
            if existing_docs_body != docs_body_hash:
                # Hash differs → must ingest (overwrite)
                return False, slugs_checked
            # Hash matches for this page — continue checking others
            # (all must match to skip)
    # If we found at least one matching page and no mismatches → skip
    if slugs_checked:
        return True, slugs_checked
    # No pages found → fresh write
    return False, []


# ---------------------------------------------------------------------------
# Per-Source pipeline helpers (shared by sync and async paths)
# ---------------------------------------------------------------------------


def _synthesize_concept_drafts(sections: list) -> list:
    """Generate one WikiPageDraft per Section via the concept synthesis LLM call.

    Returns drafts in SECTION ORDER.  Slugs are NOT yet resolved for
    collision — callers must pass the result to ``_resolve_draft_slugs``.

    Args:
        sections: Parsed Section list for one concept Source.

    Returns:
        List of WikiPageDraft in the same order as ``sections``.
    """
    drafts = []
    for section in sections:
        section_draft = generate_page(section, "concept")
        drafts.append(section_draft)
    return drafts


def _verify_draft(draft, sections: list) -> tuple:
    """Run the grounding verifier on one WikiPageDraft.

    Returns:
        (draft, failed: bool) — draft may be mutated with
        status=failed_grounding + grounding_failure frontmatter when the
        verifier does not pass.  ``failed`` is True when the page must be
        added to IngestBatchResult.pages_with_failed_grounding.
    """
    grounding_outcome = verify(draft.body, sections)
    if grounding_outcome.passed:
        return draft, False

    reason = grounding_outcome.reason
    unsupported: list[str] = []
    if (
        grounding_outcome.result is not None
        and grounding_outcome.result.unsupported_claims
    ):
        unsupported = grounding_outcome.result.unsupported_claims

    # mypy cannot narrow grounding_outcome.reason (full 6-variant Literal)
    # to GroundingFailure.reason from the runtime guard above.
    gf = GroundingFailure(
        reason=reason,  # type: ignore[arg-type]
        unsupported_claims=unsupported,
    )
    failed_fm = draft.frontmatter.model_copy(
        update={"status": "failed_grounding", "grounding_failure": gf}
    )
    draft = draft.model_copy(update={"frontmatter": failed_fm})
    return draft, True


def _resolve_draft_slugs(drafts: list, sections: list, used_slugs: set) -> list:
    """Assign collision-resolved slugs to a list of drafts (in section order).

    Calls ``resolve_slug_collision(used_slugs, slugify(section.heading))`` for
    each draft in order, then returns a new list of drafts with ``slug`` and
    ``frontmatter.id`` updated.  Mutates ``used_slugs`` in place.

    Both the sync and async paths MUST use this same sequential post-pass so
    slug assignment is deterministic and identical across both paths.

    Args:
        drafts:     Drafts in section order (output of ``_synthesize_concept_drafts``
                    or the async gather step).
        sections:   The original Section list — provides the heading for slug
                    derivation in the same order as ``drafts``.
        used_slugs: The batch-level slug collision set; mutated by this call.

    Returns:
        New list of WikiPageDraft objects with resolved slugs.
    """
    resolved = []
    for draft, section in zip(drafts, sections):
        raw_slug = slugify(section.heading)
        final_slug = resolve_slug_collision(used_slugs, raw_slug)
        updated_fm = draft.frontmatter.model_copy(update={"id": final_slug})
        resolved.append(
            draft.model_copy(update={"slug": final_slug, "frontmatter": updated_fm})
        )
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_sources(
    source_filenames: list[str] | None,
    *,
    docs_dir: Path | None = None,
    wiki_dir: Path | None = None,
    force: bool = False,
) -> IngestBatchResult:
    """Ingest one or more Sources and write synthesis pages to wiki/.

    Args:
        source_filenames: List of bare filenames to ingest (e.g.
            ``["refund_policy.md"]``).  Pass ``None`` to batch-ingest ALL
            Sources found under docs_dir (glob ``"**/*.md"``).
        docs_dir: Override the docs directory (used by tests).
        wiki_dir: Override the wiki directory (used by tests).
        force: When True, bypass hash-skip idempotency and re-ingest even if
            the docs_body_hash matches the existing wiki frontmatter. Default
            False. Corresponds to IngestRequest.force (Phase 3 amendment #93).

    Returns:
        IngestBatchResult with per-Source outcomes.
    """
    from . import indexer as _indexer_module

    if docs_dir is None:
        docs_dir = DOCS_DIR

    # Resolve wiki_dir early so orphan/created lookups can use it.
    # Import at call time to pick up monkeypatched WIKI_DIR in tests.
    resolved_wiki_dir: Path = wiki_dir if wiki_dir is not None else _indexer_module.WIKI_DIR

    # Batch mode: discover all Sources under docs_dir
    if source_filenames is None:
        # Pairs of (bare_filename, absolute_path)
        source_pairs: list[tuple[str, Path]] = _resolve_docs_files(docs_dir)
    else:
        # Single-source mode: each entry is a bare filename; look it up directly.
        # For flat docs/ this is just docs_dir/filename.  Nested paths are not
        # supported in single-source mode (caller must pass bare filenames).
        source_pairs = [(name, docs_dir / name) for name in source_filenames]

    batch = IngestBatchResult()
    # Slug collision tracking: single global set so cross-type collisions
    # (e.g. entity "foo" + concept "foo") are also detected.  The second
    # source to claim a slug — regardless of type — receives the -2 suffix.
    # This makes bare-slug citations globally unambiguous (Slice 4-3b, #54).
    used_slugs: set[str] = set()

    # --- ingest_batch_started ---
    log_event(
        "ingest_batch_started",
        f"sources={len(source_pairs)}",
    )

    for source_name, source_path in source_pairs:
        if not source_path.exists():
            log_event(
                "ingest_error",
                f"source={source_name} error=source_not_found",
            )
            batch.failed_sources.append(source_name)
            continue

        try:
            sections = parse_markdown(source_path)
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_error",
                f"source={source_name} error={type(exc).__name__}:parse_error",
            )
            batch.failed_sources.append(source_name)
            continue

        if not sections:
            log_event(
                "ingest_error",
                f"source={source_name} error=no_sections",
            )
            batch.failed_sources.append(source_name)
            continue

        # Per-section HARD token cap — fail fast before any LLM call.
        #
        # classify_source operates on build_outline(content), which bounds the
        # classifier's input, but the entity synthesis path still sends each
        # section to the LLM individually.  A section that is individually
        # oversized would blow the synthesis call's context window.  Reject the
        # whole Source with a clear reason; the batch continues.
        section_cap = _max_section_tokens()
        oversized_section: str | None = None
        for _sec in sections:
            if _estimate_tokens(_sec.content) > section_cap:
                oversized_section = _sec.heading
                break
        if oversized_section is not None:
            log_event(
                "ingest_error",
                f"source={source_name} error=section_too_large heading={oversized_section[:60]!r}",
            )
            batch.failed_sources.append(source_name)
            batch.failed_reasons[source_name] = (
                f"Source too large to ingest: section {oversized_section!r} "
                f"exceeds the {section_cap}-token per-section limit "
                f"(KB_INGEST_MAX_SECTION_TOKENS). Split the Source into smaller "
                f"files or shorten the oversized section."
            )
            continue

        # Step 3 (Phase 3 amendment #93): compute docs_body_hash and check
        # for hash-skip idempotency before making any LLM calls.
        #
        # docs_body_hash = sha256(source content as UTF-8 text).
        # If an existing wiki page has source_hashes[source_name]["docs_body"]
        # matching this hash AND force=False → skip (no LLM call).
        # Empty/missing source_hashes → treat as "unknown drift" → do NOT skip.
        docs_body_hash = _compute_docs_body_hash(source_path)
        should_skip, slugs_checked = _should_skip_source(
            source_name, docs_body_hash, resolved_wiki_dir, force
        )
        if should_skip:
            log_event(
                "ingest_skipped",
                f"source={source_name} slugs_checked={slugs_checked} docs_body_hash={docs_body_hash}",
            )
            batch.skipped_sources.append(
                IngestSourceResult(
                    source=source_name,
                    pages_written=[],
                    status="skipped",
                )
            )
            continue

        # Step 4: classify the Source
        #
        # Strip the leading YAML frontmatter before handing the text to the
        # classifier LLM (issue #106): after Phase 7, imported docs/*.md carry
        # provenance frontmatter (imported_from/original_format/imported_at/
        # content_sha256) that is metadata ABOUT the file, not content OF the
        # Source. Feeding it raw lets the LLM misread provenance as Source facts.
        # The synthesis path is already clean — it consumes Section.content from
        # parse_markdown, which strips frontmatter per Rule 2.
        #
        # LLMError is re-raised (ADR-0015): it is a domain exception, not a
        # recoverable source-level failure.  Callers (HTTP route, MCP adapter,
        # CLI adapter) catch LLMError and map it to their transport representation.
        try:
            raw_source_text = source_path.read_text(encoding="utf-8")
            _, source_content = split_frontmatter(raw_source_text)
            source_type = classify_source(source_content)
            batch._llm_call_count += 1
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_error",
                f"source={source_name} error={type(exc).__name__}:classify_failed",
            )
            batch.failed_sources.append(source_name)
            continue

        # Step 5: generate draft(s)
        try:
            if source_type == "entity":
                source_stem = Path(source_name).stem
                if _should_route_async(source_content):
                    # Large entity: per-section synthesis (one "entity" page per Section).
                    # Avoids sending the whole concatenated Source to a single LLM call,
                    # which would overflow the context window for documents above the
                    # soft cap.  The async routing seam (_should_route_async) is wired
                    # here for Fix 2; Fix 1b will add the scheduler path.
                    # NOTE: each per-section page therefore cites only its own Section
                    # (section.file#slug), not the whole-Source citation list that the
                    # single-page entity branch collapses below — an intentional
                    # consequence of splitting one entity into N pages, not a bug.
                    drafts = []
                    for section in sections:
                        raw_slug = slugify(section.heading)
                        final_slug = resolve_slug_collision(used_slugs, raw_slug)
                        section_draft = generate_page(section, "entity")
                        batch._llm_call_count += 1
                        updated_fm = section_draft.frontmatter.model_copy(
                            update={"id": final_slug}
                        )
                        section_draft = section_draft.model_copy(
                            update={"slug": final_slug, "frontmatter": updated_fm}
                        )
                        drafts.append(section_draft)
                else:
                    # Normal-size entity: single page collapsing the whole Source.
                    raw_slug = slugify(source_stem)
                    final_slug = resolve_slug_collision(used_slugs, raw_slug)
                    draft = generate_entity_page(
                        sections,
                        source_stem=source_stem,
                        source_filename=source_name,
                    )
                    # Override the slug with the collision-resolved one
                    draft = draft.model_copy(update={"slug": final_slug})
                    drafts = [draft]
                    batch._llm_call_count += 1
            else:
                # concept: 1:N — one page per Section via shared helper.
                # _synthesize_concept_drafts returns drafts in section order
                # with unresolved slugs; _resolve_draft_slugs assigns
                # collision-resolved slugs deterministically.
                raw_drafts = _synthesize_concept_drafts(sections)
                batch._llm_call_count += len(raw_drafts)
                drafts = _resolve_draft_slugs(raw_drafts, sections, used_slugs)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_error",
                f"source={source_name} error={type(exc).__name__}:generate_failed",
            )
            batch.failed_sources.append(source_name)
            continue

        # Step 6: Preserve `created` timestamp for pages that already exist.
        #
        # For each draft whose target file already exists on disk, read the
        # existing frontmatter and carry forward the original `created` value.
        # The `updated` value is set by the LLM/templates layer and is NOT
        # overridden here — it represents "now" from the caller's perspective.
        drafts_with_preserved_timestamps: list = []
        for draft in drafts:
            subdir_name = "entities" if draft.frontmatter.type == "entity" else "concepts"
            page_path = resolved_wiki_dir / subdir_name / f"{draft.slug}.md"
            existing_fm = read_existing_frontmatter(page_path)
            if existing_fm is not None and "created" in existing_fm:
                # Preserve the original created timestamp
                preserved_fm = draft.frontmatter.model_copy(
                    update={"created": existing_fm["created"]}
                )
                draft = draft.model_copy(update={"frontmatter": preserved_fm})
            drafts_with_preserved_timestamps.append(draft)

        # Step 7 (Phase 3 amendment #93): populate source_hashes in each draft's
        # frontmatter. This is done AFTER slug resolution and created-preservation
        # so source_hashes carries the correct hash for the current ingest run.
        #
        # source_hashes[source_name] = {
        #     "raw":       content_sha256 from docs frontmatter (or null),
        #     "docs_body": sha256(source_path.read_text('utf-8').encode()),
        # }
        #
        # `raw` is the hash written by importer.py (Phase 7-3) for the original
        # raw bytes. Hand-authored docs that never passed through /import have no
        # content_sha256 in their frontmatter — record null in that case.
        #
        # All Sections from a Source share the same docs frontmatter metadata
        # (indexer.parse_markdown attaches the file's YAML to every Section), so
        # reading sections[0].metadata is sufficient.
        raw_hash: str | None = None
        if sections:
            raw_hash = sections[0].metadata.get("content_sha256")

        source_hashes_entry: dict[str, str | None] = {
            "raw": raw_hash,
            "docs_body": docs_body_hash,
        }
        new_source_hashes: dict[str, dict[str, str | None]] = {source_name: source_hashes_entry}

        drafts_with_source_hashes: list = []
        for draft in drafts_with_preserved_timestamps:
            updated_fm = draft.frontmatter.model_copy(update={"source_hashes": new_source_hashes})
            drafts_with_source_hashes.append(draft.model_copy(update={"frontmatter": updated_fm}))

        # Step 9: Run grounding verifier on each draft (ADR-0004 fail-soft).
        #
        # Verifier uses OPENAI_VERIFIER_MODEL (independent of OPENAI_INGEST_MODEL).
        # On claim_supported: status stays "live".
        # On claim_unsupported or verifier_unavailable: status="failed_grounding"
        # + grounding_failure frontmatter block.  Page is still written (fail-soft).
        # CitableContent Protocol: each draft's body acts as the "draft answer";
        # sections from the source act as the "citable content".
        drafts_with_grounding: list = []
        for draft in drafts_with_source_hashes:
            draft, grounding_failed = _verify_draft(draft, sections)
            drafts_with_grounding.append(draft)
            if grounding_failed:
                gf = draft.frontmatter.grounding_failure
                reason_str = gf.reason if gf else "unknown"
                claims_str = gf.unsupported_claims if gf else []
                log_event(
                    "ingest_grounding_failed",
                    f"page={draft.slug} reason={reason_str} claims={claims_str}",
                )
                batch.pages_with_failed_grounding.append(draft.slug)

        # Step 10+11: Under the index lock, delete orphans then write pages.
        #
        # Orphan scope is per-Source: only pages derived from source_name are
        # considered.  current_page_ids is the set of slugs in the target set.
        current_page_ids = {d.slug for d in drafts_with_grounding}

        with _index_lock:
            deleted = delete_orphans(
                source_name,
                current_page_ids,
                wiki_dir=resolved_wiki_dir,
            )
            write_result = write_pages_for_source(
                source_name,
                drafts_with_grounding,
                wiki_dir=resolved_wiki_dir,
            )

        if write_result.errors:
            slug, err_msg = write_result.errors[0]
            # err_msg is formatted by wiki_writer as "<ClassName>: <message>" so it
            # already carries the real exception type; drop the previous
            # type(Exception()).__name__ literal (which always resolved to
            # "Exception") and surface the message instead.
            log_event(
                "ingest_error",
                f"source={source_name} error=write_error:{slug} detail={err_msg}",
            )
            batch.failed_sources.append(source_name)
            continue

        # --- ingest_source (only on success) ---
        log_event(
            "ingest_source",
            f"source={source_name} type={source_type}"
            f" pages_created={len(write_result.pages_created)}"
            f" pages_updated={len(write_result.pages_updated)}"
            f" pages_deleted={len(deleted)}",
        )
        # Determine status for this IngestSourceResult: 'created' if any pages
        # were freshly written, 'updated' if existing pages were overwritten.
        # If both occurred (multi-section concept source), use 'updated' to
        # indicate at least one page was not fresh.
        result_status = "updated" if write_result.pages_updated else "created"

        batch.results.append(
            IngestSourceResult(
                source=source_name,
                pages_written=write_result.pages_written,
                pages_created=write_result.pages_created,
                pages_updated=write_result.pages_updated,
                pages_deleted=deleted,
                status=result_status,
            )
        )

    total_pages = sum(len(r.pages_written) for r in batch.results)
    failed_grounding = len(batch.pages_with_failed_grounding)

    # --- ingest_batch_completed ---
    log_event(
        "ingest_batch_completed",
        f"sources={len(source_pairs)}"
        f" total_pages={total_pages}"
        f" llm_calls={batch._llm_call_count}"
        f" cost_usd=0.00"
        f" failed_grounding={failed_grounding}",
    )

    return batch
