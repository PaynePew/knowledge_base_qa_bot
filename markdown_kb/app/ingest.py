"""Deep module per Ousterhout. Public surface: ``ingest_sources``.

Ingest coordinator — Source → wiki synthesis page pipeline.

Provides `ingest_sources(source_filenames)` which orchestrates the ingest
pipeline for one or more Sources:

    1. Resolve Source path(s) under docs/ (batch: glob("**/*.md"))
    2. Parse each Source into Sections via indexer.parse_markdown
    3. Classify Source type via templates.classify_source
    4a. concept → one WikiPageDraft per Section (1:N expansion)
    4b. entity  → one WikiPageDraft collapsing the whole Source
    5. Resolve slug collisions across Sources (-2, -3 suffix)
    6. Preserve `created` timestamp for pages that already exist on disk
    7. Run grounding verifier on each generated page (Slice #4 — fail-soft)
    8. Delete orphan pages (per-Source scoped) via wiki_writer.delete_orphans
    9. Write pages via wiki_writer.write_pages_for_source
    10. Return an IngestBatchResult summarising outcomes with meaningful
        pages_created / pages_updated / pages_deleted per Source

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

Wiki Log (Slice #4 — five new kind values):
- ingest_batch_started / ingest_batch_completed bracket the whole batch.
- ingest_source emitted per successful Source.
- ingest_grounding_failed emitted per page with failed grounding.
- ingest_error emitted per Source-level failure (replaces prior ad-hoc kinds).

See PRD #28 for the full pipeline design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._paths import DOCS_DIR
from .grounding import verify
from .indexer import _index_lock, parse_markdown, slugify
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
    `_llm_call_count` tracks total LLM calls for the batch_completed log entry.
    """

    results: list[IngestSourceResult] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)
    pages_with_failed_grounding: list[str] = field(default_factory=list)
    _llm_call_count: int = 0


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_sources(
    source_filenames: list[str] | None,
    *,
    docs_dir: Path | None = None,
    wiki_dir: Path | None = None,
) -> IngestBatchResult:
    """Ingest one or more Sources and write synthesis pages to wiki/.

    Args:
        source_filenames: List of bare filenames to ingest (e.g.
            ``["refund_policy.md"]``).  Pass ``None`` to batch-ingest ALL
            Sources found under docs_dir (glob ``"**/*.md"``).
        docs_dir: Override the docs directory (used by tests).
        wiki_dir: Override the wiki directory (used by tests).

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

        # Step 3: classify the Source
        try:
            source_content = source_path.read_text(encoding="utf-8")
            source_type = classify_source(source_content)
            batch._llm_call_count += 1
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_error",
                f"source={source_name} error={type(exc).__name__}:classify_failed",
            )
            batch.failed_sources.append(source_name)
            continue

        # Step 4: generate draft(s)
        try:
            if source_type == "entity":
                source_stem = Path(source_name).stem
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
                # concept: 1:N — one page per Section
                drafts = []
                for section in sections:
                    raw_slug = slugify(section.heading)
                    final_slug = resolve_slug_collision(used_slugs, raw_slug)
                    section_draft = generate_page(section, "concept")
                    batch._llm_call_count += 1
                    # Override slug and frontmatter.id with collision-resolved value
                    updated_fm = section_draft.frontmatter.model_copy(update={"id": final_slug})
                    section_draft = section_draft.model_copy(
                        update={"slug": final_slug, "frontmatter": updated_fm}
                    )
                    drafts.append(section_draft)
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_error",
                f"source={source_name} error={type(exc).__name__}:generate_failed",
            )
            batch.failed_sources.append(source_name)
            continue

        # Step 5: Preserve `created` timestamp for pages that already exist.
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

        # Step 6: Run grounding verifier on each draft (ADR-0004 fail-soft).
        #
        # Verifier uses OPENAI_VERIFIER_MODEL (independent of OPENAI_INGEST_MODEL).
        # On claim_supported: status stays "live".
        # On claim_unsupported or verifier_unavailable: status="failed_grounding"
        # + grounding_failure frontmatter block.  Page is still written (fail-soft).
        # CitableContent Protocol: each draft's body acts as the "draft answer";
        # sections from the source act as the "citable content".
        drafts_with_grounding: list = []
        for draft in drafts_with_preserved_timestamps:
            grounding_outcome = verify(draft.body, sections)
            if grounding_outcome.passed:
                # claim_supported — keep status=live
                drafts_with_grounding.append(draft)
            else:
                # claim_unsupported or verifier_unavailable — fail-soft
                reason = grounding_outcome.reason
                unsupported: list[str] = []
                if (
                    grounding_outcome.result is not None
                    and grounding_outcome.result.unsupported_claims
                ):
                    unsupported = grounding_outcome.result.unsupported_claims

                # mypy cannot narrow grounding_outcome.reason (full 6-variant Literal)
                # to GroundingFailure.reason ({"claim_unsupported", "verifier_unavailable"})
                # from the runtime `not grounding_outcome.passed` guard above — the
                # narrowing is provable from the verify() implementation but invisible
                # to the static checker.
                gf = GroundingFailure(
                    reason=reason,  # type: ignore[arg-type]
                    unsupported_claims=unsupported,
                )
                failed_fm = draft.frontmatter.model_copy(
                    update={"status": "failed_grounding", "grounding_failure": gf}
                )
                draft = draft.model_copy(update={"frontmatter": failed_fm})
                drafts_with_grounding.append(draft)

                log_event(
                    "ingest_grounding_failed",
                    f"page={draft.slug} reason={reason} claims={unsupported}",
                )
                batch.pages_with_failed_grounding.append(draft.slug)

        # Step 7+8: Under the index lock, delete orphans then write pages.
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
        batch.results.append(
            IngestSourceResult(
                source=source_name,
                pages_written=write_result.pages_written,
                pages_created=write_result.pages_created,
                pages_updated=write_result.pages_updated,
                pages_deleted=deleted,
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
