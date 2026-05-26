"""Ingest coordinator — Source → wiki synthesis page pipeline.

Provides `ingest_sources(source_filenames)` which orchestrates the ingest
pipeline for one or more Sources:

    1. Resolve Source path(s) under docs/ (batch: glob("**/*.md"))
    2. Parse each Source into Sections via indexer.parse_markdown
    3. Classify Source type via templates.classify_source
    4a. concept → one WikiPageDraft per Section (1:N expansion)
    4b. entity  → one WikiPageDraft collapsing the whole Source
    5. Resolve slug collisions across Sources (-2, -3 suffix)
    6. Write pages via wiki_writer.write_pages_for_source
    7. Return an IngestBatchResult summarising outcomes

Continue-on-error: a Source that throws at any stage is recorded in
`failed_sources` but does not stop the batch (Q3 grill decision).

Hardcodes still in place (per Slice #2 spec):
- `status=live` always (Slice #4 introduces `failed_grounding`)
- No verifier call (Slice #4 adds Grounding Check)
- No red link rules in prompt (Slice #4)
- No ingest log event kinds beyond existing ones (Slice #4 adds 5)

See PRD #28 for the full pipeline design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .indexer import DOCS_DIR, parse_markdown
from .logger import log_event
from .schemas import IngestSourceResult
from .templates import classify_source, generate_entity_page, generate_page
from .wiki_writer import resolve_slug_collision, write_pages_for_source

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestBatchResult:
    """Aggregated outcome of ingest_sources.

    `results` lists one IngestSourceResult per successfully-processed Source.
    `failed_sources` lists bare filenames that failed (Source not found, parse
    error, LLM call failure, or write failure).
    """

    results: list[IngestSourceResult] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)


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
    if docs_dir is None:
        docs_dir = DOCS_DIR

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
    # Slug collision tracking: "concepts/overview" and "entities/overview" are
    # tracked separately so entity slugs don't collide with concept slugs.
    used_slugs: dict[str, set[str]] = {
        "concept": set(),
        "entity": set(),
    }

    for source_name, source_path in source_pairs:
        if not source_path.exists():
            log_event(
                "ingest_source_not_found",
                f"source={source_name}",
            )
            batch.failed_sources.append(source_name)
            continue

        try:
            sections = parse_markdown(source_path)
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_parse_error",
                f"source={source_name} exc={type(exc).__name__}",
            )
            batch.failed_sources.append(source_name)
            continue

        if not sections:
            log_event(
                "ingest_no_sections",
                f"source={source_name}",
            )
            batch.failed_sources.append(source_name)
            continue

        # Step 3: classify the Source
        try:
            source_content = source_path.read_text(encoding="utf-8")
            source_type = classify_source(source_content)
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_llm_error",
                f"source={source_name} stage=classify exc={type(exc).__name__}",
            )
            batch.failed_sources.append(source_name)
            continue

        # Step 4: generate draft(s)
        try:
            if source_type == "entity":
                from .indexer import slugify

                source_stem = Path(source_name).stem
                raw_slug = slugify(source_stem)
                final_slug = resolve_slug_collision(used_slugs["entity"], raw_slug)
                draft = generate_entity_page(
                    sections,
                    source_stem=source_stem,
                    source_filename=source_name,
                )
                # Override the slug with the collision-resolved one
                draft = draft.model_copy(update={"slug": final_slug})
                drafts = [draft]
            else:
                # concept: 1:N — one page per Section
                from .indexer import slugify

                drafts = []
                for section in sections:
                    raw_slug = slugify(section.heading)
                    final_slug = resolve_slug_collision(used_slugs["concept"], raw_slug)
                    section_draft = generate_page(section, "concept")
                    # Override slug and frontmatter.id with collision-resolved value
                    updated_fm = section_draft.frontmatter.model_copy(update={"id": final_slug})
                    section_draft = section_draft.model_copy(
                        update={"slug": final_slug, "frontmatter": updated_fm}
                    )
                    drafts.append(section_draft)
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_llm_error",
                f"source={source_name} stage=generate exc={type(exc).__name__}",
            )
            batch.failed_sources.append(source_name)
            continue

        # Step 5+6: write pages
        write_result = write_pages_for_source(
            source_name,
            drafts,
            wiki_dir=wiki_dir,
        )

        if write_result.errors:
            slug, err_msg = write_result.errors[0]
            log_event(
                "ingest_write_error",
                f"source={source_name} slug={slug} err={err_msg}",
            )
            batch.failed_sources.append(source_name)
            continue

        log_event(
            "ingest_complete",
            f"source={source_name} type={source_type} pages={len(write_result.pages_written)}",
        )
        batch.results.append(
            IngestSourceResult(
                source=source_name,
                pages_written=write_result.pages_written,
                pages_created=write_result.pages_written,
            )
        )

    return batch
