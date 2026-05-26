"""Ingest coordinator — Source → wiki synthesis page pipeline.

Provides `ingest_sources(source_filenames)` which orchestrates the ingest
pipeline for one or more Sources:

    1. Resolve Source path under docs/
    2. Parse the Source into Sections via indexer.parse_markdown
    3. Synthesise a wiki page for the first Section via templates.generate_page
       (Slice #1 hardcode — Slice #2 adds 1:N expansion for all Sections)
    4. Write the page(s) to wiki/concepts/ via wiki_writer.write_pages_for_source
    5. Return an IngestBatchResult summarising outcomes

Hardcodes in this slice (per Slice #1 spec):
- `type=concept` for any Source (no classifier call — Slice #2)
- Only the first Section produces a page (Slice #2 adds full expansion)
- `status=live` always (Slice #4 introduces `failed_grounding`)
- Batch mode (source_filenames=None) raises NotImplementedError — Slice #2

See PRD #28 for the full pipeline design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .indexer import DOCS_DIR, parse_markdown
from .logger import log_event
from .schemas import IngestSourceResult
from .templates import generate_page
from .wiki_writer import write_pages_for_source

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
# Public API
# ---------------------------------------------------------------------------


def ingest_sources(
    source_filenames: list[str] | None,
    *,
    docs_dir: Path | None = None,
    wiki_dir: Path | None = None,
) -> IngestBatchResult:
    """Ingest one or more Sources and write synthesis pages to wiki/concepts/.

    Args:
        source_filenames: List of bare filenames to ingest (e.g.
            ``["refund_policy.md"]``).  ``None`` triggers batch mode, which is
            not yet implemented in this slice.
        docs_dir: Override the docs directory (used by tests).
        wiki_dir: Override the wiki directory (used by tests).

    Returns:
        IngestBatchResult with per-Source outcomes.

    Raises:
        NotImplementedError: When ``source_filenames`` is ``None`` (batch mode
            deferred to Slice #2).
    """
    if source_filenames is None:
        raise NotImplementedError(
            "Batch mode (ingest all docs/) is not yet implemented; "
            "provide source_filenames=[...] for a single-source ingest."
        )

    if docs_dir is None:
        docs_dir = DOCS_DIR

    batch = IngestBatchResult()

    for source_name in source_filenames:
        source_path = docs_dir / source_name
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

        # Slice #1 hardcode: only the first Section produces a page.
        # Slice #2 will iterate all sections for the 1:N concept expansion.
        first_section = sections[0]

        try:
            # Slice #1 hardcode: type="concept" — Slice #2 adds classify_source().
            draft = generate_page(first_section, "concept")
        except Exception as exc:  # noqa: BLE001
            log_event(
                "ingest_llm_error",
                f"source={source_name} section={first_section.id} exc={type(exc).__name__}",
            )
            batch.failed_sources.append(source_name)
            continue

        write_result = write_pages_for_source(
            source_name,
            [draft],
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
            f"source={source_name} pages={len(write_result.pages_written)}",
        )
        batch.results.append(
            IngestSourceResult(
                source=source_name,
                pages_written=write_result.pages_written,
            )
        )

    return batch
