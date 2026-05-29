"""Deep module per Ousterhout. Public surface: ``run_lint``, ``_check_c11_orphan``, ``_check_c3_failed_grounding``, ``_check_c4a_slug_collision``, ``_check_c6_stale``, ``_check_c2_red_links``, ``_check_c1_coverage_gaps``, ``_canonicalise``, ``_candidate_pairs``, ``_judge_page_pair``, ``_check_c5_page_pair``, ``_load_wiki_pages``, ``get_lint_llm``, ``_check_c8_promotion_candidates``, ``_check_c9_qa_staleness``, ``_check_c10_qa_schema_validity``.

Lint orchestrator — POST /lint health check for the wiki.

Provides ``run_lint(*, wiki_dir, docs_dir, log_path)`` which orchestrates the
lint checks and writes ``wiki/lint-report.md``.

Slice 5-1 scope
---------------
C11 (orphan pages) is wired.

Slice 5-2 scope
---------------
C3 (failed-grounding sweep) and C4-a (slug collision groups) are added.
Subsequent slices add the remaining checks without changing the orchestrator
contract or the continue-on-error semantics established here.

Slice 5-3 scope
---------------
C6 (mtime-based stale detection) and C2 (red link backlog) are wired.

Slice 5-4 scope
---------------
C1 (coverage gap aggregation from chat_fallback log) is wired.  Reads
``wiki/log.md`` line by line, clusters ``chat_fallback`` and
``chat_grounding_fallback`` entries by reason and canonical query key, and
surfaces the result as a curator-actionable coverage backlog.

Slice 5-5 scope
---------------
C5 (page-pair LLM contradiction detection) is wired.  Uses F1 ∪ F3 candidate
filter before calling the LLM via ``get_lint_llm()`` lazy singleton.
``OPENAI_LINT_MODEL`` env var with 3-layer fallback.  ``temperature=0`` for
determinism.  ``KB_LINT_BM25_TOP_K`` and ``KB_LINT_BM25_THRESHOLD`` control the
F3 filter.  Pairs are symmetrically canonicalised (sorted slug names) so each
pair is judged exactly once.  Continue-on-error: LLM errors mid-batch retain
prior findings.  Cost accounting via response metadata (best-effort token counts).

Slice 6-5 scope (Phase 5 amendment — PRD #78)
---------------------------------------------
Three new read-only checks scan ``wiki/qa/*.md`` and one modifier excludes
``type=qa`` pages from C5 pair generation:

- **C5 modifier** — ``_candidate_pairs`` filters ``frontmatter.type == "qa"``
  BEFORE F1/F3 candidate computation, preserving C5 LLM call budget and
  preventing trivially-true ``duplicate`` findings between qa pages and their
  source entities (PRD #78 Q1 + Q6).
- **C8 promotion candidates** — surfaces ``status: draft`` Filed Answers ranked
  by ``count`` desc / ``updated`` desc to ``lint-report.md`` §
  ``## Promotion Candidates``. Capped by ``KB_LINT_PROMOTION_TOP_N`` env var
  (default 10). Read-only — the actual draft→live mutation is owned by
  Phase 6 ``POST /qa/{slug}/promote``.
- **C9 qa-staleness** — for each ``status: live`` Filed Answer, compares each
  cited entity file's mtime against ``frontmatter.updated``. Newer entities
  surface to ``## Stale Filed Answers``. Closes Q6b "entity re-ingested, qa
  stranded" failure mode.
- **C10 qa-schema-validity** — sweeps ``wiki/qa/`` for invalid frontmatter
  (``status`` outside ``{live, draft, stale, superseded}``, missing/empty
  ``question``, ``type != "qa"``, missing/non-positive ``count``). Surfaces to
  ``## Invalid qa Schema``. Closes Q8d "curator-typo orphan zombie" failure
  mode (third layer of the indexer-log + filing-refuse + lint defence stack).

All four amendments preserve PRD #65 Q3 read-only invariant — they read
frontmatter and write only ``lint-report.md``, never page frontmatter.

Read-only invariant
-------------------
``run_lint()`` does NOT modify wiki page frontmatter.  It writes only:
  - ``wiki/lint-report.md``   (Generated index artifact, gitignored)
  - ``wiki/log.md``           (Runtime trace, append-only)

Concurrency
-----------
``run_lint()`` holds ``indexer._index_lock`` for the full duration so a
concurrent ``/ingest`` (which also holds the lock) cannot produce mid-write
state for lint to observe.  ``/chat`` reads are not blocked.

Continue-on-error
-----------------
If a check raises, the exception is caught, a ``lint_check_error`` log entry is
written, and the error is recorded in ``LintResponse.check_errors``.  Other
checks still run.  The report is always written.

Check execution order (cheapest to most expensive)
---------------------------------------------------
1. C11 orphan (read frontmatter only)
2. C3 failed-grounding (read frontmatter only)
3. C4-a slug collision (filename list only)
4. C6 stale pages (stat docs/ files)
5. C2 red links (scan wiki page bodies)
6. C1 coverage gap (read log.md only)
7. C8 promotion candidates (read qa frontmatter)
8. C9 qa-staleness (stat entity files vs qa updated)
9. C10 qa-schema-validity (read qa frontmatter)
10. C5 page-pair LLM (F1∪F3 filter + LLM) — most expensive last

Authorised by PRD #65 (Phase 5), GitHub issue #66 (Slice 5-1), GitHub issue #67 (Slice 5-2), GitHub issue #68 (Slice 5-3), GitHub issue #69 (Slice 5-4), GitHub issue #70 (Slice 5-5), PRD #78 (Phase 6), and GitHub issue #82 (Slice 6-5 Phase 5 amendment).
"""

from __future__ import annotations

import datetime
import os
import re
import string
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ._paths import DOCS_DIR, WIKI_DIR
from .indexer import _index_lock
from .logger import LOG_PATH, log_event
from .schemas import (
    CoverageGapFinding,
    FailedGroundingFinding,
    InvalidQaSchemaFinding,
    LintFindings,
    LintResponse,
    LintSummary,
    OrphanPageFinding,
    PagePairFinding,
    PromotionCandidateFinding,
    QaStalenessFinding,
    RedLinkFinding,
    SlugCollisionFinding,
    StalePageFinding,
)

# ---------------------------------------------------------------------------
# C5 — Lazy LLM singleton (ADR-0005 pattern)
# ---------------------------------------------------------------------------

# Module-level sentinel; monkeypatched in tests.
_lint_llm = None

# Best-effort LLM call counter for cost accounting. Uses a mutable list so it
# can be incremented inside _judge_page_pair and reset in run_lint without
# relying on global statement (consistent with existing module-level patterns).
# Index 0 holds the current count; run_lint reads and resets it after C5 runs.
_c5_llm_call_counter: list[int] = [0]


def get_lint_llm():
    """Return the lazy singleton ChatOpenAI for C5 page-pair contradiction detection.

    Model resolution (three-layer fallback, mirroring OPENAI_INGEST_MODEL):
        OPENAI_LINT_MODEL  →  OPENAI_MODEL  →  "gpt-4o-mini"

    temperature=0 for determinism (structured output, reproducible runs).
    """
    global _lint_llm
    if _lint_llm is None:
        from langchain_openai import ChatOpenAI

        model_name = os.getenv(
            "OPENAI_LINT_MODEL",
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
        _lint_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=60,
            max_retries=1,
        )
    return _lint_llm


# Regex that matches a trailing ``-N`` suffix where N is an integer >= 2.
# Used by C4-a to strip the ingest-appended collision suffix from a slug.
_COLLISION_SUFFIX_RE = re.compile(r"^(.+)-(\d+)$")

# ---------------------------------------------------------------------------
# Module-level default paths (monkeypatched in tests)
# ---------------------------------------------------------------------------

# Re-exported so tests can monkeypatch ``app.lint.WIKI_DIR`` and ``app.lint.DOCS_DIR``
# using the same setattr pattern as the parent conftest's ``_redirect_paths_to_tmp``.
# These must be module attributes (not local variables) for monkeypatch to work.

# WIKI_DIR and DOCS_DIR are imported from ._paths above.
# LOG_PATH is imported from .logger above.


# ---------------------------------------------------------------------------
# C11 — Orphan page detection
# ---------------------------------------------------------------------------


def _check_c11_orphan(
    wiki_dir: Path,
    docs_dir: Path,
) -> list[OrphanPageFinding]:
    """Return orphan findings for every wiki page with a missing source file.

    A wiki page is an *orphan* when at least one entry in ``frontmatter.sources``
    references a file (the portion before ``#``) that does not exist anywhere
    under ``docs_dir`` (including nested subdirectories, per ``glob("**/*.md")``).

    The check reads ``wiki/entities/*.md`` and ``wiki/concepts/*.md``.

    Algorithm:
    1. Collect the set of all source filenames (stems with extension) present
       under ``docs_dir`` using ``glob("**/*.md")``.
    2. For each wiki page in ``entities/`` and ``concepts/``, parse its YAML
       frontmatter and read the ``sources`` list.
    3. For each source citation ``<file>#<anchor>``, extract the file portion.
       If the filename is NOT in the docs set, add it to ``missing_sources``.
    4. If any sources are missing, emit one ``OrphanPageFinding`` per page.
    5. Return findings sorted alphabetically by ``page_slug``.

    Only the basename of each source file is matched against the docs glob
    results.  This matches the pattern used by ``/ingest`` (which references
    sources as bare filenames, e.g. ``refund_policy.md#cancellation-window``).
    """
    # Build set of all source file basenames under docs_dir
    docs_filenames: set[str] = {p.name for p in docs_dir.glob("**/*.md")}

    findings: list[OrphanPageFinding] = []

    # Scan both wiki subdirs
    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            slug = page_path.stem
            sources = _read_frontmatter_sources(page_path)
            if not sources:
                continue
            missing: list[str] = []
            for citation in sources:
                # citation format: "filename.md#anchor"  or just "filename.md"
                file_part = citation.split("#")[0].strip()
                if not file_part:
                    continue
                basename = Path(file_part).name
                if basename not in docs_filenames:
                    missing.append(basename)
            if missing:
                # Deduplicate while preserving order
                seen: set[str] = set()
                deduped: list[str] = []
                for m in missing:
                    if m not in seen:
                        seen.add(m)
                        deduped.append(m)
                findings.append(
                    OrphanPageFinding(
                        page_slug=slug,
                        missing_sources=deduped,
                        suggested_action=(
                            f"The source(s) {deduped!r} referenced by '{slug}' no longer "
                            f"exist under docs/. If the Source was renamed, update this "
                            f"page's frontmatter sources field and re-ingest. If the Source "
                            f"was deleted, delete this wiki page as it has no valid source."
                        ),
                    )
                )

    findings.sort(key=lambda f: f.page_slug)
    return findings


def _read_frontmatter_sources(page_path: Path) -> list[str]:
    """Parse the YAML frontmatter of a wiki page and return the ``sources`` list.

    Returns an empty list if the page has no frontmatter, the frontmatter
    cannot be parsed, or the ``sources`` field is absent/empty.
    """
    try:
        text = page_path.read_text(encoding="utf-8")
    except OSError:
        return []

    if not text.startswith("---"):
        return []

    # Extract frontmatter block between first --- and second ---
    parts = text.split("---", 2)
    if len(parts) < 3:
        return []

    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return []

    if not isinstance(fm, dict):
        return []

    sources = fm.get("sources", [])
    if not isinstance(sources, list):
        return []
    return [str(s) for s in sources if s]


def _parse_frontmatter(page_path: Path) -> dict | None:
    """Parse the YAML frontmatter of a wiki page; return the dict or None.

    Locates the first ``---`` … ``---`` fence by scanning for fence *lines* (a
    line that is exactly ``---`` after stripping) rather than requiring the file
    to *start* with ``---``. Filed qa pages are written with a leading sentinel
    HTML comment before the fence (``<!-- Auto-filed by POST /chat… -->``), so a
    ``startswith("---")`` test skipped every filed draft — which made C8/C9/C10
    silently ignore real Filed Answers and mis-flag them as invalid schema (#4).
    This mirrors ``qa._read_frontmatter``, which already parses these files.

    Returns None if the page has no fence pair, the block cannot be parsed, or
    the result is not a dict.
    """
    try:
        text = page_path.read_text(encoding="utf-8")
    except OSError:
        return None

    lines = text.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(dash_indices) < 2:
        return None

    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return None

    if not isinstance(fm, dict):
        return None
    return fm


def _iter_wiki_pages(wiki_dir: Path):
    """Yield (slug, page_path) for every .md page under entities/ and concepts/."""
    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            yield page_path.stem, page_path


# ---------------------------------------------------------------------------
# C3 — Failed-grounding sweep
# ---------------------------------------------------------------------------


def _check_c3_failed_grounding(
    wiki_dir: Path,
) -> list[FailedGroundingFinding]:
    """Return findings for every wiki page with ``frontmatter.status == "failed_grounding"``.

    Phase 3 fail-soft ingest writes these pages when the grounding verifier
    rejects claims or is unavailable.  Phase 4 W1 silently excludes them from
    ``/chat`` retrieval.  C3 surfaces them so the curator can decide whether to
    review the Source and re-ingest or simply delete the page.

    Algorithm:
    1. Iterate ``wiki/entities/*.md`` and ``wiki/concepts/*.md``.
    2. For each page, parse YAML frontmatter.
    3. Skip pages whose ``status`` is not ``"failed_grounding"``.
    4. Build a ``FailedGroundingFinding`` from ``sources[0]``, the nested
       ``grounding_failure`` block (``reason`` + ``unsupported_claims``), and a
       suggested action.
    5. Return findings sorted alphabetically by ``page_slug``.

    If ``grounding_failure`` is absent or malformed, the finding still records
    ``reason="verifier_unavailable"`` and an empty ``unsupported_claims`` list
    rather than raising — defensive because older ingest code may not have
    written the block consistently.
    """
    findings: list[FailedGroundingFinding] = []

    for slug, page_path in _iter_wiki_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            continue
        if fm.get("status") != "failed_grounding":
            continue

        sources = fm.get("sources", [])
        source_ref = str(sources[0]) if sources else ""

        # Extract grounding_failure sub-block defensively
        gf_raw = fm.get("grounding_failure")
        if isinstance(gf_raw, dict):
            reason = gf_raw.get("reason", "verifier_unavailable")
            if reason not in ("claim_unsupported", "verifier_unavailable"):
                reason = "verifier_unavailable"
            raw_claims = gf_raw.get("unsupported_claims", [])
            unsupported_claims = (
                [str(c) for c in raw_claims] if isinstance(raw_claims, list) else []
            )
        else:
            reason = "verifier_unavailable"
            unsupported_claims = []

        suggested_action = (
            f"Page '{slug}' failed grounding check (reason: {reason}). "
            f"Review the source '{source_ref}' to confirm the claims are supported, "
            f"then re-ingest the source to regenerate this page. "
            f"If the source no longer covers these claims, delete this page."
        )

        findings.append(
            FailedGroundingFinding(
                page_slug=slug,
                source=source_ref,
                reason=reason,  # type: ignore[arg-type]
                unsupported_claims=unsupported_claims,
                suggested_action=suggested_action,
            )
        )

    findings.sort(key=lambda f: f.page_slug)
    return findings


# ---------------------------------------------------------------------------
# C4-a — Slug collision groups
# ---------------------------------------------------------------------------


def _check_c4a_slug_collision(
    wiki_dir: Path,
) -> list[SlugCollisionFinding]:
    """Return collision groups for slugs sharing a common base (stripped ``-N`` suffix).

    Phase 3 ingest appends ``-2``, ``-3``, ... suffixes to avoid overwriting
    existing pages.  These collisions indicate that two pages cover the same
    concept (or nearly so) and a curator should review them for merge or
    heading rename.

    Only suffixes with N >= 2 trigger grouping (``-1`` is not an ingest-appended
    collision suffix).

    Algorithm:
    1. Collect all page slugs from ``wiki/entities/*.md`` and ``wiki/concepts/*.md``.
    2. For each slug, test ``_COLLISION_SUFFIX_RE`` (matches ``<base>-<N>`` where
       N is a numeric string).  If N >= 2, the slug belongs to the group for
       ``<base>``; otherwise the slug is itself a base slug.
    3. A group must contain at least 2 members (the base slug + at least one
       suffixed variant, or two suffixed variants with a shared base).
    4. Emit one ``SlugCollisionFinding`` per qualifying group.
    5. Sort: group size descending; alphabetical by ``base_slug`` for ties.

    Cross-directory collisions are included (a slug in ``entities/`` and a
    suffixed variant in ``concepts/`` are grouped together) since the ingest
    uniqueness guarantee is wiki-wide, not per-subdirectory.
    """
    # Map base_slug → set of member slugs
    groups: dict[str, set[str]] = {}

    for slug, _page_path in _iter_wiki_pages(wiki_dir):
        m = _COLLISION_SUFFIX_RE.match(slug)
        if m and int(m.group(2)) >= 2:
            # Suffixed variant (`pricing-2`, `pricing-3`, ...): group under its base.
            # Do NOT seed the group with {base} — the unsuffixed base page may not
            # exist on disk; report only slugs that actually exist.
            groups.setdefault(m.group(1), set()).add(slug)
        else:
            # No suffix (or -1, which is not a collision-appended variant): the slug
            # is its own base. Always add so iteration order does not affect which
            # slugs land in a pre-existing group keyed by this same base.
            groups.setdefault(slug, set()).add(slug)

    findings: list[SlugCollisionFinding] = []
    for base_slug, members in groups.items():
        if len(members) < 2:
            continue
        pages_in_group = sorted(members)
        suggested_action = (
            f"Slug collision: {len(members)} pages share the base slug '{base_slug}' "
            f"({', '.join(pages_in_group)}). "
            f"Review the pages and either merge them into a single page or rename their "
            f"headings to be more specific so ingest assigns distinct slugs."
        )
        findings.append(
            SlugCollisionFinding(
                base_slug=base_slug,
                pages_in_group=pages_in_group,
                suggested_action=suggested_action,
            )
        )

    # Sort: group size descending, then alphabetical by base_slug for ties
    findings.sort(key=lambda f: (-len(f.pages_in_group), f.base_slug))
    return findings


# ---------------------------------------------------------------------------
# C6 — mtime-based stale detection
# ---------------------------------------------------------------------------


def _check_c6_stale(
    wiki_dir: Path,
    docs_dir: Path,
) -> list[StalePageFinding]:
    """Return stale findings for every wiki page whose Source file is newer.

    A wiki page is *stale* when:
    - ``frontmatter.sources[0]`` references a Source file that EXISTS under ``docs_dir``
    - The Source file's filesystem mtime is later than the page's ``frontmatter.updated``
      timestamp

    Pages whose Source file does NOT exist are handled by C11 (orphan check).
    C6 explicitly skips them to avoid double-reporting.

    Algorithm:
    1. For each wiki page in ``entities/`` and ``concepts/``:
       a. Parse frontmatter; skip if no sources.
       b. Take ``sources[0]``; strip ``#anchor`` to get the Source filename.
       c. Resolve ``docs_dir / <filename>``; if the file does not exist, skip (C11's job).
       d. Read the Source file's mtime as a UTC datetime.
       e. Parse ``frontmatter.updated`` as a UTC datetime.
       f. If ``source_mtime > page_updated``, emit ``StalePageFinding``.
    2. Return findings sorted by ``drift_days`` descending.
    """
    findings: list[StalePageFinding] = []

    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            slug = page_path.stem
            fm = _parse_frontmatter(page_path)
            if fm is None:
                continue

            sources = fm.get("sources", [])
            if not isinstance(sources, list) or not sources:
                continue

            # C6 uses only the first source citation
            first_citation = str(sources[0]) if sources[0] else ""
            if not first_citation:
                continue

            # Strip anchor to get the Source filename
            file_part = first_citation.split("#")[0].strip()
            if not file_part:
                continue
            source_filename = Path(file_part).name

            # Resolve Source file; skip if missing (C11's job)
            source_path = docs_dir / source_filename
            if not source_path.exists():
                # Also try nested lookup using glob
                matches = list(docs_dir.glob(f"**/{source_filename}"))
                if not matches:
                    continue
                source_path = matches[0]

            # Get Source mtime as UTC datetime
            source_mtime_ts = source_path.stat().st_mtime
            source_mtime = datetime.datetime.fromtimestamp(source_mtime_ts, tz=datetime.UTC)

            # Parse page's updated timestamp
            updated_str = fm.get("updated", "")
            if not updated_str:
                continue
            try:
                page_updated = datetime.datetime.fromisoformat(
                    str(updated_str).replace("Z", "+00:00")
                )
            except ValueError:
                continue

            # Ensure both are timezone-aware UTC for comparison
            if page_updated.tzinfo is None:
                page_updated = page_updated.replace(tzinfo=datetime.UTC)

            if source_mtime > page_updated:
                drift_seconds = (source_mtime - page_updated).total_seconds()
                drift_days = drift_seconds / 86400.0
                findings.append(
                    StalePageFinding(
                        page_slug=slug,
                        source=source_filename,
                        source_mtime=source_mtime,
                        page_updated=page_updated,
                        drift_days=drift_days,
                        suggested_action=(
                            f"Source '{source_filename}' was modified {drift_days:.1f} day(s) after "
                            f"wiki page '{slug}' was last updated. Re-ingest the Source to "
                            f'synchronise the wiki page: POST /ingest {{"source": "{source_filename}"}}.'
                        ),
                    )
                )

    # Sort by drift_days descending
    findings.sort(key=lambda f: f.drift_days, reverse=True)
    return findings


# ---------------------------------------------------------------------------
# C2 — Red link backlog
# ---------------------------------------------------------------------------

# Regex from the AC: captures slug portions of [[slug]] and [[slug#anchor]] and [[slug|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")

# Files in wiki root that must NEVER contribute red links (self-feeding loop + noise)
_C2_EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        "lint-report.md",
        "index.md",
        "log.md",
        "hot.md",
        "README.md",
    }
)


def _check_c2_red_links(
    wiki_dir: Path,
) -> list[RedLinkFinding]:
    """Return red link findings for every unresolved ``[[wikilink]]`` slug.

    Scans ``wiki/entities/`` and ``wiki/concepts/`` ONLY (matching ADR-0006 SOURCE_DIRS).
    ``wiki/.archive/*`` and root-level special files are explicitly excluded.

    Algorithm:
    1. Build the set of existing page slugs from ``entities/*.md`` + ``concepts/*.md``.
    2. For each page in those dirs, scan the page body for ``[[...]]`` patterns.
       Skip files in ``_C2_EXCLUDED_FILENAMES`` (by basename).
       Skip files in ``wiki/.archive/``.
    3. For each wikilink, extract the slug portion (drop ``#anchor`` and ``|alias``).
       If the slug is NOT in the existing slugs set, it is a red link.
    4. Aggregate by slug: count total occurrences; track pages that reference it;
       capture ~50-char context from the first occurrence.
    5. Return findings sorted by ``mention_count`` descending, alphabetical by ``slug`` for ties.

    Heading anchors (``[[slug#heading]]``) are captured but only the slug is checked.
    """
    # Build the set of known page slugs
    existing_slugs: set[str] = set()
    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in subdir.glob("*.md"):
            existing_slugs.add(page_path.stem)

    # Per-slug aggregation: mention_count, referenced_by set, first context
    slug_counts: dict[str, int] = defaultdict(int)
    slug_pages: dict[str, set[str]] = defaultdict(set)
    slug_first_context: dict[str, str | None] = {}

    # TODO: consolidate ("entities", "concepts") with ADR-0006 SOURCE_DIRS string-name companion
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            # Exclusion: skip files by name
            if page_path.name in _C2_EXCLUDED_FILENAMES:
                continue
            # Exclusion: skip .archive/ files
            if ".archive" in page_path.parts:
                continue

            page_slug = page_path.stem
            try:
                body = page_path.read_text(encoding="utf-8")
            except OSError:
                continue

            # Find all wikilinks in the body
            for match in _WIKILINK_RE.finditer(body):
                target_slug = match.group(1).strip()
                if not target_slug:
                    continue
                if target_slug in existing_slugs:
                    # Resolved — not a red link
                    continue
                # Unresolved red link
                slug_counts[target_slug] += 1
                slug_pages[target_slug].add(page_slug)
                # Capture context from first occurrence only
                if target_slug not in slug_first_context:
                    start = match.start()
                    # Take ~25 chars before and ~25 chars after the match
                    ctx_start = max(0, start - 25)
                    ctx_end = min(len(body), match.end() + 25)
                    slug_first_context[target_slug] = body[ctx_start:ctx_end]

    findings: list[RedLinkFinding] = []
    for slug, count in slug_counts.items():
        findings.append(
            RedLinkFinding(
                slug=slug,
                mention_count=count,
                referenced_by=sorted(slug_pages[slug]),
                sample_context=slug_first_context.get(slug),
            )
        )

    # Sort by mention_count descending, then alphabetical by slug for ties
    findings.sort(key=lambda f: (-f.mention_count, f.slug))
    return findings


# ---------------------------------------------------------------------------
# C1 — Coverage gap aggregation from chat_fallback / chat_grounding_fallback log
# ---------------------------------------------------------------------------

# Regex to match a log line produced by logger.log_event:
#   ## [<ISO-8601>] <kind> | <summary>
_LOG_LINE_RE = re.compile(r"^## \[(?P<ts>[^\]]+)\] (?P<kind>[^ ]+) \| (?P<summary>.*)$")

# Regex to extract the leading double-quoted query from a summary field.
# Matches: "<query>" ... (query may contain single-quotes; the retrieval
# module replaces double-quotes inside queries with single-quotes before logging)
_SUMMARY_QUERY_RE = re.compile(r'^"(?P<query>[^"]*)"')

# Log kinds consumed by C1
_C1_KINDS = frozenset({"chat_fallback", "chat_grounding_fallback"})

# Reason values that C1 handles explicitly
_C1_HANDLED_REASONS = frozenset({"retrieval_empty", "below_threshold", "claim_unsupported"})


def _canonicalise(q: str) -> str:
    """Return the canonical cluster key for a query string.

    Rules (per issue #69 AC):
    - Lowercase
    - Strip leading and trailing punctuation characters
    - Collapse internal whitespace to single spaces
    - Strip outer whitespace
    - No stop-word removal
    - No token sorting
    """
    # Lowercase
    result = q.lower()
    # Strip outer whitespace
    result = result.strip()
    # Strip leading/trailing punctuation (all chars in string.punctuation)
    result = result.strip(string.punctuation)
    # Strip any remaining outer whitespace after punctuation stripping
    result = result.strip()
    # Collapse internal whitespace to single spaces
    result = re.sub(r"\s+", " ", result)
    return result


def _parse_kv(summary: str) -> dict[str, str]:
    """Parse key=value pairs from a log summary string.

    Supports bare values (no quotes) and ignores the leading quoted query
    string.  Returns a dict of all key=value pairs found.
    """
    # Remove the leading quoted query (may contain spaces) first
    remainder = _SUMMARY_QUERY_RE.sub("", summary).strip()
    pairs: dict[str, str] = {}
    for match in re.finditer(r"(\w+)=(\S+)", remainder):
        pairs[match.group(1)] = match.group(2)
    return pairs


def _check_c1_coverage_gaps(log_path: Path) -> list[CoverageGapFinding]:
    """Parse ``wiki/log.md`` and aggregate coverage gap findings.

    Reads the log file line by line. Matches ``chat_fallback`` and
    ``chat_grounding_fallback`` entries. Groups them into clusters according
    to the per-reason cluster key table in issue #69:

    - ``retrieval_empty``: key = ``_canonicalise(q)``
    - ``below_threshold``: key = ``(_canonicalise(q), top_section)``
    - ``claim_unsupported``: key = ``(_canonicalise(q), tuple(sorted(cited_pages)))``
    - ``wiki_layer_empty``: skipped entirely (filtered by kind, not reason)

    Env var ``KB_LINT_MIN_HITS`` (default 1): clusters with ``hit_count <
    KB_LINT_MIN_HITS`` are excluded from the returned list.  Read at function
    entry (not at module load) to mirror the ``KB_SCORE_THRESHOLD`` pattern.

    Counter semantics:
    - ``malformed_lines``: lines that are in a C1 kind but whose structure
      could not be parsed (missing query field, regex non-match, parse exception).
      If ``malformed_lines > 0`` a ``lint_check_error`` is written.
    - ``out_of_scope_reasons``: lines parsed cleanly but whose ``reason=`` value
      is not one C1 handles (e.g. ``not_indexed``, ``verifier_unavailable``).
      These are silently ignored — they are legitimately not C1's concern.

    Returns findings sorted:
    - Primary: fixed group order (``retrieval_empty``, ``below_threshold``,
      ``claim_unsupported``)
    - Within group: ``hit_count`` descending; ties broken alphabetically by
      ``query_canonical``
    """
    min_hits = int(os.getenv("KB_LINT_MIN_HITS", "1"))

    if not log_path.exists():
        return []

    # Cluster accumulators keyed by (reason, cluster_key)
    # Each value: {"hits": int, "raw_queries_seen": [str], "timestamps": [str],
    #              "top_section": str|None, "cited_pages": list[str]|None}
    clusters: dict[tuple[str, Any], dict[str, Any]] = {}

    malformed_lines = 0
    out_of_scope_reasons = 0

    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _LOG_LINE_RE.match(line)
        if not m:
            # Not a structured log line (e.g., blank header, arbitrary text); skip
            continue

        ts = m.group("ts")
        kind = m.group("kind")
        summary = m.group("summary")

        if kind not in _C1_KINDS:
            continue

        # Extract leading quoted query
        qm = _SUMMARY_QUERY_RE.match(summary)
        if not qm:
            malformed_lines += 1
            continue
        raw_query = qm.group("query")

        # Parse key=value pairs from remainder
        try:
            kv = _parse_kv(summary)
        except Exception:  # noqa: BLE001
            malformed_lines += 1
            continue

        reason = kv.get("reason", "")

        if reason not in _C1_HANDLED_REASONS:
            # Parsed cleanly but not a reason C1 handles — not a parse failure
            out_of_scope_reasons += 1
            continue

        canonical = _canonicalise(raw_query)

        if reason == "retrieval_empty":
            cluster_key: Any = canonical
            top_section: str | None = None
            cited_pages: list[str] | None = None

        elif reason == "below_threshold":
            top_section = kv.get("top_section")
            cluster_key = (canonical, top_section)
            cited_pages = None

        else:  # claim_unsupported
            cited_raw = kv.get("cited", "")
            cited_list: list[str] = [c for c in cited_raw.split(",") if c] if cited_raw else []
            cited_pages = cited_list if cited_list else None
            cluster_key = (canonical, tuple(sorted(cited_list)))
            top_section = None

        full_key = (reason, cluster_key)

        if full_key not in clusters:
            clusters[full_key] = {
                "reason": reason,
                "query_canonical": canonical,
                "hits": 0,
                "raw_queries_seen": [],  # ordered, deduplicated up to 3
                "raw_seen_set": set(),
                "timestamps": [],
                "top_section": top_section,
                "cited_pages": cited_pages,
            }

        entry = clusters[full_key]
        entry["hits"] += 1
        entry["timestamps"].append(ts)

        # Collect up to 3 unique raw queries
        if len(entry["raw_queries_seen"]) < 3 and raw_query not in entry["raw_seen_set"]:
            entry["raw_queries_seen"].append(raw_query)
            entry["raw_seen_set"].add(raw_query)

    if malformed_lines > 0:
        log_event(
            "lint_check_error",
            f"check=c1 malformed_lines={malformed_lines} out_of_scope_reasons={out_of_scope_reasons} reason=malformed_log_lines",
            log_path=log_path,
        )

    # Build CoverageGapFinding list, applying min_hits filter
    findings: list[CoverageGapFinding] = []
    for (_reason, _ck), entry in clusters.items():
        if entry["hits"] < min_hits:
            continue
        reason = entry["reason"]
        canonical = entry["query_canonical"]
        raw_q = entry["raw_queries_seen"]
        timestamps = sorted(entry["timestamps"])
        first_seen = timestamps[0] if timestamps else ""
        last_seen = timestamps[-1] if timestamps else ""

        ts_ = entry["top_section"]
        cp = entry["cited_pages"]

        if reason == "retrieval_empty":
            action = f"Create a new wiki page covering {canonical}"
        elif reason == "below_threshold":
            section_ref = ts_ or "<unknown>"
            action = f"Extend page `[[{section_ref}]]` to cover {canonical}"
        else:  # claim_unsupported
            pages_str = ", ".join(cp) if cp else "<unknown>"
            action = f"Review: KB gap or verifier issue? cited: {pages_str}"

        findings.append(
            CoverageGapFinding(
                reason=reason,
                query_canonical=canonical,
                sample_raw_queries=raw_q,
                hit_count=entry["hits"],
                first_seen=first_seen,
                last_seen=last_seen,
                top_section=ts_,
                cited_pages=cp,
                suggested_action=action,
            )
        )

    # Sort: fixed group order, then hit_count desc, then query_canonical asc
    _GROUP_ORDER = {"retrieval_empty": 0, "below_threshold": 1, "claim_unsupported": 2}
    findings.sort(key=lambda f: (_GROUP_ORDER.get(f.reason, 99), -f.hit_count, f.query_canonical))

    return findings


# ---------------------------------------------------------------------------
# C5 — Page-pair LLM contradiction detection
# ---------------------------------------------------------------------------

# env vars for C5 BM25 candidate filter
_KB_LINT_BM25_TOP_K_DEFAULT = 3
_KB_LINT_BM25_THRESHOLD_DEFAULT = 1.0


def _load_wiki_pages(wiki_dir: Path) -> dict[str, dict]:
    """Load all wiki pages from entities/, concepts/, and qa/ into a dict.

    Returns a dict mapping slug → {
        "slug": str,
        "body": str (full file text after frontmatter),
        "sources": list[str],
        "path": Path,
        "type": str | None ("entity" | "concept" | "qa", from frontmatter),
    }

    Used by _candidate_pairs and _check_c5_page_pair to avoid re-parsing
    frontmatter multiple times. The ``type`` entry is what the Slice 6-5 C5
    modifier reads to exclude ``type == "qa"`` pages from candidate pair
    generation (PRD #78 Phase 5 amendment).

    qa pages are included here so the C5 modifier filter sees them and drops
    them; downstream consumers (e.g. ``_candidate_pairs``) own the filter. The
    set returned is the entire wiki page corpus (entities/concepts/qa), not the
    filtered subset.
    """
    pages: dict[str, dict] = {}
    for slug, page_path in _iter_all_wiki_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        sources: list[str] = []
        page_type: str | None = None
        if fm:
            raw_sources = fm.get("sources", [])
            if isinstance(raw_sources, list):
                sources = [str(s) for s in raw_sources if s]
            raw_type = fm.get("type")
            if isinstance(raw_type, str):
                page_type = raw_type
        try:
            full_text = page_path.read_text(encoding="utf-8")
        except OSError:
            full_text = ""
        # Strip frontmatter to get body
        body = full_text
        if full_text.startswith("---"):
            parts = full_text.split("---", 2)
            body = parts[2] if len(parts) >= 3 else full_text
        pages[slug] = {
            "slug": slug,
            "body": body.strip(),
            "sources": sources,
            "path": page_path,
            "type": page_type,
        }
    return pages


def _iter_all_wiki_pages(wiki_dir: Path):
    """Yield (slug, page_path) for every .md page under entities/, concepts/, qa/.

    Distinct from ``_iter_wiki_pages`` (entities + concepts only) — used by
    ``_load_wiki_pages`` so the C5 modifier (Slice 6-5) sees qa pages and can
    filter them out at the pair-generation layer.
    """
    # Phase 6 Slice 6-5: include qa/ so the modifier filter can see and drop them.
    for subdir_name in ("entities", "concepts", "qa"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in sorted(subdir.glob("*.md")):
            yield page_path.stem, page_path


def _candidate_pairs(
    pages: dict[str, dict],
    wiki_dir: Path,
) -> set[tuple[str, str]]:
    """Return the F1 ∪ F3 candidate pair set for C5 LLM judgement.

    F1: pairs of pages whose ``frontmatter.sources`` lists share at least one
        source citation.

    F3: for each page, treat its body as a BM25 query and call
        ``indexer.search(body, k=KB_LINT_BM25_TOP_K)``.  Any returned Section
        whose score exceeds ``KB_LINT_BM25_THRESHOLD`` and whose page slug is
        different from the query page contributes a candidate pair.

    All pair tuples are canonicalised: ``(min(a, b), max(a, b))`` so that
    ``(A, B)`` and ``(B, A)`` deduplicate to a single pair.  This is the
    symmetric pair short-circuit invariant — each pair is judged exactly once.

    ``KB_LINT_BM25_TOP_K`` env var (default 3) controls per-page BM25 hits.
    ``KB_LINT_BM25_THRESHOLD`` env var (default 1.0) controls the score gate.
    Both are read at call time (not module load) to mirror KB_SCORE_THRESHOLD.

    Phase 6 Slice 6-5 (C5 modifier — PRD #78)
    -----------------------------------------
    Pages whose ``frontmatter.type == "qa"`` are excluded from candidate pair
    generation BEFORE F1/F3 are computed. Filed Answers are structurally
    derivative of their source entity pages (they share the same
    ``frontmatter.sources``), so without this filter the C5 LLM would be called
    on every (qa, entity-source) pair only to return ``severity=duplicate``,
    flooding ``lint-report.md`` and burning LLM tokens. Filtering at the page
    set level means the qa pages neither appear in F1 source-intersection nor
    as F3 BM25 queries, *and* the F3 BM25 hit-side filter rejects sections
    whose owner is a qa page — qa is fully invisible to C5 by construction.
    """
    # Read env vars at call time
    bm25_top_k = int(os.getenv("KB_LINT_BM25_TOP_K", str(_KB_LINT_BM25_TOP_K_DEFAULT)))
    bm25_threshold = float(
        os.getenv("KB_LINT_BM25_THRESHOLD", str(_KB_LINT_BM25_THRESHOLD_DEFAULT))
    )

    # C5 modifier (Slice 6-5): drop type=qa pages from the candidate pool BEFORE
    # F1/F3 candidate computation so the LLM call budget excludes qa pairs entirely.
    non_qa_pages: dict[str, dict] = {
        slug: data for slug, data in pages.items() if data.get("type") != "qa"
    }

    pairs: set[tuple[str, str]] = set()
    slug_list = list(non_qa_pages.keys())

    # F1: shared sources (only among non-qa pages)
    for i, slug_a in enumerate(slug_list):
        srcs_a = set(non_qa_pages[slug_a]["sources"])
        if not srcs_a:
            continue
        for slug_b in slug_list[i + 1 :]:
            srcs_b = set(non_qa_pages[slug_b]["sources"])
            if srcs_a & srcs_b:
                pair = (min(slug_a, slug_b), max(slug_a, slug_b))
                pairs.add(pair)

    # F3: BM25 self-query
    # Build a temporary index from wiki pages so we can BM25-query body text.
    # We reuse indexer.parse_markdown via a tmp directory approach, but since the
    # index is already built from wiki pages (by /index), we use indexer.search
    # directly which queries the in-memory sections list.
    #
    # NOTE: The in-memory index must be populated before calling run_lint().
    # This is the normal case (bot needs an index to function). If the index is
    # empty (sections list empty), F3 produces no pairs — safe degradation.
    from .indexer import search as bm25_search

    for slug_a, data_a in non_qa_pages.items():
        body_a = data_a["body"]
        if not body_a.strip():
            continue
        try:
            hits = bm25_search(body_a, k=bm25_top_k)
        except Exception:  # noqa: BLE001
            continue
        for section, score in hits:
            if score < bm25_threshold:
                continue
            # section.file is the bare slug (ADR-0006: wiki pages indexed with slug as source_id)
            slug_b = section.file
            if slug_b == slug_a:
                continue
            # Only add pairs where both slugs exist in the non-qa page set —
            # this is the hit-side half of the C5 qa filter: even if the index
            # surfaces a qa-page section, it cannot enter the candidate pool.
            if slug_b not in non_qa_pages:
                continue
            pair = (min(slug_a, slug_b), max(slug_a, slug_b))
            pairs.add(pair)

    return pairs


# Prompt for the C5 LLM call — instructs the model to compare two wiki page bodies
_C5_SYSTEM_PROMPT = """You are a knowledge-base health auditor. Two wiki pages from the same knowledge base are shown below. Your task is to judge whether they contradict, overlap, or duplicate each other.

Output a structured finding with:
- severity: one of "direct" (explicit factual conflict — different numbers, different policies), "tension" (same topic, scope or wording differences that could confuse readers), "duplicate" (same concept covered in two pages without contradiction), or "none" (no meaningful overlap or conflict found).
- page_a_claim: a direct quote from Page A's body that is relevant to the comparison. Use the exact text.
- page_b_claim: a direct quote from Page B's body that is relevant to the comparison. Use the exact text.
- summary: a one-to-two sentence explanation of why you assigned this severity.
- suggested_action: a concrete curator action (e.g. "Reconcile sources", "Merge pages", "Review and dismiss").

If severity is "none", still provide page_a_claim and page_b_claim — pick any representative sentence from each page.

Be conservative: only assign "direct" for clear factual disagreement (different numbers, dates, policy terms). Use "tension" for ambiguous cases."""


def _judge_page_pair(
    slug_a: str,
    body_a: str,
    slug_b: str,
    body_b: str,
) -> PagePairFinding:
    """Call the LLM to judge whether two wiki pages contradict, overlap, or duplicate.

    Uses ``get_lint_llm().with_structured_output(PagePairFinding)`` per ADR-0005.
    temperature=0 (set on the singleton in ``get_lint_llm()``).

    Returns a ``PagePairFinding`` with canonical slug ordering enforced:
    ``page_a`` is always the lexicographically-smaller slug.

    LangChain types are confined to this function — callers see only
    ``PagePairFinding`` (ADR-0005 § Consequences).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Ensure canonical slug order
    if slug_a > slug_b:
        slug_a, slug_b = slug_b, slug_a
        body_a, body_b = body_b, body_a

    llm = get_lint_llm()
    chain = llm.with_structured_output(PagePairFinding)

    messages = [
        SystemMessage(content=_C5_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"**Page A** (slug: `{slug_a}`):\n\n{body_a}\n\n"
                f"---\n\n**Page B** (slug: `{slug_b}`):\n\n{body_b}"
            )
        ),
    ]

    finding: PagePairFinding = chain.invoke(messages)
    # Increment cost counter (best-effort; one call per pair)
    _c5_llm_call_counter[0] += 1

    # Enforce canonical slug order in the finding (LLM may swap them)
    if finding.page_a != slug_a or finding.page_b != slug_b:
        finding = PagePairFinding(
            severity=finding.severity,
            page_a=slug_a,
            page_b=slug_b,
            page_a_claim=finding.page_a_claim,
            page_b_claim=finding.page_b_claim,
            summary=finding.summary,
            suggested_action=finding.suggested_action,
        )

    return finding


def _check_c5_page_pair(
    wiki_dir: Path,
) -> list[PagePairFinding]:
    """Run C5 page-pair contradiction detection over the wiki.

    Steps:
    1. Load all wiki pages (slug, body, sources) via ``_load_wiki_pages``.
    2. Build candidate pairs via ``_candidate_pairs`` (F1 ∪ F3 filter).
    3. For each candidate pair, call ``_judge_page_pair``.
    4. Filter out findings with severity == "none".
    5. Continue-on-error: if the LLM raises for a pair, log the pair skipped
       and retain prior findings.

    Returns findings sorted by severity order (direct → tension → duplicate),
    then alphabetically by page_a slug.
    """
    pages = _load_wiki_pages(wiki_dir)
    pairs = _candidate_pairs(pages, wiki_dir)

    findings: list[PagePairFinding] = []
    errors: list[str] = []

    for slug_a, slug_b in sorted(pairs):
        data_a = pages.get(slug_a)
        data_b = pages.get(slug_b)
        if not data_a or not data_b:
            continue
        try:
            finding = _judge_page_pair(slug_a, data_a["body"], slug_b, data_b["body"])
            if finding.severity != "none":
                findings.append(finding)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"({slug_a},{slug_b}): {type(exc).__name__}: {exc}")
            continue

    if errors:
        # Log errors without breaking; the check returns partial results
        log_event(
            "lint_check_error",
            f"check=c5 pairs_failed={len(errors)} first_error={errors[0][:200]}",
        )

    # Sort: severity order, then alphabetical by page_a
    _SEVERITY_ORDER = {"direct": 0, "tension": 1, "duplicate": 2, "none": 3}
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 99), f.page_a, f.page_b))

    return findings


# ---------------------------------------------------------------------------
# C8 / C9 / C10 — Phase 5 amendment for Phase 6 Filed Answers (PRD #78)
# ---------------------------------------------------------------------------

# Valid frontmatter.status values for Filed Answer pages (Slice 6-1 schema).
# Used by C10 to validate ``status`` field; ``status`` outside this set is a
# curator-typo orphan-zombie risk (Q8d in PRD #78).
_VALID_QA_STATUS_VALUES: frozenset[str] = frozenset({"live", "draft", "stale", "superseded"})


def _iter_qa_pages(wiki_dir: Path):
    """Yield (slug, page_path) for every .md page under wiki/qa/.

    Separate iterator from ``_iter_wiki_pages`` because the qa lifecycle is
    distinct (Filed Answer lifecycle owned by PRD #78, not the entity/concept
    ingest lifecycle) and the three Phase 5 amendment checks (C8/C9/C10) scan
    only qa/, never entities/ or concepts/.
    """
    subdir = wiki_dir / "qa"
    if not subdir.exists():
        return
    for page_path in sorted(subdir.glob("*.md")):
        yield page_path.stem, page_path


# Default cap on Promotion Candidates surfaced per /lint run.
_KB_LINT_PROMOTION_TOP_N_DEFAULT = 10

# Truncation length for Promotion Candidate question field (keeps the report
# table column width manageable; readers can open the page for the full text).
_PROMOTION_CANDIDATE_QUESTION_MAXLEN = 80


def _check_c8_promotion_candidates(
    wiki_dir: Path,
) -> list[PromotionCandidateFinding]:
    """Surface ``status: draft`` Filed Answers ranked by re-ask popularity.

    PRD #78 Phase 5 amendment §"C8 — promotion candidates". Read-only — never
    mutates frontmatter; the actual draft→live promotion is Phase 6's
    ``POST /qa/{slug}/promote`` (Slice 6-4).

    Algorithm:
    1. Scan ``wiki/qa/*.md``.
    2. Skip pages whose frontmatter cannot be parsed (defensive — C10 surfaces
       schema breakage independently).
    3. Skip pages whose ``status`` is not ``"draft"``.
    4. Build PromotionCandidateFinding with ``slug``, truncated ``question``,
       ``count``, ``age_days`` (now - created in UTC), and ``cited_count``.
    5. Sort by ``count`` desc, then ``updated`` desc (tiebreak).
    6. Cap at the top ``KB_LINT_PROMOTION_TOP_N`` (env var, default 10).

    Returns the ranked, capped list. Empty list when no qa pages exist or no
    drafts are present.
    """
    top_n = int(os.getenv("KB_LINT_PROMOTION_TOP_N", str(_KB_LINT_PROMOTION_TOP_N_DEFAULT)))

    # Each candidate carries a sort tuple so we can rank, then strip back to the
    # Finding shape for the public return.
    candidates: list[tuple[int, str, PromotionCandidateFinding]] = []

    now_utc = datetime.datetime.now(datetime.UTC)

    for slug, page_path in _iter_qa_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            continue
        if fm.get("status") != "draft":
            continue

        # Question text — truncate for the report column. Falls back to the
        # slug if question is missing (defensive; C10 separately surfaces a
        # missing_question finding).
        raw_question = fm.get("question") or ""
        question = str(raw_question).strip()
        if not question:
            question = f"(missing question — slug `{slug}`)"
        if len(question) > _PROMOTION_CANDIDATE_QUESTION_MAXLEN:
            question = question[: _PROMOTION_CANDIDATE_QUESTION_MAXLEN - 1] + "…"

        # Count — coerce defensively; missing/invalid -> 1 (the schema default
        # value, also documented in WikiPageFrontmatter).
        raw_count = fm.get("count", 1)
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 1

        # Cited count — number of citation entries in frontmatter.sources.
        raw_sources = fm.get("sources", [])
        cited_count = len(raw_sources) if isinstance(raw_sources, list) else 0

        # Age in days — now() - frontmatter.created. Missing/unparseable falls
        # back to 0.0 (rendered as 0.0 days; non-fatal because C10 surfaces the
        # underlying schema issue if any field is wrong).
        age_days = 0.0
        created_str = fm.get("created", "")
        if created_str:
            try:
                created_dt = datetime.datetime.fromisoformat(
                    str(created_str).replace("Z", "+00:00")
                )
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=datetime.UTC)
                age_days = (now_utc - created_dt).total_seconds() / 86400.0
            except ValueError:
                age_days = 0.0

        # Sort tiebreaker — frontmatter.updated as ISO string. Lexicographic
        # comparison is correct for ISO-8601 UTC strings; missing -> empty
        # string which sorts last.
        updated_key = str(fm.get("updated", ""))

        finding = PromotionCandidateFinding(
            slug=slug,
            question=question,
            count=count,
            age_days=round(age_days, 1),
            cited_count=cited_count,
        )
        candidates.append((count, updated_key, finding))

    # Rank with two stable passes (Python's sort is stable, so secondary keys
    # propagate cleanly): first sort ascending by slug (final tiebreaker),
    # then ascending by updated (so a desc-updated stable sort by count desc
    # produces the AC-specified order: count desc, then updated desc).
    # Single-pass tuple sort would require negating the string updated key,
    # which is brittle for unequal-length ISO timestamps; two-pass is clearer.
    candidates.sort(key=lambda t: t[2].slug)  # tertiary: slug asc
    candidates.sort(key=lambda t: t[1], reverse=True)  # secondary: updated desc
    candidates.sort(key=lambda t: t[0], reverse=True)  # primary: count desc

    return [finding for _c, _u, finding in candidates[:top_n]]


def _check_c9_qa_staleness(
    wiki_dir: Path,
) -> list[QaStalenessFinding]:
    """Flag live Filed Answers whose cited entity pages have a newer mtime.

    PRD #78 Phase 5 amendment §"C9 — qa-staleness". Read-only — surfaced to
    ``lint-report.md`` §``## Stale Filed Answers``. Closes Q6b "entity
    re-ingested, qa stranded" failure mode.

    Algorithm:
    1. For each ``wiki/qa/*.md`` with ``frontmatter.status == "live"``:
       a. Parse ``frontmatter.sources``; for each citation
          ``"<entity-slug>#<heading-slug>"`` extract the bare entity slug.
       b. For each entity slug, locate the entity file: try
          ``wiki/entities/<slug>.md`` first, then ``wiki/concepts/<slug>.md``.
       c. Compare the entity file mtime against ``qa.frontmatter.updated``.
          If entity mtime > qa updated, the citation is "stale".
       d. If at least one citation is stale, emit a QaStalenessFinding with
          all stale citations and the max drift in days.
    2. Return findings sorted by ``max_drift_days`` desc, then slug asc.

    Pages whose ``frontmatter.updated`` cannot be parsed are skipped (defensive;
    C10 surfaces schema breakage separately).
    """
    findings: list[QaStalenessFinding] = []

    qa_dir = wiki_dir / "qa"
    if not qa_dir.exists():
        return findings

    # Build a lookup map: entity-slug -> Path (entities/ first, concepts/ second).
    # Filed Answer citations point at wiki entity/concept slugs (per ADR-0006
    # §Citation contract — wiki Section citations look like "<page-slug>#<heading>").
    entity_lookup: dict[str, Path] = {}
    for subdir_name in ("entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for page_path in subdir.glob("*.md"):
            entity_slug = page_path.stem
            # entities/ wins over concepts/ on collision (deterministic).
            entity_lookup.setdefault(entity_slug, page_path)

    for slug, page_path in _iter_qa_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            continue
        if fm.get("status") != "live":
            continue

        updated_str = fm.get("updated", "")
        if not updated_str:
            continue
        try:
            qa_updated = datetime.datetime.fromisoformat(str(updated_str).replace("Z", "+00:00"))
        except ValueError:
            continue
        if qa_updated.tzinfo is None:
            qa_updated = qa_updated.replace(tzinfo=datetime.UTC)

        raw_sources = fm.get("sources", [])
        if not isinstance(raw_sources, list) or not raw_sources:
            continue

        stale_citations: list[str] = []
        max_drift_seconds = 0.0
        for citation in raw_sources:
            if not citation:
                continue
            citation_str = str(citation)
            # Citation shape: "<entity-slug>#<heading-slug>" OR bare "<slug>"
            entity_slug = citation_str.split("#", 1)[0].strip()
            if not entity_slug:
                continue
            entity_path = entity_lookup.get(entity_slug)
            if entity_path is None or not entity_path.exists():
                # Missing entity file — not C9's concern; C9 only flags newer-mtime
                # drift. (No C-check currently flags qa→missing-entity; that's a
                # potential future C9.b or new check.)
                continue
            try:
                entity_mtime_ts = entity_path.stat().st_mtime
            except OSError:
                continue
            entity_mtime = datetime.datetime.fromtimestamp(entity_mtime_ts, tz=datetime.UTC)
            if entity_mtime > qa_updated:
                stale_citations.append(citation_str)
                drift_seconds = (entity_mtime - qa_updated).total_seconds()
                if drift_seconds > max_drift_seconds:
                    max_drift_seconds = drift_seconds

        if stale_citations:
            findings.append(
                QaStalenessFinding(
                    page_slug=slug,
                    stale_citations=stale_citations,
                    max_drift_days=round(max_drift_seconds / 86400.0, 1),
                )
            )

    findings.sort(key=lambda f: (-f.max_drift_days, f.page_slug))
    return findings


def _check_c10_qa_schema_validity(
    wiki_dir: Path,
) -> list[InvalidQaSchemaFinding]:
    """Surface qa pages with schema invalidities (status, type, question, count).

    PRD #78 Phase 5 amendment §"C10 — qa-schema-validity". Layer 3 of the
    orphan-visibility defence (PRD #78 Q8d): the indexer logs invalid status,
    the filing layer refuses to touch invalid pages, and C10 surfaces them in
    ``lint-report.md`` where curators habitually look.

    Validation rules (one finding per (page, broken property) tuple):
    - ``status``    ∈ ``{live, draft, stale, superseded}``
    - ``type``      == ``"qa"`` (mismatched page type under wiki/qa/)
    - ``question``  is present and non-empty string
    - ``count``     is present and a positive integer

    Pages whose frontmatter cannot be parsed (no ``---`` fence or YAML error)
    produce a single finding flagging ``frontmatter`` as the broken property —
    the curator sees the page surfaces and can investigate manually.

    Returns findings sorted by ``page_slug``, then ``property_name``.
    """
    findings: list[InvalidQaSchemaFinding] = []

    for slug, page_path in _iter_qa_pages(wiki_dir):
        fm = _parse_frontmatter(page_path)
        if fm is None:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="frontmatter",
                    offending_value="<unparseable>",
                )
            )
            continue

        # status
        if "status" not in fm:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="status",
                    offending_value="<missing>",
                )
            )
        else:
            status_val = fm.get("status")
            if not isinstance(status_val, str) or status_val not in _VALID_QA_STATUS_VALUES:
                findings.append(
                    InvalidQaSchemaFinding(
                        page_slug=slug,
                        property_name="status",
                        offending_value=str(status_val),
                    )
                )

        # type
        if "type" not in fm:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="type",
                    offending_value="<missing>",
                )
            )
        else:
            type_val = fm.get("type")
            if type_val != "qa":
                findings.append(
                    InvalidQaSchemaFinding(
                        page_slug=slug,
                        property_name="type",
                        offending_value=str(type_val),
                    )
                )

        # question — required + non-empty string
        raw_question = fm.get("question")
        if raw_question is None:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="question",
                    offending_value="<missing>",
                )
            )
        elif not isinstance(raw_question, str) or not raw_question.strip():
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="question",
                    offending_value=repr(raw_question),
                )
            )

        # count — required + positive integer
        if "count" not in fm:
            findings.append(
                InvalidQaSchemaFinding(
                    page_slug=slug,
                    property_name="count",
                    offending_value="<missing>",
                )
            )
        else:
            raw_count = fm.get("count")
            valid_count = False
            if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count > 0:
                valid_count = True
            if not valid_count:
                findings.append(
                    InvalidQaSchemaFinding(
                        page_slug=slug,
                        property_name="count",
                        offending_value=repr(raw_count),
                    )
                )

    findings.sort(key=lambda f: (f.page_slug, f.property_name))
    return findings


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_report_markdown(
    findings: LintFindings,
    summary: LintSummary,
    check_errors: dict[str, str],
) -> str:
    """Render the human-readable lint report as a markdown string.

    The report satisfies the following AC requirements:
    - Starts with sentinel HTML comment ``<!-- Auto-generated by POST /lint``
    - Contains ``# Lint Report`` heading
    - Contains a summary blockquote
    - Contains a ``## C11 Orphan pages`` section (empty when zero findings)
    - Contains a ``## C3 Failed grounding (<N> pages)`` section
    - Contains a ``## C4 Slug collision groups (<N> groups)`` section
    - Contains a ``## C6 Stale pages`` section with markdown table (Slice 5-3)
    - Contains a ``## C2 Red links`` section with markdown table (Slice 5-3)
    - Contains a ``## C1 Coverage gaps`` section (empty when zero findings)
    """
    lines: list[str] = []

    lines.append("<!-- Auto-generated by POST /lint — manual edits will be overwritten. -->")
    lines.append("")
    lines.append("# Lint Report")
    lines.append("")
    lines.append(
        f"> Generated at {summary.generated_at} · total findings: {summary.total_findings}"
    )
    lines.append("")

    # C11 Orphan pages section
    lines.append("## C11 Orphan pages")
    lines.append("")
    if not findings.orphans:
        lines.append("_No orphan pages found._")
    else:
        for orphan in findings.orphans:
            lines.append(f"### `{orphan.page_slug}`")
            lines.append("")
            lines.append(
                f"**Missing sources:** {', '.join(f'`{s}`' for s in orphan.missing_sources)}"
            )
            lines.append("")
            lines.append(f"**Suggested action:** {orphan.suggested_action}")
            lines.append("")

    lines.append("")

    # C3 Failed grounding section
    n_c3 = len(findings.failed_grounding)
    lines.append(f"## C3 Failed grounding ({n_c3} pages)")
    lines.append("")
    if not findings.failed_grounding:
        lines.append("_No failed-grounding pages found._")
    else:
        lines.append("| Page slug | Source | Reason | Unsupported claims |")
        lines.append("| --- | --- | --- | --- |")
        for fg in findings.failed_grounding:
            claims_cell = "; ".join(fg.unsupported_claims) if fg.unsupported_claims else "—"
            lines.append(f"| `{fg.page_slug}` | `{fg.source}` | {fg.reason} | {claims_cell} |")
        lines.append("")
        for fg in findings.failed_grounding:
            lines.append(f"**`{fg.page_slug}`** — {fg.suggested_action}")
            lines.append("")

    lines.append("")

    # C4 Slug collision groups section
    n_c4 = len(findings.slug_collisions)
    lines.append(f"## C4 Slug collision groups ({n_c4} groups)")
    lines.append("")
    if not findings.slug_collisions:
        lines.append("_No slug collision groups found._")
    else:
        lines.append("| Base slug | Pages in group | Group size |")
        lines.append("| --- | --- | --- |")
        for sc in findings.slug_collisions:
            pages_cell = ", ".join(f"`{p}`" for p in sc.pages_in_group)
            lines.append(f"| `{sc.base_slug}` | {pages_cell} | {len(sc.pages_in_group)} |")
        lines.append("")
        for sc in findings.slug_collisions:
            lines.append(f"**`{sc.base_slug}`** — {sc.suggested_action}")
            lines.append("")

    lines.append("")

    # C6 Stale pages section
    n_stale = len(findings.stale_pages)
    lines.append(f"## C6 Stale pages ({n_stale} page{'s' if n_stale != 1 else ''})")
    lines.append("")
    if not findings.stale_pages:
        lines.append("_No stale pages found._")
    else:
        lines.append("| Page | Source | Source mtime | Page updated | Drift (days) | Action |")
        lines.append("|------|--------|-------------|--------------|-------------|--------|")
        for stale in findings.stale_pages:
            src_mtime_str = stale.source_mtime.strftime("%Y-%m-%d")
            pg_updated_str = stale.page_updated.strftime("%Y-%m-%d")
            lines.append(
                f"| `{stale.page_slug}` | `{stale.source}` | {src_mtime_str} | {pg_updated_str}"
                f" | {stale.drift_days:.1f} | {stale.suggested_action} |"
            )
    lines.append("")

    # C2 Red links section
    n_red = len(findings.red_links)
    lines.append(f"## C2 Red links ({n_red} backlog item{'s' if n_red != 1 else ''})")
    lines.append("")
    if not findings.red_links:
        lines.append("_No red links found._")
    else:
        lines.append("| Target slug | Mentions | Referenced by | Sample context |")
        lines.append("|-------------|---------|---------------|----------------|")
        for rl in findings.red_links:
            # referenced_by rendered as comma-separated [[slug]] wikilinks
            ref_by_str = ", ".join(f"[[{s}]]" for s in rl.referenced_by)
            ctx_str = rl.sample_context.replace("|", "\\|") if rl.sample_context else ""
            lines.append(f"| `{rl.slug}` | {rl.mention_count} | {ref_by_str} | {ctx_str} |")
    lines.append("")

    # C1 Coverage gaps section
    n_c1 = len(findings.coverage_gaps)
    lines.append(f"## C1 Coverage gaps ({n_c1} findings)")
    lines.append("")
    if not findings.coverage_gaps:
        lines.append("_No coverage gaps found._")
        lines.append("")
    else:
        # Sub-group by reason in fixed order
        from collections import defaultdict as _dd

        by_reason: dict[str, list[CoverageGapFinding]] = _dd(list)
        for gap in findings.coverage_gaps:
            by_reason[gap.reason].append(gap)

        for reason in ("retrieval_empty", "below_threshold", "claim_unsupported"):
            group = by_reason.get(reason, [])
            if not group:
                continue
            lines.append(f"### Repeated {reason} ({len(group)})")
            lines.append("")
            for gap in group:
                lines.append(f"- **`{gap.query_canonical}`** (×{gap.hit_count})")
                lines.append(f"  - *{gap.suggested_action}*")
                if gap.sample_raw_queries:
                    samples = "; ".join(f'"{q}"' for q in gap.sample_raw_queries[:3])
                    lines.append(f"  - Sample queries: {samples}")
                lines.append(f"  - First seen: {gap.first_seen}  Last seen: {gap.last_seen}")
                lines.append("")

    # C5 Contradictions section
    n_c5 = len(findings.page_pairs)
    lines.append(f"## C5 Contradictions ({n_c5} findings)")
    lines.append("")
    if not findings.page_pairs:
        lines.append("_No page-pair contradictions found._")
        lines.append("")
    else:
        # Sub-group by severity in fixed order: direct → tension → duplicate
        from collections import defaultdict as _dd2

        by_sev: dict[str, list[PagePairFinding]] = _dd2(list)
        for ppf in findings.page_pairs:
            by_sev[ppf.severity].append(ppf)

        for sev in ("direct", "tension", "duplicate"):
            group = by_sev.get(sev, [])
            if not group:
                continue
            lines.append(f"### {sev.capitalize()} ({len(group)})")
            lines.append("")
            lines.append("| Page A | Page B | Page A claim | Page B claim | Suggested action |")
            lines.append("|--------|--------|-------------|-------------|------------------|")
            for ppf in group:
                claim_a = ppf.page_a_claim.replace("|", "\\|").replace("\n", " ")[:80]
                claim_b = ppf.page_b_claim.replace("|", "\\|").replace("\n", " ")[:80]
                action = ppf.suggested_action.replace("|", "\\|").replace("\n", " ")[:80]
                lines.append(
                    f"| `{ppf.page_a}` | `{ppf.page_b}` | {claim_a} | {claim_b} | {action} |"
                )
            lines.append("")
            for ppf in group:
                lines.append(f"**`{ppf.page_a}` ↔ `{ppf.page_b}`** — {ppf.summary}")
                lines.append("")

    # ---- Phase 6 Slice 6-5 sections (PRD #78 Phase 5 amendment) ----
    # Empty-findings → section omitted entirely, matching the existing pattern
    # used by the Coverage Gaps / Contradictions sub-groups above. This keeps
    # noise out of the report when the qa lifecycle is dormant.

    # ## Promotion Candidates (C8)
    if findings.promotion_candidates:
        lines.append("## Promotion Candidates")
        lines.append("")
        lines.append("| Slug | Question | Count | Age (days) | Cited |")
        lines.append("|------|----------|-------|------------|-------|")
        for pc in findings.promotion_candidates:
            q_cell = pc.question.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{pc.slug}` | {q_cell} | {pc.count} | {pc.age_days:.1f} | {pc.cited_count} |"
            )
        lines.append("")

    # ## Stale Filed Answers (C9)
    if findings.stale_filed_answers:
        lines.append("## Stale Filed Answers")
        lines.append("")
        lines.append("| Slug | Stale Entity Citations | Days Drift |")
        lines.append("|------|------------------------|------------|")
        for sf in findings.stale_filed_answers:
            cites_cell = ", ".join(f"`{c}`" for c in sf.stale_citations)
            lines.append(f"| `{sf.page_slug}` | {cites_cell} | {sf.max_drift_days:.1f} |")
        lines.append("")

    # ## Invalid qa Schema (C10)
    if findings.invalid_qa_schemas:
        lines.append("## Invalid qa Schema")
        lines.append("")
        lines.append("| Slug | Property | Offending Value |")
        lines.append("|------|----------|-----------------|")
        for inv in findings.invalid_qa_schemas:
            val_cell = inv.offending_value.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{inv.page_slug}` | `{inv.property_name}` | {val_cell} |")
        lines.append("")

    if check_errors:
        lines.append("")
        lines.append("## Check errors")
        lines.append("")
        for check_id, err_msg in check_errors.items():
            lines.append(f"- **{check_id}**: {err_msg}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Report writer (atomic)
# ---------------------------------------------------------------------------


def _write_report(report_path: Path, content: str) -> None:
    """Write lint-report.md atomically (tmp-file + os.replace)."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=report_path.parent,
        delete=False,
        suffix=".tmp",
    ) as fh:
        fh.write(content)
        tmp_name = fh.name
    os.replace(tmp_name, report_path)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def run_lint(
    *,
    wiki_dir: Path | None = None,
    docs_dir: Path | None = None,
    log_path: Path | None = None,
    include_c5: bool = True,
) -> LintResponse:
    """Run the lint checks; write wiki/lint-report.md; return LintResponse.

    Read-only with respect to wiki page frontmatter.
    Holds ``indexer._index_lock`` for the full duration.
    Continue-on-error: a check that raises is recorded in
    ``LintResponse.check_errors``; other checks still run.

    Parameters default to the module-level constants (``WIKI_DIR``, ``DOCS_DIR``,
    ``LOG_PATH``) so tests can monkeypatch those attributes without passing kwargs.

    ``include_c5`` (default True) controls the only LLM-backed check, C5
    (page-pair contradiction detection). C5 makes one LLM call per candidate
    page-pair, so it dominates lint wall-time and cost on a large wiki. Callers
    that only need the fast, local checks — notably the Console's Curation
    Queue, which reads C8/C9/C10 — pass ``include_c5=False`` to skip it; the
    response then carries ``page_pairs == []`` and ``llm_calls == 0``.

    Returns a LintResponse (which is also JSON-serialisable via FastAPI).
    """
    resolved_wiki = wiki_dir if wiki_dir is not None else WIKI_DIR
    resolved_docs = docs_dir if docs_dir is not None else DOCS_DIR
    resolved_log = log_path if log_path is not None else LOG_PATH

    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    log_event("lint_started", "", log_path=resolved_log)

    check_errors: dict[str, str] = {}

    # Hold the indexer lock for the entire check sequence so /ingest cannot mutate
    # wiki pages mid-snapshot. Lint is read-only, so the lock is purely a
    # consistency guard, not a write barrier on lint's side.
    with _index_lock:
        # --- C11 Orphan pages ---
        orphans: list[OrphanPageFinding] = []
        try:
            orphans = _check_c11_orphan(resolved_wiki, resolved_docs)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c11"] = err_msg
            log_event("lint_check_error", f"check=c11 exc={err_msg}", log_path=resolved_log)

        # --- C3 Failed-grounding sweep ---
        failed_grounding: list[FailedGroundingFinding] = []
        try:
            failed_grounding = _check_c3_failed_grounding(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c3"] = err_msg
            log_event("lint_check_error", f"check=c3 exc={err_msg}", log_path=resolved_log)

        # --- C4-a Slug collision groups ---
        slug_collisions: list[SlugCollisionFinding] = []
        try:
            slug_collisions = _check_c4a_slug_collision(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c4a"] = err_msg
            log_event("lint_check_error", f"check=c4a exc={err_msg}", log_path=resolved_log)

        # --- C6 Stale pages ---
        stale_pages: list[StalePageFinding] = []
        try:
            stale_pages = _check_c6_stale(resolved_wiki, resolved_docs)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c6"] = err_msg
            log_event("lint_check_error", f"check=c6 exc={err_msg}", log_path=resolved_log)

        # --- C2 Red links ---
        red_links: list[RedLinkFinding] = []
        try:
            red_links = _check_c2_red_links(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c2"] = err_msg
            log_event("lint_check_error", f"check=c2 exc={err_msg}", log_path=resolved_log)

        # --- C1 Coverage gaps ---
        coverage_gaps: list[CoverageGapFinding] = []
        try:
            coverage_gaps = _check_c1_coverage_gaps(resolved_log)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c1"] = err_msg
            log_event("lint_check_error", f"check=c1 exc={err_msg}", log_path=resolved_log)

        # --- C8 Promotion candidates (read-only qa scan) ---
        promotion_candidates: list[PromotionCandidateFinding] = []
        try:
            promotion_candidates = _check_c8_promotion_candidates(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c8"] = err_msg
            log_event("lint_check_error", f"check=c8 exc={err_msg}", log_path=resolved_log)

        # --- C9 qa-staleness (read-only entity-vs-qa mtime comparison) ---
        stale_filed_answers: list[QaStalenessFinding] = []
        try:
            stale_filed_answers = _check_c9_qa_staleness(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c9"] = err_msg
            log_event("lint_check_error", f"check=c9 exc={err_msg}", log_path=resolved_log)

        # --- C10 qa-schema-validity (read-only schema sweep) ---
        invalid_qa_schemas: list[InvalidQaSchemaFinding] = []
        try:
            invalid_qa_schemas = _check_c10_qa_schema_validity(resolved_wiki)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {exc}"
            check_errors["c10"] = err_msg
            log_event("lint_check_error", f"check=c10 exc={err_msg}", log_path=resolved_log)

        # --- C5 Page-pair contradiction detection (most expensive — last) ---
        # The only LLM-backed check: one call per candidate page-pair, so it
        # dominates wall-time and cost on a large wiki. Skipped wholesale when
        # include_c5 is False (fast path for the Curation Queue) — page_pairs,
        # llm_calls and cost_usd then keep their zero defaults below.
        page_pairs: list[PagePairFinding] = []
        llm_calls: int = 0
        cost_usd: float = 0.0
        if include_c5:
            try:
                page_pairs = _check_c5_page_pair(resolved_wiki)
                # Cost accounting (best-effort): count candidate pairs judged
                # _check_c5_page_pair calls _judge_page_pair once per pair.
                # We approximate llm_calls by counting all findings + "none" judgements
                # via the candidate pair count. Since we don't have direct access to
                # that count post-run, we count findings (severity != none) as a
                # lower bound and note this is approximate.
                # A more exact count would require _check_c5_page_pair to return the
                # call count; that is the content-hash-cache trigger improvement.
                # For now: count all page_pairs as each required one LLM call.
                # Non-"none" findings + estimated filtered pairs is tracked via the
                # module-level _c5_llm_call_counter which _judge_page_pair updates.
                llm_calls = _c5_llm_call_counter[0]
                # gpt-4o-mini pricing (2025): $0.15/1M input tokens, $0.60/1M output tokens
                # Rough estimate: ~500 input + 150 output tokens per pair call = ~$0.000165/call
                # Best-effort: $0.000165 * llm_calls
                cost_usd = round(llm_calls * 0.000165, 6)
                # Reset counter for next run
                _c5_llm_call_counter[0] = 0
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                check_errors["c5"] = err_msg
                log_event("lint_check_error", f"check=c5 exc={err_msg}", log_path=resolved_log)

    # --- Aggregate findings ---
    findings = LintFindings(
        orphans=orphans,
        failed_grounding=failed_grounding,
        slug_collisions=slug_collisions,
        stale_pages=stale_pages,
        red_links=red_links,
        coverage_gaps=coverage_gaps,
        page_pairs=page_pairs,
        promotion_candidates=promotion_candidates,
        stale_filed_answers=stale_filed_answers,
        invalid_qa_schemas=invalid_qa_schemas,
    )
    total = (
        len(orphans)
        + len(failed_grounding)
        + len(slug_collisions)
        + len(stale_pages)
        + len(red_links)
        + len(coverage_gaps)
        + len(page_pairs)
        + len(promotion_candidates)
        + len(stale_filed_answers)
        + len(invalid_qa_schemas)
    )
    findings_by_check: dict[str, int] = {
        "c11": len(orphans),
        "c3": len(failed_grounding),
        "c4a": len(slug_collisions),
        "c6": len(stale_pages),
        "c2": len(red_links),
        "c1": len(coverage_gaps),
        "c5": len(page_pairs),
        "c8": len(promotion_candidates),
        "c9": len(stale_filed_answers),
        "c10": len(invalid_qa_schemas),
    }

    summary = LintSummary(
        total_findings=total,
        findings_by_check=findings_by_check,
        llm_calls=llm_calls,
        cost_usd=cost_usd,
        generated_at=generated_at,
    )

    # --- Write report ---
    report_path = resolved_wiki / "lint-report.md"
    report_content = _render_report_markdown(findings, summary, check_errors)
    _write_report(report_path, report_content)

    # --- Log completed ---
    by_check_str = ",".join(f"{k}:{v}" for k, v in findings_by_check.items())
    log_event(
        "lint_completed",
        f"findings={total} by_check={by_check_str} llm_calls={llm_calls} cost_usd={cost_usd:.6f} errors={len(check_errors)}",
        log_path=resolved_log,
    )

    return LintResponse(
        report_path=str(report_path.relative_to(resolved_wiki.parent))
        if report_path.is_relative_to(resolved_wiki.parent)
        else str(report_path),
        findings=findings,
        summary=summary,
        check_errors=check_errors,
    )
