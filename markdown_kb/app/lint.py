"""Deep module per Ousterhout. Public surface: ``run_lint``, ``_check_c11_orphan``, ``_check_c3_failed_grounding``, ``_check_c4a_slug_collision``, ``_check_c6_stale``, ``_check_c2_red_links``, ``_check_c1_coverage_gaps``, ``_canonicalise``.

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
7. Future slices: C5

Authorised by PRD #65 (Phase 5), GitHub issue #66 (Slice 5-1), GitHub issue #67 (Slice 5-2), GitHub issue #68 (Slice 5-3), and GitHub issue #69 (Slice 5-4).
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
    LintFindings,
    LintResponse,
    LintSummary,
    OrphanPageFinding,
    RedLinkFinding,
    SlugCollisionFinding,
    StalePageFinding,
)

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

    Returns None if the page has no frontmatter block, the block cannot be
    parsed, or the result is not a dict.
    """
    try:
        text = page_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        fm = yaml.safe_load(parts[1])
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
) -> LintResponse:
    """Run all lint checks; write wiki/lint-report.md; return LintResponse.

    Read-only with respect to wiki page frontmatter.
    Holds ``indexer._index_lock`` for the full duration.
    Continue-on-error: a check that raises is recorded in
    ``LintResponse.check_errors``; other checks still run.

    Parameters default to the module-level constants (``WIKI_DIR``, ``DOCS_DIR``,
    ``LOG_PATH``) so tests can monkeypatch those attributes without passing kwargs.

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

    # --- Aggregate findings ---
    findings = LintFindings(
        orphans=orphans,
        failed_grounding=failed_grounding,
        slug_collisions=slug_collisions,
        stale_pages=stale_pages,
        red_links=red_links,
        coverage_gaps=coverage_gaps,
    )
    total = (
        len(orphans)
        + len(failed_grounding)
        + len(slug_collisions)
        + len(stale_pages)
        + len(red_links)
        + len(coverage_gaps)
    )
    findings_by_check: dict[str, int] = {
        "c11": len(orphans),
        "c3": len(failed_grounding),
        "c4a": len(slug_collisions),
        "c6": len(stale_pages),
        "c2": len(red_links),
        "c1": len(coverage_gaps),
    }

    summary = LintSummary(
        total_findings=total,
        findings_by_check=findings_by_check,
        llm_calls=0,
        cost_usd=0.0,
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
        f"findings={total} by_check={by_check_str} llm_calls=0 cost_usd=0.000 errors={len(check_errors)}",
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
