"""Deep module per Ousterhout. Public surface: ``run_lint``, ``_check_c11_orphan``, ``_check_c3_failed_grounding``, ``_check_c4a_slug_collision``.

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
4–7. Future slices: C6, C2, C1, C5

Authorised by PRD #65 (Phase 5), GitHub issue #66 (Slice 5-1), GitHub issue #67 (Slice 5-2).
"""

from __future__ import annotations

import datetime
import os
import re
import tempfile
from pathlib import Path

import yaml

from ._paths import DOCS_DIR, WIKI_DIR
from .indexer import _index_lock
from .logger import LOG_PATH, log_event
from .schemas import (
    FailedGroundingFinding,
    LintFindings,
    LintResponse,
    LintSummary,
    OrphanPageFinding,
    SlugCollisionFinding,
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

    # --- Aggregate findings ---
    findings = LintFindings(
        orphans=orphans,
        failed_grounding=failed_grounding,
        slug_collisions=slug_collisions,
    )
    total = len(orphans) + len(failed_grounding) + len(slug_collisions)
    findings_by_check: dict[str, int] = {
        "c11": len(orphans),
        "c3": len(failed_grounding),
        "c4a": len(slug_collisions),
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
