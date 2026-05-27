"""Deep module per Ousterhout. Public surface: ``run_lint``, ``_check_c11_orphan``.

Lint orchestrator — POST /lint health check for the wiki.

Provides ``run_lint(*, wiki_dir, docs_dir, log_path)`` which orchestrates the
lint checks and writes ``wiki/lint-report.md``.

Slice 5-1 scope
---------------
Only C11 (orphan pages) is wired.  Subsequent slices add the remaining six
checks without changing the orchestrator contract or the continue-on-error
semantics established here.

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
written, and the error is recorded in ``LintReport.check_errors``.  Other checks
still run.  The report is always written.

Check execution order (cheapest to most expensive)
---------------------------------------------------
1. C11 orphan (read frontmatter only)         ← Slice 5-1 only
2–7. Future slices: C3, C4-a, C6, C2, C1, C5

Authorised by PRD #65 (Phase 5) and GitHub issue #66 (Slice 5-1).
"""

from __future__ import annotations

import datetime
import os
import tempfile
from pathlib import Path

import yaml

from ._paths import DOCS_DIR, WIKI_DIR
from .indexer import _index_lock
from .logger import LOG_PATH, log_event
from .schemas import (
    LintFindings,
    LintResponse,
    LintSummary,
    OrphanPageFinding,
)

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

    if check_errors:
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

    # --- C11 Orphan pages ---
    orphans: list[OrphanPageFinding] = []
    try:
        with _index_lock:
            orphans = _check_c11_orphan(resolved_wiki, resolved_docs)
    except Exception as exc:  # noqa: BLE001
        err_msg = f"{type(exc).__name__}: {exc}"
        check_errors["c11"] = err_msg
        log_event("lint_check_error", f"check=c11 exc={err_msg}", log_path=resolved_log)

    # --- Aggregate findings ---
    findings = LintFindings(orphans=orphans)
    total = len(orphans)
    findings_by_check: dict[str, int] = {"c11": len(orphans)}

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
