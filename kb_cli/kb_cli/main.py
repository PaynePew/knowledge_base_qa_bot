"""CLI entry point for the knowledge base.

Phase 12 Slice 3 (ADR-0016).

Dual mode:
  - One-shot subcommands: ``kb ask "Q"`` and ``kb index``
  - Bare ``kb`` (no subcommand): interactive REPL with a warm index

The CLI wraps the same ``markdown_kb`` deep modules as the MCP server (ADR-0016:
direct deep-module adapter, NOT the Gateway).  ``reload_if_stale`` from
``kb_mcp.freshness`` is the shared mtime-reload mechanism for the REPL's
warm-index pattern.

``LLMError`` (ADR-0015) renders as a non-zero exit code plus a human-readable
message to stderr — no traceback.

``--stack rag`` routes to the vector_rag retrieval arm and ``--stack hybrid`` to
the Phase 13 Hybrid arm (``hybrid_kb``, BM25 + dense over wiki/ fused by RRF —
ADR-0018); both are called directly, stacks stay independent (ADR-0002).

Human-readable output: the ``ask`` path and the REPL render the full
``markdown_kb.app.retrieval.query`` result shape (``answer`` + ``sources`` +
``grounding_outcome``) via ``_print_result``.  This is the grounded-answer
shape, not the search-result shape that ``kb_mcp.normalizer`` maps for the MCP
``kb_search_v1`` tool — the normalizer is therefore not on the CLI path.

``kb qa`` (issue #377 / ADR-0026 decision 3) is a thin sub-command group over
``markdown_kb.app.qa`` for the Filed Answer curation loop — ``list`` / ``show``
/ ``promote`` / ``discard``, mirroring the Operator Console's gate semantics
(promote reindexes; discard refuses a live page).  This is deliberately a CLI
capability with **no MCP equivalent**: gates resolve on human surfaces only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv

# markdown_kb / vector_rag are PEP 420 namespace packages and `package = false`
# workspace members (root pyproject) — never installed, so they import only when the
# repo root is on sys.path. pytest provides it via `pythonpath` and `python -m` via
# cwd, but the installed `kb` console script has neither (sys.path[0] is the launcher
# dir), so the lazy `import markdown_kb...` in the command bodies dies with
# ModuleNotFoundError. Insert the repo root (this file is
# <repo>/kb_cli/kb_cli/main.py → parents[2]) so `kb` works from any cwd. Relies on
# the editable workspace layout, which always holds for this dev-only CLI.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The `kb` console script does not go through uv's env-file, so without this the
# grounded-answer path (`kb ask` / REPL) fails with an LLM auth error unless
# OPENAI_API_KEY is already exported.  Load `.env` from the cwd here — parity with
# markdown_kb.app.main / gateway.app.main / kb_mcp.__main__.  The env-reading
# modules (retrieval / KB_SCORE_THRESHOLD) are lazy-imported inside the command
# functions, so loading at import time suffices.
load_dotenv(find_dotenv(usecwd=True))

app = typer.Typer(
    name="kb",
    help=(
        "Knowledge-base CLI.  Bare ``kb`` enters the interactive REPL; "
        "``kb ask`` / ``kb index`` are one-shot subcommands."
    ),
    add_completion=False,
    invoke_without_command=True,
)

# Typer stack literal — kept as a plain string enum rather than Literal so
# the help text renders clearly. ``hybrid`` is the Phase 13 Stack C (ADR-0018:
# BM25 + dense over wiki/, fused by RRF), dispatched alongside wiki / rag.
_VALID_STACKS = ("wiki", "rag", "hybrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ask_one(question: str, stack: str) -> int:
    """Run one grounded-answer query and print the result to stdout.

    Returns the shell exit code: 0 on success, 1 on LLMError.
    """
    from markdown_kb.app.errors import LLMError

    try:
        result = _do_query(question, stack)
    except LLMError as exc:
        code_label = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        typer.echo(f"Error [{code_label}]: {exc.message}", err=True)
        return 1

    _print_result(result, stack)
    return 0


def _do_query(question: str, stack: str) -> dict:
    """Dispatch a question to the requested retrieval stack.

    Returns the raw retrieval result dict with keys:
        answer, sources, grounding_outcome
    """
    if stack == "wiki":
        from kb_mcp.freshness import reload_if_stale
        from markdown_kb.app.retrieval import query as wiki_query

        reload_if_stale()
        return wiki_query(question)
    elif stack == "hybrid":
        # ADR-0018 Stack C: BM25 + dense over wiki/, fused by RRF. reload_if_stale
        # warms the shared BM25 arm (same .kb/index.json as the wiki stack); the
        # dense arm is lazy-loaded inside hybrid query() on first use.
        from hybrid_kb.app.query import query as hybrid_query
        from kb_mcp.freshness import reload_if_stale

        reload_if_stale()
        return hybrid_query(question)
    else:
        from vector_rag.app.retrieval import query as rag_query  # type: ignore[import-untyped]

        return rag_query(question)


def _print_result(result: dict, stack: str) -> None:
    """Render a retrieval result dict to stdout in human-readable format.

    Format:
        Stack: wiki
        Answer: <text>

        Citations:
          [1] <section-id> (score: <score>)
              <content excerpt>

        Grounding: passed / cannot confirm (reason: <reason>)
    """
    grounding = result.get("grounding_outcome")
    sources = result.get("sources", [])
    answer = result.get("answer", "")

    typer.echo(f"Stack: {stack}")
    typer.echo(f"Answer: {answer}")

    if sources:
        typer.echo("\nCitations:")
        for i, src in enumerate(sources, 1):
            source_id = src.get("source", src.get("id", "?"))
            score = src.get("score")
            score_str = f" (score: {score:.3f})" if score is not None else ""
            content = src.get("content", "")[:120]
            typer.echo(f"  [{i}] {source_id}{score_str}")
            if content:
                typer.echo(f"      {content}")

    if grounding is not None:
        passed = grounding.passed
        reason = grounding.reason
        if passed:
            typer.echo("\nGrounding: passed")
        else:
            typer.echo(f"\nGrounding: cannot confirm (reason: {reason})")


# ---------------------------------------------------------------------------
# Subcommand: kb ask
# ---------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the knowledge base."),
    stack: str = typer.Option(
        "wiki",
        "--stack",
        help=(
            "Retrieval stack: 'wiki' (BM25, default), 'rag' (Vector RAG), "
            "or 'hybrid' (BM25 + dense over wiki, fused by RRF)."
        ),
    ),
) -> None:
    """Ask the knowledge base a question and print a grounded answer.

    On ``LLMError``, prints the error to stderr and exits with code 1.
    """
    if stack not in _VALID_STACKS:
        typer.echo(f"Error: --stack must be one of {_VALID_STACKS}", err=True)
        raise typer.Exit(code=2)

    exit_code = _ask_one(question, stack)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# ---------------------------------------------------------------------------
# Subcommand: kb index
# ---------------------------------------------------------------------------


@app.command(name="index")
def index_cmd() -> None:
    """(Re)build the Section Index from the wiki corpus.

    Scans the wiki SOURCE_DIRS, rebuilds the BM25 index in memory, and
    persists it to ``.kb/index.json``.
    """
    from markdown_kb.app.indexer import build_index

    files_indexed, sections_indexed = build_index()
    typer.echo(f"Indexed {files_indexed} file(s), {sections_indexed} section(s).")


# ---------------------------------------------------------------------------
# Subcommand: kb import
# ---------------------------------------------------------------------------


@app.command(name="import")
def import_cmd(
    path: str = typer.Argument(..., help="Path to the local file to import (.html, .txt, .md)."),
) -> None:
    """Import a local file into the knowledge base.

    Stages the file into ``raw/`` under its basename, then converts it to a
    ``docs/`` Source programmatically (HTML → Markdown, txt passthrough,
    .md recognised as canonical).  PDF is wired but the extractor is not yet
    available — a clear error is printed instead.

    Exit codes follow the ADR-0015 CLI contract:
        0   — success
        1   — import failure (invalid file, traversal, unsupported format)
        2   — bad CLI usage (missing argument, etc.)
    """
    from pathlib import Path

    from markdown_kb.app.importer import ImportPathError, import_path

    input_path = Path(path)
    try:
        result = import_path(input_path)
    except ImportPathError as exc:
        typer.echo(f"Error [{exc.error_type}]: {exc.message}", err=True)
        raise typer.Exit(code=1) from None
    except OSError as exc:
        typer.echo(f"Error [IOError]: {exc}", err=True)
        raise typer.Exit(code=1) from None

    # Concise success output: format, status, and docs path basename
    docs_basename = Path(result.docs_path).name
    typer.echo(
        f"Imported: {input_path.name} → docs/{docs_basename} "
        f"[{result.original_format}] status={result.status}"
    )


# ---------------------------------------------------------------------------
# Subcommand: kb ingest [source]
# ---------------------------------------------------------------------------


def _print_ingest_batch(batch: object) -> None:
    """Render an ``IngestBatchResult`` to stdout with per-source progress lines.

    Format per successful source:
        Ingested refund_policy.md: 2 page(s) created.

    Format for grounding-failed pages (fail-soft — page still written):
        Warning: page 'cancellation-window' failed grounding check.

    Format for failed sources:
        Failed: broken_source.md — could not be processed.

    Called after ``ingest_sources`` returns so all per-source outcomes are
    available.  The grounding-failure lines follow the source summary so a
    shell script can ``grep "Warning:"`` to detect non-fatal issues.
    """
    from markdown_kb.app.ingest import IngestBatchResult

    assert isinstance(batch, IngestBatchResult)

    # Per successfully processed source.  ``status`` is "created" or "updated"
    # here — skipped sources are carried in ``skipped_sources``, not ``results``.
    for src_result in batch.results:
        page_count = len(src_result.pages_written)
        typer.echo(f"Ingested {src_result.source}: {page_count} page(s) {src_result.status}.")

    # Per skipped source (hash-match no-op)
    for skipped in batch.skipped_sources:
        typer.echo(f"Skipped {skipped.source}: source unchanged (hash match).")

    # Per failed source.  Surface the deep module's reason (e.g. the size guard)
    # when present, otherwise the generic line.
    for failed in batch.failed_sources:
        reason = batch.failed_reasons.get(failed)
        if reason:
            typer.echo(f"Failed: {failed} — {reason}", err=True)
        else:
            typer.echo(f"Failed: {failed} — could not be processed.", err=True)

    # Per page with failed grounding (fail-soft — page still written, slug
    # tracked in batch.pages_with_failed_grounding rather than a status field).
    for slug in batch.pages_with_failed_grounding:
        typer.echo(f"Warning: page '{slug}' failed grounding check.")


@app.command(name="ingest")
def ingest_cmd(
    source: str | None = typer.Argument(
        None,
        help=(
            "Bare filename of a single Source to ingest (e.g. 'refund_policy.md'). "
            "Omit to batch-ingest all Sources under docs/."
        ),
    ),
) -> None:
    """Synthesise one or all docs/ Sources into wiki page(s).

    ``kb ingest <source>`` ingests a single named Source.
    ``kb ingest`` (no argument) batch-ingests all Sources under docs/.

    Execution is synchronous.  Per-source progress is printed to stdout as
    each source completes.  A grounding-failed page is reported but does NOT
    cause a non-zero exit (fail-soft per ADR-0004).

    On ``LLMError``, prints the error to stderr and exits with code 1.
    """
    from markdown_kb.app.errors import LLMError
    from markdown_kb.app.ingest import ingest_sources

    source_filenames = [source] if source is not None else None

    if source is not None:
        typer.echo(f"Ingesting {source}...")
    else:
        typer.echo("Batch-ingesting all docs/ Sources...")

    try:
        batch = ingest_sources(source_filenames)
    except LLMError as exc:
        code_label = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        typer.echo(f"Error [{code_label}]: {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    _print_ingest_batch(batch)


# ---------------------------------------------------------------------------
# Subcommand: kb lint
# ---------------------------------------------------------------------------


def _format_c11_orphans(findings: object, label: str) -> list[str]:
    """C11 Orphan pages lines, or ``[]`` when there are none."""
    if not findings.orphans:  # type: ignore[attr-defined]
        return []
    lines = [f"C11 Orphan pages ({len(findings.orphans)}) — {label}:"]  # type: ignore[attr-defined]
    for f in findings.orphans:  # type: ignore[attr-defined]
        lines.append(f"  • {f.page_slug} — {f.suggested_action}")
    lines.append("")
    return lines


def _format_c3_failed_grounding(findings: object, label: str) -> list[str]:
    """C3 Failed grounding lines, or ``[]`` when there are none."""
    if not findings.failed_grounding:  # type: ignore[attr-defined]
        return []
    lines = [
        f"C3 Failed grounding ({len(findings.failed_grounding)}) — {label}:"  # type: ignore[attr-defined]
    ]
    for f in findings.failed_grounding:  # type: ignore[attr-defined]
        lines.append(f"  • {f.page_slug} (reason: {f.reason}) — {f.suggested_action}")
    lines.append("")
    return lines


def _format_c4_slug_collisions(findings: object, label: str) -> list[str]:
    """C4 Slug collisions lines, or ``[]`` when there are none."""
    if not findings.slug_collisions:  # type: ignore[attr-defined]
        return []
    lines = [
        f"C4 Slug collisions ({len(findings.slug_collisions)}) — {label}:"  # type: ignore[attr-defined]
    ]
    for f in findings.slug_collisions:  # type: ignore[attr-defined]
        pages = ", ".join(f.pages_in_group)
        lines.append(f"  • {f.base_slug}: {pages}")
    lines.append("")
    return lines


def _format_c6_stale_pages(findings: object, label: str) -> list[str]:
    """C6 Stale pages lines, or ``[]`` when there are none."""
    if not findings.stale_pages:  # type: ignore[attr-defined]
        return []
    lines = [f"C6 Stale pages ({len(findings.stale_pages)}) — {label}:"]  # type: ignore[attr-defined]
    for f in findings.stale_pages:  # type: ignore[attr-defined]
        lines.append(f"  • {f.page_slug} — drift {f.drift_days:.1f} day(s)")
    lines.append("")
    return lines


def _format_c2_red_links(findings: object, label: str) -> list[str]:
    """C2 Red links lines, or ``[]`` when there are none."""
    if not findings.red_links:  # type: ignore[attr-defined]
        return []
    lines = [f"C2 Red links ({len(findings.red_links)}) — {label}:"]  # type: ignore[attr-defined]
    for f in findings.red_links:  # type: ignore[attr-defined]
        lines.append(f"  • [[{f.slug}]] — {f.mention_count} mention(s)")
    lines.append("")
    return lines


def _format_c1_coverage_gaps(findings: object, label: str) -> list[str]:
    """C1 Coverage gaps lines, or ``[]`` when there are none."""
    if not findings.coverage_gaps:  # type: ignore[attr-defined]
        return []
    lines = [f"C1 Coverage gaps ({len(findings.coverage_gaps)}) — {label}:"]  # type: ignore[attr-defined]
    for f in findings.coverage_gaps:  # type: ignore[attr-defined]
        lines.append(f"  • {f.query_canonical} (×{f.hit_count}) — {f.suggested_action}")
    lines.append("")
    return lines


def _format_c5_contradictions(findings: object, label: str) -> list[str]:
    """C5 Contradictions lines, or ``[]`` when there are none."""
    if not findings.page_pairs:  # type: ignore[attr-defined]
        return []
    lines = [f"C5 Contradictions ({len(findings.page_pairs)}) — {label}:"]  # type: ignore[attr-defined]
    for f in findings.page_pairs:  # type: ignore[attr-defined]
        lines.append(f"  • {f.page_a} ↔ {f.page_b} [{f.severity}] — {f.summary}")
    lines.append("")
    return lines


def _format_c8_promotion_candidates(findings: object, label: str) -> list[str]:
    """C8 Promotion candidates lines, or ``[]`` when there are none.

    Includes ``question`` and the on-disk ``path`` per finding (issue #377 /
    ADR-0026 decision 3: any surface — CLI or MCP — can inspect what needs
    curating). ``path`` is pure string formatting (``qa_view.display_path``);
    no file read is required for C8 since ``PromotionCandidateFinding``
    already carries ``question``.
    """
    if not findings.promotion_candidates:  # type: ignore[attr-defined]
        return []
    from kb_mcp import qa_view

    lines = [
        f"C8 Promotion candidates ({len(findings.promotion_candidates)}) — {label}:"  # type: ignore[attr-defined]
    ]
    for f in findings.promotion_candidates:  # type: ignore[attr-defined]
        lines.append(f"  • {f.slug} — count {f.count}, age {f.age_days:.1f} day(s)")
        lines.append(f'    question: "{f.question}"')
        lines.append(f"    path: {qa_view.display_path(f.slug)}")
    lines.append("")
    return lines


def _c9_question_or_placeholder(slug: str) -> str:
    """The C9-slug's question, backfilled from disk, or a placeholder.

    ``QaStalenessFinding`` carries no ``question`` field, so both the ``kb
    lint`` C9 renderer and ``kb qa list`` backfill it the same way via
    ``qa_view.read_qa_page`` — falls back to a placeholder when the page is
    no longer readable (e.g. deleted since the lint scan) rather than raising.
    """
    from kb_mcp import qa_view

    page = qa_view.read_qa_page(slug)
    if page is not None and page.question:
        return page.question
    return "(question unavailable)"


def _format_c9_stale_filed_answers(findings: object, label: str) -> list[str]:
    """C9 Stale filed answers lines, or ``[]`` when there are none.

    Includes ``question`` and the on-disk ``path`` per finding (issue #377 /
    ADR-0026 decision 3).
    """
    if not findings.stale_filed_answers:  # type: ignore[attr-defined]
        return []
    from kb_mcp import qa_view

    lines = [
        f"C9 Stale filed answers ({len(findings.stale_filed_answers)}) — {label}:"  # type: ignore[attr-defined]
    ]
    for f in findings.stale_filed_answers:  # type: ignore[attr-defined]
        lines.append(f"  • {f.page_slug} — drift {f.max_drift_days:.1f} day(s)")
        lines.append(f'    question: "{_c9_question_or_placeholder(f.page_slug)}"')
        lines.append(f"    path: {qa_view.display_path(f.page_slug)}")
    lines.append("")
    return lines


def _format_c10_invalid_qa_schemas(findings: object, label: str) -> list[str]:
    """C10 Invalid qa schemas lines, or ``[]`` when there are none."""
    if not findings.invalid_qa_schemas:  # type: ignore[attr-defined]
        return []
    lines = [
        f"C10 Invalid qa schemas ({len(findings.invalid_qa_schemas)}) — {label}:"  # type: ignore[attr-defined]
    ]
    for f in findings.invalid_qa_schemas:  # type: ignore[attr-defined]
        lines.append(f"  • {f.page_slug}.{f.property_name} = {f.offending_value}")
    lines.append("")
    return lines


# code -> formatter, so _print_lint_result's axis loop below only decides
# ordering (via group_findings_by_axis), not per-check rendering.
_LINT_CHECK_FORMATTERS: dict[str, Callable[[object, str], list[str]]] = {
    "C11": _format_c11_orphans,
    "C3": _format_c3_failed_grounding,
    "C4": _format_c4_slug_collisions,
    "C6": _format_c6_stale_pages,
    "C2": _format_c2_red_links,
    "C1": _format_c1_coverage_gaps,
    "C5": _format_c5_contradictions,
    "C8": _format_c8_promotion_candidates,
    "C9": _format_c9_stale_filed_answers,
    "C10": _format_c10_invalid_qa_schemas,
}


def _print_lint_result(response: object) -> None:
    """Render a LintResponse to stdout, grouped under its four Lint Axis headers.

    Format:
        Lint Pass — <total> finding(s)  [or "No findings — KB is clean."]

        == Freshness ==

        C6 Stale pages (<N>) — stale:
          • <page_slug> — drift <N> day(s)

        C3 Failed grounding (<N>) — failed-grounding:
          • <page_slug> — <reason>
          ...

        == Coherence ==
        ...

        Report written to: <report_path>

    Axis grouping and check order (Freshness -> Coherence -> Coverage ->
    Lifecycle, checks in taxonomy order) come from
    ``markdown_kb.app.lint.group_findings_by_axis`` and its
    ``LINT_CHECK_TAXONOMY`` (issue #361 S1) — this renderer does not
    re-derive the mapping, per ADR-0017 interface parity with the report and
    MCP surfaces. An axis header is omitted when every check beneath it has
    zero findings, matching the report's empty-axis elision.
    """
    from markdown_kb.app.lint import group_findings_by_axis

    findings = response.findings  # type: ignore[attr-defined]
    summary = response.summary  # type: ignore[attr-defined]
    report_path = response.report_path  # type: ignore[attr-defined]
    total = summary.total_findings

    if total == 0:
        typer.echo("No findings — KB is clean.")
    else:
        typer.echo(f"Lint Pass — {total} finding(s)")

    typer.echo("")

    for axis_group in group_findings_by_axis(findings):
        axis_lines: list[str] = []
        for meta, _findings_list in axis_group.checks:
            axis_lines.extend(_LINT_CHECK_FORMATTERS[meta.code](findings, meta.label))
        if not axis_lines:
            continue
        typer.echo(f"== {axis_group.axis} ==")
        typer.echo("")
        for line in axis_lines:
            typer.echo(line)

    typer.echo(f"Report written to: {report_path}")


@app.command(name="lint")
def lint_cmd() -> None:
    """Run the Lint Pass health check and render findings to stdout.

    Checks the wiki for contradictions, stale claims, orphan pages, slug
    collisions, coverage gaps, and red-link backlog — the same health check
    the Browser exposes via ``POST /lint``.

    On ``LLMError`` (raised by the C5 contradiction check which calls the LLM),
    prints the error to stderr and exits with code 1 (ADR-0015 CLI contract).
    """
    from markdown_kb.app.errors import LLMError
    from markdown_kb.app.lint import run_lint

    try:
        response = run_lint()
    except LLMError as exc:
        code_label = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
        typer.echo(f"Error [{code_label}]: {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    _print_lint_result(response)


# ---------------------------------------------------------------------------
# Subcommand group: kb qa (list / show / promote / discard)
# ---------------------------------------------------------------------------
# Issue #377 / ADR-0026 decision 3: gates resolve on human surfaces only. This
# group is a thin wrapper over markdown_kb.app.qa's PUBLIC promote/delete
# (mirroring POST /qa/{slug}/promote and DELETE /qa/{slug}) plus the read-only
# kb_mcp.qa_view helper for list/show. There is deliberately no MCP
# equivalent — see kb_mcp/kb_mcp/server.py's kb_lint_v1 visibility note and
# kb_mcp/tests/test_no_gate_resolving_tools.py.

qa_app = typer.Typer(
    name="qa",
    help=(
        "Curate Filed Answers (wiki/qa/) — list C8/C9 candidates, inspect "
        "one, promote a draft to live, or discard an inert page. Mirrors the "
        "Operator Console's gate semantics (promote reindexes; discard "
        "refuses a live page)."
    ),
    add_completion=False,
)
app.add_typer(qa_app, name="qa")


def _print_qa_candidate(slug: str, status: str, question: str, detail: str, path: str) -> None:
    """Shared one-candidate rendering line for ``kb qa list``."""
    typer.echo(f"  • {slug} [{status}] — {detail}")
    typer.echo(f'    question: "{question}"')
    typer.echo(f"    path: {path}")


@qa_app.command(name="list")
def qa_list_cmd() -> None:
    """List C8 promotion candidates (draft) and C9 stale Filed Answers (live).

    Runs only the fast local Lint checks (``include_c5=False`` — no LLM
    call), the same call shape the Operator Console's Curation Queue uses
    (``markdown_kb.app.lint.run_lint`` docstring). C9's question is backfilled
    by reading the page (``QaStalenessFinding`` carries no question field).
    """
    from kb_mcp import qa_view
    from markdown_kb.app.lint import run_lint

    response = run_lint(include_c5=False)
    candidates = response.findings.promotion_candidates
    stale = response.findings.stale_filed_answers

    if not candidates and not stale:
        typer.echo("No Filed Answers need curation — queue is empty.")
        return

    typer.echo(
        f"Filed Answers — {len(candidates)} promotion candidate(s), {len(stale)} stale live page(s)"
    )
    typer.echo("")

    if candidates:
        typer.echo("C8 Promotion candidates (draft):")
        for c in candidates:
            _print_qa_candidate(
                c.slug,
                "draft",
                c.question,
                f"count={c.count}, age={c.age_days:.1f}d",
                qa_view.display_path(c.slug),
            )
        typer.echo("")

    if stale:
        typer.echo("C9 Stale filed answers (live):")
        for s in stale:
            _print_qa_candidate(
                s.page_slug,
                "live",
                _c9_question_or_placeholder(s.page_slug),
                f"drift={s.max_drift_days:.1f}d",
                qa_view.display_path(s.page_slug),
            )
        typer.echo("")


@qa_app.command(name="show")
def qa_show_cmd(
    slug: str = typer.Argument(..., help="The wiki/qa/<slug>.md slug to inspect."),
) -> None:
    """Show one Filed Answer's question, answer body, sources, and status.

    Exit codes follow the ADR-0015 CLI contract: 0 on success, 1 when the
    slug has never been filed.
    """
    from kb_mcp import qa_view

    page = qa_view.read_qa_page(slug)
    if page is None:
        typer.echo(f"Error: wiki/qa/{slug}.md not found.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Slug: {page.slug}")
    typer.echo(f"Status: {page.status}")
    typer.echo(f"Question: {page.question}")
    typer.echo(f"Sources: {', '.join(page.sources) if page.sources else '(none)'}")
    typer.echo(f"Path: {page.path}")
    typer.echo("")
    typer.echo("Answer:")
    typer.echo(page.body)


@qa_app.command(name="promote")
def qa_promote_cmd(
    slug: str = typer.Argument(..., help="The wiki/qa/<slug>.md slug to promote to live."),
) -> None:
    """Promote a draft Filed Answer to live, then reindex.

    Mirrors ``POST /qa/{slug}/promote``: flips ``status: draft -> live`` via
    ``markdown_kb.app.qa.promote`` (idempotent on an already-live page), then
    calls ``build_index()`` so the page is retrievable immediately — the same
    two-step contract the HTTP route uses (ADR-0020 Consequence 1).

    Exit codes: 0 on success, 1 when the slug was never filed or its
    frontmatter is corrupt (ADR-0015 CLI contract).
    """
    from markdown_kb.app import qa as qa_module
    from markdown_kb.app.indexer import build_index

    try:
        result = qa_module.promote(slug)
    except qa_module.QaPageNotFound:
        typer.echo(f"Error: wiki/qa/{slug}.md not found.", err=True)
        raise typer.Exit(code=1) from None
    except qa_module.QaPageCorrupt as exc:
        typer.echo(f"Error: wiki/qa/{slug}.md has corrupt frontmatter: {exc}", err=True)
        raise typer.Exit(code=1) from None

    files_indexed, sections_indexed = build_index()
    typer.echo(f"Promoted {result.slug}: status={result.status}, count={result.count}")
    typer.echo(f"Reindexed {files_indexed} file(s), {sections_indexed} section(s).")


@qa_app.command(name="discard")
def qa_discard_cmd(
    slug: str = typer.Argument(..., help="The wiki/qa/<slug>.md slug to discard."),
) -> None:
    """Discard an inert (non-live) Filed Answer.

    Mirrors ``DELETE /qa/{slug}`` via ``markdown_kb.app.qa.delete``: any
    non-live page (draft, or schema-invalid/unparseable frontmatter) may be
    discarded; a ``status: live`` page is refused with a clear message (ADR-0012
    — live pages are the served corpus and are not removed by a one-click
    action).

    Exit codes: 0 on success, 1 when the slug was never filed or is live
    (ADR-0015 CLI contract).
    """
    from markdown_kb.app import qa as qa_module

    try:
        result = qa_module.delete(slug)
    except qa_module.QaPageNotFound:
        typer.echo(f"Error: wiki/qa/{slug}.md not found.", err=True)
        raise typer.Exit(code=1) from None
    except qa_module.QaPageLive:
        typer.echo(
            f"Error: wiki/qa/{slug}.md has status=live; discard refused. "
            "Live pages are the served corpus — re-ingest to refresh a stale "
            "live page instead of discarding it.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    typer.echo(f"Discarded {result.slug} (was status={result.prev_status}).")


# ---------------------------------------------------------------------------
# REPL — bare ``kb`` (no subcommand)
# ---------------------------------------------------------------------------


def _run_repl(stack: str = "wiki") -> None:
    """Interactive REPL: loads the index warm once, reuses it across queries.

    Prompt: ``kb>``
    Commands:
        :stack <wiki|rag|hybrid>  — toggle the retrieval stack
        quit / exit / :q          — exit the REPL
        <anything else>     — treated as a query

    The index is loaded once on REPL start via ``reload_if_stale()`` so the
    first query does not pay the cold-start I/O cost.  Subsequent queries
    reuse the in-process index; ``reload_if_stale()`` before each query checks
    the mtime and only reloads when the operator has rebuilt ``.kb/index.json``.

    ``LLMError`` is caught per-query and printed to stderr without crashing
    the REPL — the user can try again or quit.
    """
    from kb_mcp.freshness import reload_if_stale

    # Warm the index once at REPL start.
    reload_if_stale()

    typer.echo(
        "Knowledge base REPL.  Type a question, ':stack <wiki|rag|hybrid>' to toggle, or 'quit'."
    )

    current_stack = stack

    while True:
        try:
            line = input(f"kb[{current_stack}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nBye.")
            break

        if not line:
            continue

        # Exit commands
        if line.lower() in ("quit", "exit", ":q", "q"):
            typer.echo("Bye.")
            break

        # Stack toggle
        if line.startswith(":stack"):
            parts = line.split()
            if len(parts) == 2 and parts[1] in _VALID_STACKS:
                current_stack = parts[1]
                typer.echo(f"Stack switched to: {current_stack}")
            else:
                typer.echo(f"Usage: :stack <{'|'.join(_VALID_STACKS)}>", err=True)
            continue

        # Query
        from markdown_kb.app.errors import LLMError

        try:
            result = _do_query(line, current_stack)
        except LLMError as exc:
            code_label = "LLM_UNAVAILABLE" if exc.retryable else "LLM_ERROR"
            typer.echo(f"Error [{code_label}]: {exc.message}", err=True)
            continue

        _print_result(result, current_stack)
        typer.echo("")  # blank line between answers


# ---------------------------------------------------------------------------
# Root callback — enter REPL when no subcommand is given
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _root_callback(ctx: typer.Context) -> None:
    """Entry point for bare ``kb``.  Enters the REPL when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        _run_repl()
