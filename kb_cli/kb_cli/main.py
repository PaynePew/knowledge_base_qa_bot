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
"""

from __future__ import annotations

import sys
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


def _print_lint_result(response: object) -> None:
    """Render a LintResponse to stdout in human-readable form.

    Format:
        Lint Pass — <total> finding(s)  [or "No findings — KB is clean."]

        C11 Orphan pages (<N>):
          • <page_slug> — <suggested_action>

        C3 Failed grounding (<N>):
          • <page_slug> — <reason>
          ...

        C5 Contradictions (<N>):
          • <page_a> ↔ <page_b> [<severity>] — <summary>

        Report written to: <report_path>

    Sections with zero findings are omitted from output to reduce noise.
    """
    findings = response.findings  # type: ignore[attr-defined]
    summary = response.summary  # type: ignore[attr-defined]
    report_path = response.report_path  # type: ignore[attr-defined]
    total = summary.total_findings

    if total == 0:
        typer.echo("No findings — KB is clean.")
    else:
        typer.echo(f"Lint Pass — {total} finding(s)")

    typer.echo("")

    # C11 Orphan pages
    if findings.orphans:
        typer.echo(f"C11 Orphan pages ({len(findings.orphans)}):")
        for f in findings.orphans:
            typer.echo(f"  • {f.page_slug} — {f.suggested_action}")
        typer.echo("")

    # C3 Failed grounding
    if findings.failed_grounding:
        typer.echo(f"C3 Failed grounding ({len(findings.failed_grounding)}):")
        for f in findings.failed_grounding:
            typer.echo(f"  • {f.page_slug} (reason: {f.reason}) — {f.suggested_action}")
        typer.echo("")

    # C4 Slug collisions
    if findings.slug_collisions:
        typer.echo(f"C4 Slug collisions ({len(findings.slug_collisions)}):")
        for f in findings.slug_collisions:
            pages = ", ".join(f.pages_in_group)
            typer.echo(f"  • {f.base_slug}: {pages}")
        typer.echo("")

    # C6 Stale pages
    if findings.stale_pages:
        typer.echo(f"C6 Stale pages ({len(findings.stale_pages)}):")
        for f in findings.stale_pages:
            typer.echo(f"  • {f.page_slug} — drift {f.drift_days:.1f} day(s)")
        typer.echo("")

    # C2 Red links
    if findings.red_links:
        typer.echo(f"C2 Red links ({len(findings.red_links)}):")
        for f in findings.red_links:
            typer.echo(f"  • [[{f.slug}]] — {f.mention_count} mention(s)")
        typer.echo("")

    # C1 Coverage gaps
    if findings.coverage_gaps:
        typer.echo(f"C1 Coverage gaps ({len(findings.coverage_gaps)}):")
        for f in findings.coverage_gaps:
            typer.echo(f"  • {f.query_canonical} (×{f.hit_count}) — {f.suggested_action}")
        typer.echo("")

    # C5 Contradictions
    if findings.page_pairs:
        typer.echo(f"C5 Contradictions ({len(findings.page_pairs)}):")
        for f in findings.page_pairs:
            typer.echo(f"  • {f.page_a} ↔ {f.page_b} [{f.severity}] — {f.summary}")
        typer.echo("")

    # C8 Promotion candidates
    if findings.promotion_candidates:
        typer.echo(f"C8 Promotion candidates ({len(findings.promotion_candidates)}):")
        for f in findings.promotion_candidates:
            typer.echo(f"  • {f.slug} — count {f.count}, age {f.age_days:.1f} day(s)")
        typer.echo("")

    # C9 Stale filed answers
    if findings.stale_filed_answers:
        typer.echo(f"C9 Stale filed answers ({len(findings.stale_filed_answers)}):")
        for f in findings.stale_filed_answers:
            typer.echo(f"  • {f.page_slug} — drift {f.max_drift_days:.1f} day(s)")
        typer.echo("")

    # C10 Invalid qa schemas
    if findings.invalid_qa_schemas:
        typer.echo(f"C10 Invalid qa schemas ({len(findings.invalid_qa_schemas)}):")
        for f in findings.invalid_qa_schemas:
            typer.echo(f"  • {f.page_slug}.{f.property_name} = {f.offending_value}")
        typer.echo("")

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
