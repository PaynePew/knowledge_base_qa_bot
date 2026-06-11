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

``--stack rag`` routes to the vector_rag retrieval arm (ADR-0002: stacks stay
independent, called directly).

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
# the help text renders clearly.
_VALID_STACKS = ("wiki", "rag")


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
        help="Retrieval stack: 'wiki' (BM25, default) or 'rag' (Vector RAG).",
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
# REPL — bare ``kb`` (no subcommand)
# ---------------------------------------------------------------------------


def _run_repl(stack: str = "wiki") -> None:
    """Interactive REPL: loads the index warm once, reuses it across queries.

    Prompt: ``kb>``
    Commands:
        :stack <wiki|rag>   — toggle the retrieval stack
        quit / exit / :q    — exit the REPL
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

    typer.echo("Knowledge base REPL.  Type a question, ':stack <wiki|rag>' to toggle, or 'quit'.")

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
