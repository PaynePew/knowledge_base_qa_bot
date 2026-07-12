"""Deep module per Ousterhout. Public surface: ``run_contaminated_session``,
``render_report``, ``main``.

Run the contaminated-session drift eval (#608, split from #579's "(b)" report
finding) and render the trust-marked report.

The ONE LLM call this eval can make is ``rewrite_query`` on the CONTAMINATED
history arm — the clean-history control arm is always empty history, so
``rewrite_query`` takes its turn-1-passthrough branch (no LLM call there).
Retrieval + gating (``driver._gate``) is always LLM-free and deterministic.
This eval never adds a second live test for the Query Rewriting surface
(CODING_STANDARD §6.4): it drives the real ``rewrite_query`` deep module
directly from this CLI script, exactly as ``eval.negative_case`` drives
``retrieval._retrieve_and_gate`` — never from pytest.

Trust level (CODING_STANDARD §6.6, issue #328 precedent): a real run (real
``OPENAI_API_KEY``, real ``rewrite_query``) writes the canonical
``report.md``. Without a key (or ``--fake``), a deterministic stand-in
rewrite runs instead and the report is routed to
``report.offline-tracer.md`` behind the loud placeholder header — it proves
the harness's plumbing end to end, it is NOT a real drift measurement.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger

from .driver import ContaminatedSessionOutcome, RewriteFn, evaluate_case, index_corpus
from .sessions import CASES, ContaminatedSessionCase

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "report.md"

# ---------------------------------------------------------------------------
# Offline-tracer trust-level artifact (CODING_STANDARD §6.6, issue #328)
# ---------------------------------------------------------------------------
# A run with no real OPENAI_API_KEY (or --fake) must write to THIS path, never
# to REPORT_PATH. Kept as a module constant (not inlined) so tests can
# monkeypatch both the CLI routing and the header assertion, mirroring
# eval.paraphrase_comparison.runner's OFFLINE_TRACER_REPORT_PATH / HEADER.
OFFLINE_TRACER_REPORT_PATH = _PKG_ROOT / "report.offline-tracer.md"

OFFLINE_TRACER_HEADER = (
    "⚠️ PLACEHOLDER — NOT REAL DATA. The rewrite step below used a "
    "deterministic stand-in, not the real gateway.app.query_rewriting LLM "
    "call (no OPENAI_API_KEY, or --fake was passed). Do not interpret these "
    "numbers as real drift measurements."
)


def _offline_rewrite_stub(raw_query: str, *, history: list[dict]) -> str:
    """Deterministic, LLM-free stand-in for ``rewrite_query`` (offline-tracer path only).

    Preserves the real function's turn-1-passthrough contract (empty history
    -> query unchanged). With history, naively appends the prior turn's
    stored question as bracketed context — enough to exercise the
    drift/flip plumbing end to end without an API call. Never a substitute
    for the real rewrite's judgment; a real run's numbers will differ.

    ``history`` is keyword-only to match ``RewriteFn`` /
    ``gateway.app.query_rewriting.rewrite_query`` exactly (#608): a
    positional-only stub here would pass every test in this suite while
    crashing against the real seam.
    """
    if not history:
        return raw_query
    return f"{raw_query} [{history[-1]['question']}]"


def run_contaminated_session(
    rewrite_fn: RewriteFn,
    cases: tuple[ContaminatedSessionCase, ...] = CASES,
    corpus_dir: Path | None = None,
) -> list[ContaminatedSessionOutcome]:
    """Build the corpus index, run every case, return per-case outcomes.

    Assumes the caller has isolated production paths (see
    ``_isolate_production_paths`` / the test suite's autouse conftest).
    """
    index_corpus(corpus_dir)
    return [evaluate_case(case, rewrite_fn) for case in cases]


def render_report(outcomes: list[ContaminatedSessionOutcome], *, real: bool) -> str:
    """Render the drift + flip results as Markdown.

    ``real`` decides whether ``OFFLINE_TRACER_HEADER`` is prepended — the
    caller (``main``) derives ``real`` from the run mode; this function never
    inspects the environment itself (CODING_STANDARD §6.6: the writer picks
    the path/header from the run mode, not from re-detection here).
    """
    lines: list[str] = []
    if not real:
        lines.append(OFFLINE_TRACER_HEADER)
        lines.append("")
    lines += [
        "# Contaminated-session rewrite drift (#608)",
        "",
        "An earlier WRONG answer sits in a session's history; a later",
        "on-topic follow-up is re-asked. Two LLM-free, deterministic",
        "measurements per case (only the contaminated rewrite itself can be",
        "a real LLM call — see module docstring):",
        "",
        "- **Rewrite drift** — token-overlap Jaccard + length ratio between",
        "  the CONTAMINATED rewrite and the user's literal follow-up (lower",
        "  overlap / higher length ratio means the rewrite pulled in more",
        "  than what was actually asked).",
        "- **Answer flip** — does the retrieval gate's top Section / outcome",
        "  reason differ between the contaminated rewrite and the",
        "  clean-history control (no prior turn) for the SAME literal",
        "  follow-up?",
        "",
    ]
    flipped_count = sum(1 for o in outcomes if o.flipped)
    lines.append(
        f"**{flipped_count}/{len(outcomes)} case(s) flipped** the retrieval "
        "outcome under contamination."
    )
    lines += [
        "",
        "## Per-case detail",
        "",
        "| Case | Literal follow-up | Contaminated rewrite | Token overlap | Length ratio | Flipped? |",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        mark = "🔀 yes" if o.flipped else "no"
        lines.append(
            f"| {o.case.name} | {o.case.followup_question} | "
            f"{o.contaminated_rewrite} | {o.drift.token_overlap:.0%} | "
            f"{o.drift.length_ratio:.2f} | {mark} |"
        )
    lines += [
        "",
        "## Gate outcomes",
        "",
        "| Case | Contaminated top source | Contaminated reason | Clean top source | Clean reason |",
        "|---|---|---|---|---|",
    ]
    for o in outcomes:
        lines.append(
            f"| {o.case.name} | {o.contaminated_gate.top_source or '—'} | "
            f"{o.contaminated_gate.reason} | {o.clean_gate.top_source or '—'} | "
            f"{o.clean_gate.reason} |"
        )
    lines += ["", "## Case notes", ""]
    for o in outcomes:
        lines.append(f"- **{o.case.name}**: {o.case.note}")
    lines.append("")
    return "\n".join(lines)


@contextmanager
def _isolate_production_paths():
    """Redirect markdown_kb's index/wiki/log writes to a tmp dir for the CLI run.

    Mirrors ``eval.negative_case.runner._isolate_production_paths`` —
    without this, a CLI run would overwrite production ``.kb/`` / ``wiki/``
    with the tiny characterization corpus.
    """
    index_path, wiki_dir, log_path = (
        mk_indexer.INDEX_PATH,
        mk_indexer.WIKI_DIR,
        mk_logger.LOG_PATH,
    )
    source_dirs = mk_indexer.SOURCE_DIRS
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mk_indexer.INDEX_PATH = tmp_path / ".kb" / "index.json"
        mk_indexer.WIKI_DIR = tmp_path / "wiki"
        mk_logger.LOG_PATH = tmp_path / "wiki" / "log.md"
        try:
            yield
        finally:
            mk_indexer.INDEX_PATH = index_path
            mk_indexer.WIKI_DIR = wiki_dir
            mk_logger.LOG_PATH = log_path
            mk_indexer.SOURCE_DIRS = source_dirs
            mk_indexer.sections.clear()


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run the eval, write the trust-appropriate report.

    ``--fake`` forces the offline stand-in even with a key set (parity with
    ``eval.paraphrase_comparison.run_comparison --fake-embeddings``); without
    the flag, a missing ``OPENAI_API_KEY`` falls back to the stand-in
    automatically so the harness stays exercisable with no spend.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force the deterministic offline stand-in rewrite (no API call).",
    )
    args = parser.parse_args(argv)

    fake = args.fake or not os.getenv("OPENAI_API_KEY")
    rewrite_fn: RewriteFn
    if fake:
        rewrite_fn = _offline_rewrite_stub
    else:
        # Function-scope import: only the real (non-fake) path needs
        # gateway's LangChain-backed rewriter, so the offline/--fake path
        # (the default with no OPENAI_API_KEY, e.g. CI) never pays for
        # importing langchain_openai at all.
        from gateway.app.query_rewriting import rewrite_query

        rewrite_fn = rewrite_query

    with _isolate_production_paths():
        outcomes = run_contaminated_session(rewrite_fn)

    report = render_report(outcomes, real=not fake)
    report_path = OFFLINE_TRACER_REPORT_PATH if fake else REPORT_PATH
    report_path.write_text(report, encoding="utf-8")
    print(f"{'OFFLINE-TRACER' if fake else 'Real'} report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
