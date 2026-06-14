"""Deep module per Ousterhout. Public surface: ``run_negative_case``, ``render_report``, ``main``.

Run the negative-case eval and render the fallback-rate report.

``run_negative_case`` does the work (build the in-scope corpus index, drive every
out-of-scope query through the pre-LLM gate, compute the correct-refusal rate) and
assumes the caller has isolated production paths. ``main`` is the CLI entry: it
wraps the run in ``_isolate_production_paths`` so ``build_index``'s write to
``.kb/`` / ``wiki/`` lands in a tmp dir, then writes ``report.md``.

The whole eval is LLM-free and runs offline (no OPENAI_API_KEY).
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger

from .cases import NEGATIVE_CASES
from .driver import evaluate_case, index_corpus
from .metric import correct_refusal_rate
from .models import NegativeCase, RefusalOutcome

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "report.md"


@dataclass
class NegativeCaseReport:
    """Aggregated outcome of one negative-case run."""

    rate: float
    by_category: dict[str, float] = field(default_factory=dict)
    outcomes: list[tuple[NegativeCase, RefusalOutcome]] = field(default_factory=list)


def run_negative_case(
    corpus_dir: Path | None = None,
    negative_cases: Sequence[NegativeCase] = NEGATIVE_CASES,
) -> NegativeCaseReport:
    """Build the corpus index, run every negative case, compute the fallback rate.

    ``negative_cases`` defaults to the committed English set, so existing call sites
    (and ``test_runner``) are unchanged; ``main`` passes a different language's set
    via ``lang.resolve_lang``. Assumes production paths are already isolated (see
    ``main`` / ``_isolate_production_paths``); tests rely on the autouse conftest.
    """
    index_corpus(corpus_dir)

    outcomes: list[tuple[NegativeCase, RefusalOutcome]] = [
        (case, evaluate_case(case.query)) for case in negative_cases
    ]

    per_category: dict[str, list[RefusalOutcome]] = defaultdict(list)
    for case, outcome in outcomes:
        per_category[case.category].append(outcome)

    return NegativeCaseReport(
        rate=correct_refusal_rate(o for _, o in outcomes),
        by_category={
            cat: correct_refusal_rate(outs)
            for cat, outs in sorted(per_category.items())
        },
        outcomes=outcomes,
    )


def render_report(report: NegativeCaseReport, lang: str = "en") -> str:
    """Render the report as Markdown (the committed deliverable).

    ``lang`` defaults to ``en`` (byte-identical to the baseline); ``zh`` only
    localises the title — the table structure stays identical for comparability.
    """
    title = (
        "# Negative-case eval — fallback rate · Traditional Chinese (#256)"
        if lang == "zh"
        else "# Negative-case eval — fallback rate (Week 6 FM4)"
    )
    lines = [
        title,
        "",
        "Measures whether the bot correctly **refuses** (Cannot Confirm) out-of-scope",
        "queries the KB cannot answer. The refusal decision is the production pre-LLM",
        "gate (`retrieval._retrieve_and_gate`: BM25 + `KB_SCORE_THRESHOLD`), so this is",
        "deterministic and LLM-free. A *low* rate means the threshold is too permissive",
        "(the bot answers things it should refuse).",
        "",
        f"**Correct-refusal rate: {report.rate:.0%}** "
        f"({sum(1 for _, o in report.outcomes if o.refused)}/{len(report.outcomes)} refused)",
        "",
        "## By category",
        "",
        "| Category | Refusal rate |",
        "|---|---|",
    ]
    for cat, rate in report.by_category.items():
        lines.append(f"| {cat} | {rate:.0%} |")
    lines += [
        "",
        "## Per-case detail",
        "",
        "| Query | Category | Refused? | Reason | Top BM25 score |",
        "|---|---|---|---|---|",
    ]
    for case, outcome in report.outcomes:
        mark = "✅" if outcome.refused else "❌ leaked"
        lines.append(
            f"| {case.query} | {case.category} | {mark} | {outcome.reason} | "
            f"{outcome.top_score:.3f} |"
        )
    lines += [
        "",
        "> A `❌ leaked` row is an out-of-scope query that cleared the threshold — the",
        "> raw material for calibrating `KB_SCORE_THRESHOLD` (the `top_score` column",
        "> shows how far over the 0.5 default it landed).",
        "",
    ]
    return "\n".join(lines)


@contextmanager
def _isolate_production_paths():
    """Redirect markdown_kb's index/wiki/log writes to a tmp dir for the run.

    ``build_index`` persists to ``INDEX_PATH`` and writes the wiki index under
    ``WIKI_DIR``; without this a CLI run would overwrite production ``.kb/`` /
    ``wiki/`` with the tiny eval corpus (the KB-pollution failure mode from
    large-file-ingest-size-limit-findings §6).
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


def main() -> None:
    """CLI entry: run the eval for one language under production isolation.

    Language comes from ``KB_EVAL_LANG`` (default ``en``); the zh run uses a ``_zh``
    report suffix so it never clobbers the committed English report.md.
    """
    from .lang import resolve_lang

    cfg = resolve_lang()
    with _isolate_production_paths():
        report = run_negative_case(cfg.corpus_dir, cfg.negative_cases)
    report_path = REPORT_PATH.with_name(f"report{cfg.report_suffix}.md")
    report_path.write_text(render_report(report, lang=cfg.lang), encoding="utf-8")
    print(f"[{cfg.lang}] Correct-refusal rate: {report.rate:.0%}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
