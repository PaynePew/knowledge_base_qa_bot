"""Deep module per Ousterhout. Public surface: ``run_comparison``, ``build_test_cases``, ``StackScores``, ``REPORT_PATH``.

In-process DeepEval comparison runner for the Phase 8 retrieval comparison
(CONTEXT.md § Phase 8 > Retrieval Stack, PRD #100).

Drives both Retrieval Stacks through the same Paraphrase set in one process
(no HTTP), scores each retrieval with the deterministic C5c ``HitRateAtK``
metric, and renders a per-Paraphrase-Type ``report.md`` row (Stack A vs Stack B
hit_rate@k). Slice 1 reports a single ``synonym_swap`` row; the structure
generalises to all seven types.

Production isolation (PRD #100 acceptance): index building points markdown_kb
``SOURCE_DIRS`` and vector_rag ``DOCS_DIR`` at the eval fixtures, and redirects
markdown_kb ``INDEX_PATH`` / ``WIKI_DIR`` (and disables its wiki-index side
write) to a tmp directory, so production ``wiki/`` / ``docs/`` / ``.kb/`` are
never read or written.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.logger as mk_logger
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
from deepeval.test_case import LLMTestCase

from . import stacks
from .loader import load_paraphrases, write_text_atomic
from .metric import DEFAULT_K, HitRateAtK
from .models import Paraphrase, RetrievedItem

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "report.md"

# A Retrieval Stack's retrieval entry point.
StackRetrieval = Callable[[str, int], list[RetrievedItem]]


@dataclass(frozen=True)
class StackScores:
    """Per-Paraphrase-Type hit_rate@k and MRR for one Retrieval Stack.

    ``by_type`` is the per-type hit_rate@k (mean of per-Paraphrase 1.0/0.0 hits);
    ``mrr_by_type`` is the per-type MRR (mean of per-Paraphrase reciprocal ranks
    of the first top-k hit). ``n_by_type`` is the Paraphrase count per type, so
    the report can render the ``n`` column and a Paraphrase-weighted Core
    macro-average (PRD #100).
    """

    stack: str
    k: int
    by_type: dict[str, float]
    mrr_by_type: dict[str, float] = field(default_factory=dict)
    n_by_type: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Test-case assembly + scoring (the DeepEval seam)
# ---------------------------------------------------------------------------
def build_test_cases(
    paraphrases: list[Paraphrase],
    retrieve: StackRetrieval,
    k: int = DEFAULT_K,
) -> list[tuple[Paraphrase, LLMTestCase]]:
    """Run ``retrieve`` for each Paraphrase and pack a DeepEval LLMTestCase.

    The retrieved items and the Paraphrase's Key Tokens travel in ``metadata``
    so the deterministic C5c metric reads them without re-running retrieval.
    ``retrieval_context`` carries the source ids for DeepEval's own display.
    """
    cases: list[tuple[Paraphrase, LLMTestCase]] = []
    for para in paraphrases:
        items = retrieve(para.text, k)
        case = LLMTestCase(
            input=para.text,
            actual_output="",  # retrieval-only comparison; no generated answer
            expected_output=para.gold_docs_section_id,
            retrieval_context=[it.source_section_id for it in items] or ["<none>"],
            metadata={
                "retrieved_items": items,
                "key_tokens": sorted(para.key_tokens),
                "paraphrase_type": para.paraphrase_type,
            },
        )
        cases.append((para, case))
    return cases


def score_stack(
    stack_name: str,
    paraphrases: list[Paraphrase],
    retrieve: StackRetrieval,
    k: int = DEFAULT_K,
) -> StackScores:
    """Score one Stack over all Paraphrases, aggregating hit_rate@k AND MRR per type."""
    metric = HitRateAtK(k=k)
    per_type_hits: dict[str, list[float]] = defaultdict(list)
    per_type_rr: dict[str, list[float]] = defaultdict(list)
    for para, case in build_test_cases(paraphrases, retrieve, k):
        metric.measure(case)
        per_type_hits[para.paraphrase_type].append(metric.score)
        per_type_rr[para.paraphrase_type].append(metric.reciprocal_rank)
    by_type = {
        ptype: sum(scores) / len(scores) for ptype, scores in per_type_hits.items()
    }
    mrr_by_type = {
        ptype: sum(rrs) / len(rrs) for ptype, rrs in per_type_rr.items()
    }
    n_by_type = {ptype: len(scores) for ptype, scores in per_type_hits.items()}
    return StackScores(
        stack=stack_name,
        k=k,
        by_type=by_type,
        mrr_by_type=mrr_by_type,
        n_by_type=n_by_type,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_report(
    stack_a: StackScores,
    stack_b: StackScores,
    embedding_mode: str = "real",
) -> str:
    """Render the per-type comparison table (Stack A vs Stack B hit_rate@k).

    ``embedding_mode`` annotates how Stack B's vectors were produced ("real"
    OpenAI embeddings vs a "fake" deterministic offline stand-in) so a reader
    never mistakes an offline tracer number for a real-embedding result.
    """
    k = stack_a.k
    types = sorted(set(stack_a.by_type) | set(stack_b.by_type))
    lines = [
        "# Paraphrase Comparison Report",
        "",
        "Phase 8 retrieval comparison (PRD #100). Stack A = Wiki + BM25; "
        "Stack B = Vector RAG. Numbers are the deterministic C5c hit metric "
        "(source-match AND Key-Token overlap).",
        "",
        f"Stack B embedding mode: **{embedding_mode}** "
        "(`fake` = deterministic offline stand-in used when OPENAI_API_KEY is "
        "absent; `real` = OpenAI `text-embedding-3-small`).",
        "",
        f"| Paraphrase Type | Stack A hit_rate@{k} | Stack B hit_rate@{k} |",
        "|---|---|---|",
    ]
    for ptype in types:
        a = stack_a.by_type.get(ptype, 0.0)
        b = stack_b.by_type.get(ptype, 0.0)
        lines.append(f"| {ptype} | {a:.3f} | {b:.3f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_comparison(
    k: int = DEFAULT_K,
    report_path: Path = REPORT_PATH,
    embedding_mode: str = "real",
) -> tuple[StackScores, StackScores]:
    """Index both Stacks over the eval fixtures, score them, write report.md.

    Production isolation is enforced for the duration of the run (see module
    docstring). Requires OPENAI_API_KEY for Stack B's real embeddings; offline
    callers swap ``vector_rag.app.indexer._build_faiss`` first and pass
    ``embedding_mode="fake"`` so the report records it.
    """
    paraphrases = load_paraphrases()
    stack_a, stack_b = _run_scored(paraphrases, k)
    write_text_atomic(report_path, render_report(stack_a, stack_b, embedding_mode))
    return stack_a, stack_b


def _run_scored(
    paraphrases: list[Paraphrase], k: int
) -> tuple[StackScores, StackScores]:
    """Build both indexes under production isolation, then score each Stack."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _isolate_production_paths(tmp_path)
        stacks.index_stack_a()
        stacks.index_stack_b()
        a = score_stack("Stack A", paraphrases, stacks.stack_a_retrieval, k)
        b = score_stack("Stack B", paraphrases, stacks.stack_b_retrieval, k)
    return a, b


def _isolate_production_paths(tmp_path: Path) -> None:
    """Redirect both Stacks' persistent-state targets to ``tmp_path``.

    SOURCE_DIRS / DOCS_DIR are repointed inside ``stacks.index_stack_{a,b}``;
    here we redirect every persistence + log target so the builds' atomic-write
    side effects land in tmp, never in production ``.kb/`` / ``wiki/`` /
    ``vector_rag/log.md``. vector_rag's ``build_index`` now persists the FAISS
    index on success (issue #103), so its ``FAISS_INDEX_DIR`` must be isolated
    here too.
    """
    mk_indexer.INDEX_PATH = tmp_path / ".kb" / "index.json"
    mk_indexer.WIKI_DIR = tmp_path / "wiki"
    mk_logger.LOG_PATH = tmp_path / "wiki" / "log.md"
    vr_indexer.FAISS_INDEX_DIR = tmp_path / ".kb" / "faiss_index"
    vr_logger.LOG_PATH = tmp_path / "vector_rag" / "log.md"
