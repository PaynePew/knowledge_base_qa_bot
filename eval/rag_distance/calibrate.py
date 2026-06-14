"""Deep module per Ousterhout. Public surface: ``sweep``, ``recommend``, ``collect_distances``, ``render_report``, ``main``.

Distance-ceiling calibration for ``KB_RAG_DISTANCE_THRESHOLD`` (#257 gate, shipped
OFF in #258). The RAG pre-LLM Cannot Confirm gate refuses when the CLOSEST chunk's
FAISS distance is *above* the ceiling. That distance is ceiling-independent, so the
whole trade-off is a pure function of the per-query min distances:

  - a NEGATIVE (out-of-scope) case is *correctly refused* at ceiling ``C`` iff its
    min distance > C  → lowering C helps;
  - a POSITIVE (in-scope) case is *over-refused* at ``C`` iff its min distance > C
    → lowering C hurts.

This is the mirror image of the BM25 sweep (``eval.negative_case.calibrate``):
there a *low* score is bad, here a *high* distance is bad. ``recommend`` maximises
Youden's J (correct-refusal − over-refusal).

``text-embedding-3-small`` returns unit-norm vectors, so the FAISS L2-squared
distance equals ``2 - 2·cos`` ∈ [0, 4] — a distance ceiling is therefore a cosine
floor, i.e. the raw distance the gate already uses is well-behaved and directly
calibratable (no cosine/top-gap reformulation needed). Reuses the negative-case
corpus + case sets so the RAG gate is calibrated on the same data as the BM25 gate.

Unlike the BM25 sweep this needs REAL embeddings, so ``main`` is a manual,
quota-spending run; the tests use deterministic offline fake embeddings.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import markdown_kb.app.logger as mk_logger
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger

from eval.negative_case.cases import NEGATIVE_CASES
from eval.negative_case.driver import CORPUS_DIR
from eval.negative_case.models import NegativeCase, PositiveCase
from eval.negative_case.positive_cases import POSITIVE_CASES

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "calibration_report.md"

# FAISS L2-squared distance over unit-norm embeddings ∈ [0, 4]; in-scope hits
# cluster low, out-of-scope high. This grid spans that range; the real run's
# raw min/max is printed so the grid can be widened if a corpus lands outside it.
DEFAULT_CEILINGS: tuple[float, ...] = (
    0.6,
    0.8,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.8,
    2.0,
)


@dataclass(frozen=True)
class CeilingPoint:
    """One row of the sweep: the two error rates at a candidate distance ceiling."""

    ceiling: float
    correct_refusal_rate: float  # negatives refused / negatives  (higher is better)
    over_refusal_rate: float  # positives refused / positives  (lower is better)

    @property
    def youden_j(self) -> float:
        """Separation score: correct-refusal − over-refusal (1.0 = perfect split)."""
        return self.correct_refusal_rate - self.over_refusal_rate


def _rate_above(distances: Sequence[float], ceiling: float) -> float:
    """Fraction of ``distances`` strictly above ``ceiling`` (the gate's refusal rule)."""
    if not distances:
        return 0.0
    return sum(1 for d in distances if d > ceiling) / len(distances)


def sweep(
    positive_distances: Sequence[float],
    negative_distances: Sequence[float],
    ceilings: Sequence[float] = DEFAULT_CEILINGS,
) -> list[CeilingPoint]:
    """Compute correct-refusal and over-refusal at each candidate distance ceiling."""
    return [
        CeilingPoint(
            ceiling=c,
            correct_refusal_rate=_rate_above(negative_distances, c),
            over_refusal_rate=_rate_above(positive_distances, c),
        )
        for c in ceilings
    ]


def recommend(points: Sequence[CeilingPoint]) -> CeilingPoint:
    """Pick the best ceiling from the sweep.

    Maximise Youden's J. The gate has no shipped default to preserve (it ships OFF),
    so within the optimal *plateau* pick the median — the maximum margin to either
    error boundary, the most robust single value.
    """
    best_j = max(p.youden_j for p in points)
    optimal = sorted(
        (p for p in points if p.youden_j == best_j), key=lambda p: p.ceiling
    )
    return optimal[len(optimal) // 2]


def _min_distance(query: str, k: int) -> float:
    """Closest chunk's FAISS distance for ``query`` (the signal the gate reads)."""
    results = vr_indexer.search_with_distance(query, k=k)
    return min(d for _, d in results) if results else float("inf")


def collect_distances(
    corpus_dir: Path | None = None,
    positive_cases: Sequence[PositiveCase] = POSITIVE_CASES,
    negative_cases: Sequence[NegativeCase] = NEGATIVE_CASES,
    k: int = 3,
) -> tuple[list[float], list[float]]:
    """Build the RAG index over the corpus and return ``(positive, negative)`` min distances.

    Mirrors ``negative_case.calibrate.collect_scores`` but drives ``vector_rag``'s
    FAISS index instead of the BM25 Section Index. Assumes production paths are
    isolated (``_isolate_vector_rag_paths`` or the test conftest). Real embeddings
    unless ``get_embeddings`` is faked (tests).
    """
    corpus = corpus_dir or CORPUS_DIR
    vr_indexer.build_index(corpus)
    positive = [_min_distance(c.query, k) for c in positive_cases]
    negative = [_min_distance(c.query, k) for c in negative_cases]
    return positive, negative


def render_report(
    points: Sequence[CeilingPoint],
    recommended: CeilingPoint,
    positive_distances: Sequence[float],
    negative_distances: Sequence[float],
) -> str:
    """Render the distance-ceiling sweep + recommendation as Markdown."""
    pos_sorted = sorted(positive_distances)
    neg_sorted = sorted(negative_distances)
    lines = [
        "# KB_RAG_DISTANCE_THRESHOLD calibration (#257 / #258 follow-up)",
        "",
        "The RAG pre-LLM Cannot Confirm gate refuses when the closest chunk's FAISS",
        "distance is **above** `KB_RAG_DISTANCE_THRESHOLD` (lower distance = closer).",
        "Below is the trade-off between **correct-refusal** (rejecting out-of-scope",
        "queries) and **over-refusal** (wrongly rejecting in-scope queries), swept over",
        "the per-query min distances. Embeddings are `text-embedding-3-small` (unit-norm,",
        "so L2² = 2 − 2·cos ∈ [0, 4]: this ceiling is equivalently a cosine floor).",
        "",
        f"- Positive (in-scope) cases: {len(pos_sorted)}; "
        f"distance range [{min(pos_sorted):.3f}, {max(pos_sorted):.3f}], "
        f"max in-scope distance = **{max(pos_sorted):.3f}**.",
        f"- Negative (out-of-scope) cases: {len(neg_sorted)}; "
        f"distance range [{min(neg_sorted):.3f}, {max(neg_sorted):.3f}], "
        f"min out-of-scope distance = **{min(neg_sorted):.3f}**.",
        "",
        f"**Recommended ceiling: {recommended.ceiling} "
        f"(Youden's J = {recommended.youden_j:.2f}; "
        f"correct-refusal {recommended.correct_refusal_rate:.0%}, "
        f"over-refusal {recommended.over_refusal_rate:.0%}).**",
        "",
        "## Sweep",
        "",
        "| Ceiling | Correct-refusal (neg) | Over-refusal (pos) | Youden J |",
        "|---|---|---|---|",
    ]
    for p in points:
        marker = " ⭐" if p.ceiling == recommended.ceiling else ""
        lines.append(
            f"| {p.ceiling}{marker} | {p.correct_refusal_rate:.0%} | "
            f"{p.over_refusal_rate:.0%} | {p.youden_j:.2f} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "A clean **gap** between the max in-scope distance and the min out-of-scope",
        "distance means a ceiling in that gap separates them perfectly (J = 1.0). If",
        "they overlap, no single ceiling separates every case — the overlap needs",
        "semantic reranking (Phase 13), not ceiling tuning. The recommended ceiling is",
        "the plateau median (max margin to both error boundaries).",
        "",
    ]
    return "\n".join(lines)


@contextmanager
def _isolate_vector_rag_paths():
    """Redirect vector_rag's FAISS index + log writes to a tmp dir for the run.

    ``build_index`` persists to ``FAISS_INDEX_DIR`` and the logger writes
    ``LOG_PATH``; without this a CLI run would overwrite production ``.kb/`` with the
    tiny eval corpus (mirrors ``negative_case.runner._isolate_production_paths`` and
    the vector_rag test conftest). markdown_kb's LOG_PATH is redirected too because
    the shared parser path can log there.
    """
    faiss_dir = vr_indexer.FAISS_INDEX_DIR
    vr_log, mk_log = vr_logger.LOG_PATH, mk_logger.LOG_PATH
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        vr_indexer.FAISS_INDEX_DIR = tmp_path / ".kb" / "faiss_index"
        vr_logger.LOG_PATH = tmp_path / "vector_rag" / "log.md"
        mk_logger.LOG_PATH = tmp_path / "wiki" / "log.md"
        try:
            yield
        finally:
            vr_indexer.FAISS_INDEX_DIR = faiss_dir
            vr_logger.LOG_PATH = vr_log
            mk_logger.LOG_PATH = mk_log
            vr_indexer.vectorstore = None
            vr_indexer.files_indexed = 0
            vr_indexer.chunks_indexed = 0


def main() -> None:
    """CLI entry: real-embedding sweep under production isolation; write the report.

    Spends OpenAI quota (real ``text-embedding-3-small`` calls). Loads ``.env`` for
    ``OPENAI_API_KEY`` the same way uvicorn does.
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    with _isolate_vector_rag_paths():
        positive, negative = collect_distances()
    print(f"Positive (in-scope) min distances: {sorted(round(d, 3) for d in positive)}")
    print(
        f"Negative (out-of-scope) min distances: {sorted(round(d, 3) for d in negative)}"
    )
    points = sweep(positive, negative)
    best = recommend(points)
    REPORT_PATH.write_text(
        render_report(points, best, positive, negative), encoding="utf-8"
    )
    print(f"Recommended ceiling: {best.ceiling} (Youden J = {best.youden_j:.2f})")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
