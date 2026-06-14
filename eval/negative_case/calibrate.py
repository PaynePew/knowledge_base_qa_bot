"""Deep module per Ousterhout. Public surface: ``sweep``, ``recommend``, ``collect_scores``, ``render_calibration_report``, ``main``.

Threshold calibration for KB_SCORE_THRESHOLD (#253).

The pre-LLM Cannot Confirm gate refuses a query when its top BM25 score falls
below ``KB_SCORE_THRESHOLD``. That score is threshold-independent, so the whole
trade-off is a pure function of the per-query top scores:

  - a NEGATIVE (out-of-scope) case is *correctly refused* at threshold ``T`` iff
    its top score < T  → raising T helps;
  - a POSITIVE (in-scope) case is *over-refused* at ``T`` iff its top score < T
    → raising T hurts.

``sweep`` computes both rates across a grid; ``recommend`` picks the threshold
maximising Youden's J (correct-refusal − over-refusal). The whole thing is
LLM-free, like the rest of the negative-case eval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .cases import NEGATIVE_CASES
from .driver import evaluate_case, index_corpus
from .positive_cases import POSITIVE_CASES

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "calibration_report.md"

# BM25 scores are unbounded; this grid spans the observed range for the committed
# corpus (clearly-out-of-scope ~0, real hits ~1.5+). CURRENT_DEFAULT mirrors
# retrieval._KB_SCORE_THRESHOLD_DEFAULT, included for side-by-side comparison.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
CURRENT_DEFAULT = 0.5


@dataclass(frozen=True)
class ThresholdPoint:
    """One row of the sweep: the two error rates at a candidate threshold."""

    threshold: float
    correct_refusal_rate: float  # negatives refused / negatives  (higher is better)
    over_refusal_rate: float  # positives refused / positives  (lower is better)

    @property
    def youden_j(self) -> float:
        """Separation score: correct-refusal − over-refusal (1.0 = perfect split)."""
        return self.correct_refusal_rate - self.over_refusal_rate


def _rate_below(scores: Sequence[float], threshold: float) -> float:
    """Fraction of ``scores`` strictly below ``threshold`` (the gate's refusal rule)."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s < threshold) / len(scores)


def sweep(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[ThresholdPoint]:
    """Compute correct-refusal and over-refusal at each candidate threshold."""
    return [
        ThresholdPoint(
            threshold=t,
            correct_refusal_rate=_rate_below(negative_scores, t),
            over_refusal_rate=_rate_below(positive_scores, t),
        )
        for t in thresholds
    ]


def recommend(points: Sequence[ThresholdPoint]) -> ThresholdPoint:
    """Pick the best threshold from the sweep.

    Maximise Youden's J. Ties are common — a flat optimal *plateau* of thresholds
    that all separate the data equally well. Within that plateau:
      - if the current production default is itself optimal, keep it (a working
        default shouldn't churn just to land on a different equally-good value);
      - otherwise pick the plateau median (maximum margin to either error boundary,
        the most robust single value).
    """
    best_j = max(p.youden_j for p in points)
    optimal = sorted(
        (p for p in points if p.youden_j == best_j), key=lambda p: p.threshold
    )
    for p in optimal:
        if p.threshold == CURRENT_DEFAULT:
            return p
    return optimal[len(optimal) // 2]


def collect_scores(corpus_dir: Path | None = None) -> tuple[list[float], list[float]]:
    """Index the corpus and return ``(positive_scores, negative_scores)`` top scores.

    Assumes production paths are isolated (the runner's ``_isolate_production_paths``
    or the test conftest).
    """
    index_corpus(corpus_dir)
    positive = [evaluate_case(c.query).top_score for c in POSITIVE_CASES]
    negative = [evaluate_case(c.query).top_score for c in NEGATIVE_CASES]
    return positive, negative


def render_calibration_report(
    points: Sequence[ThresholdPoint],
    recommended: ThresholdPoint,
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> str:
    """Render the calibration sweep + recommendation as Markdown."""
    pos_sorted = sorted(positive_scores)
    neg_nonzero = sorted(s for s in negative_scores if s > 0.0)
    lines = [
        "# KB_SCORE_THRESHOLD calibration (#253)",
        "",
        "The Cannot Confirm gate refuses when the top BM25 score < `KB_SCORE_THRESHOLD`.",
        "Below is the trade-off between **correct-refusal** (rejecting out-of-scope",
        "queries) and **over-refusal** (wrongly rejecting in-scope queries), swept over",
        "the per-query top scores. LLM-free and deterministic.",
        "",
        f"- Positive (in-scope) cases: {len(positive_scores)}; "
        f"score range [{min(pos_sorted):.3f}, {max(pos_sorted):.3f}], "
        f"min in-scope score = **{min(pos_sorted):.3f}**.",
        f"- Negative (out-of-scope) cases: {len(negative_scores)}; "
        f"{len(negative_scores) - len(neg_nonzero)} score 0.0 (no overlap), "
        f"non-zero leaks at {[round(s, 3) for s in neg_nonzero]}.",
        "",
        f"**Recommended threshold: {recommended.threshold} "
        f"(Youden's J = {recommended.youden_j:.2f}; "
        f"correct-refusal {recommended.correct_refusal_rate:.0%}, "
        f"over-refusal {recommended.over_refusal_rate:.0%}).** "
        f"Current default is {CURRENT_DEFAULT}.",
        "",
        "## Sweep",
        "",
        "| Threshold | Correct-refusal (neg) | Over-refusal (pos) | Youden J |",
        "|---|---|---|---|",
    ]
    for p in points:
        marker = ""
        if p.threshold == recommended.threshold:
            marker = " ⭐"
        elif p.threshold == CURRENT_DEFAULT:
            marker = " (current)"
        lines.append(
            f"| {p.threshold}{marker} | {p.correct_refusal_rate:.0%} | "
            f"{p.over_refusal_rate:.0%} | {p.youden_j:.2f} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "If the non-zero negative leaks fall **inside** the positive score range, no",
        "threshold can separate them from real answers — those leaks need semantic",
        "reranking (Phase 13 hybrid / FM2), not threshold tuning. The threshold's job",
        "is only to reject the ~0-scoring clearly-out-of-scope queries while admitting",
        "every real hit.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry: run the sweep under production isolation and write the report."""
    from .runner import _isolate_production_paths

    with _isolate_production_paths():
        positive, negative = collect_scores()
    points = sweep(positive, negative)
    best = recommend(points)
    REPORT_PATH.write_text(
        render_calibration_report(points, best, positive, negative), encoding="utf-8"
    )
    print(f"Recommended threshold: {best.threshold} (Youden J = {best.youden_j:.2f})")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
