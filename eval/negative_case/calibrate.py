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
from .models import NegativeCase, PositiveCase
from .positive_cases import POSITIVE_CASES

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "calibration_report.md"

# BM25 scores are unbounded; this grid spans the observed range for the committed
# corpus (clearly-out-of-scope ~0, real hits ~1.5+). CURRENT_DEFAULT mirrors
# retrieval._KB_SCORE_THRESHOLD_DEFAULT, included for side-by-side comparison.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
# Chinese (bigram) BM25 top-scores land in a higher band than English. #261's
# enlarged 10-topic corpus puts the in-scope min ~4.9 (hits up to ~16.9) and the
# catchable adjacent-absent leaks at ~1.9-2.8, so the English grid is far too short
# to bracket the Chinese separating gap. This grid keeps 0.5 (side-by-side with the
# English default), resolves the (max-catchable-leak, min-in-scope] gap ~(2.85, 4.9]
# finely, and extends past min-in-scope so the over-refusal onset is visible.
DEFAULT_THRESHOLDS_ZH: tuple[float, ...] = (
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    6.0,
)
CURRENT_DEFAULT = 0.5
# English baseline's min in-scope BM25 score, surfaced in the zh cross-language
# verdict for a magnitude comparison. Mirrors the committed English
# calibration_report.md ("min in-scope score = 1.406"); keep the two in sync.
_EN_MIN_IN_SCOPE = 1.406


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


def collect_scores(
    corpus_dir: Path | None = None,
    positive_cases: Sequence[PositiveCase] = POSITIVE_CASES,
    negative_cases: Sequence[NegativeCase] = NEGATIVE_CASES,
) -> tuple[list[float], list[float]]:
    """Index the corpus and return ``(positive_scores, negative_scores)`` top scores.

    The case sets default to the committed English constants, so existing call
    sites (and ``test_calibrate``) are unchanged; ``main`` passes a different
    language's sets via ``lang.resolve_lang``. Assumes production paths are isolated
    (the runner's ``_isolate_production_paths`` or the test conftest).
    """
    index_corpus(corpus_dir)
    positive = [evaluate_case(c.query).top_score for c in positive_cases]
    negative = [evaluate_case(c.query).top_score for c in negative_cases]
    return positive, negative


def _cross_language_verdict(
    recommended: ThresholdPoint,
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> list[str]:
    """The #256 / #261 AC: does one global threshold serve both languages? (zh report only).

    Compares the Chinese score distribution to the English baseline, then splits the
    non-zero leaks into:
      - *catchable* leaks below the min in-scope score — a per-language threshold
        separates them (the #261 magnitude-mismatch fix), and
      - *residual* leaks at/above the min in-scope score — inside the real-answer
        band, where no threshold separates them (semantic-reranking / Phase 13 territory).

    Four outcomes:
      - 0.5 itself J-optimal → one global threshold serves both;
      - all leaks catchable (clean gap) → a per-language threshold cleanly separates;
      - some catchable + some residual (the representative-corpus case) → a per-language
        threshold fixes the magnitude mismatch, residual leaks → Phase 13;
      - none catchable (every leak inside the in-scope range) → threshold tuning cannot
        help at all; Phase 13 only.
    """
    min_pos = min(positive_scores)
    leaks = sorted(s for s in negative_scores if s > 0.0)
    catchable = [s for s in leaks if s < min_pos]
    residual = [s for s in leaks if s >= min_pos]
    refusal_at_default = _rate_below(negative_scores, CURRENT_DEFAULT)
    lines = [
        "## Cross-language verdict (#256 / #261)",
        "",
        f"Chinese BM25 top-scores sit in a **higher band** than English: min in-scope "
        f"**{min_pos:.3f}** (English baseline {_EN_MIN_IN_SCOPE}), "
        f"hits up to {max(positive_scores):.3f}. "
        "Bigram tokenisation emits more tokens per query, inflating raw scores "
        "(ADR-0014), so the English-calibrated 0.5 is not transferable as-is.",
        "",
    ]
    if recommended.threshold == CURRENT_DEFAULT:
        lines.append(
            f"✅ `KB_SCORE_THRESHOLD={CURRENT_DEFAULT}` is itself Youden-J-optimal for "
            "Chinese — **one global threshold serves both languages**; no change needed."
        )
    elif not catchable:
        # No non-zero leaks, or every leak sits inside the in-scope range — a
        # per-language threshold cannot improve on the global default.
        lines.append(
            f"⚠️ At {CURRENT_DEFAULT} correct-refusal is {refusal_at_default:.0%}; the "
            f"{len(residual)} non-zero leaks ({[round(s, 3) for s in residual]}) fall "
            f"**inside** the in-scope range (≥ {min_pos:.3f}), so — like the English "
            "`adjacent_absent` leaks — **no threshold separates them**. This needs "
            "semantic reranking (Phase 13 hybrid / FM2), not threshold tuning or a "
            "per-language value."
        )
    elif not residual:
        # Clean gap: every non-zero leak is below the in-scope range.
        lines.append(
            f"❌ **One global `KB_SCORE_THRESHOLD={CURRENT_DEFAULT}` does NOT serve "
            f"Chinese.** At {CURRENT_DEFAULT} correct-refusal is only "
            f"{refusal_at_default:.0%}: the {len(catchable)} adjacent-absent leaks "
            f"({[round(s, 3) for s in catchable]}) clear the gate. Unlike English, the "
            f"Chinese leaks fall **below** the min in-scope score, leaving a clean gap "
            f"({max(catchable):.3f}, {min_pos:.3f}]; the sweep recommends "
            f"**{recommended.threshold}** (correct-refusal "
            f"{recommended.correct_refusal_rate:.0%}, over-refusal "
            f"{recommended.over_refusal_rate:.0%})."
        )
        lines += [
            "",
            "**Recommendation:** adopt a per-language Chinese threshold "
            "(`KB_SCORE_THRESHOLD_ZH`). The magnitude gap is robust; re-sweep whenever "
            "the Chinese corpus grows.",
        ]
    else:
        # Mixed (the representative-corpus case, #261): some leaks are catchable by a
        # per-language threshold, some sit inside the in-scope range (Phase 13 only).
        lines.append(
            f"❌ **One global `KB_SCORE_THRESHOLD={CURRENT_DEFAULT}` does NOT serve "
            f"Chinese.** At {CURRENT_DEFAULT} correct-refusal is only "
            f"{refusal_at_default:.0%}. The {len(catchable)} catchable adjacent-absent "
            f"leaks ({[round(s, 3) for s in catchable]}) sit **below** the min in-scope "
            f"score ({min_pos:.3f}), so a per-language threshold separates them: the "
            f"sweep recommends **{recommended.threshold}** (correct-refusal "
            f"{recommended.correct_refusal_rate:.0%}, over-refusal "
            f"{recommended.over_refusal_rate:.0%})."
        )
        lines += [
            "",
            f"The remaining {len(residual)} leak(s) "
            f"({[round(s, 3) for s in residual]}) fall **inside** the in-scope range "
            f"(≥ {min_pos:.3f}) — e.g. an order-page query whose surface tokens match a "
            "real Section but whose specific ask is absent — so, exactly like the English "
            "`adjacent_absent` leaks, **no threshold (per-language or not) separates "
            "them**; they are semantic-reranking (Phase 13 hybrid / FM2) territory in "
            "both languages.",
            "",
            "**Recommendation:** adopt a per-language Chinese threshold "
            "(`KB_SCORE_THRESHOLD_ZH`) to fix the magnitude mismatch — an **interim** "
            "measure, superseded by the Phase 13 reranker (which also catches the "
            "in-scope-range residual). Re-sweep whenever the Chinese corpus grows.",
        ]
    lines.append("")
    return lines


def render_calibration_report(
    points: Sequence[ThresholdPoint],
    recommended: ThresholdPoint,
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    lang: str = "en",
) -> str:
    """Render the calibration sweep + recommendation as Markdown.

    ``lang`` defaults to ``en`` (the report is byte-identical to the #253 baseline);
    ``zh`` localises the title and appends the cross-language verdict (#256).
    """
    pos_sorted = sorted(positive_scores)
    neg_nonzero = sorted(s for s in negative_scores if s > 0.0)
    title = (
        "# KB_SCORE_THRESHOLD calibration — Traditional Chinese (#256 / #261)"
        if lang == "zh"
        else "# KB_SCORE_THRESHOLD calibration (#253)"
    )
    lines = [
        title,
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
    if lang == "zh":
        lines += _cross_language_verdict(recommended, positive_scores, negative_scores)
    return "\n".join(lines)


def main() -> None:
    """CLI entry: sweep one language under production isolation and write its report.

    Language comes from ``KB_EVAL_LANG`` (default ``en``); the zh run uses the wider
    Chinese grid and a ``_zh`` report suffix so it never clobbers the English report.
    """
    from .lang import resolve_lang
    from .runner import _isolate_production_paths

    cfg = resolve_lang()
    thresholds = DEFAULT_THRESHOLDS_ZH if cfg.lang == "zh" else DEFAULT_THRESHOLDS
    with _isolate_production_paths():
        positive, negative = collect_scores(
            cfg.corpus_dir, cfg.positive_cases, cfg.negative_cases
        )
    points = sweep(positive, negative, thresholds)
    best = recommend(points)
    report_path = REPORT_PATH.with_name(f"calibration_report{cfg.report_suffix}.md")
    report_path.write_text(
        render_calibration_report(points, best, positive, negative, lang=cfg.lang),
        encoding="utf-8",
    )
    print(
        f"[{cfg.lang}] Recommended threshold: {best.threshold} "
        f"(Youden J = {best.youden_j:.2f})"
    )
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
