"""Weight calibration for ``QA_QUESTION_TOKEN_WEIGHT`` (#578).

Builds a synthetic corpus in-memory (no file I/O — direct ``Section``
construction, like ``markdown_kb/tests/test_indexer_qa_question_weight.py``)
that reproduces two things at once, over one shared index:

  - the #578 collision AT SCALE: one real fact-carrying Section competing
    against a real-shaped distractor qa page (mirrors the reported
    production case: a payment qa page out-ranking the shipping-countries
    fact for a "你們配送到哪些國家？" query) PLUS a configurable number of
    synthetic noise qa pages sharing the same generic interrogative bigrams,
    simulating the "hundreds of pages" scale the issue names;
  - the #570 own-question invariant a downweight must NOT break: a qa page
    whose body shares no distinctive token with its own question must still
    be retrievable by that question.

For each, the trade-off is a pure function of two per-query outcomes, so
``sweep``/``recommend`` follow the same Youden-J-style shape as
``eval/negative_case/calibrate.py`` (#253 precedent): maximise
(own-question hit rate − pollution rate), subject to the own-question hit
rate never dropping below its best-achieved value (a hard floor, not a
trade-off — see ``recommend``'s docstring), breaking ties toward the
plateau median for robustness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import markdown_kb.app.indexer as indexer

_PKG_ROOT = Path(__file__).resolve().parent
REPORT_PATH = _PKG_ROOT / "calibration_report.md"

DEFAULT_WEIGHTS: tuple[float, ...] = (
    1.0,
    0.75,
    0.5,
    0.4,
    0.3,
    0.2,
    0.15,
    0.1,
    0.05,
    0.02,
    0.0,
)

# Real production content (issue #578's reported case + #570's own regression
# fixture), copied verbatim so this eval stays meaningful even if the live
# wiki/ corpus drifts — see module docstring.
_REAL_FACT_HEADING = "配送地區"
_REAL_FACT_BODY = (
    "本節說明國際配送的可配送地區：日本、韓國、香港、澳門、新加坡、馬來西亞、美國、加拿大。"
    "除上述地區外，暫不提供國際配送。"
)
_DISTRACTOR_QUESTION = "你們接受哪些付款方式？"
_DISTRACTOR_BODY = (
    "ACME 商店接受的付款方式包括 VISA、MasterCard 和 JCB 的信用卡付款，"
    "並且可一次付清或在符合滿額門檻的情況下選擇分期付款。"
    "不支援的付款方式包括銀行轉帳、支票與外幣付款。"
)
_POLLUTION_QUERY = "你們配送到哪些國家？"

_OWN_QUESTION = "Which countries do you ship to?"
_OWN_QUESTION_BODY = "Delivered worldwide, allegedly."
_OWN_QUESTION_FILLER_BODY = "Returns are accepted within 30 days of delivery."

_NOISE_TOPICS = (
    "門市營業時間",
    "會員等級",
    "退貨流程",
    "優惠券使用",
    "客服聯絡方式",
    "商品保固",
    "禮品包裝",
    "訂單修改",
    "發票開立",
    "點數兌換",
)


def _make_qa_section(*, sec_id: str, question: str, body: str) -> indexer.Section:
    """Build one ``type: qa`` Section the way rule 2a (#570) would tokenize it.

    A filed qa page's body is plain prose with no Markdown heading, so it
    parses under rule 7 (zero-heading Source): ``tokens = tokenize(body)``
    ONLY — the heading/id slug is never tokenized (see
    ``parse_markdown_body``'s zero-heading branch). Rule 2a then prepends the
    question tokens. Using ``tokenize(sec_id)`` here would leak the slug's
    words into the body signal and understate the real #570 collision this
    corpus is modelling.
    """
    question_tokens = indexer.tokenize(question)
    body_tokens = indexer.tokenize(body)
    return indexer.Section(
        id=sec_id,
        file=sec_id,
        heading=sec_id,
        heading_path=[sec_id],
        content=body,
        tokens=question_tokens + body_tokens,
        metadata={"type": "qa", "question": question},
        question_tokens=question_tokens,
    )


def _make_concept_section(*, sec_id: str, heading: str, body: str) -> indexer.Section:
    return indexer.Section(
        id=f"{sec_id}#{heading}",
        file=sec_id,
        heading=heading,
        heading_path=[sec_id, heading],
        content=body,
        tokens=indexer.tokenize(heading) + indexer.tokenize(body),
        metadata={"type": "concept"},
    )


def build_synthetic_corpus(
    noise_count: int = 30,
) -> tuple[list[indexer.Section], str, str]:
    """Build the corpus in-memory and return ``(sections, real_id, own_question_id)``.

    ``noise_count`` simulates the issue's "at hundreds of pages" scale concern
    at a reproducible size — every noise page shares the pollution query's
    generic interrogative bigrams ("你們"/"哪些") but its body is about an
    unrelated topic, matching the production shape (rule 2a injects the whole
    question, body is the answer text alone).
    """
    real = _make_concept_section(
        sec_id="配送地區", heading=_REAL_FACT_HEADING, body=_REAL_FACT_BODY
    )
    distractor = _make_qa_section(
        sec_id="qa-payment-zh", question=_DISTRACTOR_QUESTION, body=_DISTRACTOR_BODY
    )
    noise_pages = [
        _make_qa_section(
            sec_id=f"qa-noise-zh-{i:03d}",
            question=f"你們哪些{_NOISE_TOPICS[i % len(_NOISE_TOPICS)]}項目第{i}次有調整？",
            body=f"{_NOISE_TOPICS[i % len(_NOISE_TOPICS)]}的詳細規則如下，僅供內部客服參考使用。",
        )
        for i in range(noise_count)
    ]
    # Deliberately opaque id/slug (unlike the real filed-page convention of
    # slugifying the question itself) so this probe isolates what
    # ``qa_question_weight`` alone protects, uncontaminated by
    # ``bm25_score``'s separate heading-path boost (which tokenizes
    # ``heading_path`` independently of the weight and would otherwise let a
    # question-derived slug quietly cover for a weight that is too low).
    own_question_page = _make_qa_section(
        sec_id="qa-f70fda",
        question=_OWN_QUESTION,
        body=_OWN_QUESTION_BODY,
    )
    own_question_filler = _make_concept_section(
        sec_id="returns", heading="Returns", body=_OWN_QUESTION_FILLER_BODY
    )
    sections = [real, distractor, *noise_pages, own_question_page, own_question_filler]
    return sections, real.file, own_question_page.id


@dataclass(frozen=True)
class WeightPoint:
    """One row of the sweep: the two competing rates at a candidate weight."""

    weight: float
    own_question_hit_rate: (
        float  # own-question probe still retrievable (higher is better)
    )
    pollution_rate: (
        float  # fraction of top-k that is noise/distractor (lower is better)
    )

    @property
    def separation(self) -> float:
        """Youden-J-style separation: own-question hit rate − pollution rate."""
        return self.own_question_hit_rate - self.pollution_rate


def _pollution_rate(
    hits: Sequence[tuple[indexer.Section, float]], real_file: str
) -> float:
    if not hits:
        return 0.0
    return sum(1 for sec, _ in hits if sec.file != real_file) / len(hits)


def _own_question_hit(
    hits: Sequence[tuple[indexer.Section, float]], own_question_id: str
) -> bool:
    return any(sec.id == own_question_id for sec, _ in hits)


def evaluate_weight(
    weight: float, real_file: str, own_question_id: str, *, k: int = 3
) -> WeightPoint:
    """Run both probe queries at ``weight`` against the already-built module index."""
    pollution_hits = indexer.search(_POLLUTION_QUERY, k, qa_question_weight=weight)
    own_hits = indexer.search(_OWN_QUESTION, k, qa_question_weight=weight)
    return WeightPoint(
        weight=weight,
        own_question_hit_rate=float(_own_question_hit(own_hits, own_question_id)),
        pollution_rate=_pollution_rate(pollution_hits, real_file),
    )


def sweep(
    real_file: str,
    own_question_id: str,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> list[WeightPoint]:
    """Evaluate every candidate weight against the already-built module index."""
    return [evaluate_weight(w, real_file, own_question_id) for w in weights]


def recommend(points: Sequence[WeightPoint]) -> WeightPoint:
    """Pick the best weight: max separation, subject to the own-question floor.

    The own-question hit rate is a hard invariant (#570 must not regress), not
    a trade-off knob — a weight that drops it is disqualified outright, even if
    its pollution rate is lower. Among the weights that hold the floor, a
    small/coarse probe corpus (few real competing candidates) tends to produce
    a wide plateau of tied-best separation rather than a single clear winner
    (own-question score decays smoothly with weight but only reaches exactly
    zero AT weight=0.0, and pollution is a step function of how many candidates
    happen to score above zero) — so, following the ``eval/negative_case/
    calibrate.py`` (#253) precedent, ties are broken toward the plateau
    MEDIAN rather than either extreme: the most robust single value, with
    margin away from the weight=0.0 cliff where the floor breaks outright.
    If no weight holds the floor, fall back to the safest available (highest
    own-question hit rate, then highest weight) rather than silently picking
    a broken one.
    """
    floor = max(p.own_question_hit_rate for p in points)
    safe = [p for p in points if p.own_question_hit_rate == floor]
    best_separation = max(p.separation for p in safe)
    optimal = sorted(
        (p for p in safe if p.separation == best_separation), key=lambda p: p.weight
    )
    return optimal[len(optimal) // 2]


def score_margin_by_weight(
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> list[tuple[float, float, float]]:
    """Return ``(weight, real_score, distractor_score)`` at TODAY's real scale.

    Uses ``noise_count=0`` — just the real fact-carrier and the one real-shaped
    distractor qa page, no synthetic swarm — because the committed corpus has
    only 2 zh qa pages today (#578 is about future scale, not a currently
    reproducing collision: the reported production ranking no longer
    reproduces against the current, larger committed wiki/ corpus). Complements
    the at-scale ``sweep`` above: pollution-in-top-3 there is a step function
    (any nonzero weight keeps SOME low-scoring noise page in the k=3 window
    once there are enough of them), but the score MARGIN between the real
    content and the one realistic distractor shrinks smoothly and
    substantially as the weight decreases — the effect this eval is meant to
    demonstrate at the scale that matters today.
    """
    sections, real_file, _own_id = build_synthetic_corpus(noise_count=0)
    indexer.sections = sections
    indexer.rebuild_stats()
    rows = []
    for w in weights:
        hits = {
            sec.file: score
            for sec, score in indexer.search(_POLLUTION_QUERY, 5, qa_question_weight=w)
        }
        rows.append((w, hits.get(real_file, 0.0), hits.get("qa-payment-zh", 0.0)))
    return rows


def render_calibration_report(
    points: Sequence[WeightPoint], recommended: WeightPoint
) -> str:
    """Render the sweep + recommendation as Markdown."""
    lines = [
        "# QA_QUESTION_TOKEN_WEIGHT calibration (#578)",
        "",
        "Rule 2a (#570) joins a qa page's frontmatter `question:` into its BM25",
        "tokens. `QA_QUESTION_TOKEN_WEIGHT` scales the term-frequency contribution",
        "of matches that come ONLY from that injected question, never from real",
        "body content. Two competing rates, swept over a synthetic corpus that",
        "reproduces the reported collision at scale (LLM-free, deterministic):",
        "",
        "- **Pollution rate** — fraction of the top-3 window for"
        f" `{_POLLUTION_QUERY}` occupied by a qa page that is NOT the real"
        " fact-carrier (the distractor qa page + 30 synthetic noise qa pages,"
        " all sharing only the query's generic interrogative bigrams). Lower is"
        " better.",
        "- **Own-question hit rate** — whether the #570 regression fixture (a qa"
        f" page with zero body/question token overlap) is still retrievable by"
        f" its own question (`{_OWN_QUESTION}`). This is a hard floor, not a"
        " trade-off: a weight that drops it is disqualified regardless of its"
        " pollution rate.",
        "",
        f"**Recommended weight: {recommended.weight}** "
        f"(separation = {recommended.separation:.2f}; "
        f"own-question hit rate {recommended.own_question_hit_rate:.0%}, "
        f"pollution rate {recommended.pollution_rate:.0%}).",
        "",
        "## Sweep",
        "",
        "| Weight | Own-question hit rate | Pollution rate | Separation |",
        "|---|---|---|---|",
    ]
    for p in points:
        marker = " ⭐" if p.weight == recommended.weight else ""
        lines.append(
            f"| {p.weight}{marker} | {p.own_question_hit_rate:.0%} | "
            f"{p.pollution_rate:.0%} | {p.separation:.2f} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "At `weight=1.0` (the pre-#578 behaviour) the distractor/noise qa pages'",
        "shared-interrogative matches count at full term frequency, same as real",
        "body content — the collision the issue reports. Decreasing the weight",
        "shrinks their contribution; the own-question hit rate is the floor below",
        "which rule 2a's #570 fix itself starts to regress.",
        "",
        "**Limitation of the pollution-rate metric above:** with 30 identical-shape",
        "noise pages sharing the same two bigrams, their tied scores only clear",
        "exactly at `weight=0.0` — any smaller-but-nonzero weight still leaves two",
        "of them in the k=3 window, so the metric cannot distinguish 0.02 from",
        "1.0 in THIS adversarial construction. That is itself a finding, not a",
        "bug: pure per-token downweighting caps out against a large-enough swarm",
        "of pages sharing a generic interrogative. The issue's own scope decision",
        "names the fallback for that case — a CJK-interrogative stopword list —",
        "as a stopgap gated on eval evidence of pressure AT THE CURRENT corpus",
        "scale. Today's committed corpus has only 2 zh qa pages (not the ~30+",
        "simulated here) and the reported production collision no longer",
        "reproduces against it, so that gate is not tripped; the table below",
        "shows the downweight is doing real, graduated work at today's actual",
        "scale.",
        "",
        "## Score margin at today's real scale (no synthetic swarm)",
        "",
        "`noise_count=0` — just the real fact-carrier vs. the one real-shaped",
        "distractor qa page (the committed corpus has no more zh qa pages than",
        "this today). Score decreases smoothly and substantially with weight,",
        "well before the own-question floor is at risk:",
        "",
        "| Weight | Real content score | Distractor qa score | Margin |",
        "|---|---|---|---|",
    ]
    for w, real_score, distractor_score in score_margin_by_weight():
        marker = " ⭐" if w == recommended.weight else ""
        lines.append(
            f"| {w}{marker} | {real_score:.3f} | {distractor_score:.3f} | "
            f"{real_score - distractor_score:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry: sweep, recommend, and write ``calibration_report.md``."""
    sections, real_file, own_question_id = build_synthetic_corpus()
    indexer.sections = sections
    indexer.rebuild_stats()
    points = sweep(real_file, own_question_id)
    best = recommend(points)
    REPORT_PATH.write_text(render_calibration_report(points, best), encoding="utf-8")
    print(
        f"Recommended QA_QUESTION_TOKEN_WEIGHT: {best.weight} (separation={best.separation:.2f})"
    )
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
