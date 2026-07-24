"""Deep module per Ousterhout. Public surface: ``PairwiseComparison``,
``ClauseVerdict``, ``SurvivalEntry``, ``DecisionMatrixCell``,
``DecisionMatrixRow``, ``AXIS_HIGHER_IS_BETTER``, ``kill_clause_verdict``,
``demote_clause_verdict``, ``survival_entries``, ``render_verdict_report``.

Report-assembly and ADR-0045 clause-walkthrough logic for the corpus v3
verdict report (issue #662 AC 1 / AC 3). Every function here is pure --
it consumes already-computed axis results (rates + significance tests over
canned or real per-query outcomes) and renders Markdown; nothing calls an
LLM or a stack adapter, which is what lets this module be unit-tested on
canned axis results exactly as the AC requires, independent of whether a
live run ever happened.

The clause text quoted in the walkthrough is copied VERBATIM from
``project-docs/adr/0045-wiki-retrieval-arm-kill-criteria-preregistered.md``
§ Decision -- issue #662 Consequences: "its report must state each clause's
verdict explicitly (kill / demote / survive, per axis, with test
statistics)". A clause's wording must never be paraphrased here: if ADR-0045
is ever superseded, the constants below are the single place to update, and
a diff against that ADR is how a reviewer confirms fidelity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

ALPHA = 0.05

AxisName = Literal[
    "contradiction_leak_rate", "grounding_pass_rate", "correct_refusal_rate"
]
REQUIRED_AXES: frozenset[str] = frozenset(
    {"contradiction_leak_rate", "grounding_pass_rate", "correct_refusal_rate"}
)

# Axis polarity: for grounding pass / correct refusal, HIGHER is better; for
# the leak rate, LOWER is better (it is a rate of a bad outcome).
AXIS_HIGHER_IS_BETTER: dict[str, bool] = {
    "grounding_pass_rate": True,
    "correct_refusal_rate": True,
    "contradiction_leak_rate": False,
}

EvidenceGrade = Literal["measured-local", "measured-analogue", "argued"]

# ---------------------------------------------------------------------------
# ADR-0045 § Decision — verbatim clause text (do not paraphrase; see docstring)
# ---------------------------------------------------------------------------
KILL_CLAUSE_TEXT = (
    "**Kill the retrieval arm** if, on corpus v3, `stack=wiki` shows no "
    "statistically significant advantage over `stack=rag` on **all three** "
    "content axes — contradiction-leak rate, grounding pass rate, "
    "correct-refusal rate. Consequence: `stack=wiki` is retired as a "
    "standalone retrieval option; the wiki layer remains as the hybrid "
    "embedding substrate and governance surface. (Per ADR-0003's W2 hedge, "
    "this retirement is a config/routing change, not a rewrite.)"
)
DEMOTE_CLAUSE_TEXT = (
    "**Demote the wiki layer** if C (hybrid-over-wiki) fails to show a "
    "statistically significant advantage over B (dense-over-raw-docs) on "
    "**contradiction-leak rate** — the curated layer's home axis. "
    "Consequence: the layer is repositioned honestly as a demonstration "
    "artifact of a KB-governance workflow, not as a measured quality win; "
    "code is retained (the demo and Console depend on it) but every "
    "narrative claim of retrieval or grounding superiority is dropped."
)
SURVIVAL_CLAUSE_TEXT = (
    "**Survival:** any axis on which a wiki-backed stack significantly beats "
    "B becomes the lead narrative for that stack, with the corpus v3 numbers "
    "as backing."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairwiseComparison:
    """One axis's paired significance test between two arms, on one stratum.

    ``stratum`` is a free-text label (``eval.corpus_v3.aggregation``'s
    stratum key, or ``"macro"``) -- this module does not depend on the
    aggregation module's types, only on the numbers it produces, so it can
    be unit-tested with hand-built comparisons independent of a real
    aggregation run.
    """

    axis: str
    stratum: str
    arm_a: str
    arm_b: str
    rate_a: float
    rate_b: float
    n: int
    p_value: float
    test_name: str  # "mcnemar" | "bootstrap"

    def significant(self, alpha: float = ALPHA) -> bool:
        return self.p_value < alpha

    def advantage_arm(self) -> str | None:
        """The arm with the axis-polarity-correct better rate, or ``None`` on
        a tie. Raises ``ValueError`` if ``axis`` is not a known content axis
        (fail-fast: a typo'd axis name must never silently read as "no
        advantage")."""
        if self.axis not in AXIS_HIGHER_IS_BETTER:
            raise ValueError(
                f"unknown axis {self.axis!r}; must be one of {sorted(AXIS_HIGHER_IS_BETTER)}"
            )
        if self.rate_a == self.rate_b:
            return None
        higher_is_better = AXIS_HIGHER_IS_BETTER[self.axis]
        a_is_better = (self.rate_a > self.rate_b) == higher_is_better
        return self.arm_a if a_is_better else self.arm_b

    def significant_advantage(self, arm: str) -> bool:
        """True iff ``arm`` has a statistically significant, direction-correct
        advantage over the other arm in this pair (ADR-0045: "a
        non-significant *advantage* still kills" -- both conditions required)."""
        return self.significant() and self.advantage_arm() == arm

    def _arms(self) -> frozenset[str]:
        return frozenset({self.arm_a, self.arm_b})


@dataclass(frozen=True)
class ClauseVerdict:
    """The mechanical outcome of one ADR-0045 clause (kill or demote)."""

    clause: Literal["kill", "demote"]
    subject: str
    outcome: str
    comparisons: list[PairwiseComparison]
    clause_text: str


@dataclass(frozen=True)
class SurvivalEntry:
    """One (arm, axis) pair where ADR-0045's survival clause fires."""

    arm: str
    axis: str
    comparison: PairwiseComparison


@dataclass(frozen=True)
class DecisionMatrixCell:
    value: str
    evidence_grade: EvidenceGrade


@dataclass(frozen=True)
class DecisionMatrixRow:
    label: str
    cells: dict[str, DecisionMatrixCell]


# ---------------------------------------------------------------------------
# Public API — clause walkthrough (issue #662 AC 1 / AC 3)
# ---------------------------------------------------------------------------
def kill_clause_verdict(
    comparisons: Iterable[PairwiseComparison], *, wiki_arm: str, baseline_arm: str
) -> ClauseVerdict:
    """Mechanically evaluate ADR-0045's kill clause for ``wiki_arm`` vs
    ``baseline_arm``: killed only when ``wiki_arm`` shows a statistically
    significant advantage on **none** of the three content axes ("no ...
    advantage ... on all three" = fails everywhere). A single significantly
    won axis survives the arm — per the ADR's Survival clause it becomes
    that stack's lead narrative, which is incompatible with retiring it.

    Raises ``ValueError`` if ``comparisons`` does not carry exactly one
    ``wiki_arm``-vs-``baseline_arm`` comparison per required axis -- a
    missing axis must never silently read as "not killed" (or "killed").
    """
    by_axis = {
        c.axis: c
        for c in comparisons
        if c._arms() == frozenset({wiki_arm, baseline_arm}) and c.axis in REQUIRED_AXES
    }
    missing = REQUIRED_AXES - by_axis.keys()
    if missing:
        raise ValueError(
            f"kill_clause_verdict missing {wiki_arm!r} vs {baseline_arm!r} "
            f"comparisons for axes: {sorted(missing)}"
        )
    advantage_on_any = any(
        by_axis[axis].significant_advantage(wiki_arm) for axis in REQUIRED_AXES
    )
    killed = not advantage_on_any
    return ClauseVerdict(
        clause="kill",
        subject=wiki_arm,
        outcome="killed" if killed else "survives_kill_clause",
        comparisons=[by_axis[axis] for axis in sorted(REQUIRED_AXES)],
        clause_text=KILL_CLAUSE_TEXT,
    )


def demote_clause_verdict(
    comparison: PairwiseComparison, *, hybrid_arm: str, baseline_arm: str
) -> ClauseVerdict:
    """Mechanically evaluate ADR-0045's demote clause: demoted unless
    ``hybrid_arm`` shows a statistically significant contradiction-leak-rate
    advantage over ``baseline_arm`` -- "ties and non-significant differences
    both demote" (ADR-0045).

    Raises ``ValueError`` if ``comparison`` is not a
    ``contradiction_leak_rate`` comparison between exactly these two arms.
    """
    if comparison.axis != "contradiction_leak_rate":
        raise ValueError(
            f"demote_clause_verdict requires a contradiction_leak_rate "
            f"comparison, got axis={comparison.axis!r}"
        )
    if comparison._arms() != frozenset({hybrid_arm, baseline_arm}):
        raise ValueError(
            f"comparison arms {sorted(comparison._arms())} do not match "
            f"({hybrid_arm!r}, {baseline_arm!r})"
        )
    advantage = comparison.significant_advantage(hybrid_arm)
    demoted = not advantage
    return ClauseVerdict(
        clause="demote",
        subject=hybrid_arm,
        outcome="demoted" if demoted else "retains_narrative_claim",
        comparisons=[comparison],
        clause_text=DEMOTE_CLAUSE_TEXT,
    )


def survival_entries(
    comparisons: Iterable[PairwiseComparison],
    *,
    wiki_backed_arms: Iterable[str],
    baseline_arm: str,
) -> list[SurvivalEntry]:
    """Every (arm, axis) pair where a wiki-backed arm significantly beats
    ``baseline_arm`` (ADR-0045's survival clause) -- becomes that stack's
    lead narrative for that axis. Ordered by (axis, arm) for determinism.
    """
    wiki_backed = set(wiki_backed_arms)
    entries = []
    for c in comparisons:
        if baseline_arm not in (c.arm_a, c.arm_b):
            continue
        subject = c.arm_a if c.arm_b == baseline_arm else c.arm_b
        if subject not in wiki_backed:
            continue
        if c.significant_advantage(subject):
            entries.append(SurvivalEntry(arm=subject, axis=c.axis, comparison=c))
    entries.sort(key=lambda e: (e.axis, e.arm))
    return entries


# ---------------------------------------------------------------------------
# Public API — rendering
# ---------------------------------------------------------------------------
def _tag(text: str, grade: EvidenceGrade) -> str:
    """Suffix ``text`` with its evidence-grade tag (PRD #654 user story 24:
    "every cell tagged measured-local / measured-analogue / argued")."""
    return f"{text} **[{grade}]**"


def render_comparison_row(c: PairwiseComparison) -> str:
    sig = "✓" if c.significant() else "—"
    return (
        f"| {c.stratum} | {c.arm_a} {c.rate_a:.3f} vs {c.arm_b} {c.rate_b:.3f} "
        f"| n={c.n} | {c.test_name} p={c.p_value:.4f} | {sig} |"
    )


def render_axis_stratum_tables(
    comparisons: Iterable[PairwiseComparison],
) -> str:
    """Per-axis per-stratum tables (issue #662: "per-axis per-stratum tables
    with test statistics")."""
    comparisons = list(comparisons)
    lines = ["## Per-axis, per-stratum results", ""]
    for axis in sorted({c.axis for c in comparisons}):
        lines.append(f"### {axis}")
        lines.append("")
        lines.append("| Stratum | Rates | n | Test | sig (p<0.05) |")
        lines.append("|---|---|---|---|---|")
        for c in [c for c in comparisons if c.axis == axis]:
            lines.append(render_comparison_row(c))
        lines.append("")
    return "\n".join(lines)


def render_clause_walkthrough(
    kill: ClauseVerdict, demote: ClauseVerdict, survivals: list[SurvivalEntry]
) -> str:
    """ADR-0045 clause walkthrough (issue #662 AC 3: "walks every ADR-0045
    clause verbatim with test statistics; kill / demote / survive, per axis")."""
    lines = ["## ADR-0045 clause walkthrough", ""]

    lines += [
        "### Kill clause",
        "",
        f"> {kill.clause_text}",
        "",
        f"**Verdict: `{kill.subject}` {kill.outcome.replace('_', ' ')}.**",
        "",
        "| Stratum | Rates | n | Test | sig (p<0.05) |",
        "|---|---|---|---|---|",
    ]
    lines += [render_comparison_row(c) for c in kill.comparisons]
    lines.append("")

    lines += [
        "### Demote clause",
        "",
        f"> {demote.clause_text}",
        "",
        f"**Verdict: `{demote.subject}` {demote.outcome.replace('_', ' ')}.**",
        "",
        "| Stratum | Rates | n | Test | sig (p<0.05) |",
        "|---|---|---|---|---|",
    ]
    lines += [render_comparison_row(c) for c in demote.comparisons]
    lines.append("")

    lines += ["### Survival clause", "", f"> {SURVIVAL_CLAUSE_TEXT}", ""]
    if survivals:
        lines += [
            "| Arm | Axis | Stratum | Rates | n | Test | sig (p<0.05) |",
            "|---|---|---|---|---|---|---|",
        ]
        for entry in survivals:
            c = entry.comparison
            sig = "✓" if c.significant() else "—"
            lines.append(
                f"| {entry.arm} | {entry.axis} | {c.stratum} | "
                f"{c.arm_a} {c.rate_a:.3f} vs {c.arm_b} {c.rate_b:.3f} | n={c.n} "
                f"| {c.test_name} p={c.p_value:.4f} | {sig} |"
            )
    else:
        lines.append("No axis met the survival bar in this run.")
    lines.append("")
    return "\n".join(lines)


def render_cost_chapter(
    *,
    cost_per_grounded_correct: dict[str, float | None],
    amortization_curves: dict[str, dict[int, float]],
) -> str:
    """Cost chapter (issue #662: "cost-per-grounded-correct-answer and a
    build-cost amortization curve, so cost and quality argue in the same
    chart")."""
    lines = ["## Cost chapter", "", "### Cost per grounded-correct answer", ""]
    lines += ["| Arm | USD / grounded-correct answer |", "|---|---|"]
    for arm in sorted(cost_per_grounded_correct):
        value = cost_per_grounded_correct[arm]
        rendered = (
            "n/a (0 grounded-correct answers)" if value is None else f"${value:.4f}"
        )
        lines.append(f"| {arm} | {rendered} |")
    lines.append("")

    lines.append("### Build-cost amortization curve")
    lines.append("")
    volumes = sorted({v for curve in amortization_curves.values() for v in curve})
    header = "| Arm | " + " | ".join(f"n={v}" for v in volumes) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(volumes) + 1))
    for arm in sorted(amortization_curves):
        curve = amortization_curves[arm]
        row = " | ".join(f"${curve[v]:.4f}" if v in curve else "n/a" for v in volumes)
        lines.append(f"| {arm} | {row} |")
    lines.append("")
    return "\n".join(lines)


def render_decision_matrix(columns: list[str], rows: list[DecisionMatrixRow]) -> str:
    """Method-comparison decision matrix, evidence-tagged per cell (PRD #654
    user story 24)."""
    lines = ["## Method-comparison decision matrix (updated, evidence-graded)", ""]
    lines.append("| | " + " | ".join(columns) + " |")
    lines.append("|---|" + "---|" * len(columns))
    for row in rows:
        cells = [
            _tag(row.cells[col].value, row.cells[col].evidence_grade)
            if col in row.cells
            else "—"
            for col in columns
        ]
        lines.append(f"| {row.label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_honest_limits(limits: list[str]) -> str:
    """Honest-limits section (issue #662 AC 3: "structural analogues,
    calculated MDD, unmeasured high-churn/ACL cells")."""
    lines = ["## Honest limits", ""]
    lines += [f"{i}. {limit}" for i, limit in enumerate(limits, start=1)]
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class VerdictReportInput:
    """Everything :func:`render_verdict_report` needs to assemble the report."""

    title: str
    tldr: str
    comparisons: list[PairwiseComparison]
    kill: ClauseVerdict
    demote: ClauseVerdict
    survivals: list[SurvivalEntry]
    cost_per_grounded_correct: dict[str, float | None]
    amortization_curves: dict[str, dict[int, float]]
    decision_matrix_columns: list[str]
    decision_matrix_rows: list[DecisionMatrixRow]
    honest_limits: list[str]
    trust_note: str | None = field(default=None)


def render_verdict_report(report_input: VerdictReportInput) -> str:
    """Assemble the full committed Markdown verdict report (issue #662's
    definition of done). ``trust_note``, when set, is rendered as a loud
    top-of-file placeholder header (CODING_STANDARD §6.6) -- the caller
    (``run_verdict.py``) decides whether this run's data is real or an
    offline tracer; this function only renders what it is given.
    """
    sections = []
    if report_input.trust_note:
        sections.append(report_input.trust_note)
        sections.append("")
    sections.append(f"# {report_input.title}")
    sections.append("")
    sections.append("## TL;DR")
    sections.append("")
    sections.append(report_input.tldr)
    sections.append("")
    sections.append(render_axis_stratum_tables(report_input.comparisons))
    sections.append(
        render_clause_walkthrough(
            report_input.kill, report_input.demote, report_input.survivals
        )
    )
    sections.append(
        render_cost_chapter(
            cost_per_grounded_correct=report_input.cost_per_grounded_correct,
            amortization_curves=report_input.amortization_curves,
        )
    )
    sections.append(
        render_decision_matrix(
            report_input.decision_matrix_columns, report_input.decision_matrix_rows
        )
    )
    sections.append(render_honest_limits(report_input.honest_limits))
    return "\n".join(sections)
