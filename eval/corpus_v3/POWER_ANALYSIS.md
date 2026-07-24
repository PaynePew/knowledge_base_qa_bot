# Corpus v3 query-set power analysis

> Design note for issue #660 (ADR-0045 Prerequisite 4). The verdict report
> (a later slice, PRD #654) cites this note for its query-set-size
> justification. The calculation itself lives in `eval/corpus_v3/power.py`
> (pure, unit-tested — `eval/corpus_v3/tests/test_power.py`); this note
> records the chosen inputs and the resulting `n`, not the math.

## Why this exists

The v2 eval (`eval/paraphrase_comparison/report.md`, n=260) could not detect
its own observed A-B gap: the minimal detectable difference (MDD) at n=260
is roughly 6-7 hit@3 points (`eval/fairness_review/literature.md` §2),
larger than the observed 5.6-point gap. "Not significant" meant
"underpowered", not "no difference" (ADR-0045). Corpus v3's query-set size
must be picked so this cannot happen again: the MDD at the chosen `n` must
be smaller than the effect size that mattered last time.

## Method

Standard closed-form paired-proportions (McNemar) sample-size formula
(Connor 1987 — the same normal-approximation family Sakai's topic-set-size
design uses for IR test-collection sizing):

```
n = (z_(alpha/2) * sqrt(psi) + z_power * sqrt(psi - mdd**2))**2 / mdd**2
```

`psi` (the discordant-pair proportion) is estimated from one baseline
proportion under an independence assumption between the two arms'
per-query outcomes (`discordant_proportion_under_independence`) —
independence never *under*-estimates `psi`, so it never under-estimates the
required `n` (positive correlation, which paired same-query comparisons
typically have, would only shrink `psi`). This is the conservative
direction given ADR-0045's "burden of proof is on the wiki" stance.

This targets the three ADR-0045 kill-clause content axes — grounding pass
rate, correct-refusal rate, contradiction-leak rate — all binary per-query
outcomes tested with `eval.corpus_v3.statistics.mcnemar_test`. The bootstrap
wrapper (rate metrics, e.g. reciprocal rank) is not separately sized here;
its query-count requirement is of the same order and is covered by the same
`n`.

## Chosen inputs (primary, per scenario stratum, English)

| Input | Value | Rationale |
|---|---|---|
| `alpha` | 0.05 | ADR-0045: "p < 0.05" |
| `power` | 0.80 | Conventional default; no ADR-specified override |
| `mdd` | 0.05 (5 points) | Below the v2 MDD (6-7 pts) *and* below the observed 5.6-pt gap that v2 couldn't confirm — the whole point of this exercise |
| `p_baseline` | 0.85 | Round number inside the v2 observed hit@3 range (0.880-0.936), chosen slightly conservative (not pinned to the ceiling, where `psi` shrinks and `n` drops — see sensitivity table) |

`required_n_paired_proportions(PowerInputs(alpha=0.05, power=0.80, mdd=0.05,
p_baseline=0.85))` → **`n = 909`** paired queries per scenario stratum
(`discordant_proportion ≈ 0.290`; `achieved_power_paired_proportions(909,
...) ≈ 0.8004`, confirming the ceiling rounding did not undershoot the
target).

Applied via `per_stratum_requirements` to the four scenario strata
(`factoid`, `cross_doc`, `version_conflict`, `unanswerable`) with the same
`PowerInputs`, every stratum gets the same `n = 909` (identical inputs
assumption, stated explicitly — a future slice with per-stratum baseline
estimates can pass per-stratum overrides instead).

## zh slice — its own, explicitly relaxed gate

ADR-0045 Prerequisite 3 calls for "a zh query slice with its own gates."
Full parity with the English `n = 909` per stratum is not adopted for the
zh slice — a resource-constrained relaxation, stated openly rather than
hidden in an unlabeled smaller sample (the same honesty standard PRD #654's
"Further Notes" applies to its other known limits):

| Input | Value |
|---|---|
| `alpha` | 0.05 |
| `power` | 0.70 (relaxed from 0.80) |
| `mdd` | 0.10 (10 points, relaxed from 0.05) |
| `p_baseline` | 0.85 |

`required_n_paired_proportions(PowerInputs(alpha=0.05, power=0.70, mdd=0.10,
p_baseline=0.85))` → **`n = 200`**. This is still smaller than the v2 MDD
(10 pts vs a possible v2-zh figure, which does not exist — v2 was
English-only, so there is no prior zh baseline to beat) but is explicitly
*not* held to the same standard as the English primary axes. The verdict
report must state the zh axis's lower power in its per-language table, not
only its point estimate.

## Sensitivity (for a downstream corpus-build budget call)

The primary `n = 909` per scenario stratum, times 4 strata, is ~3,636
English queries for one content axis's fully-powered per-stratum test —
large for a hand-/LLM-generated demo corpus. This table is the honest
trade-off surface a later corpus-build issue must reconcile against actual
budget; it is not itself a decision:

| `mdd` | `p_baseline` | `n` |
|---|---|---|
| 0.05 | 0.80 | 1097 |
| 0.05 | 0.85 | 909 |
| 0.05 | 0.90 | 689 |
| 0.06 | 0.85 | 646 |
| 0.08 | 0.85 | 380 |
| 0.10 | 0.85 | 253 |

Any cell with `mdd < 0.06` still improves on the v2 eval's demonstrated
failure mode (a 6-7 pt MDD that missed the real 5.6 pt gap); `mdd >= 0.06`
does not and must not be presented as "power-analysis-derived" without that
caveat. If the corpus-build issue must trade down for budget, the honest
move is to shrink the number of *fully-powered* scenario strata (treat some
as exploratory, reported without a significance claim) rather than silently
loosening `mdd` past the v2 failure point.

## Reproducing these numbers

```python
from eval.corpus_v3.power import PowerInputs, required_n_paired_proportions

result = required_n_paired_proportions(
    PowerInputs(alpha=0.05, power=0.80, mdd=0.05, p_baseline=0.85)
)
result.required_n  # 909
```
