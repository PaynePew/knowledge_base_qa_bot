# QA_QUESTION_TOKEN_WEIGHT calibration (#578)

Rule 2a (#570) joins a qa page's frontmatter `question:` into its BM25
tokens. `QA_QUESTION_TOKEN_WEIGHT` scales the term-frequency contribution
of matches that come ONLY from that injected question, never from real
body content. Two competing rates, swept over a synthetic corpus that
reproduces the reported collision at scale (LLM-free, deterministic):

- **Pollution rate** — fraction of the top-3 window for `你們配送到哪些國家？` occupied by a qa page that is NOT the real fact-carrier (the distractor qa page + 30 synthetic noise qa pages, all sharing only the query's generic interrogative bigrams). Lower is better.
- **Own-question hit rate** — whether the #570 regression fixture (a qa page with zero body/question token overlap) is still retrievable by its own question (`Which countries do you ship to?`). This is a hard floor, not a trade-off: a weight that drops it is disqualified regardless of its pollution rate.

**Recommended weight: 0.3** (separation = 0.33; own-question hit rate 100%, pollution rate 67%).

## Sweep

| Weight | Own-question hit rate | Pollution rate | Separation |
|---|---|---|---|
| 1.0 | 100% | 67% | 0.33 |
| 0.75 | 100% | 67% | 0.33 |
| 0.5 | 100% | 67% | 0.33 |
| 0.4 | 100% | 67% | 0.33 |
| 0.3 ⭐ | 100% | 67% | 0.33 |
| 0.2 | 100% | 67% | 0.33 |
| 0.15 | 100% | 67% | 0.33 |
| 0.1 | 100% | 67% | 0.33 |
| 0.05 | 100% | 67% | 0.33 |
| 0.02 | 100% | 67% | 0.33 |
| 0.0 | 0% | 0% | 0.00 |

## Reading this

At `weight=1.0` (the pre-#578 behaviour) the distractor/noise qa pages'
shared-interrogative matches count at full term frequency, same as real
body content — the collision the issue reports. Decreasing the weight
shrinks their contribution; the own-question hit rate is the floor below
which rule 2a's #570 fix itself starts to regress.

**Limitation of the pollution-rate metric above:** with 30 identical-shape
noise pages sharing the same two bigrams, their tied scores only clear
exactly at `weight=0.0` — any smaller-but-nonzero weight still leaves two
of them in the k=3 window, so the metric cannot distinguish 0.02 from
1.0 in THIS adversarial construction. That is itself a finding, not a
bug: pure per-token downweighting caps out against a large-enough swarm
of pages sharing a generic interrogative. The issue's own scope decision
names the fallback for that case — a CJK-interrogative stopword list —
as a stopgap gated on eval evidence of pressure AT THE CURRENT corpus
scale. Today's committed corpus has only 2 zh qa pages (not the ~30+
simulated here) and the reported production collision no longer
reproduces against it, so that gate is not tripped; the table below
shows the downweight is doing real, graduated work at today's actual
scale.

## Score margin at today's real scale (no synthetic swarm)

`noise_count=0` — just the real fact-carrier vs. the one real-shaped
distractor qa page (the committed corpus has no more zh qa pages than
this today). Score decreases smoothly and substantially with weight,
well before the own-question floor is at risk:

| Weight | Real content score | Distractor qa score | Margin |
|---|---|---|---|
| 1.0 | 2.570 | 1.517 | 1.053 |
| 0.75 | 2.570 | 1.215 | 1.356 |
| 0.5 | 2.570 | 0.868 | 1.702 |
| 0.4 | 2.570 | 0.715 | 1.855 |
| 0.3 ⭐ | 2.570 | 0.553 | 2.018 |
| 0.2 | 2.570 | 0.380 | 2.190 |
| 0.15 | 2.570 | 0.290 | 2.281 |
| 0.1 | 2.570 | 0.196 | 2.374 |
| 0.05 | 2.570 | 0.100 | 2.471 |
| 0.02 | 2.570 | 0.040 | 2.530 |
| 0.0 | 2.570 | 0.000 | 2.570 |
