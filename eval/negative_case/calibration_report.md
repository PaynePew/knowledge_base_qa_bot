# KB_SCORE_THRESHOLD calibration (#253)

The Cannot Confirm gate refuses when the top BM25 score < `KB_SCORE_THRESHOLD`.
Below is the trade-off between **correct-refusal** (rejecting out-of-scope
queries) and **over-refusal** (wrongly rejecting in-scope queries), swept over
the per-query top scores. LLM-free and deterministic.

- Positive (in-scope) cases: 10; score range [1.406, 7.268], min in-scope score = **1.406**.
- Negative (out-of-scope) cases: 15; 13 score 0.0 (no overlap), non-zero leaks at [1.489, 1.703].

**Recommended threshold: 0.5 (Youden's J = 0.87; correct-refusal 87%, over-refusal 0%).** Current default is 0.5.

## Sweep

| Threshold | Correct-refusal (neg) | Over-refusal (pos) | Youden J |
|---|---|---|---|
| 0.25 | 87% | 0% | 0.87 |
| 0.5 ⭐ | 87% | 0% | 0.87 |
| 0.75 | 87% | 0% | 0.87 |
| 1.0 | 87% | 0% | 0.87 |
| 1.25 | 87% | 0% | 0.87 |
| 1.5 | 93% | 10% | 0.83 |
| 1.75 | 100% | 20% | 0.80 |
| 2.0 | 100% | 20% | 0.80 |

## Reading this

If the non-zero negative leaks fall **inside** the positive score range, no
threshold can separate them from real answers — those leaks need semantic
reranking (Phase 13 hybrid / FM2), not threshold tuning. The threshold's job
is only to reject the ~0-scoring clearly-out-of-scope queries while admitting
every real hit.
