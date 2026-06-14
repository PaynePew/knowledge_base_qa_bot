# KB_SCORE_THRESHOLD calibration — Traditional Chinese (#256)

The Cannot Confirm gate refuses when the top BM25 score < `KB_SCORE_THRESHOLD`.
Below is the trade-off between **correct-refusal** (rejecting out-of-scope
queries) and **over-refusal** (wrongly rejecting in-scope queries), swept over
the per-query top scores. LLM-free and deterministic.

- Positive (in-scope) cases: 10; score range [2.159, 9.877], min in-scope score = **2.159**.
- Negative (out-of-scope) cases: 15; 12 score 0.0 (no overlap), non-zero leaks at [1.52, 1.582, 1.582].

**Recommended threshold: 1.875 (Youden's J = 1.00; correct-refusal 100%, over-refusal 0%).** Current default is 0.5.

## Sweep

| Threshold | Correct-refusal (neg) | Over-refusal (pos) | Youden J |
|---|---|---|---|
| 0.25 | 80% | 0% | 0.80 |
| 0.5 (current) | 80% | 0% | 0.80 |
| 0.75 | 80% | 0% | 0.80 |
| 1.0 | 80% | 0% | 0.80 |
| 1.25 | 80% | 0% | 0.80 |
| 1.5 | 80% | 0% | 0.80 |
| 1.75 | 100% | 0% | 1.00 |
| 1.875 ⭐ | 100% | 0% | 1.00 |
| 2.0 | 100% | 0% | 1.00 |
| 2.25 | 100% | 10% | 0.90 |
| 2.5 | 100% | 10% | 0.90 |
| 3.0 | 100% | 10% | 0.90 |
| 4.0 | 100% | 10% | 0.90 |
| 5.0 | 100% | 40% | 0.60 |

## Reading this

If the non-zero negative leaks fall **inside** the positive score range, no
threshold can separate them from real answers — those leaks need semantic
reranking (Phase 13 hybrid / FM2), not threshold tuning. The threshold's job
is only to reject the ~0-scoring clearly-out-of-scope queries while admitting
every real hit.

## Cross-language verdict (#256)

Chinese BM25 top-scores sit in a **higher band** than English: min in-scope **2.159** (English baseline 1.406), hits up to 9.877. Bigram tokenisation emits more tokens per query, inflating raw scores (ADR-0014), so the English-calibrated 0.5 is not transferable as-is.

❌ **One global `KB_SCORE_THRESHOLD=0.5` does NOT serve Chinese.** At 0.5 correct-refusal is only 80%: the 3 adjacent-absent leaks ([1.52, 1.582, 1.582]) clear the gate. Unlike English, the Chinese leaks fall **below** the min in-scope score, leaving a clean gap (1.582, 2.159]; the sweep recommends **1.875** (correct-refusal 100%, over-refusal 0%).

**Recommendation:** adopt a per-language Chinese threshold. Because `retrieval._SCORE_THRESHOLD` is read globally at import time, this means either a per-language override (a `KB_SCORE_THRESHOLD_ZH` consulted when the query is detected as CJK) or BM25 score normalisation so a single threshold spans both languages. The magnitude gap is robust; the exact value is illustrative on this small corpus and should be re-swept on a larger Chinese KB before shipping a production default.
