# KB_SCORE_THRESHOLD calibration — Traditional Chinese (#256 / #261)

The Cannot Confirm gate refuses when the top BM25 score < `KB_SCORE_THRESHOLD`.
Below is the trade-off between **correct-refusal** (rejecting out-of-scope
queries) and **over-refusal** (wrongly rejecting in-scope queries), swept over
the per-query top scores. LLM-free and deterministic.

- Positive (in-scope) cases: 20; score range [4.903, 16.943], min in-scope score = **4.903**.
- Negative (out-of-scope) cases: 21; 16 score 0.0 (no overlap), non-zero leaks at [1.889, 1.889, 2.093, 2.849, 8.771].

**Recommended threshold: 4.0 (Youden's J = 0.95; correct-refusal 95%, over-refusal 0%).** Current default is 0.5.

## Sweep

| Threshold | Correct-refusal (neg) | Over-refusal (pos) | Youden J |
|---|---|---|---|
| 0.5 (current) | 76% | 0% | 0.76 |
| 1.0 | 76% | 0% | 0.76 |
| 1.5 | 76% | 0% | 0.76 |
| 2.0 | 86% | 0% | 0.86 |
| 2.5 | 90% | 0% | 0.90 |
| 3.0 | 95% | 0% | 0.95 |
| 3.5 | 95% | 0% | 0.95 |
| 4.0 ⭐ | 95% | 0% | 0.95 |
| 4.5 | 95% | 0% | 0.95 |
| 5.0 | 95% | 5% | 0.90 |
| 6.0 | 95% | 10% | 0.85 |

## Reading this

If the non-zero negative leaks fall **inside** the positive score range, no
threshold can separate them from real answers — those leaks need semantic
reranking (Phase 13 hybrid / FM2), not threshold tuning. The threshold's job
is only to reject the ~0-scoring clearly-out-of-scope queries while admitting
every real hit.

## Cross-language verdict (#256 / #261)

Chinese BM25 top-scores sit in a **higher band** than English: min in-scope **4.903** (English baseline 1.406), hits up to 16.943. Bigram tokenisation emits more tokens per query, inflating raw scores (ADR-0014), so the English-calibrated 0.5 is not transferable as-is.

❌ **One global `KB_SCORE_THRESHOLD=0.5` does NOT serve Chinese.** At 0.5 correct-refusal is only 76%. The 4 catchable adjacent-absent leaks ([1.889, 1.889, 2.093, 2.849]) sit **below** the min in-scope score (4.903), so a per-language threshold separates them: the sweep recommends **4.0** (correct-refusal 95%, over-refusal 0%).

The remaining 1 leak(s) ([8.771]) fall **inside** the in-scope range (≥ 4.903) — e.g. an order-page query whose surface tokens match a real Section but whose specific ask is absent — so, exactly like the English `adjacent_absent` leaks, **no threshold (per-language or not) separates them**; they are semantic-reranking (Phase 13 hybrid / FM2) territory in both languages.

**Recommendation:** adopt a per-language Chinese threshold (`KB_SCORE_THRESHOLD_ZH`) to fix the magnitude mismatch — an **interim** measure, superseded by the Phase 13 reranker (which also catches the in-scope-range residual). Re-sweep whenever the Chinese corpus grows.
