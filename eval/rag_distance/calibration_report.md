# KB_RAG_DISTANCE_THRESHOLD calibration (#257 / #258 follow-up)

The RAG pre-LLM Cannot Confirm gate refuses when the closest chunk's FAISS
distance is **above** `KB_RAG_DISTANCE_THRESHOLD` (lower distance = closer).
Below is the trade-off between **correct-refusal** (rejecting out-of-scope
queries) and **over-refusal** (wrongly rejecting in-scope queries), swept over
the per-query min distances. Embeddings are `text-embedding-3-small` (unit-norm,
so L2² = 2 − 2·cos ∈ [0, 4]: this ceiling is equivalently a cosine floor).

- Positive (in-scope) cases: 10; distance range [0.620, 0.989], max in-scope distance = **0.989**.
- Negative (out-of-scope) cases: 15; distance range [1.278, 1.904], min out-of-scope distance = **1.278**.

**Recommended ceiling: 1.1 (Youden's J = 1.00; correct-refusal 100%, over-refusal 0%).**

## Sweep

| Ceiling | Correct-refusal (neg) | Over-refusal (pos) | Youden J |
|---|---|---|---|
| 0.6 | 100% | 100% | 0.00 |
| 0.8 | 100% | 60% | 0.40 |
| 1.0 | 100% | 0% | 1.00 |
| 1.1 ⭐ | 100% | 0% | 1.00 |
| 1.2 | 100% | 0% | 1.00 |
| 1.3 | 93% | 0% | 0.93 |
| 1.4 | 87% | 0% | 0.87 |
| 1.5 | 80% | 0% | 0.80 |
| 1.6 | 67% | 0% | 0.67 |
| 1.8 | 47% | 0% | 0.47 |
| 2.0 | 0% | 0% | 0.00 |

## Reading this

A clean **gap** between the max in-scope distance and the min out-of-scope
distance means a ceiling in that gap separates them perfectly (J = 1.0). If
they overlap, no single ceiling separates every case — the overlap needs
semantic reranking (Phase 13), not ceiling tuning. The recommended ceiling is
the plateau median (max margin to both error boundaries).
