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

## Spec parity vs Stack A (#656)

- **Gate setting:** `KB_RAG_DISTANCE_THRESHOLD` production default = **1.1** (`vector_rag/app/retrieval.py _KB_RAG_DISTANCE_THRESHOLD_DEFAULT`), enabled by default since #258.
- **Spec calibrated against:** the same `eval.negative_case` positive/negative case sets as Stack A's `KB_SCORE_THRESHOLD` (`eval/negative_case/calibration_report.md`) — `collect_distances` takes `POSITIVE_CASES` / `NEGATIVE_CASES` from that package directly, so both gates are calibrated on identical data.
- **Parity:** same Youden's J sweep + plateau-median selection; `recommend()` now also prefers the current production default when it is itself on the optimal plateau, mirroring `eval.negative_case.calibrate.recommend` — a no-op on the English data above (the plateau median already is 1.1), but it keeps a future re-calibration from churning a working default for no reason.
- **zh coverage:** not yet run here. Stack A has a Traditional-Chinese negative-case spec (`eval/negative_case/cases_zh.py` + `corpus_zh/`); this module now supports it via `KB_EVAL_LANG=zh` (`eval.negative_case.lang.resolve_lang`), mirroring `eval.negative_case.calibrate`. Sweeping it needs real `text-embedding-3-small` calls — a manual, quota-spending run like this English sweep — so it was not run in #656 (no `OPENAI_API_KEY` in that session). **Follow-up:** run `KB_EVAL_LANG=zh uv run python -m eval.rag_distance.calibrate` and commit the resulting `calibration_report_zh.md` before ADR-0045 Prerequisite 2 is treated as satisfied for zh.
