# Corpus v3 pilot batch report (issue #674)

Generated: 2026-07-24T14:41:22Z. Real pilot batch through the wired AnswerFn seam (`answer_fn.build_answer_fn`): every arm's production `query()` surface, real retrieval over the committed corpus v3 fixtures, real LLM calls.

## Batch size and arm coverage

- 32 English queries: the first 8 per scenario stratum in `query_id` order (deterministic pre-registered enumeration; zh excluded because `PLANNED_LIVE_CALLS` is en-only).
- All 4 arms answered every query: 128 answers, 221 recorded LLM calls, models gpt-4o-mini.
- Pilot spend: $0.0356

## Per-arm results

| arm | answers | LLM calls | USD | negatives refused | of which pre-LLM gate | answerable answered |
|---|---|---|---|---|---|---|
| wiki | 32 | 52 | $0.0083 | 8/8 | 0 | 20/24 |
| rag | 32 | 64 | $0.0101 | 8/8 | 0 | 19/24 |
| hybrid | 32 | 52 | $0.0085 | 8/8 | 0 | 20/24 |
| dense_over_wiki | 32 | 53 | $0.0087 | 7/8 | 0 | 20/24 |

## Refusal-gate fairness (ADR-0045 Prerequisite 2)

Per-arm verdicts on the corpus v3 negative set (the pilot's `unanswerable` slice, n=8 per arm):

- **wiki** (Cannot-Confirm score threshold (calibrated #253/#261)): refused 8/8 negatives, 0 at the pre-LLM gate; answered 20/24 answerable queries.
- **rag** (distance ceiling 1.1 (calibrated eval/rag_distance #257/#258)): refused 8/8 negatives, 0 at the pre-LLM gate; answered 19/24 answerable queries.
- **hybrid** (OR-gate over both calibrated gates): refused 8/8 negatives, 0 at the pre-LLM gate; answered 20/24 answerable queries.
- **dense_over_wiki** (NO calibrated pre-LLM gate (documented gap, `answer_fn.dense_over_wiki_query`)): refused 7/8 negatives, 0 at the pre-LLM gate; answered 20/24 answerable queries.

## Full-run projection input (the per-answer vs per-call correction)

- Planned ANSWERS: 14544 (`run_verdict.PLANNED_LIVE_CALLS`: 3,636 en queries x 4 arms).
- Measured LLM calls per answer: 1.727 (221 calls / 128 answers -- synthesis + grounding-verify per answered query; gate refusals cost 0).
- **Corrected `--planned-calls`: 25112** = ceil(14544 x 1.727). Projecting the raw 14544 over this ledger would undercount spend by ~42%.

Produce the canonical projection with:

```
uv run python -m eval.corpus_v3.run_verdict --mode live --confirm-live --pilot-ledger eval/corpus_v3/pilot_ledger.json --planned-calls 25112
```

## Projection output

Verbatim from the run_verdict invocation above (exit code 3 — the guard's
designed stop: projection produced, live answering not wired into the
run_verdict CLI yet, and the full live run stays a human decision):

```
cost guard: projected spend $4.05 is within the $10.00 cap -- proceeding
cost guard cleared, but the real per-arm answer_fn integration (see module docstring: AnswerFn) is not wired yet -- this issue leaves it as an explicit follow-up rather than faking a live run
```

## Recommendation (go/no-go input for the human)

- **Projected full-run spend: $4.05 of the $10.00 cap** (25,112 projected
  LLM calls at the pilot's measured $0.000161/call average) — GO is
  affordable. The projection uses the corrected per-call figure; the
  uncorrected per-answer figure (14,544) would have projected ~$2.34 and
  undercounted by ~42%.
- Refusal-gate fairness (Prerequisite 2) holds on the pilot negatives for
  every CALIBRATED gate (wiki 8/8, rag 8/8, hybrid 8/8; answerable
  answer-rates 19-20/24 show the gates are not trivially refusing
  everything). `dense_over_wiki` (no calibrated gate, documented gap)
  refused 7/8 via synthesis self-refusal + Grounding Check alone — state
  this as a caveat on its correct-refusal axis in the verdict, or calibrate
  a gate for it before the live run if per-arm parity is required.
- Note: zero negatives were refused at the PRE-LLM gate in this pilot —
  every refusal happened at the synthesis/grounding layer. The calibrated
  thresholds did not fire on corpus v3 negatives (retrieval always surfaced
  plausible sections); correct-refusal still held end-to-end.
- Remaining work before a live run: wire `answer_fn.build_answer_fn` into
  `run_verdict --mode live` past the guard (the explicit follow-up its exit
  code 3 names).
