# Corpus v3 adversarial fixtures

Issue #661, PRD #654 user stories 10-12 / 18, ADR-0045 (the curated layer's
claimed home ground). This is the fixture inventory and the honest
budget/power-analysis reconciliation the design docs deferred to "a later
corpus-build issue" (`POWER_ANALYSIS.md` § Sensitivity, `generation/SPEC.md`
§ Out of scope for #660).

## What this is / is not

This issue builds the **adversarial corpus** — raw Sources (`corpus/`) and
the curated wiki concept pages a real `/ingest` run would produce over them
(`wiki/concepts/`). It does **not** build the query set: query generation
(`generation/SPEC.md`) explicitly requires this corpus to exist first and is
a separate, later issue in PRD #654's dependency order.

## Regenerating

```bash
uv run python -m eval.corpus_v3.build_corpus
```

Deterministic and offline (no `OPENAI_API_KEY`, no LLM call — every group in
`ADVERSARIAL_GROUPS` is hand-authored data, not synthesized). Re-running
reproduces the committed `corpus/` and `wiki/concepts/` files byte-for-byte
(`tests/test_build_corpus.py::test_regenerating_reproduces_the_committed_fixtures_byte_for_byte`)
and rewrites `BUILD_COST.offline-tracer.md` (CODING_STANDARD §6.6:
trust-marked filename, since this construction method is offline). Never
writes outside this package's own `corpus/` / `wiki/concepts` dirs —
production `docs/` and `wiki/` are never touched (enforced by
`test_write_corpus_fixtures_never_writes_outside_its_own_fixture_dirs`).

## Fixture inventory

| Class | Group id | Raw Sources | Wiki page cites |
|---|---|---|---|
| redundancy | `store-hours` | `store_hours_a.md#weekday-hours`, `store_hours_b.md#weekday-hours` | both |
| redundancy | `loyalty-signup` | `loyalty_program.md#how-to-join`, `membership_faq.md#joining-the-program` | both |
| redundancy | `two-factor-setup` | `account_security_2fa.md#two-factor-setup`, `security_faq.md#enabling-2fa` | both |
| contradiction | `gift-card-expiration` | `gift_card_terms.md#expiration` (never) vs `gift_card_faq.md#expiration` (12mo) | both, `open_questions` flags the conflict |
| contradiction | `free-shipping-threshold` | `shipping_policy_v2.md#free-shipping-threshold` ($50) vs `promo_terms.md#free-shipping-threshold` ($75) | both, `open_questions` flags the conflict |
| contradiction | `restocking-fee` | `returns_policy_addendum.md#restocking-fee` (none) vs `electronics_returns.md#restocking-fee` (15%) | both, `open_questions` flags the conflict |
| version_evolution | `return-shipping-label-cost` | v1 (2025-01) → v2 (2025-11) → **v3 (2026-06, current)** | v3 only |
| version_evolution | `loyalty-gold-tier-threshold` | v1 (2024-01) → v2 (2025-06) → **v3 (2026-05, current)** | v3 only |
| version_evolution | `warranty-claim-window` | v1 (2024-03) → v2 (2025-08) → **v3 (2026-04, current)** | v3 only |

9 groups, 3 per class (`MIN_INSTANCES_PER_CLASS = 3` in `build_corpus.py`,
enforced by `test_each_adversarial_class_meets_the_minimum_instance_floor`).
21 raw Source Sections total (6 redundancy + 6 contradiction + 9
version_evolution).

Each class exercises the gold-mapping table (`gold.py`, issue #658)
differently, with no code change needed — the fixtures are the thing under
test:

- **redundancy** — the wiki id resolves both raw ids to the same gold class
  (dedup: a raw-docs stack retrieving both near-duplicates wastes two top-k
  slots on one fact; the wiki collapses them to one).
- **contradiction** — the wiki id resolves both raw ids too, but the wiki
  page picks no winner (`open_questions` names the conflict); this is the
  `contradiction-leak rate` axis's home ground — whether a stack's answer
  cites the wiki's honest "unresolved" framing or leaks one raw figure as
  fact.
- **version_evolution** — only the NEWEST raw id resolves into the wiki's
  gold class; an older version resolves to only itself, a defined non-hit
  (`test_version_evolution_groups_give_gold_coverage_to_only_the_newest_id`).
  A raw-docs stack has no such pruning — all three versions are equally
  retrievable, which is exactly the amplification the version-conflict
  scenario stratum is meant to expose.

## Power-analysis reconciliation (AC "enough instances to support the
per-stratum n")

`POWER_ANALYSIS.md` derives `n = 909` fully-powered English queries **per
scenario stratum** (`factoid` / `cross_doc` / `version_conflict` /
`unanswerable`) — a query-set-size figure, not a corpus-instance count. Its
own Sensitivity section is explicit that reaching that n at demo scale is "a
later corpus-build issue must reconcile against actual budget," and that the
honest move if budget does not stretch is to shrink the number of
*fully-powered* strata rather than silently loosening the MDD. This issue
**is** that corpus-build issue; here is the reconciliation:

- The corpus does not need 909 *Sections* per class — it needs enough
  DISTINCT topical instances that a generator (a later issue) can draw many
  queries per instance (paraphrase variants, both model families named in
  `generation/SPEC.md`, high/low lexical-overlap variants) without
  collapsing into near-duplicate queries against the same underlying fact.
  3 distinct instances per class, each with realistic, freestanding prose,
  gives that generator 3 independent topical seeds per class to multiply
  from — comparable in shape to how a single `docs/` Source Section already
  supports many differently-phrased queries in the v2 eval's query set.
- This is a **stated, explicit** trade-off in the same spirit as
  `POWER_ANALYSIS.md`'s own zh-slice relaxation: 3 instances per class is a
  starting floor (`MIN_INSTANCES_PER_CLASS`), not a claim that it alone
  reaches n=909. If the query-generation issue's actual per-instance
  paraphrase yield falls short of the target n at full power, growing
  `ADVERSARIAL_GROUPS` is a pure data addition (append an `AdversarialGroup`
  entry, rerun `build_corpus.py`) — no harness code changes required, by
  construction of this module.
- The `version_conflict` scenario stratum specifically is the one this
  corpus most directly powers (`gold_section_ids` already encodes the
  "only the newest is correct" answer key per `query_schema.py`'s
  requirement). `factoid` and `cross_doc` queries can also target these
  fixtures (e.g. a factoid query against a redundancy group's dedup content,
  a cross_doc query synthesizing across a contradiction pair), so the same
  9 groups serve more than one scenario stratum's query supply.

## Build cost

See `BUILD_COST.offline-tracer.md` (committed, regenerated by
`build_corpus.py`). The `.offline-tracer.` filename marker and its top-of-file
`⚠️ PLACEHOLDER` header follow CODING_STANDARD §6.6: this construction
method is offline and deterministic, so its ledger `build` entries are a
real zero for *this* method, but must not be read as a general "the wiki
corpus build costs $0" claim — ADR-0045 cites a real live-synthesis build
cost (~$4.4/corpus). A future live corpus-build run (LLM synthesis, e.g. to
scale up instance count) would record real non-zero entries into the same
`CostLedger` shape via `eval.cost_ledger.hooks` and would earn the canonical
(non-trust-marked) filename.
