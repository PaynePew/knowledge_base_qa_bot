# Lint Fixtures

Demo corpus for `POST /lint`. Contains planted fixtures that exercise all seven lint checks.
Used for demos and hermetic e2e testing; **never pollutes the production wiki**.

## Purpose

The production corpus (`docs/` + `fake-docs/`) is clean — running `POST /lint` against it
returns 0 findings (proof of corpus health). Demo material lives here and is opt-in.

## Demo Flow

```
# Baseline: production wiki is clean
POST /lint → 0 findings

# Load fixtures into wiki state
python scripts/load_lint_fixtures.py

# Demo: all checks fire
POST /lint → 9 findings (across 7 checks)

# Diff against golden expected output
diff wiki/lint-report.md eval/lint_fixtures/expected_lint_report.md

# Revert to production-clean state
git checkout wiki/
rm -f wiki/lint-report.md
POST /lint → 0 findings (proves no state bleed)
```

## Fixture Inventory

| # | Check | Wiki pages | Source |
|---|-------|-----------|--------|
| 1+2 | C5 direct contradiction | `refund-policy-a`, `refund-policy-b` | `policy_a.md` (5 days) vs `policy_b.md` (14 days) |
| 3 | C5 duplicate | `shipping`, `our-shipping` | `about_us.md`, `shipping_info.md` — same concept |
| 4 | C6 stale | `aged` (updated 2026-01-01) | `aged_policy.md` — loader touches mtime to now |
| 5 | C3 failed-grounding | `broken-page` | `policy_a.md` — status: failed_grounding |
| 6 | C4-a slug collision | `pricing`, `pricing-2` | — |
| 7 | C11 orphan | `legacy-faq` | `deleted_source.md` — does not exist |
| 8 | C1 retrieval_empty | — | 3 log lines: "vip membership fee" |
| 9 | C1 below_threshold | — | 2 log lines: "how long is refund" top_section=refund-timeline |

**C2 (red links)** is not planted — `shipping` and `our-shipping` already contain
`[[order-tracking]]` which resolves to a red link (no such page exists in the fixtures).

## Production vs Fixture Separation Invariant

1. `eval/lint_fixtures/sources/` are **never** copied to `docs/` or `fake-docs/`.
2. `eval/lint_fixtures/wiki/` pages are **only** loaded by `scripts/load_lint_fixtures.py`.
3. The loader is idempotent: re-running overwrites without erroring.
4. Reverting is a single command: `git checkout wiki/ && rm -f wiki/lint-report.md`.

## Files

```
eval/lint_fixtures/
├── README.md                     ← this file
├── log_entries.txt               ← 5 chat_fallback lines for C1 fixtures
├── expected_lint_report.md       ← golden expected output for e2e test
├── sources/
│   ├── policy_a.md               ← C5 direct partner A
│   ├── policy_b.md               ← C5 direct partner B
│   ├── about_us.md               ← C5 duplicate partner A
│   ├── shipping_info.md          ← C5 duplicate partner B
│   └── aged_policy.md            ← C6 stale demonstration source
└── wiki/
    └── concepts/
        ├── refund-policy-a.md    ← C5 direct (5 business days)
        ├── refund-policy-b.md    ← C5 direct (14 business days)
        ├── shipping.md           ← C5 duplicate (about_us perspective)
        ├── our-shipping.md       ← C5 duplicate (shipping_info perspective)
        ├── aged.md               ← C6 stale (updated 2026-01-01)
        ├── broken-page.md        ← C3 failed-grounding
        ├── pricing.md            ← C4-a slug collision base
        ├── pricing-2.md          ← C4-a slug collision variant
        └── legacy-faq.md         ← C11 orphan (deleted_source.md)
```
