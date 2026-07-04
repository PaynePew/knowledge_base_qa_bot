# docs/

Markdown Source files for the knowledge base.

## Layout

```
docs/
  fake-docs/      # Curated EN Source pool (demo KB + Phase 8 eval input)
  demo-zh/        # Curated zh-TW Source pool (1:1 topic parity with fake-docs)
  account_help.md # Three hermetic fixture files used by test suites
  refund_policy.md
  shipping_faq.md
```

`docs/fake-docs/` (EN) and `docs/demo-zh/` (zh-TW) form the **corpus v2**
demo pool (issue #440): 16 topics per language, hand-authored from the
canonical fact sheet `project-docs/demo-corpus/FACTS.md`. Every statement in
these files traces to an `F-*` fact entry — do not add or change a fact here
without updating FACTS.md first. The plan is mirrored in
`eval/paraphrase_comparison/generation/corpus_generator.py` (`DOC_SPECS`),
whose tests enforce name consistency and the ≥50 Gold Sections floor.

The pre-v2 generated corpus is preserved at the git tag `full-fake-corpus`.
The Phase 8 eval freezes a snapshot of `docs/fake-docs/` into
`eval/paraphrase_comparison/corpus/` at run time, so the committed eval report
stays reproducible even when demo content is later edited.

The three root-level fixture files (`account_help.md`, `refund_policy.md`,
`shipping_faq.md`) are a stable 3-file hermetic fixture for test suites that
assert exact counts. They are **not** part of the demo KB.

## Basename uniqueness constraint

**Basenames must be globally unique across all of `docs/`** (including
subdirectories). This is a hard invariant:

- `markdown_kb` indexes sections by `{basename}#{slug}` (e.g.
  `returns_policy.md#return-window`). Duplicate basenames across subdirectories
  would produce colliding section IDs.
- Phase 8 Gold Section IDs use the same `{basename}#{slug}` form; duplicate
  basenames would make section IDs ambiguous between the eval corpus snapshot
  and the source pool.

When adding a new file anywhere under `docs/`, verify that its basename does
not already exist in any sibling or parent directory within `docs/`.
