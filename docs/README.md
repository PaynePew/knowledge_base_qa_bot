# docs/

Markdown Source files for the knowledge base.

## Layout

```
docs/
  fake-docs/      # Synthetic Source pool (demo KB + Phase 8 eval input)
  account_help.md # Three hermetic fixture files used by test suites
  refund_policy.md
  shipping_faq.md
```

`docs/fake-docs/` is the single synthetic Source pool shared by the runtime
demo KB (`POST /ingest` from the root) and the Phase 8 Paraphrase Comparison
eval. The Phase 8 eval freezes a snapshot of this directory into
`eval/paraphrase_comparison/corpus/` at run time so the committed eval report
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
