# W2 layered architecture, target shape follows AgriciDaniel/claude-obsidian

When the wiki layer is added, this project takes the **layered (W2)** approach: the immutable `docs/` Sources and the LLM-maintained `wiki/` pages are *both* queryable surfaces, with `wiki/` checked first and `docs/` as the ground-truth fallback. The concrete target structure mirrors [`AgriciDaniel/claude-obsidian`](https://github.com/AgriciDaniel/claude-obsidian) (5.4K⭐, actively maintained, the most-starred reference implementation of Karpathy's LLM Wiki pattern): a `wiki/` directory with `hot.md` (Hot Cache), `index.md` (Wiki Index), `log.md` (Wiki Log), and type-segregated pages under `entities/`, `concepts/`, `comparisons/`, etc. Per-type extraction is driven by Source Templates under `_templates/`.

We chose W2 over the alternatives because W1 (strict Karpathy — only `wiki/` queryable, `docs/` reduced to provenance) would force a full retrieval rewrite when the wiki layer arrives, eliminating the cheap upgrade path; W3 (in-place enrichment of `docs/`) directly violates ADR-0001's source-grounded discipline by mutating the immutable Sources; and W4 (Q&A cache only) cannot deliver the compounding-artifact behavior that motivates the wiki pattern in the first place. W2 also degrades gracefully — if the wiki layer fails to materialize, the prototype's BM25-over-Sources retrieval keeps working unchanged.

The choice of `claude-obsidian` as the reference implementation, rather than designing the wiki layer from first principles, is deliberate: at 5.4K stars and shipping today's commits, it is the highest-signal community validation of which Karpathy-gist patterns actually compound in real use. Earlier drafts of this decision leaned on a single unverified gist commenter's tournament results; that source was wrong to elevate, and `claude-obsidian` replaces it as the primary external reference.

## Considered Options

- **W1 — strict Karpathy, only `wiki/` queryable.** Rejected: forces a full retrieval-path rewrite at the moment the wiki layer ships, with no easy fallback if the wiki proves disappointing in practice. Citation surface also becomes harder to reconcile with the PROMPT.md `filename#heading` contract.
- **W3 — LLM enriches `docs/` in place, no separate `wiki/`.** Rejected: mutating Sources contradicts ADR-0001 (Sources are immutable; the LLM never modifies them). Loses the provenance separation that the Karpathy pattern depends on.
- **W4 — `wiki/` exists only as a Q&A cache.** Rejected as a target (acceptable as a *subset* of W2): a cache layer cannot host synthesis pages, contradiction flags, or the cross-references that produce compounding value. W2 subsumes W4 (the answer-filing stretch goal lives inside W2's `wiki/`).

## Consequences

- `build_index()` is designed today with a *list* of source directories (`SOURCE_DIRS = [DOCS_DIR]`), so adding `WIKI_DIR` later requires no signature change.
- `CONTEXT.md` reserves the vocabulary up front: **Wiki**, **Wiki Index**, **Hot Cache**, **Wiki Log**, **Source Template**, **Lint Pass**, **Ingest**. The naming follows `claude-obsidian` so a future contributor (or future me) does not invent a parallel vocabulary.
- The `docs/` folder name stays (PROMPT.md verification freezes it). The `claude-obsidian` convention is `.raw/`, but renaming is a cheap follow-up if it ever matters — see Section ID stability note in `CONTEXT.md`.
- Operational patterns we want to inherit from `claude-obsidian` and the community discussion (Two-output rule, L0/L1/L2/L3 token budget, frontmatter schema, etc.) but that are not vocabulary live in [`project-docs/inspiration.md#deferred-patterns`](../inspiration.md). Re-read that section before starting any post-prototype phase.
- The deferred-pattern register has explicit phase triggers (`phase: wiki`, `phase: ingest`, `phase: conversation`), so the patterns surface naturally when we start the phase, not by memory.
