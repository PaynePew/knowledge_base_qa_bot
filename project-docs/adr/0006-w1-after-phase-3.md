# W1 retrieval: wiki/ becomes the sole query target after Phase 3

With Phase 3 (`/ingest`) shipped and `wiki/entities/` + `wiki/concepts/` materialising curated synthesis pages, the project moves from W2 (layered query — `wiki/` checked first, `docs/` as fallback) to **W1 (Karpathy strict — `wiki/` is the sole runtime retrieval surface)**. `docs/` retains its role as the immutable Source layer that `/ingest` reads; it is no longer scanned by `build_index()` and is never cited by `/chat`. When `/chat` cannot find adequately scoring wiki content, it returns `Cannot Confirm` with the existing structured reasons (`retrieval_empty` for zero overlap, `below_threshold` for weak overlap, `index_missing` for an empty wiki), which Phase 5 `/lint` aggregates into a coverage backlog for the wiki owner.

This restores the Karpathy compounding loop: gaps surface as actionable signal rather than being silently masked by fallback to raw Sources. The W2 *layered storage* decision from ADR-0003 is preserved — what changes is only the *runtime query behaviour*. ADR-0003 is therefore superseded *in part*: its identification of `claude-obsidian` as the pattern reference and its rejection of W3/W4 still stand; only its choice of W2 over W1 is reversed in this ADR.

## Re-evaluation of ADR-0003's W1 rejection

ADR-0003 rejected W1 ("strict Karpathy — only `wiki/` queryable") for three reasons. Each is re-examined against Phase 3's shipped state:

| Original rejection | Status now |
|---|---|
| "Forces a full retrieval-path rewrite at the moment the wiki layer ships, with no easy fallback if the wiki proves disappointing." | Phase 4 *is* the retrieval rewrite — the cost is paid regardless. A disappointing wiki is fixed by re-running `/ingest` with better Source Templates, not by falling back to docs/. Fallback would mask wiki quality issues, not surface them. |
| "Citation surface harder to reconcile with the PROMPT.md `filename#heading` contract." | Wiki pages already carry a stable slug in their `frontmatter.id`, and the contract's literal shape (`<filename-stem>#<heading-slug>`) is preserved under W1 — the *referent* of `filename-stem` evolves from docs Source to wiki page. See § Citation surface (A2) below. |
| "No easy fallback if the wiki proves disappointing in practice." | The "easy fallback" was a euphemism for "we keep two retrieval paths and trust whichever wins per query." This defeats the compounding-artifact motivation that justifies the wiki layer in the first place. Wiki disappointment is signal; muting it is anti-pattern. |

## Considered Options

- **W1 — wiki/ only** *(chosen)*: `SOURCE_DIRS = [WIKI_DIR/"entities", WIKI_DIR/"concepts"]`. `docs/` is read only by `/ingest`. Cannot Confirm fires when wiki is empty or query falls below threshold; the existing reason taxonomy carries the gap signal.
- **W2 layered fallback** *(rejected — was ADR-0003's choice)*: wiki/ checked first, docs/ as fallback. Masks wiki gaps; defeats Karpathy compounding loop; produces inconsistent citation surface (some answers cite wiki, some cite docs).
- **Flat merge** (`SOURCE_DIRS = [DOCS_DIR, WIKI_DIR]`): a single BM25 corpus over both. Rejected: shared `doc_freq` between layers depresses IDF for terms that appear in both wiki page and source doc — the synthesis page (which should win) and its raw source actively suppress each other's BM25 score. Also incoherent with Karpathy framing (wiki is the LLM's compiled view, not a sibling of raw Sources).
- **W1 with wiki score boost on flat merge**: keep both layers in one BM25 corpus but multiply wiki scores by ~1.3×. Rejected: lower-cost approximation of W1 that still leaves `docs/` retrievable as a "soft fallback." Carries all W2's mask-the-gap problems with extra arithmetic.

## Citation surface (A2)

Citations under W1 take the form `[Source: <slug>#<section-anchor>]` — bare slug, no type subdirectory, no `.md` extension. Example: `[Source: refund-policy#cancellation-window]` rather than `[Source: entities/refund-policy.md#cancellation-window]`.

Rationale:

- **Consistency with Red Links**: wiki page bodies already use `[[<slug>]]` for cross-page references (per [`templates.py`](../../markdown_kb/app/templates.py)). Citation surface uses the same `slug` namespace — one addressing convention for the entire wiki, not two.
- **Decoupled from filesystem layout**: future Phase 5 `/lint` may reorganise subdirectories (e.g. split `entities/` into `entities/persons/` and `entities/companies/`); slug-based citation survives such moves. Path-based citation does not.
- **Lower LLM citation-production error rate**: a single-token slug is easier for the LLM to reproduce verbatim than a slash-bearing path, reducing grounding verifier false-rejections caused by citation typos.
- **Uses `frontmatter.id`'s designed purpose**: the `id` field on `WikiPageFrontmatter` (per [`schemas.py:113`](../../markdown_kb/app/schemas.py)) is already the canonical stable identifier of a wiki page. Citation surface aligns with that identifier rather than introducing a parallel path-based scheme.

### Cross-type slug uniqueness invariant

A2 requires globally-unique slugs across `wiki/entities/` and `wiki/concepts/` (otherwise `acme-shop` could refer to either an entity or a concept page ambiguously). Phase 3's current per-type `resolve_slug_collision` state ([`ingest.py`](../../markdown_kb/app/ingest.py)) maintains separate sets per type and must be consolidated into a single global set as part of Phase 4. This is a small, contained Phase 3 cleanup landed inside the Phase 4 PR.

## Obsidian alignment scope

Obsidian is **a pattern source and syntax convention, not a runtime dependency or planned tool integration.** The roadmap (Phases 3–12) contains no Obsidian-integration phase; no Obsidian-specific dependency exists in `pyproject.toml`; the codebase references Obsidian only twice (one wikilink-syntax comment in [`templates.py`](../../markdown_kb/app/templates.py), one anchor-convention note in [`CONTEXT.md`](../../CONTEXT.md)).

`wiki/` happens to be Obsidian-friendly as a side effect — slug convention, `[[wikilink]]` syntax, YAML frontmatter, type-segregated subdirectories all render natively if a wiki owner chooses to open `wiki/` in Obsidian. The bot itself remains unaware of Obsidian. The W1 citation format `[Source: slug#heading]` is **LLM-facing, not Obsidian-facing**: a future display layer (Phase 9 browser UI, Phase 12 CLI/MCP) can post-process `[Source: X#Y]` into `[[X#Y]]` for Obsidian rendering without changing the storage format.

## W1 reason semantics

Under W1, the three pre-LLM `grounding.reason` values are re-interpreted:

| Reason | W1 meaning |
|---|---|
| `index_missing` | `sections == []` — wiki has zero pages. Operational state, not a content gap. Phase 5 `/lint` ignores. Logged separately as `wiki_layer_empty` (new log kind) so ops can tell "system has been deployed but never ingested" apart from normal cannot-confirm. |
| `retrieval_empty` | BM25 returned zero hits — wiki has pages but the query has no token overlap with any. **Severe coverage gap signal** — wiki probably doesn't cover this topic area at all. Phase 5 `/lint` aggregates these for "create a new page" backlog items. |
| `below_threshold` | BM25 returned hits but top score is below `KB_SCORE_THRESHOLD`. **Mild coverage gap signal** — wiki has nearby content but not specific enough. Phase 5 `/lint` aggregates these for "extend an existing page" backlog items. |

No new `wiki_gap` reason is introduced. Phase 5 `/lint` aggregates `chat_fallback` log entries with `retrieval_empty` or `below_threshold` reasons; that pair *is* the gap signal under W1.

## L0/L1/L2/L3 read-depth alignment

Per the `claude-obsidian` + Karpathy gist pattern (re-recorded with `phase: query` tag in [`inspiration.md`](../inspiration.md)):

| Level | File | Token budget | Phase 4 coverage |
|---|---|---|---|
| L0 | `wiki/hot.md` (Hot Cache) | ~200 | **Out of scope** — Phase 10 |
| L1 | `wiki/index.md` (Wiki Index) | ~1–2K | Already produced by Phase 2 (navigation surface for humans + agents) — **not consumed by `/chat` retrieval** in Phase 4 |
| L2 | BM25 hits (Section granularity) | ~2–5K | **Yes** — top-K Sections via [`indexer.search`](../../markdown_kb/app/indexer.py) |
| L3 | Full Wiki Pages (all sections of a hit's parent page) | ~5–20K | **Yes (B3 expansion)** — Phase 4 expands hits to include sibling sections of their parent pages, so the LLM receives page-level coherent context, not isolated section snippets |

Phase 4 ships L2 + L3 (BM25 plus page-coherent expansion). L0 (Hot Cache) is Phase 10. L1 (Wiki Index as LLM-consumed input via two-step retrieval) is deferred until the wiki passes a scale threshold that justifies the extra LLM call — small demo corpus does not.

## Log enrichment for Phase 5 `/lint`

Phase 4's only direct contract with Phase 5 is `chat_fallback` and `chat_grounding_fallback` log data. The current single-line log entries do not carry enough signal for `/lint` to localise gaps to specific pages.

| Log kind | Current format | W1 format |
|---|---|---|
| `chat_fallback` (`below_threshold` case) | `"<q>" reason=below_threshold top_score=0.3` | `"<q>" reason=below_threshold top_score=0.3 top_section=refund-timeline#refund-timeline` |
| `chat_fallback` (`retrieval_empty` case) | `"<q>" reason=retrieval_empty top_score=0.0` | unchanged (no top section to record) |
| `chat_grounding_fallback` | `"<q>" reason=claim_unsupported` | `"<q>" reason=claim_unsupported cited=acme-shop#shipping,acme-shop#overview` |
| `wiki_layer_empty` (new kind) | n/a | Emitted by `build_index()` when both `wiki/entities/` and `wiki/concepts/` resolve to zero sections. Output of `/chat` is unchanged (`index_missing` reason) — this kind only differs in log surface, to give `/lint` a distinct ops signal |

Phase 5 `/lint` clusters these entries to surface coverage gaps localised to specific wiki pages.

## PROMPT.md citation contract evolution

PROMPT.md's `{filename}#{heading-slug}` citation contract is **preserved in W1, with filename re-interpreted from the docs Source stem to the wiki page slug.** The contract's intent — "every claim cites a real file and a real heading" — holds. The contract's referent — which file — evolves with the architecture. The `.md` extension is omitted in W1 citations for prompt brevity. A one-line footnote in PROMPT.md points readers to this ADR for the W1 evolution; PROMPT.md's body remains unchanged so the prototype's verification narrative stays intact.

The companion artefact (wiki page → docs Source back-reference) already exists in `frontmatter.sources` for each wiki page. Exposing it on the `/chat` response (so a caller can chain wiki citation → docs citation for full audit) is deferred to **Phase 6 (Answer Filing)**: that phase introduces `wiki/qa/` pages whose internal references are the natural trigger for surfacing the full citation chain.

## Consequences

- `build_index()` now scans `[WIKI_DIR / "entities", WIKI_DIR / "concepts"]` (whitelist; default-deny on new top-level wiki files). Phase 6 will append `wiki/qa/` to this whitelist.
- `wiki/index.md`, `wiki/log.md`, `wiki/hot.md`, `wiki/README.md`, `wiki/.archive/*` are **excluded** from BM25 corpus as a *correctness invariant*. Including any of these creates self-referential pollution: `wiki/index.md` cites every wiki Section, so indexing it makes it a high-BM25-score artificial Section for any query; `wiki/log.md` updates on every `/chat`, so indexing it makes the corpus non-idempotent. See [§ Q4 grill record](.) for the full walkthrough.
- `Section.id` and `Section.file` for wiki-derived Sections use the bare slug (no type subdirectory, no `.md` extension). Cross-type slug uniqueness is enforced in `ingest.py` via consolidated `used_slugs` state.
- `retrieval.query()` adopts B3 page expansion: BM25 hits at Section granularity, parent pages of hits collected, all sibling sections of those pages included in the prompt CONTEXT. Citation in the LLM draft remains Section-granular.
- `NOT_INDEXED_MESSAGE` (the `index_missing` user-facing string) is updated to advise calling `/ingest` first, since under W1 an empty wiki is more commonly the cause than a missing `/index` call.
- Demo bootstrap: Phase 4 PR commits an initial `/ingest` run's output for the three canonical `docs/` Sources to `wiki/entities/` and `wiki/concepts/`. Tests do not depend on this committed seed — they use deterministic fixtures under `tests/fixtures/wiki/`. The `fake-docs/` ingest test fixture is not included in the seed.
- ADR-0003's W1-rejection clause is superseded by this ADR. ADR-0003 itself is retained (its W3/W4 rejections and `claude-obsidian` reference still apply) with an in-line supersession header pointing here.
- Phase 4 ships as five sequential slices: (4-1) ADR-0006 + CONTEXT.md + PROMPT.md footnote (docs-only); (4-2) `SOURCE_DIRS` whitelist + `wiki_layer_empty` log kind; (4-3) A2 citation format + cross-type slug uniqueness; (4-4) B3 page expansion in `retrieval.query`; (4-5) log enrichment + committed seed wiki content. Slice 4-1 lands first so subsequent slices implement against an already-aligned vocabulary.
