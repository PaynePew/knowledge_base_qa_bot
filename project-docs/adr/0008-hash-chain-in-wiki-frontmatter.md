# Hash chain in wiki frontmatter

Wiki frontmatter gains an 8th field `source_hashes: dict[str, dict[str, str]]`, mapping each source filename to its `{raw, docs_body}` hash dict. The `raw` sub-key carries the `content_sha256` propagated from docs frontmatter when available (or `null` for hand-authored docs that never went through `/import`); the `docs_body` sub-key is always computed by `/ingest` from the docs file contents. Phase 7 (PRD #89) is the trigger — Phase 7-3 (#92) writes `content_sha256` of raw bytes into docs frontmatter, and this amendment closes the chain from docs → wiki. Phase 5 lint drift detection is the downstream consumer.

We chose this design because Phase 7's hash chain ends at the docs layer, leaving wiki-side drift detection blind. A wiki page synthesised from `docs/foo.md` at hash H1 has no recorded link back to H1 — if the underlying `docs/foo.md` later drifts to H2 without re-ingest, Phase 5 lint cannot tell. Propagating only the raw_hash (option β below) handles `/import`-managed sources but fails the hand-authored case, where `content_sha256` doesn't exist to propagate. Recording both `raw` (from docs frontmatter when available, null otherwise) and `docs_body` (always computed by Phase 3 ingest from docs file content) gives Phase 5 lint a usable hash for every wiki page regardless of source provenance.

## Considered Options

### What hash, hashed where, written where (Q1)

- **α — Hash only at the raw→docs boundary (Phase 7 alone)**: `/import` writes `content_sha256` to docs frontmatter, nothing propagates beyond. Rejected: leaves wiki-side drift detection blind. The hash field becomes a dangling artifact downstream — Phase 5 lint cannot detect docs→wiki drift, only raw→docs drift, halving the chain's coverage.
- **β — Propagate raw_hash only**: Phase 3 reads `content_sha256` from docs frontmatter and writes it into wiki under `source_hashes[filename].raw`. Rejected: fails on hand-authored docs that never went through `/import` (no `content_sha256` to read). Those wiki pages would have `raw: null` and no `docs_body` fallback, leaving Phase 5 lint with no usable hash for drift detection on the hand-authored subset.
- **γ (chosen) — Hash everywhere**: write both `raw` (propagated when docs frontmatter carries `content_sha256`, `null` otherwise) and `docs_body` (always computed by Phase 3 ingest from `source_path.read_text('utf-8').encode()`) into wiki `source_hashes`. Two-level coverage: `raw` detects raw-file drift (only when chain originated from `/import`); `docs_body` detects docs-file drift (always available). Phase 5 lint can fall back to `docs_body` when `raw` is `null`, so the hand-authored subset still has a working drift signal.

See PRD #89 § Implementation Decision 5 for the cross-phase chain reasoning.

### Hash function and encoding convention (Q2)

- **(i) SHA-256, byte-level on the docs file**: `sha256(source_path.read_bytes())`. Matches Phase 7's convention exactly. Rejected for `docs_body`: docs files are always UTF-8 text and the text-level hash makes more sense for the wiki layer's drift question (which is "did the docs body change semantically?", not "did any byte flip?"). Phase 7's byte-level hash is correct for raw bytes because raw inputs may be non-UTF-8 (`UnicodeDecodeError` is a real failure mode there).
- **(ii) SHA-256, text-level via `read_text('utf-8').encode()` (chosen)**: `sha256(source_path.read_text('utf-8').encode())`. Deterministic across platforms because both `read_text` and `.encode()` default to UTF-8. Wiki-layer convention is text-level; Phase 7-layer convention remains byte-level. The convention difference is intentional and documented in the field shape: `raw` is byte-hash, `docs_body` is text-hash. They are not directly comparable.

### Default for missing `source_hashes` (Q3)

- **(a) Default `dict()` with explicit `null` sub-keys**: e.g. `source_hashes: {foo.md: {raw: null, docs_body: null}}` for legacy pages. Rejected: forces Phase 5 to distinguish "explicitly-null hash" from "missing-source-entry", which is no clearer than treating both as unknown.
- **(b) Default `{}` empty dict (chosen)**: Phase 6 legacy wiki pages have `source_hashes: {}`. Phase 5 lint MUST treat empty as "drift state unknown", NOT "no drift". On first re-ingest after this amendment lands, the dict is populated for the first time; subsequent ingests can skip-on-match.

### Backward compatibility — retroactive re-ingest of Phase 6's 9 pages (Q4)

- **(α) Retroactive re-ingest required**: a migration step that re-ingests all 9 Phase 6 pages on first amendment-aware boot. Rejected: needless work — the empty-dict-as-unknown convention already gives Phase 5 lint a correct fallback for those pages.
- **(β) No retroactive action (chosen)**: Phase 6's 9 pages remain valid with empty `source_hashes`. On the next normal `/ingest` call that touches one of them, the dict is populated for the first time. Phase 5 lint treats empty as "unknown" until then.

## Consequences

**Invariant**: wiki frontmatter is an 8-field schema. The 8th field is `source_hashes: dict[str, dict[str, str]]` with literal sub-keys `raw` (str | None) and `docs_body` (str). Empty dict is the legacy / unknown drift state.

**Invariant**: Phase 5 lint MUST treat an empty `source_hashes` dict as "drift state unknown", NOT as "no drift" — otherwise Phase 6 legacy pages (with empty dict) would falsely report as clean during the first lint run after this amendment lands.

**Invariant**: `raw` is byte-level SHA-256 (Phase 7's hash, propagated unchanged), `docs_body` is text-level SHA-256 over `source_path.read_text('utf-8').encode()`. The two sub-keys use different hash conventions and are NOT comparable — Phase 5 lint must hash like Phase 3 ingest (text-level) when recomputing `docs_body` for comparison.

**Rolling forward**: Phase 6 落地 9 wiki pages do NOT require retroactive re-ingest. On first re-ingest after this amendment lands, `source_hashes` is populated for the first time; subsequent ingests can skip-on-match.

**Extensibility**: the dict-of-dict shape reserves room for future hash sub-keys (e.g. `wiki_body` hash for inverse drift detection) without further schema change. New sub-keys must specify their hash convention in this ADR's Invariant list when added.

**For `WikiPageFrontmatter` schema (`markdown_kb/app/schemas.py`)**: the 8-field shape is encoded as `source_hashes: dict[str, dict[str, str]] = Field(default_factory=dict)`. The default-factory empty dict is the legacy / unknown drift sentinel — do not change the default to `None` (Phase 5's "unknown" check uses `len() == 0` for cheaper lookup).

**For `/ingest` skip-on-match logic**: hash compare happens BEFORE the LLM call (perf-correctness, mirroring Phase 7-3's "hash compare before markdownify" decision). When `source_hashes[<source_filename>]["docs_body"]` matches the freshly computed `docs_body_hash` and `force=False`, skip without LLM invocation; emit `ingest_skipped` Wiki Log event; record `IngestSourceResult(status="skipped")` in `IngestResponse.skipped_sources`.

**Cross-reference**: PRD #89 § Implementation Decision 5 (γ chain reasoning); supersedes the 7-field schema invariant from Phase 3 PRD (#28) and Phase 6 PRD (#78) where they previously read "7-field frontmatter schema". CONTEXT.md `Wiki Page` term is updated in the same slice to read "8-field frontmatter schema" and document the new field's shape.
