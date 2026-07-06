# Wiki Log — `kind=` enumeration

Single source of truth for every `kind` value emitted via `log_event(kind, summary)`. Adding a new kind to the codebase MUST add a row here in the same commit; the reviewer fails any PR that introduces an undocumented kind. Removing a kind requires a deprecation note (see § Deprecation).

Format on disk (per [`CODING_STANDARD.md`](CODING_STANDARD.md) § 5.2):

```
## [<ISO-8601 UTC>] <kind> | <summary>
```

`kind` is `snake_case`. `summary` is grep-friendly `key=value` pairs (queries double-quoted, never multiline).

---

## `/chat` route

Authorized by [`prd.md`](prd.md) (Phase 1).

| Kind | When fired | Summary template |
|---|---|---|
| `chat` | Successful `/chat` response written | `"<truncated query>" sources=N grounded_reason=X` |
| `chat_fallback` | Pre-LLM Cannot Confirm gate fires | `"<truncated query>" reason=<not_indexed\|below_threshold> [top_score=X] [top_section=<id>]` |
| `chat_grounding_fallback` | Post-LLM verifier returned not-passed; reply replaced with Cannot Confirm | `"<truncated query>" reason=<outcome.reason> cited=<comma_separated_section_ids>` |
| `chat_error` | OpenAI exception during `/chat`; mapped per [`CODING_STANDARD.md`](CODING_STANDARD.md) § 4.2 | `"<truncated query>" kind=<openai_transient\|openai_auth\|openai_api> exc=<ExcClass>` |

### `chat_error` `kind=` sub-tags

| Sub-kind | Exception class | HTTP | Authorized by |
|---|---|---|---|
| `openai_transient` | `APITimeoutError`, `RateLimitError` | 503 | [`prd.md`](prd.md) (Phase 1 error mapping) |
| `openai_auth` | `AuthenticationError` | 500 | [`prd.md`](prd.md) (Phase 1 error mapping) |
| `openai_api` | Other `APIError` subclasses | 500 | [`prd.md`](prd.md) (Phase 1 error mapping) |

---

## Grounding Check

Authorized by [ADR-0004](adr/0004-post-llm-grounding-check.md).

| Kind | When fired | Summary template |
|---|---|---|
| `grounding_verify` | Verifier LLM call returned a parseable `GroundingResult` | `reason=<claim_supported\|claim_unsupported\|verifier_unavailable> retries=N latency=Xs` |
| `grounding_verifier_error` | Verifier raised an exception (transient retried, refusal / auth not retried) | `error_type=<ErrType> retries=N exc=<ExcClass> [HARD_ERROR]` or `reason=verifier_unavailable error_type=<ErrType> retries=N latency=Xs exc=<ExcClass>` |

`error_type` values come from `grounding.VerifierErrorType`: `timeout`, `server_error`, `malformed_json`, `refusal`, `auth`.

---

## Indexer + Wiki Index

Authorized by [`prd.md`](prd.md) (Phase 1, indexer), [ADR-0003](adr/0003-w2-layered-wiki-target-claude-obsidian.md) (Phase 2, wiki_index projection), and [ADR-0006](adr/0006-w1-after-phase-3.md) (Phase 4, wiki whitelist + wiki_layer_empty signal).

| Kind | When fired | Summary template |
|---|---|---|
| `parse_warning` | Frontmatter unavailable / unparseable, a non-leaf heading has no body content, or a pre-heading preamble was captured as its own Section (ADR-0033 decision 1, issue #509) | `frontmatter present in <file> but PyYAML is not installed` or `frontmatter parse failed in <file>: <ExcType>` or `non-leaf heading with no body in <file>: '<heading>'` or `preamble captured as Section in <file>: '<section_id>'` |
| `index_loaded` | Section Index rehydrated from `.kb/index.json` on app startup | `files=N sections=M` |
| `index_built` | `build_index()` completed successfully | `files=N sections=M` |
| `wiki_layer_empty` | `build_index()` ran on the default `SOURCE_DIRS` and both `wiki/entities/` and `wiki/concepts/` resolved to zero sections (ADR-0006). Distinct ops signal from `index_missing` so Phase 5 `/lint` can tell "system deployed but never ingested" apart from a normal cannot-confirm. `/chat` output is unchanged. | `entities=0 concepts=0` |
| `wiki_index_error` | `write_wiki_index()` failed during `build_index()`; non-blocking (index still serves) | `reason=<ExcType>: <message>` |

---

## `/ingest` route

Authorized by GitHub issue #28 (Phase 3 PRD), § Q9.

| Kind | When fired | Summary template |
|---|---|---|
| `ingest_batch_started` | Start of an `ingest_sources()` call | `sources=N` |
| `ingest_batch_completed` | End of `ingest_sources()`; emitted even when some Sources failed | `sources=N total_pages=M llm_calls=K cost_usd=X failed_grounding=F` |
| `ingest_source` | One Source completed successfully (skipped on failure) | `source=<file> type=<entity\|concept> pages_created=A pages_updated=B pages_deleted=C sections_count=N uncarried_chars=U enriched_chars=E` (issue #511, ADR-0033 observability decision — `enriched_chars` is always 0 until issue #512 ships) |
| `ingest_error` | One Source failed at any stage (continue-on-error semantics) | `source=<file> error=<reason>` |
| `ingest_grounding_failed` | A draft page failed the grounding verifier; page still written with `status: failed_grounding` (ADR-0004 fail-soft) | `page=<slug> reason=<outcome.reason> claims=[<list>]` |
| `ingest_skipped` | _(Phase 3 amendment #93)_ Hash-match no-op: existing wiki page `source_hashes[source_name]["docs_body"]` matched the freshly computed docs_body_hash and `force=False`; no LLM call made | `source=<file> slugs_checked=[<list>] docs_body_hash=<hex>` |

### `ingest_error` `error=` sub-tags

| Sub-error | When | Source |
|---|---|---|
| `source_not_found` | File missing under `docs/` | path resolution |
| `<ExcType>:parse_error` | `parse_markdown()` raised | indexer |
| `no_sections` | Source parsed to zero Sections | indexer (after parse) |
| `<ExcType>:classify_failed` | Classifier LLM raised | templates |
| `<ExcType>:generate_failed` | Generator LLM raised | templates |
| `write_error:<slug> detail=<msg>` | `wiki_writer.write_pages_for_source()` returned an error | wiki_writer |

---

## `/import` route

Authorized by GitHub issue #89 (Phase 7 PRD). Slice 7-1 (#90) introduces the three end-of-batch / per-source kinds below; slice 7-2 (#91) adds `import_error`, slice 7-3 (#92) adds `import_skipped`. PRD #414 / ADR-0031 (issues #415, #416) add `.pdf` as a fourth `original_format` and three PDF-specific `import_error` sub-tags.

| Kind | When fired | Summary template |
|---|---|---|
| `import_batch_started` | Start of an `import_sources()` call | `mode=<batch\|single> source=<filename\|*>` |
| `import_source` | One raw file successfully imported to `docs/<basename>.md` | `source=<basename> docs=<docs_filename> format=<html\|txt\|md\|pdf>` |
| `import_batch_completed` | End of `import_sources()`; emitted even when some sources failed | `imported=A skipped=B failed=C duration_ms=N` |
| `import_error` | _(Slice 7-2)_ One raw file failed at any stage (continue-on-error semantics) | `raw=<raw_path> error_type=<one of the typed errors below> error_message=<truncated≤200>` |
| `import_skipped` | Re-import no-op when `content_sha256` matches existing docs frontmatter | `raw=<raw_path> docs=<docs_path> content_sha256=<hex>` |

### `import_error` `error_type=` sub-tags (slice 7-2, plus PDF-specific additions)

Full enumeration landed with slice 7-2 (#91); PDF-specific modes added by issue #415 (`NoTextLayer`, `PdfExtractionError`) and issue #416 (`EncryptedPdf`). All typed failure modes are emitted via `_emit_import_error` in `importer.py`.

| `error_type` | Trigger |
|---|---|
| `HandAuthoredCollision` | `docs/<filename>.md` exists without `imported_from` frontmatter (hand-authored) — refuse to overwrite |
| `UnicodeDecodeError` | raw file not UTF-8 (Big5, Shift_JIS, etc.) |
| `EmptySource` | raw file size 0 bytes |
| `OversizedSource` | raw file > 10 MB hard limit (protects markdownify from in-memory OOM) |
| `UnsupportedExtension` | single-mode source has unsupported extension (`.docx`, etc.); batch mode silently skips |
| `FileNotFoundError` | single-mode source doesn't exist in `raw/` |
| `MarkdownifyError` | markdownify internal exception (rare; BS4 is highly tolerant) |
| `IOError` | atomic-write `os.replace` failure (disk full, permission) |
| `InvalidFilename` | basename contains a rejected character class (`#`, `/`, `\`, `:`, control chars, bidi control chars U+202A-E/U+2066-9 per CVE-2021-42574) |
| `InvalidSourcePath` | single-mode `source` format violation (absolute path, `..` traversal, `raw/` prefix) |
| `FilenameCollision` | two raw files in the same batch produce the same docs basename; first wins, second fails |
| `NoTextLayer` | _(#415)_ PDF extraction yielded an empty/whitespace body (scanned/image-only PDF); curator must OCR externally and re-import |
| `PdfExtractionError` | _(#415)_ MarkItDown internal exception (corrupt/truncated PDF), distinct from an encryption failure |
| `EncryptedPdf` | _(#416)_ password-protected PDF (pdfminer's `PDFEncryptionError`/`PDFPasswordIncorrect` on open); curator must supply a decrypted copy and re-import |

---

## `/transcribe` route + probe-routed Transcribe successes (issue #426, ADR-0032)

Authorized by GitHub issue #426 and [ADR-0032](adr/0032-transcribe-model-assisted-pdf-conversion.md). Mirrors the `import_*` family in shape (`transcriber.py` owns these emitters): `transcribe_batch_started`/`transcribe_batch_completed` bracket one call to `transcribe_source` (single-source only — Transcribe has no batch mode), and `transcribe_source`/`transcribe_skipped`/`transcribe_error` mirror `import_source`/`import_skipped`/`import_error` per-file. A text-less PDF auto-routed to Transcribe from `POST /import` / `kb import` / batch `import_sources` also emits `transcribe_source` (not `import_source`) for its per-file success line, so the log distinguishes model-derived conversions from mechanical ones regardless of entry point — `import_batch_started`/`import_batch_completed` still bracket that call unchanged (one batch, mixed mechanical + transcribed successes).

| Kind | When fired | Summary template |
|---|---|---|
| `transcribe_batch_started` | Start of a `transcribe_source()` call (force entry: `POST /transcribe`, `kb transcribe`) | `mode=single source=<filename>` |
| `transcribe_source` | One PDF successfully transcribed to `docs/<basename>.md` — fired both by the force entry and by probe-routed auto-transcription inside `import_sources`/`import_path` | `source=<basename> docs=<docs_filename> model=<transcribe_model> status=<created\|updated>` |
| `transcribe_batch_completed` | End of a `transcribe_source()` call; emitted even on failure | `transcribed=<0\|1> skipped=<0\|1> failed=<0\|1> duration_ms=N` |
| `transcribe_skipped` | Re-transcribe no-op when `content_sha256` matches existing docs frontmatter | `raw=<raw_path> docs=<docs_path> content_sha256=<hex>` |
| `transcribe_error` | The force entry failed at any stage (validation, availability, page limit, model failure) | `raw=<raw_path> error_type=<one of TranscribeUnavailable\|TranscribePageLimitExceeded\|TranscribeError\|the reused Import validation types> error_message=<truncated≤200>` |

Auto-routed failures (a text-less PDF that fails `TranscribePageLimitExceeded` / `TranscribeError` while `import_sources`/`import_path` is processing it) are logged as `import_error` with that `error_type`, not `transcribe_error` — they are still failures of the `/import` batch/call, not of a `transcribe_source()` call. The unavailable case (Transcribe not configured) stays `NoTextLayer` under `import_error`, per the updated message contract (ADR-0032).

---

## Structure Enrichment (ADR-0033 decision 2, issue #512)

Authorized by GitHub issue #512 and [ADR-0033](adr/0033-longform-structure-enrichment-hub-page.md). Emitted from `markdown_kb/app/structure_enrichment.py::enrich_structure`, called from BOTH `importer._process_one_source` (the mechanical + auto-routed-Transcribe path) and `transcriber._force_transcribe` (the forced `/transcribe` entry) — same two kinds regardless of which caller reached it. Only fires when the longform predicate (`is_longform`) gates enrichment in; a well-headed Source bypasses byte-identically with no log entry at all.

| Kind | When fired | Summary template |
|---|---|---|
| `structure_enrichment_applied` | Enrichment succeeded: page furniture stripped and LLM-proposed chapter headings materialized into the body; `docs/` frontmatter gains `structure: enriched` | `source=<basename> chapters=N furniture_lines_removed=M` |
| `structure_enrichment_failed` | Enrichment was gated in (longform) but failed — LLM error, malformed proposal, or an unfindable boundary anchor; the caller falls back to the un-enriched body (`structure: enriched` is NOT written) | `source=<basename> reason=<truncated≤200 repr>` |

---

## `/upload` route (Phase 15 Operator Console)

Authorized by GitHub issue #168 (Phase 15 PRD) + [ADR-0011](adr/0011-upload-separate-from-import.md). Slice S1 (#169) introduces all five kinds below. Upload is a system boundary: it stages dropped browser bytes onto the server (`.html`/`.txt` → `raw/`, `.md` → `docs/`) and never converts (Import stays unchanged). `filename` and `reason` fields are rendered via `repr()` so embedded quotes/spaces stay unambiguous in the grep-able log line.

| Kind | When fired | Summary template |
|---|---|---|
| `upload_batch_started` | Start of an `upload_files()` call | `files=N` |
| `upload_file` | One file staged successfully to `raw/` or `docs/` | `filename=<repr> target=<repr target dir>` |
| `upload_rejected` | One file failed validation (type allow-list, size limit, traversal-safe filename) | `filename=<repr> reason=<repr>` |
| `upload_error` | One file failed at the atomic-write stage (OS-level, e.g. disk full / permission) | `filename=<repr> reason=<repr ≤200>` |
| `upload_batch_completed` | End of `upload_files()`; emitted even when some files were rejected / errored | `written=A rejected=B errors=C duration_ms=N` |

---

## Phase 6 Answer Filing

Authorized by GitHub issue #78 (Phase 6 PRD) §"Reflect step" (Q5) and
§"Orphan-visibility three-layer defence" (Q8d). Slice 6-1 (#79) documents the
three kinds here; the emitters arrive across slices 6-1 / 6-2 / 6-3.

| Kind | When fired | Summary template |
|---|---|---|
| `qa_invalid_status` | `indexer._passes_index_filter` skipped a `wiki/qa/*.md` whose `frontmatter.status` was set but not in `{"draft", "live"}` — the indexer-layer member of the three-layer orphan-visibility defence (curator-typo `status: Live`, forward-compat values like `stale` / `superseded`, missing `status` key, etc.). Slice 6-1. | `file=<filename> status=<repr-of-value>` |
| `qa_reflect` | A `wiki/qa/*.md` page was mutated by Phase 6 — emitted atomically inside the same `_filing_lock` critical section as the file write. Five `op` variants share this kind so reflective-log readers can grep `qa_reflect` once and slice by `op=`. Slices 6-2 (created/touched), 6-3 (promoted), tier-B S3 (edited), and tier-B S4 (refiled). | `slug=<slug> op=created question="<truncated>" cited=<comma_separated_citations> count=1` _(op=created — first filing for this question)_ <br>or `slug=<slug> op=touched cited_delta=added:<list>,dropped:<list> count=<N>` (or `cited_delta=none`) _(op=touched — re-ask of an existing question; body preserved, count incremented; cited_delta is the drift signal core)_ <br>or `slug=<slug> op=promoted by=curator` _(op=promoted — `POST /qa/{slug}/promote` flipped `status: draft -> live`)_ <br>or `slug=<slug> op=edited count=<N>` _(op=edited — `PUT /qa/{slug}` rewrote question/body after a passing grounding re-check; `status` stays `draft`)_ <br>or `slug=<slug> op=refiled count=<N>` _(op=refiled — `POST /qa/{slug}/refile` overwrote the page with a freshly re-synthesized, passing answer; `status` becomes `draft`)_ |
| `qa_filing_error` | `qa.maybe_file_answer` (Slice 6-2) hit an unrecoverable error during the filing critical section — IOError (disk full, permission denied), an attempted touch against an invalid-status orphan page (defence layer 2 of the three-layer defence), or a frontmatter-read failure on the touch path. F3 fail-soft: `/chat` still returns the answer and `response.filed = None`. Slice 6-2. | `slug=<slug> reason=<io_error\|orphan_status\|frontmatter_read_error> exc=<ExcClass>: <message>` |
| `qa_edit_rejected` | `PUT /qa/{slug}` (tier-B S3, issue #379, ADR-0026) refused because the LLM-free grounding re-check failed for the submitted body (HTTP 422 with the failure list). Nothing is written. | `slug=<slug> failures=<count>` |
| `qa_refile_rejected` | `POST /qa/{slug}/refile` (tier-B S4, issue #380, ADR-0026 decision 1) refused because the fresh re-synthesis failed the (LLM-based) Grounding Check (HTTP 422). Nothing is written — the old live page keeps serving and the C9 finding stays. | `slug=<slug> reason=<GroundingOutcome.reason>` |

### `qa_reflect` `op=` sub-tags

| Sub-op | When | Authorized by |
|---|---|---|
| `created` | First filing of a question (slug did not exist on disk) | PRD #78 §"Reflect step" (Q5) |
| `touched` | Re-ask of an existing question (B2 touch semantics — body preserved, `count` and `updated` bumped) | PRD #78 §"Reflect step" (Q5) |
| `promoted` | `POST /qa/{slug}/promote` flipped `status: draft -> live` | PRD #78 §"Two-stage curation lifecycle" (Q1) |
| `edited` | `PUT /qa/{slug}` rewrote question/body on a passing grounding re-check (draft-only) | issue #379 / [ADR-0026](adr/0026-curation-gate-refile-edit-human-surface-only.md) |
| `refiled` | `POST /qa/{slug}/refile` overwrote the page in place with a freshly re-synthesized, passing answer (`status` becomes `draft`) | issue #380 / [ADR-0026](adr/0026-curation-gate-refile-edit-human-surface-only.md) |

### `qa_deleted` kind

| Kind | When fired | Summary template |
|---|---|---|
| `qa_deleted` | `DELETE /qa/{slug}` successfully removed an inert `wiki/qa/<slug>.md` page. Only fires when the delete succeeds — `QaPageLive` refusals and `QaPageNotFound` errors are surfaced via HTTP status and do not emit a log entry. Authorized by [ADR-0012](adr/0012-delete-inert-filed-answers-only.md) (Phase 15 Slice 6, issue #174). | `slug=<slug> prev_status=<draft\|<unparseable>\|<other-invalid-status>>` |

### `qa_filing_error` `reason=` sub-tags

| Sub-reason | When | Source |
|---|---|---|
| `io_error` | OS-level write failure (disk full, permission denied, ENOSPC, etc.) | filesystem write in `qa.maybe_file_answer` |
| `orphan_status` | Touch attempted against a page whose existing `status` is not in `{"draft", "live"}` — the filing-layer defence refuses to bump `count` on an orphan zombie | `qa.maybe_file_answer` touch path |
| `frontmatter_read_error` | Existing `wiki/qa/<slug>.md` could not be parsed for the touch decision (corrupt YAML, missing fences, etc.) | `qa.maybe_file_answer` touch path |

---

## `/lint` route

Authorized by GitHub issue #65 (Phase 5 PRD), issue #66 (Slice 5-1), and issue #446 (C5 content-hash verdict cache).

| Kind | When fired | Summary template |
|---|---|---|
| `lint_started` | `run_lint()` enters | _(no payload beyond timestamp)_ |
| `lint_completed` | `run_lint()` exits (including when some checks failed) | `findings=N by_check=c11:A,c3:B,c4a:C,c6:D,c2:E,c1:F,c5:G llm_calls=K cost_usd=X c5_cache_hits=H errors=M` — `llm_calls` counts only actual C5 cache MISSES (issue #446); `c5_cache_hits` is the judged pairs reused from `.kb/c5_verdict_cache.json` with zero LLM calls, so `cost_usd` reflects real spend, not the pre-cache judged-pair count. |
| `lint_check_error` | An individual check raises (continue-on-error; other checks still run); also fired (non-fatally) when the C5 verdict cache write fails | `check=<name> exc=<ExcClass>: <msg>` — `<name>` is `c5_cache` for a cache write failure (issue #446), distinct from `c5` (the LLM judging check itself) |

No per-finding log entries — findings live in `wiki/lint-report.md` as the source of truth (deliberate noise reduction, per PRD #65 Implementation Decision §Wiki Log entries).

---

## `/pages/reconcile` route (tier-B S1)

Authorized by GitHub issue #376 (tier-B S1) and [ADR-0028](adr/0028-reconcile-stateless-two-phase-apply-revalidation.md).

| Kind | When fired | Summary template |
|---|---|---|
| `reconcile_generate` | `reconcile.generate_reconcile()` returns a draft (writes nothing to disk) | `page_a=<slug> page_b=<slug> passed=<bool> reason=<GroundingInfo.reason>` |
| `reconcile_apply_refused` | `reconcile.apply_reconcile()` refused because the apply-time grounding re-check failed for either page's submitted content (HTTP 422) | `page_a=<slug> page_b=<slug> reason=<GroundingInfo.reason>` |
| `reconcile_applied` | `reconcile.apply_reconcile()` successfully rewrote both pages | `page_a=<slug> page_b=<slug>` |

No log entry is emitted for the `ReconcileHashMismatch` (409), `PageNotFound` (404) / `ReconcileInvalidPair` (400), or `PageCorrupt` (500) refusal paths — those are request-shape / staleness / data-integrity rejections surfaced directly via HTTP status, not domain events (mirrors `qa_deleted`'s "only fires on success" convention).

---

## `/pages/{slug}` route (tier-B S5)

Authorized by GitHub issue #381 (tier-B S5) and [ADR-0025](adr/0025-delete-live-orphans-full-orphan-predicate.md).

| Kind | When fired | Summary template |
|---|---|---|
| `orphan_page_deleted` | `pages.delete_full_orphan()` successfully hard-deleted an entities/concepts page after re-verifying the ADR-0025 full-orphan predicate at delete time. Only fires when the delete succeeds — `PageNotFound` (404), `PageCorrupt` (500), and `PageNotFullOrphan` (409, the predicate no longer holding) refusals are surfaced via HTTP status and do not emit a log entry (mirrors `qa_deleted`'s "only fires on success" convention). | `slug=<slug>` |

---

## `/pages/{slug}/aliases` route (issue #409)

Authorized by GitHub issue #409 and [ADR-0030](adr/0030-alias-frontmatter-link-layer-resolver-linkify.md) decision 3.

| Kind | When fired | Summary template |
|---|---|---|
| `alias_assigned` | `pages.add_alias()` successfully assigned a new alias to an entities/concepts page's frontmatter. Only fires on an actual write — the idempotent no-op path (alias already assigned to this page) and every refusal (`PageNotFound` 404, `PageCorrupt` 500, `InvalidAlias` 422, `AliasCollision` 409) do not emit a log entry (mirrors `orphan_page_deleted`'s "only fires on success" convention). | `slug=<slug> alias=<alias>` |

---

## Vector RAG (Stack B)

Authorized by GitHub issue #103 (Phase 8 Slice 3). Stack B stays decoupled from
`markdown_kb`, so it owns its own append-only log channel at `vector_rag/log.md`
(written via `vector_rag/app/logger.py::log_event`) — the same `## [<ISO-8601 UTC>] <kind> | <summary>`
format and the same single-channel discipline (CODING_STANDARD §5.1), just a
separate file. The kinds reuse markdown_kb's names so log readers parse both
channels identically; they are listed here (not duplicated under the sections
above) because they are emitted by a different module against a different file.

| Kind | When fired | Summary template |
|---|---|---|
| `index_built` | `vector_rag.indexer.build_index()` completed (including the empty-corpus no-op) | `files=N chunks=M` |
| `index_loaded` | Persisted FAISS index rehydrated from `.kb/faiss_index/` on app startup | `files=N chunks=M` |
| `chat` | Successful `/chat` response written | `"<truncated query>" top=<chunk source> count=N` |
| `chat_fallback` | Pre-LLM gate fires (index missing, vector search empty, or closest chunk distance over the ceiling) | `"<truncated query>" reason=<not_indexed\|retrieval_empty\|below_threshold> [rag_distance=X]` |
| `chat_grounding_fallback` | Post-LLM verifier returned not-passed; reply replaced with Cannot Confirm | `"<truncated query>" reason=<outcome.reason> cited=<comma_separated_chunk_sources>` |
| `chat_error` | OpenAI exception during `/chat` or the `/index` embedding call; mapped per [`CODING_STANDARD.md`](CODING_STANDARD.md) § 4.2 | `"<truncated query>" kind=<openai_transient\|openai_auth\|openai_api> exc=<ExcClass>` (chat) or `op=index kind=<…> exc=<ExcClass>` (index embedding) |

The verifier-side kinds (`grounding_verify`, `grounding_verifier_error`) are
emitted by `markdown_kb`'s `grounding.py` — which Stack B adopts unchanged — so
they write to `markdown_kb`'s `wiki/log.md`, not `vector_rag/log.md`. They are
already enumerated under the Grounding Check section above and not re-listed here.

---

## Hybrid Retrieval (Stack C)

Authorized by GitHub issue #311 (Phase 13 Slice S1) and [ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md). Stack C is additive and stays decoupled from `markdown_kb` (Stack A) and `vector_rag` (Stack B), so it owns its own append-only log channel at `hybrid_kb/log.md` (written via `hybrid_kb/app/logger.py::log_event`) — the same `## [<ISO-8601 UTC>] <kind> | <summary>` format and the same single-channel discipline (CODING_STANDARD §5.1), just a separate file. Slice S1 introduces the two dense-index lifecycle kinds below; the kinds reuse the existing `index_*` naming convention with a `dense_` prefix so a log reader can tell the dense-over-wiki seed apart from the BM25 (`index_built`) and docs FAISS (`index_built`) channels.

| Kind | When fired | Summary template |
|---|---|---|
| `dense_index_built` | `hybrid_kb.dense_index.build_index()` completed (including the empty-corpus no-op) | `sections=N` |
| `dense_index_loaded` | Persisted dense-over-wiki seed rehydrated from `.kb/hybrid_dense/` on startup | `sections=N` |

Slice S3 (GitHub issue #313 / [ADR-0018](adr/0018-hybrid-retrieval-third-stack-rrf-over-wiki.md)) adds the `hybrid_kb.query()` answer-synthesis surface, which reuses the Wiki/RAG `chat*` kinds verbatim on Stack C's own channel (same templates as the `/chat` route, just a `hybrid_kb/log.md` file). Page expansion, prompt building, and `grounding.verify` are imported from `markdown_kb`; only the answer LLM call and these log lines are owned here.

| Kind | When fired | Summary template |
|---|---|---|
| `chat` | Successful `hybrid_kb.query()` response written | `"<truncated query>" top=<top section id> count=N` |
| `chat_fallback` | Pre-LLM Cannot Confirm fires — the per-arm OR-gate refused before any LLM call | `"<truncated query>" reason=<below_threshold>` |
| `chat_grounding_fallback` | Post-LLM verifier returned not-passed (or the model self-refused); reply replaced with Cannot Confirm | `"<truncated query>" reason=<outcome.reason> cited=<comma_separated_section_ids>` |
| `chat_error` | OpenAI exception during synthesis; mapped per [`CODING_STANDARD.md`](CODING_STANDARD.md) § 4.2 | `"<truncated query>" kind=<openai_transient\|openai_auth\|openai_api> exc=<ExcClass>` |

#### `chat_error` `kind=` sub-tags

| Sub-kind | Exception class | LLMError.retryable | Authorized by |
|---|---|---|---|
| `openai_transient` | `APITimeoutError`, `RateLimitError` | `True` | issue #313 (Stack C synthesis, mirrors the Phase 1 mapping) |
| `openai_auth` | `AuthenticationError` | `False` | issue #313 |
| `openai_api` | Other `APIError` subclasses | `False` | issue #313 |

---

## Gateway

Authorized by GitHub issue #158 (Phase 11 PRD — Conversation Memory) and
GitHub issue #161 (Phase 11 Slice 3 — Rewrite observability). The Gateway owns
its own append-only log channel at `gateway/log.md` (written via
`gateway/app/logger.py::log_event`) — the same `## [<ISO-8601 UTC>] <kind> | <summary>`
format and the same single-channel discipline (CODING_STANDARD §5.1), just a
separate file from `wiki/log.md` and `vector_rag/log.md`.

The `chat_rewrite` kind is gateway-specific (query rewriting is a gateway
operation, stack-agnostic). It is emitted **only when a rewrite actually
happened** (turn 2+, history non-empty); a turn-1 passthrough writes no entry.
Phase 5 `/lint` reads `wiki/log.md` only (kinds `chat_fallback` /
`chat_grounding_fallback`) and is therefore unaffected by this channel.

The production overload + cost-protection kinds (`budget_block`,
`overload_shed`, `provider_quota_503`) are gateway-specific, authorized by
GitHub issue #269 (deploy S1 — Gateway production middleware). They are emitted
by `gateway/app/middleware.py::ProdMiddleware` when a heavy request is rejected
by one of the three demo guards (daily USD budget, concurrency cap, provider
quota), so the `gateway/log.md` carries an operator-facing audit of every shed.

| Kind | When fired | Summary template |
|---|---|---|
| `chat_rewrite` | Turn 2+ query rewriting succeeded inside `_sse_generator`; emitted right after `rewrite_query()` returns | `session=<uuid> raw="<60-char-bounded raw follow-up>" rewritten="<60-char-bounded self-contained query>"` |
| `budget_block` | A heavy request is rejected because the UTC-day cost estimate has reached `KB_DAILY_USD_CAP` | `path=<mounted-path> cap=<usd>` |
| `overload_shed` | A heavy request is rejected because its semaphore (read or admin) is fully held | `path=<mounted-path> kind=<read\|admin>` |
| `provider_quota_503` | A non-streaming heavy request raised an OpenAI `insufficient_quota` / 429, mapped to a friendly 503 | `path=<mounted-path> exc=<ExceptionClassName>` |

`raw` and `rewritten` are truncated to 60 chars and have inner `"` replaced
with `'` (per CODING_STANDARD §5.3 bounded-summary idiom).

---

## Startup warmup (issue #439, failure-timing symmetry issue #457)

Authorized by GitHub issue #439 (gateway: warm hybrid dense index + OpenAI
clients at startup). Fixes the post-deploy `/chat` cold start: the Gateway
`lifespan` (`gateway/app/main.py`) loads the Hybrid (Stack C) dense index at
boot (`gateway/app/warmup.py::warm_hybrid_indexes`, token-free — a pure disk
load), and, only when `KB_WARMUP_PING` is truthy, fires one tiny ping per
distinct OpenAI client (`warm_openai_clients`) so connection-priming cost
lands at boot instead of on a user's first question.

The `startup_warmup` kind is emitted from **whichever package owns the target**
— it is one instrumentation concept written into four different channels, not
four different kinds:

| Emitted from | `target=` values |
|---|---|
| `gateway/log.md` (`gateway/app/warmup.py`) | `hybrid_dense_index` (only when `OPENAI_API_KEY` is absent — `status=skipped`; issue #457. A successful load emits `dense_index_loaded` instead, see Hybrid Retrieval (Stack C) below, and a load failure with a key present now PROPAGATES out of the Gateway lifespan rather than being logged here) |
| `wiki/log.md` (`markdown_kb/app/retrieval.py::warm_llm_client`) | `wiki_llm` |
| `vector_rag/log.md` (`vector_rag/app/retrieval.py::warm_llm_client`, `vector_rag/app/indexer.py::warm_embeddings_client`) | `rag_llm`, `rag_embeddings` |
| `hybrid_kb/log.md` (`hybrid_kb/app/query.py::warm_llm_client`, `hybrid_kb/app/dense_index.py::warm_embeddings_client`) | `hybrid_llm`, `hybrid_embeddings` |

| Kind | When fired | Summary template |
|---|---|---|
| `startup_warmup` | A client ping (`KB_WARMUP_PING` on) succeeded or failed, OR the hybrid dense-index warm was skipped for a keyless boot | `client=<target> status=ok` or `client=<target> status=failed exc=<ExceptionClassName>` (client pings) / `target=hybrid_dense_index status=skipped reason=no_openai_api_key` (keyless-boot skip) |

The client pings stay best-effort: a failure is logged, never raised —
Gateway startup never blocks on a single client ping problem, and the
existing per-request lazy-load fallbacks (issue #133 RAG, issue #326 Hybrid)
still apply on the first real request either way. The hybrid dense-index warm
is DIFFERENT since issue #457: a missing key is a soft skip (logged, boot
stays green), but any other failure — most notably a corrupt or missing
committed dense seed — now PROPAGATES out of `warm_hybrid_indexes` and fails
the Gateway boot, matching the Wiki/RAG sub-apps' existing fail-fast
lifespans (neither of which ever caught its own loader's exceptions either).

---

## Adding a new kind

1. Pick a `snake_case` name that names the **event**, not the outcome (so failures and successes can share a kind with `reason=` discrimination — see `grounding_verify` returning either `claim_supported` or `claim_unsupported` under one kind).
2. Add a row to the relevant section above in the same commit that introduces the `log_event()` call.
3. If the kind has sub-tags (like `chat_error kind=` or `ingest_error error=`), add the sub-tag table too.
4. Reference the authorizing ADR or PRD in the section header so future readers can see *why* the kind exists.

## Deprecation

Removing a kind requires:

1. A grep across the codebase confirming no production caller still emits it.
2. A note in this file under the affected section like `~~`kind_name`~~ — removed in <PR> per <reason>`. Keep the strikethrough row for one phase before deleting, so log readers parsing historical `wiki/log.md` content know what the kind meant.

## Why a separate file (not in `prd.md` or `CODING_STANDARD.md`)

Earlier the enumeration lived in `prd.md` § Log entry conventions. That created a perverse update tax: adding a kind in Phase 3 required editing the Phase 1 PRD. Centralising here keeps `prd.md` phase-historical, lets `CODING_STANDARD.md` stay principles-only, and gives the reviewer one file to grep.
