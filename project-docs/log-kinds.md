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
| `parse_warning` | Frontmatter unavailable / unparseable, or a non-leaf heading has no body content | `frontmatter present in <file> but PyYAML is not installed` or `frontmatter parse failed in <file>: <ExcType>` or `non-leaf heading with no body in <file>: '<heading>'` |
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
| `ingest_source` | One Source completed successfully (skipped on failure) | `source=<file> type=<entity\|concept> pages_created=A pages_updated=B pages_deleted=C` |
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

Authorized by GitHub issue #89 (Phase 7 PRD). Slice 7-1 (#90) introduces the three end-of-batch / per-source kinds below; slice 7-2 (#91) adds `import_error`, slice 7-3 (#92) adds `import_skipped`.

| Kind | When fired | Summary template |
|---|---|---|
| `import_batch_started` | Start of an `import_sources()` call | `mode=<batch\|single> source=<filename\|*>` |
| `import_source` | One raw file successfully imported to `docs/<basename>.md` | `source=<basename> docs=<docs_filename> format=<html\|txt>` |
| `import_batch_completed` | End of `import_sources()`; emitted even when some sources failed | `imported=A skipped=B failed=C duration_ms=N` |
| `import_error` | _(Slice 7-2)_ One raw file failed at any stage (continue-on-error semantics) | `raw=<raw_path> error_type=<one of 12 typed errors> error_message=<truncated≤200>` |
| `import_skipped` | Re-import no-op when `content_sha256` matches existing docs frontmatter | `raw=<raw_path> docs=<docs_path> content_sha256=<hex>` |

### `import_error` `error_type=` sub-tags (slice 7-2)

Full enumeration landed with slice 7-2 (#91). All 11 typed failure modes are emitted via `_emit_import_error` in `importer.py`.

| `error_type` | Trigger |
|---|---|
| `HandAuthoredCollision` | `docs/<filename>.md` exists without `imported_from` frontmatter (hand-authored) — refuse to overwrite |
| `UnicodeDecodeError` | raw file not UTF-8 (Big5, Shift_JIS, etc.) |
| `EmptySource` | raw file size 0 bytes |
| `OversizedSource` | raw file > 10 MB hard limit (protects markdownify from in-memory OOM) |
| `UnsupportedExtension` | single-mode source has unsupported extension (`.pdf`, `.docx`, etc.); batch mode silently skips |
| `FileNotFoundError` | single-mode source doesn't exist in `raw/` |
| `MarkdownifyError` | markdownify internal exception (rare; BS4 is highly tolerant) |
| `IOError` | atomic-write `os.replace` failure (disk full, permission) |
| `InvalidFilename` | basename contains a rejected character class (`#`, `/`, `\`, `:`, control chars, bidi control chars U+202A-E/U+2066-9 per CVE-2021-42574) |
| `InvalidSourcePath` | single-mode `source` format violation (absolute path, `..` traversal, `raw/` prefix) |
| `FilenameCollision` | two raw files in the same batch produce the same docs basename; first wins, second fails |

---

## Phase 6 Answer Filing

Authorized by GitHub issue #78 (Phase 6 PRD) §"Reflect step" (Q5) and
§"Orphan-visibility three-layer defence" (Q8d). Slice 6-1 (#79) documents the
three kinds here; the emitters arrive across slices 6-1 / 6-2 / 6-3.

| Kind | When fired | Summary template |
|---|---|---|
| `qa_invalid_status` | `indexer._passes_index_filter` skipped a `wiki/qa/*.md` whose `frontmatter.status` was set but not in `{"draft", "live"}` — the indexer-layer member of the three-layer orphan-visibility defence (curator-typo `status: Live`, forward-compat values like `stale` / `superseded`, missing `status` key, etc.). Slice 6-1. | `file=<filename> status=<repr-of-value>` |
| `qa_reflect` | A `wiki/qa/*.md` page was mutated by Phase 6 — emitted atomically inside the same `_filing_lock` critical section as the file write. Three `op` variants share this kind so reflective-log readers can grep `qa_reflect` once and slice by `op=`. Slices 6-2 (created/touched) and 6-3 (promoted). | `slug=<slug> op=created question="<truncated>" cited=<comma_separated_citations> count=1` _(op=created — first filing for this question)_ <br>or `slug=<slug> op=touched cited_delta=added:<list>,dropped:<list> count=<N>` (or `cited_delta=none`) _(op=touched — re-ask of an existing question; body preserved, count incremented; cited_delta is the drift signal core)_ <br>or `slug=<slug> op=promoted by=curator` _(op=promoted — `POST /qa/{slug}/promote` flipped `status: draft -> live`)_ |
| `qa_filing_error` | `qa.maybe_file_answer` (Slice 6-2) hit an unrecoverable error during the filing critical section — IOError (disk full, permission denied), an attempted touch against an invalid-status orphan page (defence layer 2 of the three-layer defence), or a frontmatter-read failure on the touch path. F3 fail-soft: `/chat` still returns the answer and `response.filed = None`. Slice 6-2. | `slug=<slug> reason=<io_error\|orphan_status\|frontmatter_read_error> exc=<ExcClass>: <message>` |

### `qa_reflect` `op=` sub-tags

| Sub-op | When | Authorized by |
|---|---|---|
| `created` | First filing of a question (slug did not exist on disk) | PRD #78 §"Reflect step" (Q5) |
| `touched` | Re-ask of an existing question (B2 touch semantics — body preserved, `count` and `updated` bumped) | PRD #78 §"Reflect step" (Q5) |
| `promoted` | `POST /qa/{slug}/promote` flipped `status: draft -> live` | PRD #78 §"Two-stage curation lifecycle" (Q1) |

### `qa_filing_error` `reason=` sub-tags

| Sub-reason | When | Source |
|---|---|---|
| `io_error` | OS-level write failure (disk full, permission denied, ENOSPC, etc.) | filesystem write in `qa.maybe_file_answer` |
| `orphan_status` | Touch attempted against a page whose existing `status` is not in `{"draft", "live"}` — the filing-layer defence refuses to bump `count` on an orphan zombie | `qa.maybe_file_answer` touch path |
| `frontmatter_read_error` | Existing `wiki/qa/<slug>.md` could not be parsed for the touch decision (corrupt YAML, missing fences, etc.) | `qa.maybe_file_answer` touch path |

---

## `/lint` route

Authorized by GitHub issue #65 (Phase 5 PRD) and issue #66 (Slice 5-1).

| Kind | When fired | Summary template |
|---|---|---|
| `lint_started` | `run_lint()` enters | _(no payload beyond timestamp)_ |
| `lint_completed` | `run_lint()` exits (including when some checks failed) | `findings=N by_check=c11:A,c3:B,c4a:C,c6:D,c2:E,c1:F,c5:G llm_calls=K cost_usd=X errors=M` |
| `lint_check_error` | An individual check raises (continue-on-error; other checks still run) | `check=<name> exc=<ExcClass>: <msg>` |

No per-finding log entries — findings live in `wiki/lint-report.md` as the source of truth (deliberate noise reduction, per PRD #65 Implementation Decision §Wiki Log entries).

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
