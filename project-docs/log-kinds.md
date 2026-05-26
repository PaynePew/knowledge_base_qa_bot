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
| `chat_fallback` | Pre-LLM Cannot Confirm gate fires | `"<truncated query>" reason=<not_indexed\|below_threshold> [top_score=X]` |
| `chat_grounding_fallback` | Post-LLM verifier returned not-passed; reply replaced with Cannot Confirm | `"<truncated query>" reason=<outcome.reason>` |
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
