# A live wiki page may be deleted only as a server-re-verified full orphan, via a new `DELETE /pages/{slug}`; ADR-0012's qa narrowing stays intact

Tier B's Confirmed Remediation ([[adr-0024]]) for **C11 orphan** needs a delete operation that [ADR-0012](0012-delete-inert-filed-answers-only.md) deliberately did not build: removing a page that is *in the retrieval corpus*. Exploring the code re-shaped the problem in four ways:

1. **C11 fires on partial orphans.** `_check_c11_orphan` flags a page when *at least one* `sources` citation's file is missing under `docs/**` — so a page with three Sources and one gone is a C11 finding whose remaining two Sources may still ground it perfectly well. Deletion is only ever correct for a **full orphan**: `sources` non-empty and *every* citation's file missing.
2. **This is not a widening of `DELETE /qa`.** C11 scans `entities/` and `concepts/`; `DELETE /qa/{slug}` governs `wiki/qa/` Filed Answers, whose draft/live `status` field entities/concepts pages do not even have — for them, existing *is* being in the BM25 corpus. This decision introduces the **first delete of a corpus-resident page**, a different act from qa discard, so ADR-0012's "refuse `status: live`" rule is left untouched.
3. **The existing cleanup path is structurally unreachable here.** `wiki_writer.delete_orphans` removes pages a *re-ingested* Source no longer produces — it only runs inside ingest of that Source. A C11 orphan's Source file is *gone*, ingest never runs for it, so no existing machinery can ever clean it (the "orphan-case is the real leak" note from the #355 grill).
4. **Index consequences are partially pre-built.** Deleting a corpus page requires one BM25 reindex; the dense arm already tolerates deletions via the dead-id serving guard (#357) and is fully cleaned by the manual dense rebuild ([[adr-0022]]).

## Decision

- **Predicate (the only condition under which a live page may be deleted):** the page is a **full orphan** — its `sources` frontmatter is non-empty and every citation's file is missing under `docs/**`.
- **Server-side re-verify at delete time.** The endpoint recomputes the predicate on receipt; it never trusts the client's lint finding, which may be stale (the Source may have been restored or re-imported since the report rendered). Predicate fails → `409 Conflict`, nothing deleted.
- **New endpoint `DELETE /pages/{slug}`.** The C11 finding carries a bare `page_slug` (no subdir) and slugs are corpus-unique (`resolve_slug_collision`), so the server resolves `entities/` vs `concepts/` itself. The endpoint is *not* a general page delete — it is a guarded lifecycle operation that only ever deletes full orphans.
- **Hard delete, then one BM25 reindex** (same pattern as promote). No dense-arm cleanup in-endpoint: the dead-id guard covers serving, the manual dense rebuild covers removal.
- **Partial orphans get no delete affordance.** Their remediation (fix the renamed citation in frontmatter, re-ingest) is a human edit and stays advisory in tier B. The Console splits C11 findings: full orphan → Confirmed delete button; partial → advisory text only.
- **Governance:** the operation is a **Confirmed Remediation** ([[adr-0024]]) — a human confirms the named irreversible operation; no LLM is involved; it never batches.

## Considered Options

- **Widen `DELETE /qa/{slug}` to accept live pages under some flag (rejected).** Category error: different page family, different lifecycle semantics (qa `status` vs corpus residency). It would also erode ADR-0012's clean "inert only" rule for Filed Answers to solve a problem that lives elsewhere.
- **Trust the client's lint finding as authorization (rejected).** The report is a point-in-time projection; a Source restored after the report renders would make the delete destroy a grounded page. The server must recompute the predicate at the moment of deletion.
- **Allow deleting partial orphans too (rejected).** A page with any surviving Source may still be validly grounded curated synthesis; deleting it destroys content that has a basis. The right fix there is repairing the citation, not removal.
- **Soft-delete / trash directory instead of hard delete (rejected).** The public demo is ephemeral with scheduled resets ([[adr-0021]]), the Confirmed gate is the safety mechanism, and a trash layer adds a second lifecycle (expiry, restore, lint visibility) to govern for negligible benefit.

## Consequences

- **ADR-0012 is amended, not repealed:** its qa-scoped "delete inert only, refuse live" decision stands verbatim; this ADR opens one precisely-shaped hole in the broader "no general page delete" narrowing — the full-orphan case — and nothing else.
- Tier A's "zero new backend endpoints" stance is deliberately broken here (first of two tier-B breaks; the other is the batch-promote endpoint ADR-0023 already deferred).
- Pages with an *empty* `sources` list are unaffected: C11 skips them today, so they produce no finding and no affordance.
- **Invariants (tagged for the reviewer, per CODING_STANDARD §2.5).** **Invariant** — `DELETE /pages/{slug}` recomputes the full-orphan predicate server-side at delete time and refuses (409) when it does not hold. **Invariant** — a successful delete triggers exactly one BM25 reindex. **Invariant** — the operation is Confirmed-class: human gate, no LLM call, never batched.
