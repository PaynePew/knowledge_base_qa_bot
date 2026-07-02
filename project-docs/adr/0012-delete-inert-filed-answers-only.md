# The Operator Console can delete inert Filed Answers, never live ones

> **Status:** Amended by [ADR-0025](0025-delete-live-orphans-full-orphan-predicate.md) — the qa-scoped rule below (delete inert only, refuse `status: live`) stands verbatim. What ADR-0025 opens is one precisely-shaped hole in the broader "no general page delete" narrowing: an `entities/`/`concepts/` page may be deleted **only** as a server-re-verified **full orphan** (every Source citation's file gone), via a new guarded `DELETE /pages/{slug}`, as a Confirmed Remediation ([ADR-0024](0024-gated-remediation-human-gate-seam.md)).

Phase 15's Curation Queue surfaces Filed Answer drafts (ranked by lint C8) and schema-invalid `wiki/qa/` pages (C10), and the operator wants a one-click "discard" for the bad ones. The backend deliberately has **no** qa delete or demote endpoint: lint is read-only/advisory ([lint.py](../../markdown_kb/app/lint.py) writes only `lint-report.md`), and the designed remediation is the manual "delete the file to re-file fresh" path the `qa.py` page sentinel documents, or `re-ingest` for entity-driven staleness.

To give the Console a usable discard action without puncturing that model, we add a narrow `DELETE /qa/{slug}` that deletes **only inert pages** (`status: draft` or schema-invalid) and **refuses `status: live`**. Live is the only state that enters the BM25 corpus and is the precious one. The new endpoint is symmetric with `POST /qa/{slug}/promote` (Slice 6-4): both are explicit, separate curator actions — neither is a side-effect of read-only lint.

## Considered Options

- **Console never deletes; stale/invalid are advisory hints only** (operator edits the filesystem by hand). Rejected: poor demo UX — telling a browser user to go delete a file on disk defeats the point of the console.
- **General delete/demote/edit on any qa page, including live.** Rejected: removing live content is consequential and belongs behind a deliberate demote flow, not a console button; it widens Phase 15 scope well beyond the demo.

## Consequences

- `DELETE` refuses `live`, so C9 staleness (which is a *live* page whose source entity changed) is remediated via `re-ingest`, not delete.
- Lint's read-only invariant (PRD #65 Q3 / #78) is preserved: deletion is a distinct explicit action, never a lint mutation.
- Deleting an inert page is safe and regenerable — a discarded draft re-files fresh on the next grounded `/chat`, and the page was never in the retrieval corpus, so no re-index is required.
