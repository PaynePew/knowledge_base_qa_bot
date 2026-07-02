# Coherence reconcile is a stateless two-phase Authored flow (generate → preview/edit → apply-with-revalidation); C5 rewrites both pages in place, C4 merge is reference-guarded

Tier B's flagship Authored Remediation ([ADR-0023](0023-lint-remediation-direct-vs-authored.md): C5/C4 Reconcile, grounding-satisfiable from the **union of the two pages' Sources**) hit two structural facts during the 2026-07-02 grill:

1. **entities/concepts pages have no draft status.** Unlike `wiki/qa/` (whose `status: draft` lets a page sit on disk outside the corpus awaiting review), an entities/concepts page is in the corpus by existing. An Authored reconcile draft has **no disk home** — staging one would mean inventing a page lifecycle (a status field or a staging dir, plus indexer filtering, lint exclusions, C11/C2 false-positive defence) for an artifact that costs one LLM call to regenerate.
2. **Killing a slug has invisible fallout.** A deleted page turns referencing `[[links]]` into C2 red-links (whose Routed remediation would misdirect the curator toward *importing* — wrong fix for a rename), and — worse — **qa citations to the dead slug are invisible to lint**: C9 explicitly skips missing entities (lint.py notes this as a potential future "C9.b"). An unguarded merge-delete manufactures dangling references lint cannot see.

## Decision

- **Two-phase, stateless, server-revalidated.**
  - `POST /pages/reconcile` — the LLM drafts from the union of the two pages' Sources, the draft is grounding-checked, and the response returns draft + grounding report + the two pages' content hashes. **Nothing is written to disk.**
  - The Console renders the two existing pages side-by-side with an **editable** draft preview (same spirit as edit-before-promote, [[adr-0026]]).
  - `POST /pages/reconcile/apply` — the client submits the final content (possibly human-edited) plus the content hashes from generate time. The server **re-runs the grounding-check on the exact submitted content** and **refuses (409) if either page changed since generate** (hash mismatch — the finding may no longer hold). Only then: write, one BM25 reindex. The trust model is uniform across tier B: *the server revalidates everything at commit time* (ADR-0025's delete predicate re-check, `PUT /qa`'s grounding re-check, and now this).
  - A draft lost to a page refresh is regenerated for one LLM call — deliberately cheaper than owning a staging lifecycle.
- **C5 reconcile rewrites both pages in place; both slugs live.** Contradiction repair means making the two pages consistent against their Sources' union, not making fewer pages. No deletion → no red-link fallout, no dangling citations. The rewrite advances both pages' `updated` timestamps, so qa pages citing them trip C9 on the next lint and flow into the [[re-file]] loop — the checks compose, which is exactly what the flagship should demonstrate.
- **C4 offers the curator both documented resolutions** (the check's own docstring: "merge or heading rename"):
  - **Merge into base** — merged content lands on the unsuffixed slug; the suffixed variants are deleted *inside apply*, behind a **reference guard**: the server refuses when any variant has inbound `[[links]]` or qa citations, listing them. Auto-suffixed duplicates typically have no inbound references, so the guard rarely fires; when it does, that collision genuinely needs a human. This merge-delete is a distinct deletion path from [[adr-0025]]'s full-orphan `DELETE /pages/{slug}` — different predicate (superseded + reference-free), different operation, no endpoint sharing.
  - **Differentiate** — both pages rewritten in place to be complementary and more specific; nobody dies.
- **Slicing:** C5 Reconcile is the tracer bullet (it drives the full two-phase API + preview/edit/apply UX end to end); C4's dual resolution follows. **C9.b** (a lint check for qa citations pointing at missing entity pages) is filed as its own future issue, not smuggled into tier B.

## Considered Options

- **Stage the draft on disk (status field on entities/concepts, or `wiki/.staging/`) (rejected).** Invents a page lifecycle with a cascade of consumers to fix (indexer, lint's C11/C2, orphan handling) to persist an artifact that is one LLM call to recreate.
- **Trust the client's submitted draft at apply (rejected).** The human may have edited it; grounded answers are the project's identity; and the two pages may have changed underneath the preview. Revalidate + hash-check at commit, always.
- **C5 resolves by merging into one page (rejected).** Kills a slug for no reason, with red-link and lint-invisible citation fallout; consistency, not consolidation, is the contradiction fix.
- **Unguarded C4 merge-delete (rejected).** Manufactures the exact dangling-citation state lint cannot currently see (C9.b blindspot).
- **Auto-rewrite inbound links when merging (rejected).** Mutates pages the curator never reviewed — against the gate's spirit; the guard turns the situation over to a human instead.

## Consequences

- Tier B's new-endpoint ledger closes at six, each separately decided: `DELETE /pages/{slug}` ([[adr-0025]]), `POST /qa/promote-batch` (ADR-0023 pre-authorization), `PUT /qa/{slug}` + `POST /qa/{slug}/refile` ([[adr-0026]]), `POST /pages/reconcile` + `POST /pages/reconcile/apply` (this ADR).
- The reconcile flow is the only tier-B operation that can write entities/concepts pages; everything it writes has passed union-of-Sources grounding twice (generate and apply).
- **Invariants (tagged for the reviewer, per CODING_STANDARD §2.5).** **Invariant** — `POST /pages/reconcile` writes nothing to disk. **Invariant** — apply re-runs the grounding-check server-side on the submitted content and refuses on page-hash mismatch. **Invariant** — C4 merge refuses to delete a variant with inbound `[[links]]` or qa citations. **Invariant** — reconcile never rewrites referencing pages.
