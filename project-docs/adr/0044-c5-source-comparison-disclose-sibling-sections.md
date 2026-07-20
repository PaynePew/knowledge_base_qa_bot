# The C5 Source comparison discloses the cited files' sibling sections — the whole-file evidence must be on screen

Issue #635 (operator report, live use of [ADR-0043](0043-c5-in-modal-source-editor-staged-batch-byte-source.md)'s editor). The Reconcile grounding/convergence evidence is **whole-file** ([ADR-0036](0036-c5-source-rooted-contradiction-routed-fix-source.md) decision 7 — deliberately, to catch cross-section contradictions), but the Source comparison view rendered **only the cited sections** (issue #534 payload, ADR-0036 decision 3). For the demo pair `退款流程 / 退貨期限提醒`, the root of the contradiction — "14 天" in `docs/demo-zh/退款與退貨.md#退款申請窗口`, a sibling section the page does not cite — participated in the verdict but never appeared on screen, while the grounding-failure list quoted it. The curator saw an accusation citing invisible evidence, read it as a mis-judgment ("only 30 天 exists anywhere I can see — why would I edit toward a 14 that doesn't exist?"), and rationally refused to edit. An information gap, not a judgment error.

## Decision

1. **`_cited_sections_for_page` also returns the non-cited sibling sections of every resolved cited Source file**, flagged `CitedSourceSection.cited = False` (new field, default `True` so every pre-existing consumer keeps its meaning). Siblings are appended AFTER the cited entries — the cited-first ordering issue #534's consumers rely on is untouched — deduplicated against them, in file order per file. Zero extra I/O and zero LLM: the files are already parsed for the citation resolution, and the grounding union already reads them whole.
2. **The Source comparison card renders cited sections expanded (as today) and sibling sections collapsed** to a heading row tagged "not cited by this page", expandable in place. An expanded sibling gets the same View / Fix-this-Source affordances (the ADR-0043 editor loads the whole file either way). Unresolved citations contribute no siblings (nothing was parsed).
3. **The modal names its evidence scope.** Under the grounding-failure claim list, a note states the check runs against the whole Source files, so a flagged claim may come from a non-cited section; the sources-disagree note points at the collapsed sections. Without this, a claim quoted from an off-screen sibling reads as a hallucination.

## Considered and rejected

- **Per-claim provenance from the verifier** (extend the grounding output schema so each unsupported claim cites its source section). Rejected: ADR-0038 demoted grounding to a faithfulness-only check that is unseeded and flaps run-to-run; LLM-attributed provenance on top of a flapping signal manufactures new misdirection. Whole-file disclosure is deterministic and sufficient — the human can see every sentence that participated.
- **Whole-file `<pre>` dump instead of sectioned collapse.** Rejected: loses the anchored per-section View/Fix affordances and the cited/uncited distinction that tells the curator where to look first.

## Consequences

- ADR-0036 decision 3's display contract is amended (note appended there): the Source comparison shows the whole cited files, sectioned, cited-first. Decision 7 (whole-file union) is unchanged — this ADR is its display-side completion: what the judge can see, the curator can now see.
- **Invariants (tagged for the reviewer, per CODING_STANDARD §2.5).** **Invariant** — every section of every resolved cited Source file appears in the payload exactly once, cited entries first. **Invariant** — sibling disclosure adds no fetch, no LLM call, and no change to the grounding union or the convergence signal. **Invariant** — an unresolved citation still yields its honest unresolved entry and never phantom siblings.
