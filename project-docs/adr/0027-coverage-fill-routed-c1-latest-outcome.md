# Coverage Fill is a Routed remediation (a third class), and C1 resolution is latest-outcome

[ADR-0023](0023-lint-remediation-direct-vs-authored.md) classified C1 coverage-gap / C2 red-link "fill" as Authored, and [[adr-0024]] carried that placement into the Gated class. Designing the actual flow broke the classification, and exposed that neither Coverage check could ever close its remediation loop as specified.

**Why fill is not Authored.** Fill routes through the existing Upload → Import → Ingest pipeline (the ADR-0023 decision that stands: never LLM-from-nothing, because a coverage gap has no [[source]] to ground against). Walking that flow: the human supplies the missing file; the LLM synthesis happens *inside ingest* — Direct-class machinery, re-derivable from the Source, whose output pages enter the corpus with no review today. **At no point does a draft exist for a curator to approve.** Keeping fill in Authored would force either inventing an approval gate on ingest output (scope explosion, and a change to Direct-class trust tier A already settled) or shipping an "Authored" remediation whose defining gate gates nothing.

**Why the loop could not close.** Two code facts:
- **C1 aggregates the whole `wiki/log.md`** — no time window, no resolution marker. Old `chat_fallback` entries survive any fix, so even a perfect fill re-surfaces the finding on the next lint. Tier A's "remediate → re-lint → finding visibly disappears" pattern is structurally unavailable to C1.
- **C2 closure is not guaranteed** — the red-link clears only if ingest happens to mint a page with exactly the linked slug, and ingest's classify-on-outline chooses slugs freely.

## Decision

1. **A third remediation class: Routed.** A Routed Remediation is one the system cannot execute at all — the missing ingredient is knowledge only the human can supply. The affordance is a **navigation, not an execution**: it carries the finding's context into the existing workflow. C1/C2 flip from `authored` to `routed` in `_REMEDIATION_TAXONOMY` (descriptor carries the route, e.g. `route="import"`); the Console's disabled "Authored (tier B)" placeholders on Coverage findings become active "Fill via Import" buttons that open the pipeline stepper with a context banner (C2: slug + `referenced_by` + `sample_context`; C1: `sample_raw_queries` + `hit_count`). Routed has no gate (nothing to approve) and no batch (nothing to run); it commits nothing itself.
2. **C1 resolution is latest-outcome.** Every failing ask writes a fallback entry, and every ask writes a `chat` entry (retrieval.py `_write_chat_log`), so: a cluster is **suppressed when a `chat` entry for the same canonical query is newer than the cluster's newest fallback entry** — the latest ask succeeded. C1's meaning sharpens from "queries that ever failed" to "queries whose **latest outcome** is still a failure". This also fixes organic healing (a gap answered by later content growth stops nagging without any fill). Post-fill, the finding card offers **"Verify: re-ask"** — one user-triggered `/chat` call that simultaneously proves closure to the human and writes the resolving log entry. Self-healing, zero new state files.
3. **C2 reports honest misses, and ingest's contract is untouched.** After the fill flow, re-lint: the red-link clears iff the slug now resolves. When it does not, the card reports "imported and ingested — new pages A, B — but `[[slug]]` is still unresolved"; renaming the link or the page is a curator judgment, done by hand. No slug-hint parameter is added to ingest.

## Considered Options

- **Keep C1/C2 in Authored (rejected).** No draft exists to approve; see above.
- **LLM-draft the missing page anyway (rejected, re-affirmed from ADR-0023).** Ungroundable by construction; fails [[grounding-check]] before it exists.
- **Clear C1 by log truncation/rotation (rejected).** Destroys the operational history every other log consumer reads, and clears unresolved gaps along with resolved ones.
- **Time-window C1 (rejected).** A window eventually hides resolved gaps but also hides *unresolved* old gaps (false negatives) and keeps resolved ones nagging until the window passes (false positives). Latest-outcome is the precise predicate; a window is a blur.
- **Slug-hint to ingest so C2 closure is guaranteed (rejected).** Couples lint's finding shape into ingest's outline contract for one case, and "should the new page really be named exactly like the link?" is a curator judgment, not a constraint to force.

## Consequences

- CONTEXT.md `Remediation` gains the third class; "tier B" now reads Gated ∪ Routed (the Direct/Gated seam of [[adr-0024]] is untouched — Routed sits outside it because nothing executes).
- The C1 semantic change is user-visible on its own: clusters can disappear from the report because a later ask succeeded, with no import having run. That is the intended reading of the check.
- `_REMEDIATION_TAXONOMY` stays the single source of truth for all three surfaces ([[adr-0017]]); CLI/MCP render the route as text ("fill via: kb import …").
- **Invariants (tagged for the reviewer, per CODING_STANDARD §2.5).** **Invariant** — a Routed remediation commits nothing itself: the fill affordance only navigates; all writes happen in the existing Import/Ingest machinery under its own rules. **Invariant** — C1 suppression requires a strictly newer `chat` entry for the same canonical cluster key; a fallback entry newer than the last success re-opens the cluster. **Invariant** — C2 fill never mutates the referencing pages (no automatic link rewriting).
