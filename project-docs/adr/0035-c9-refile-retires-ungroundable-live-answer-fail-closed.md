# C9 re-file retires an un-groundable live answer instead of leaving it serving forever (fail-closed)

A `status: live` [[filed-answer]] flagged stale by C9 has exactly one remediation — [[re-file]] — and on a failed re-ground [[re-file]] wrote **nothing** (ADR-0026 § Consequences Invariant). Combined with `DELETE /qa/{slug}` refusing live pages ([[adr-0012]]), that left a real dead end: a live answer whose question the wiki can **no longer ground** keeps serving its stale text, C9 re-fires on every audit, and no console action can clear it. Observed live on `qa-store-pickup-zh-003` (re-file → `claim_unsupported` → "nothing changed, this page keeps serving", forever).

The root mistake was treating **all** re-ground failures the same. They are not:

- A **transient / operational** failure (`verifier_unavailable`, `index_missing`) says nothing about the KB's content — the verifier blipped or the index was not loaded. Writing nothing and retrying is correct; ADR-0026's instinct holds here.
- A **content** failure (`retrieval_empty`, `below_threshold`, `claim_unsupported`) says the KB itself can no longer ground a fresh answer to this question. For a **fail-closed** KB whose identity is "[[cannot-confirm]] is a success", continuing to serve a stale answer we can no longer back is *worse* than admitting we cannot confirm.

**Decision.** `qa.refile` splits its failure branch by reason. A transient failure, or any failure on a page that is not live, still raises `QaRefileRejected` (nothing written — keep serving, retry). A **content** failure (`_RETIRE_REFILE_REASONS`) on a **live** page **RETIRES** the answer: it is demoted to `status: draft` **in place, with its OLD content preserved**, so it leaves the BM25 corpus (`/chat` now returns [[cannot-confirm]] for that question), C9 stops firing (no longer live), and the draft lands in the existing Curation Queue for the curator to **salvage (edit) or discard**. The route returns `200 {retired: true}` (carrying the failing `GroundingInfo` for audit) and reindexes; the Console messages this distinctly from a fresh re-file.

This completes the "deliberate demote flow" [[adr-0012]] foresaw and ADR-0026 realised only *inside* a successful re-file — now it also covers the case a re-file cannot succeed.

## Considered Options

- **Keep ADR-0026's blanket write-nothing (rejected).** It is what produced the stuck-forever state: a live answer the KB can no longer back serves indefinitely and C9 nags forever — the opposite of the fail-closed identity.
- **A separate explicit "Retire / Delete-live" button, independent of Re-file (rejected for now).** More surface, and the curator cannot know a re-file will fail until they try it. Folding retire into the re-file *failure* path means one action does the right thing: refresh if it can, retire if it cannot. A standalone retire can be added later if a use case appears that is not downstream of a re-file attempt.
- **Retire on ALL re-ground failures, transient included (rejected).** A verifier outage is operational, not a verdict on the KB; retiring on it would demote good answers during a blip. The retire set is an explicit allow-list of content failures, and an *unrecognised* reason defaults to transient — a new failure mode never silently retires a live answer.
- **Delete the page instead of demoting (rejected).** Demote is reversible and preserves the old content for the curator to salvage; Discard (delete-on-a-draft) remains available if they want it gone. Delete of a *live* page stays refused ([[adr-0012]]).

## Consequences

- **Supersedes** ADR-0026 § Consequences Invariant *"a failed re-ground during re-file writes nothing: no demote, no draft, no reindex"* — now scoped to **transient** failures and **non-live** pages. ADR-0026's other invariants are unchanged: un-approved synthesis never enters the corpus (retire stages a *draft*; only promote commits, [[adr-0020]]).
- The retired draft carries the OLD content and its OLD `sources`; the failing `GroundingInfo` is returned for audit and logged (`qa_reflect op=retired reason=<content-failure>`). It surfaces in the Curation Queue / C8 as an ordinary draft — the curator reviews it (promote if the staleness was a false alarm and the old answer still holds, edit to fix, or discard). Blindly promoting it re-serves the old answer: that is the human's explicit gate call ([[adr-0020]]), not a silent regression.
- Dovetails with the ADR-0026 amendment (2026-07): a retired draft frequently has no fresh inline citation, and the relaxed edit gate now lets the curator edit it rather than being forced to discard.
- **Invariant** (tagged for the reviewer, CODING_STANDARD §2.5) — retire fires ONLY on a content failure (`_RETIRE_REFILE_REASONS`) AND a live page; a transient failure or a non-live page writes nothing. **Invariant** — retire demotes to draft (leaves the corpus, fail-closed); it never deletes and never promotes.
- `POST /qa/{slug}/refile` now has three outcomes: `200 {retired:false}` (fresh re-file), `200 {retired:true}` (content-failure retire, reindexed), `422` (transient failure / non-live page, nothing written).
