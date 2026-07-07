# A source-rooted C5 contradiction is detected by re-judging the reconcile drafts for cross-page convergence, not by the per-page grounding gate

[ADR-0036](0036-c5-source-rooted-contradiction-routed-fix-source.md) gave C5 a second, Routed **fix-source** exit for source-rooted contradictions and made **grounding failure** the signal that distinguishes source-rooted from wiki-rooted (§2: *"a grounding failure is the source-rooted signal"*; §7 premise: *"facing a self-contradictory union, the verifier cannot support any single 'N 天' claim, so `apply` always fails"*). Live-console testing of the deployed corpus (issue #545) disproved that premise.

## Confirmed root cause

`generate_reconcile` grounds each draft against the **whole-file union** of both pages' Sources (`reconcile.py:650-655`; `passed = outcome_a.passed AND outcome_b.passed`, `reconcile.py:500`). But grounding is an **existence** check — "is each claim supported by *some* cited Source section?" — not a **consistency** check. For a source-rooted pair the union contains *both* the 14 天 section (`docs/demo-zh/退款與退貨.md#退款申請窗口`) and the 30 天 section (`docs/planted-zh/退貨期限提醒.md#退貨期限提醒`), so it individually supports 14 天 **and** 30 天. A reconcile draft of A=14 天 / B=30 天 — still mutually contradictory — is therefore fully grounded, and `grounding.passed=true`. §7's premise is simply wrong: the self-contradictory union supports *each* claim on its own; that is precisely what grounding is designed to confirm.

Live evidence (backend API, clean `reset.yml` baseline, 2026-07-07):
- Two consecutive `POST /wiki/pages/reconcile` on the same pair with **identical `hash_a`/`hash_b`** returned `grounding.passed` **true** then **false** — an unseeded `gpt-4o-mini` (`temperature` is already pinned to 0 in `get_lint_llm`, `lint.py:472-498`; no `seed`). The signal is not just unsound, it flaps.
- In the passing run the draft even wrote "此處…存在不一致" in prose, yet grounding passed.
- **Apply is not gated on grounding at all** (`console.html:5553-5559`, always enabled except in-flight), and the server-side apply re-verify is the *same* union-existence grounding, so applying A=14/B=30 **re-passes and succeeds**, writing a fresh `direct` contradiction. The loop diverges silently — the opposite of ADR-0034's "convergent by construction."

## Decision

1. **Detect source-rooted by a convergence re-judge, not by grounding.** After `generate_reconcile` produces `content_a`/`content_b`, run the existing C5 contradiction oracle `_judge_page_pair` on the two **drafts**. `none` → the drafts agree → **converged** (wiki-rooted, reconcilable). `direct`/`tension` → the drafts still give incompatible answers → **not converged → source-rooted**. This is [ADR-0034](0034-c5-contradiction-only-retire-similarity-buckets.md)'s deferred "claim-level cross-page contradiction detection (the ceiling)", realized through the oracle we already trust rather than a new claim-extractor: detection and convergence-confirmation now use the *same* judge, which literally makes "convergent by construction" a checkable property.

2. **Grounding is demoted to faithfulness-only.** It still verifies each draft is faithful to the Sources (it catches a draft that hallucinates a value neither Source states — a genuine wiki-authoring error), but it no longer decides source-rooted. Grounding (existence / faithfulness) and convergence (cross-page consistency) are **orthogonal** — the same axis split CONTEXT.md already draws between the Grounding Check and Coherence. C5 is the page-vs-page contradiction the Grounding Check structurally cannot see; the source-rooted call belongs to a Coherence-shaped check, not to grounding.

3. **Routing table** (replaces ADR-0036 §3's "default view follows the grounding outcome"):

   | grounding | converged | meaning | modal default | Apply |
   |---|---|---|---|---|
   | fail | — | draft not faithful to its Source(s) | Wiki (with grounding report) | **disabled** |
   | pass | **false** | both faithful, but Sources disagree → **source-rooted** | **Source comparison** | **disabled** |
   | pass | true | genuine reconciliation | Wiki comparison | enabled |

4. **Explicit response field; the frontend stops inferring.** `ReconcileGenerateResponse` gains `converged: bool` (plus the re-judge `severity` and one-line `summary` for the Source-view note). The two-view modal's **default view and the Apply enablement both key on `converged`** (and grounding for the fail row), never on `grounding.passed` alone.

5. **Conservative fail-safe.** If the convergence re-judge errors, times out, or is indeterminate, treat the pair as **not converged (source-rooted) → Apply disabled**. The cost is asymmetric: a false "source-rooted" only makes the curator toggle to Wiki view (mild); a false "converged" lets Apply write a new contradiction into the corpus (bad, and invisible until the next audit). Never auto-enable Apply under uncertainty.

6. **Seed the lint LLM.** Pass a fixed `seed` in `get_lint_llm`'s `ChatOpenAI` construction to cut run-to-run flap of both the judge and the convergence re-judge. This is *not* the load-bearing fix — a deterministic grounding PASS on a contradicting draft would still be wrong; the re-judge is what makes routing correct — but it reduces curator confusion and needless verdict-cache churn.

## Considered options

- **Only pin determinism (add `seed`), keep grounding as the signal — rejected.** Determinism cannot repair an unsound signal: a *stable* `grounding.passed=true` on an A=14/B=30 draft is still the wrong answer. Seeding is kept as a secondary hardening (decision 6), not the fix.
- **A bespoke claim-level extractor + same-subject/opposite-polarity aligner (ADR-0034's literal ceiling) — deferred, not adopted now.** Highest theoretical precision, but a new prompt surface and a second LLM flap source, for a signal the existing judge already provides. If the drafts-re-judge proves too coarse in practice (e.g. misses a numeric conflict the audit judge would also miss), the claim-level aligner is the filed follow-up — the honest ceiling above this, not gold-plating today.
- **Narrow the grounding union to cited sections — rejected (as in ADR-0036 §7).** Orthogonal to this defect: the existence-check gap persists at any union scope, and whole-file union remains load-bearing for detection (pulling a sibling section into the audit) and for faithfulness grounding.
- **Gate Apply purely server-side by re-running grounding on apply — rejected.** The apply re-verify is the same union-existence grounding, so it re-passes the contradiction. The gate must be convergence, applied at both the response (Apply enablement) and, defensively, at `reconcile/apply`.

## Consequences

- **One extra LLM call per reconcile *generate*** (the drafts re-judge), on the same model/temperature/seed as the judge. Reconcile-generate is a deliberate, low-frequency curator action, not a request hot path; the cost is acceptable and bounded.
- **The C5 audit (detection) is unchanged.** This decision touches only reconcile-*generate* time; the deep-audit candidate selection, the judge, and the body-hash-keyed verdict cache are untouched.
- **ADR-0036 §2 and §7's premise are superseded** by this ADR (grounding failure is no longer the source-rooted signal). ADR-0036's other decisions stand: the fix-source Routed exit, the two-view modal, the destination-aware in-place overwrite, and the whole-file grounding union (retained — still correct for faithfulness and for detection recall). An amendment is appended to ADR-0036.
- **Verified end-to-end against the real corpus before this is called done** (the operator's stated acceptance bar): reconcile the planted `退款申請窗口 / 退貨期限提醒` pair → `converged=false` → Source view, Apply disabled; fix the Source (`docs/planted-zh/退貨期限提醒.md` 30→14) → force re-ingest → reconcile again → `converged=true`, Apply enabled; and confirm no path lets Apply write a 14/30 contradiction.
- **Invariants (tagged for the reviewer, CODING_STANDARD §2.5).** **Invariant** — Apply is enabled **iff** the reconcile drafts are convergent (re-judge `none`) **and** grounded; a non-convergent or ungrounded pair can never be Applied, at the response layer and at `reconcile/apply`. **Invariant** — source-rooted-ness is decided by the convergence re-judge, never by `grounding.passed`. **Invariant** — grounding verifies each draft's faithfulness to the Source union, never cross-page mutual consistency. **Invariant** — on convergence-re-judge error/indeterminacy the pair is treated as source-rooted (Apply disabled) — the system fails toward not-applying.
