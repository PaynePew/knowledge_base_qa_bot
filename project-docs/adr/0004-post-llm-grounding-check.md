# Post-LLM grounding check: Block & Replace, claim-level, single-step verifier

After the main LLM call drafts an answer, a second structured LLM call (the **Grounding Check**) verifies every atomic claim in the draft against the cited Sections. If any claim is unsupported, the draft is discarded and the literal `"I cannot confirm from the knowledge base."` string is returned — identical to the pre-LLM threshold gate in ADR-0001. The `/chat` response gains a unified `grounding` field that carries the outcome reason across all gate types (pre-LLM and post-LLM), so callers can branch UX deterministically without parsing the answer text.

The verifier uses `gpt-4o-mini` (same default as the main model), is separately configurable via `OPENAI_VERIFIER_MODEL`, operates in a single LLM call with a chain-of-thought scratchpad embedded in the response schema, and retries up to twice on transient errors before failing closed. The implementation lives in a new deep module `grounding.py` that consumes a `CitableContent` Protocol (not the markdown_kb-specific `Section` type), keeping `vector_rag/` adoption a no-rewrite addition.

We chose this design because the ADR-0001 strict contract rejects half-confidence: if any claim cannot be traced to a cited Section, the whole answer is unsafe to return. A Block & Replace failure contract at claim-level granularity is the only option that preserves this invariant uniformly. The unified `grounding` field closes the audit gap that existed when `Cannot Confirm` responses were indistinguishable from each other at the API surface.

## Considered Options

### Failure contract (Q2)

- **(a) Block & Replace** _(chosen)_: verifier failure → replace draft with literal `Cannot Confirm` string. Symmetric with the pre-LLM threshold gate; the API surface for "KB cannot back this answer" is always the same string.
- **(b) Block & Repair**: verifier failure → send the draft back to the main LLM with unsupported claims flagged, requesting a revised answer. Rejected: the repaired answer still has no guarantee of grounding; adds latency with no correctness bound; the loop could require multiple iterations, defeating the 5-second latency budget.
- **(c) Non-block & Annotate**: verifier failure → return the draft with a `warning` annotation, letting the caller decide. Rejected: the ADR-0001 strict contract does not permit releasing an ungrounded answer in any form; annotation moves the trust problem to the caller, which typically cannot resolve it.
- **(d) Async log only**: verifier failure → return the draft, log the failure for offline review. Rejected for the same reason as (c) — a confidently wrong answer with a server-side note is worse than `Cannot Confirm`.

### Judgment granularity (Q3)

- **(i) Answer-level**: verifier returns a single `passed: bool` for the whole draft. Rejected: coarse — does not tell the caller which claims were grounded, cannot populate the `claims` list for per-claim provenance.
- **(ii) Claim-level, any-unsupported drops whole answer** _(chosen)_: verifier extracts atomic claims, judges each, records `citing_section_ids` per supported claim. Any `supported=false` → `passed=false` for the whole response. Matches ADR-0001's rejection of partial-confidence answers.
- **(iii) Claim-level, partial pass**: return the draft with the unsupported claims redacted or flagged. Rejected: requires the API to distinguish "full answer" vs "partial answer", neither of which appears in the current contract; complicates client UX for marginal gain.

### Verifier model (Q4)

- **(1) mini/mini** _(chosen)_: `gpt-4o-mini` for both main and verifier. Cost ~NT$0.02/query at prototype scale. `OPENAI_VERIFIER_MODEL` env var allows independent override without code changes.
- **(2) 4o/main + mini/verifier**: using `gpt-4o` for the main call and `gpt-4o-mini` for verification. Rejected: at prototype scale the cost uplift (~10×) is not justified; cross-family mixing adds a second pricing dimension with no quality benefit that is measurable at corpus size.
- **(3) mini/main + 4o/verifier**: using `gpt-4o-mini` for the main call and `gpt-4o` for verification. Rejected: inverted economics — a weaker model can produce claims that a stronger verifier rejects; produces a higher-than-expected false-positive rejection rate, hurting user experience without improving safety.
- **(4) Cross-family (Claude as verifier)**: bring in the Anthropic SDK for the verification step. Rejected for this phase: adds a second dependency and complicates prompt iteration; `OPENAI_VERIFIER_MODEL` keeps the door open via configuration when an objective eval justifies the switch.

### Verifier fail-mode (Q5)

- **(a) Retry only**: keep retrying until OpenAI responds. Rejected: unbounded latency; user-facing `/chat` would hang on sustained outage.
- **(b) Fail-open**: after exhausted retries, return the unverified draft. Rejected: violates ADR-0001 strict contract — an answer must never leave without a completed grounding check.
- **(c) Retry + Fail-closed, bounded** _(chosen)_: errors are classified into four types. Transient errors (timeout, 5xx, 429) → retry up to 2 times with exponential backoff (200 ms, 800 ms). Malformed response errors (invalid JSON, missing schema fields) → retry once. Refusal (empty completion) → no retry. Hard errors (401 auth, model not found) → no retry, prominent server log. Total verifier-side latency budget: 5 seconds. On exhaustion, fail-closed: response becomes `Cannot Confirm` with `grounding.reason = "verifier_unavailable"`. ADR-0001 strict contract is preserved under degraded infrastructure.

### Module placement (Q6)

- **(α) New `grounding.py` deep module** _(chosen)_: single responsibility; clean boundary for future `Lint Pass` when the wiki layer ships; expected to grow into a `grounding/` package (`verifier.py`, `lint.py`, `schemas.py`, `prompts.py`).
- **(β) Inline in `retrieval.py`**: add the verifier call into the existing retrieval + LLM call chain in `retrieval.py`. Rejected: violates single-responsibility; `retrieval.py` already manages retrieval, prompt building, and the main LLM call. Adding verification makes it a god-module.
- **(γ) Inline in `routes.py`**: add the verifier call in the `/chat` route handler. Rejected: business logic in a route handler is the standard pattern to regret; the `grounding.verify()` function needs an independent test seam.

### Verifier prompt shape (Q7)

- **(α) Single-step with CoT `reasoning` scratchpad** _(chosen)_: one LLM call simultaneously extracts claims and judges each. The Pydantic schema includes a `reasoning: str` field that the model fills before committing structured fields. Best ROI at prototype scale; one round-trip; scratchpad improves judgment without adding latency. `reasoning` stays internal — not exposed in `ChatResponse`.
- **(β) Two-step (extraction then judgment)**: first call extracts claims; second call judges them. Rejected at this scale: doubles round-trips and latency for a draft that is typically under 200 tokens. Documented as an upgrade trigger: activate when draft length exceeds 1K tokens, claim count exceeds 8, or type-I fixture eval shows extraction recall below 90%.
- **(γ) Per-claim batched calls)**: one LLM call per claim in parallel. Rejected: at claim counts of 2–6 (typical for a focused KB answer), parallelism overhead exceeds single-call latency; also quadruples complexity.

### API surface (Q8)

- **(I) Expose everything**: include `reasoning`, `error_type`, and `retries_attempted` in `ChatResponse`. Rejected: leaks internal plumbing to callers; `reasoning` is a CoT scratchpad that is not stable across prompt iterations.
- **(II) Selective expose + unified `grounding` field** _(chosen)_: expose `passed`, `reason`, `claims`, and `unsupported_claims`. Keep `reasoning`, `error_type`, and `retries_attempted` in server logs only. The `grounding` field covers both pre-LLM and post-LLM gates using a shared `reason` literal set, unifying the audit surface.
- **(III) Separate pre-LLM and post-LLM fields**: add both a `retrieval_outcome` field and a `grounding` field. Rejected: two fields with overlapping semantics confuse callers and require callers to merge them for the common "why did I get Cannot Confirm?" question.

### Scope (Q9)

- **(A) `markdown_kb/` only** _(chosen)_: `CitableContent` Protocol defines the contract; `vector_rag/` satisfies it when reactivated, with no `grounding.py` changes needed. Extraction into a shared workspace member is deferred until a second caller exists (rule of three).
- **(B) Shared workspace member now**: extract `grounding/` into a top-level `shared/` package immediately. Rejected: premature generalisation; the API of `grounding.py` will iterate during Slices #2-#4, and abstracting before the shape is stable doubles the refactoring burden.

### Implementation order (Q10)

- **(I) Outside-in (routes → schemas → `grounding.py`)**: implement the full `/chat` wiring first, stub the verifier. Rejected: tests cannot run green until the entire chain is wired; blocked slices cannot be reviewed independently.
- **(II) Dependency-ordered 4-slice hybrid TDD** _(chosen)_: Slice #1 (design docs) → Slice #2 (schemas + fixtures, RED) → Slice #3 (verifier implementation, GREEN) → Slice #4 (route wiring + `ChatResponse` upgrade). Each slice is independently mergeable. The 7 anchor fixtures in Slice #2 constrain the verifier API before any behaviour is written.

## Consequences

**For callers of `/chat`:**
- `ChatResponse` gains a `grounding: GroundingInfo` field, always populated. Existing callers that ignore extra fields are unaffected; callers that parse the full schema need to add `grounding` handling. The `sources` field remains populated on `Cannot Confirm` responses so callers can display what the bot looked at.
- `grounding.reason` distinguishes the six situations that previously all produced the same `Cannot Confirm` string: `claim_supported`, `claim_unsupported`, `verifier_unavailable` (post-LLM); `below_threshold`, `retrieval_empty`, `index_missing` (pre-LLM).

**For operators:**
- Two LLM calls per `/chat` request in the normal (non-`Cannot Confirm`) flow. Nominal latency target: +0.5-1s for the verifier call. Worst-case (one retry + backoff): +5s hard ceiling before fail-closed.
- `OPENAI_VERIFIER_MODEL` env var allows independent verifier model upgrade. Default is `OPENAI_MODEL` (currently `gpt-4o-mini`).
- Server logs record `error_type` and `retries_attempted` for every failed verification, enabling distinction between infrastructure issues (transient) and prompt issues (malformed / refusal).

**For future contributors:**
- `grounding.py`'s public interface is intentionally one function: `verify(draft: str, sections: list[CitableContent]) -> GroundingOutcome`. All internal complexity (retry policy, prompt template, `with_structured_output` binding) is encapsulated. See `CitableContent` Protocol definition for the minimum contract a retrieval unit must satisfy.
- When `vector_rag/` is reactivated, adopting `grounding.py` requires only that the chunk type satisfies `CitableContent` (three fields: `id: str`, `heading_path: list[str]`, `content: str`). No `grounding.py` changes anticipated.
- If two-step extraction-then-judgment becomes necessary (triggers above), swap the single `verify()` call for an internal `_extract()` + `_judge()` pair; the public function signature is unchanged.
- `grounding.py` is the future home of `Lint Pass` (ADR-0003 wiki layer health checks). When the wiki layer ships, `grounding/` becomes a package with `verifier.py` and `lint.py` as siblings.

**ADR cross-references:**
- ADR-0001: This ADR operationalises layer 3 of the strict grounded contract. The literal `Cannot Confirm` string and the fail-closed invariant are inherited from ADR-0001 and not relaxed here.
- ADR-0003: Phase 2 (Wiki Index Generation) is the first concrete step toward the W2 layered wiki target. `grounding.py` is also the future `Lint Pass` module for that layer.
- ADR-0005: Slice #3 fires the pre-blessed trigger for `ChatOpenAI.with_structured_output(GroundingResult)`. No new framework dependencies are introduced; `langchain-openai==1.2.2` already covers this.
