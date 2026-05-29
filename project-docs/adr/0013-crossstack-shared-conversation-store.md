# Gateway-owned cross-stack shared Conversation Store + grounding-firewall safety

Phase 11 adds Conversation Memory at the Gateway ([ADR-0010](0010-gateway-mounts-both-apps.md)) via a per-session **Conversation Store** (in-memory, keyed by `session_id` only) and **Query Rewriting** (the Gateway's first LLM-facing module). The Phase 9 Wiki↔RAG toggle introduced in [ADR-0010](0010-gateway-mounts-both-apps.md) creates a design question: should the store be partitioned per stack, or shared?

This ADR decides: the store is **shared across stacks** (`stack` is per-turn metadata, NOT a partition key) and the **Grounding Check** — already specified in [ADR-0001](0001-strict-grounded-answers.md) and [ADR-0004](0004-post-llm-grounding-check.md) — is the safety firewall that prevents cross-stack fact leakage into answers.

## Considered Options

- **Per-stack conversation partitions** (separate session history for `stack=wiki` and `stack=rag`). Rejected: breaks toggle coherence — a user who switches stacks mid-conversation loses the conversational context that Query Rewriting needs to resolve references, defeating the purpose of the session. It would also duplicate the firewall unnecessarily (the partition itself would be an inferior, weaker substitute for the Grounding Check).
- **Shared store, no safety constraint** (treat the other stack's historical answers as facts the current stack's LLM may rely on). Rejected: a claim in a RAG answer is grounded against the RAG corpus, not the Wiki corpus. If that claim slips into the Wiki answer directly, the Wiki grounding check was bypassed — a silent ADR-0001 violation.
- **Shared store + Grounding Check as the firewall** (this decision). Accepted: the Grounding Check already enforces that *every* answer claim traces to the current answering stack's own corpus. Conversation context shapes the *question* only (via Query Rewriting — reference resolution and ellipsis filling). It cannot inject facts into the answer, because facts must still pass the grounding verifier against the answering stack's own Sections/Chunks.

## Consequences

**Invariant** — The Conversation Store is keyed by `session_id` only. `stack` is stored as per-turn metadata but is never used as a partition key. A cross-stack session holds turns from both stacks in a single ordered window.

**Invariant** — Query Rewriting receives the full cross-stack turn history. It may use any prior turn (Wiki or RAG) to resolve pronouns and fill ellipsis in the current follow-up. The output is a self-contained query string — no turn's answer text is injected into the answering stack's prompt directly.

**Invariant** — The Grounding Check (ADR-0001 + ADR-0004) is the sole cross-stack safety firewall. A RAG-stack answer is grounded against the RAG corpus; a Wiki-stack answer is grounded against the Wiki corpus. A claim only one stack's corpus supports cannot enter the other stack's answer: the grounding verifier will return `claim_unsupported`, triggering Cannot Confirm, which blocks filing. No second firewall layer is needed or added.

**Invariant** — RAG turns never file. `done.filed` is always `null` for `stack=rag`, regardless of grounding outcome (RAG has no `wiki/qa/` filing path). Wiki turns file the **rewritten self-contained query** (not the raw follow-up) on grounding pass, consistent with the Phase 6 filing contract.

**Invariant** — The production singleton `_conv_store_module.store.evict_expired()` is called at the top of `chat_stream` (before the history lookup) so idle sessions are swept at request entry. This is the only call site. The sweep iterates over a key snapshot to avoid `RuntimeError: dictionary changed size during iteration` (CODING_STANDARD §2.6).

- The Conversation Store is now the **second per-turn-mutated, TTL-swept structure** in the Gateway (the first is the Section Index swap). CODING_STANDARD §2.6 is updated to document this: single-process dict operations are GIL-atomic; the TTL sweep must not mutate during iteration (use a key snapshot, as implemented).
- The Gateway's Query Rewriting module (`gateway/app/query_rewriting.py`) is added to the LLM-facing surface enumeration in ADR-0005 § Consequences so CODING_STANDARD §2.4 (LangChain isolation) and §6.4 (one live test per surface) apply to it.
- `CONTEXT.md` terms **Query Rewriting** and **Conversation Store** are promoted from `## Reserved (not yet implemented)` to active Language (CODING_STANDARD §3.2) now that they are implemented. Their definitions are unchanged from the grill of 2026-05-29.
- This ADR extends ADR-0010 (gateway-owned cross-stack composition) and references ADR-0001 (strict grounded answers) and ADR-0004 (post-LLM grounding check) for the firewall semantics. It does NOT change the Grounding Check logic — it only confirms the existing check is sufficient as a cross-stack firewall.
