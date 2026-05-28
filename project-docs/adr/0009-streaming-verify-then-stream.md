# Streaming /chat/stream verifies before it streams

Phase 9 adds `POST /chat/stream` (SSE) to both retrieval apps. The post-LLM Grounding Check ([ADR-0004](0004-post-llm-grounding-check.md)) is a whole-draft Block & Replace gate, which is fundamentally incompatible with streaming raw LLM tokens: verifying requires the *complete* draft, and the Block & Replace contract forbids ever showing unverified content. We therefore run draft + verify **server-side** and stream only the verified answer (or `Cannot Confirm`); the answer's token-by-token delivery is a cosmetic replay of already-verified text, not real-time generation.

The genuine, non-cosmetic streaming value is **sources-first**: the `sources` event is emitted immediately after retrieval (Wiki ~instant; RAG after one query-embedding round-trip), ~4-7s before the answer — exactly PROMPT.md's "Return selected sources first, so users can see what context the bot is using."

## Considered Options

- **Optimistic streaming + retract** (stream raw tokens, emit a retract / grounding-failed event if verification fails), and its "grey verifying-state" variant. Rejected: both briefly expose fabricated content — the precise harm the exercise handout's "Layer 3 output validation" exists to prevent (a customer-support bot stating a non-existent policy → user acts on it → trust loss / complaint).
- **Drop the verifier on the stream path** (stream real tokens, grounding only via system prompt + threshold gate). Rejected: it would make the most demo-visible surface the *least*-grounded one, omitting a defense layer the handout requires.

## Consequences

- Answer latency is identical to non-streaming `/chat` (~4-7s; draft + verify are sequential LLM calls). SSE buys sources-first delivery + liveness feedback + the live Wiki-vs-RAG toggle demo — **not** a faster answer.
- This is the conservative "verify-then-show" pattern standard for high-stakes RAG. Consumer chat streams raw tokens because it moves grounding into training-time alignment and accepts residual risk via a disclaimer — an exemption this customer-support domain does not have.
- Cannot Confirm is represented uniformly: `sources → token(s) of the fixed phrase → done{passed:false, reason}` — same semantics as the non-streaming `/chat` answer field.
