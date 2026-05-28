# A gateway app mounts both retrieval apps behind one origin

Phase 9's toggle UI needs a single frontend to talk to both Retrieval Stacks (`markdown_kb` / `vector_rag`), which [ADR-0002](0002-two-parallel-retrieval-apps.md) keeps as two independent FastAPI apps. We introduce a thin **gateway** (parent ASGI app) that mounts both sub-apps (`/wiki`, `/rag`), serves the browser UI from one origin, and exposes a single `POST /chat/stream?stack=wiki|rag` that dispatches in-process to the selected stack's streaming function. Single port, zero CORS.

This became feasible only after Phase 8 unified the dependency ecosystem (`vector_rag` migrated to langchain 1.x and joined the uv workspace; `eval/paraphrase_comparison` already imports both stacks in one process). It is deliberately a thin **UI-composition** layer — **not** the pluggable-`Retriever`-protocol merge that ADR-0002 deferred (tracked as #107). The retrieval cores stay independent; the gateway only composes them.

## Considered Options

- **Two processes on two ports + CORS** (UI fetches both origins). Rejected: worse demo ergonomics (two launch commands) and CORS middleware on both apps.
- **Standalone apps + a static UI file hitting both ports.** Rejected: worst path / CORS friction, no single launch command.

## Consequences

- The gateway is the designated home for cross-stack and session-scoped concerns. Phase 11 Conversation Memory (Query Rewriting + Conversation Store) attaches here, keeping the sub-apps stateless and single-turn (see [roadmap](../roadmap.md) § Phase 11 prep note). Phase 9 writes no Phase 11 code.
- The single `POST /chat/stream?stack=` contract is forward-compatible with #107: if the pluggable `Retriever` protocol lands, `stack=wiki|rag|hybrid` selects implementations behind the same endpoint with no client change.
