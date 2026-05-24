# Two parallel retrieval apps now, plugin architecture later

The repo promotes the two retrieval scaffolds to top-level apps (`markdown_kb/` and `vector_rag/`) side-by-side, rather than refactoring them into a single application with a pluggable `Retriever` protocol. Both apps share the same external HTTP contract (`/health`, `/index`, `/chat`) so client tests and tooling can target either interchangeably.

We chose the dual-app layout over a unified `src/` with a pluggable backend because the Friday prototype deadline does not absorb a 4–6 hour interface-extraction refactor, and because keeping the two apps fully independent until both work end-to-end gives us a cleaner moment to design the protocol (after we know what the implementations actually need, not before). The `scaffold/` directory name and framing — which implied throwaway exercise code — is dropped because this repo is now being built as a real, long-lived project.

## Considered Options

- **Keep `scaffold/markdown_kb/` and `scaffold/vector_rag/` in place.** Rejected: the `scaffold/` framing actively contradicts the project's long-term intent, and `scaffold/README.md` then competes with the root `README.md`.
- **Refactor to `src/kb_qa_bot/` with a pluggable `Retriever` protocol now.** Rejected for the prototype: forces interface design before either implementation is complete, and risks the Friday deadline. Revisit after both retrievers work end-to-end.

## Consequences

- The path constant in each `app/indexer.py` changes from `parents[3]` to `parents[2]` (one directory shallower).
- The root `README.md` becomes the single entry point; `scaffold/README.md` is deleted, its content merged upward.
- `vector_rag/` ships with a `README.md` that explicitly marks it as post-prototype work, so a future reader does not mistake the TODO scaffolding for abandoned code.
- A future ADR will record the eventual unification into a pluggable architecture when (and if) we do it.
