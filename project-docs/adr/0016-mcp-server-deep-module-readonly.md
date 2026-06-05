# MCP server: direct deep-module adapter with a read-only corpus surface

Phase 12 exposes the knowledge base to LLM agents (Claude hosts) over MCP. Two structural decisions shape the server: which layer it adapts, and how much of the system it exposes.

## Decision

### Wrap the deep modules directly, not the Gateway

The MCP server (`kb_mcp/`, a new workspace member) imports `markdown_kb` and `vector_rag` deep modules directly and calls their functions. It does **not** route through the Gateway (ADR-0010).

Justification: an MCP host holds the conversation context itself and forms a self-contained query per tool call, so the Gateway's two reasons to exist on the chat path — Query Rewriting and the Conversation Store (ADR-0013) — are redundant here. (CONTEXT.md already records that the `ConversationStore.dump()` "Hot-Cache seam" framing was retired because the Hot Cache's input is the host session, not the store.) Routing through the Gateway would also force an SSE-stream-to-tool-result impedance match and would hide `kb_search` (raw evidence, no LLM) — a deep-module primitive the Gateway does not expose.

### Read-only corpus surface + Hot Cache (four tools)

The server exposes `kb_ask_v1`, `kb_search_v1`, `kb_read_hot_v1`, `kb_save_hot_v1`. It does **not** expose `kb_ingest`, `kb_index`, or `kb_lint`.

Justification: without `kb_ingest` the agent never mutates the curated corpus, which makes `kb_index` dead weight (nothing new to index). Corpus maintenance stays an operator concern, driven out-of-band through the Operator Console (Phase 15) or the CLI (Slice 3); the MCP server and the operator meet at the `.kb/index.json` file on disk. Keeping the agent out of curated state fits the enterprise-KB governance posture (ingest is controlled and auditable), keeps the tool surface small (better tool selection), and makes the server safe to expose. The one write tool, `kb_save_hot`, writes working memory (`wiki/hot.md`), which is excluded from the Section Index corpus — it is not curated content.

### `stack` is a parameter; the switch is human-driven

Retrieval-stack selection is a tool parameter (`stack`: enum `wiki`|`rag`, default `wiki`), not separate per-stack tools. Both stacks are near-symmetric (`build_index` / `search` / `query` exist on both), so the adapter dispatches trivially and honors ADR-0002 (stacks stay independent — `rag` is called directly, never via the Gateway toggle). The switch is driven by an explicit user request ("answer using rag"); the model does not auto-select, so the `stack` description can stay simple.

### stdio transport; in-process index with mtime reload

The primary transport is stdio (Claude Desktop), launched via `python -m kb_mcp`; HTTP is deferred and near-free to add from the same FastMCP definitions. Because `indexer.search` reads an in-process module global populated once from `.kb/index.json`, the long-lived server reloads the index when that file's mtime changes — the same freshness mechanism the CLI REPL (Slice 3) uses.

### Action-verb tool names with a `_v1` suffix

Tools are named for the user's verb (`ask` / `search` / `read` / `save`) and carry a mandatory `_v1` suffix. Hosts cache the tool list per conversation, so renaming a tool later breaks cached schemas; versioning from the first release (like semver) means a breaking change ships as `_v2` alongside an untouched `_v1`. (`.` and `@` are invalid in tool names, so the suffix is rendered `_v1`, not `@v1`.)

## Considered Options

### MCP wraps the Gateway

Rejected. It reintroduces the redundant multi-turn machinery (the host already holds the context), forces an SSE→tool-result impedance match, and cannot expose `kb_search`. It would make MCP an adapter over an adapter.

### A read-write surface (`kb_ingest` / `kb_index` / `kb_lint`)

Rejected for Phase 12 (deferred to a future phase). Ingest and lint are LLM-heavy, long-running, and mutate `wiki/` concurrently with the operator; letting an agent free-write curated state conflicts with the governance posture. This is the fuller "persistent agent maintains the KB" vision and deserves its own grill (concurrency, long-task handling, write-conflict policy).

### Separate per-stack tools (`kb_chat_wiki` / `kb_chat_rag`)

Rejected. The requirement is a toggle, not two distinct concepts; separate tools double the surface and worsen tool selection.

## Consequences

- The agent is a **consumer**; the corpus is maintained out-of-band by the operator. The two meet at `.kb/index.json` — a clean decoupling and a usable interview narrative ("the Operator Console curates; the MCP agent consumes; they meet at the index file").
- Hot Cache (`kb_read_hot` / `kb_save_hot`) is the only write, and it targets working memory (`wiki/hot.md`), not the curated corpus. Read is a **tool** (agent-initiated — L0 of the read-depth budget), not a resource; a `kb://hot` resource for human inspection is deferred (resources do not bloat the tool list, so it can be added free later).
- `vector_rag` is reachable from MCP via the `stack` parameter, called directly per ADR-0002 — the comparison arm is exposed without coupling the stacks.
- Error rendering follows ADR-0015 (`LLMError` → structured `isError`). `Cannot Confirm` is a successful result carrying `grounding.reason`, never `isError`.
- Relates to: ADR-0002 (independent stacks), ADR-0010 (Gateway mounts both apps), ADR-0013 (cross-stack conversation store — why it is redundant here), ADR-0015 (transport-agnostic error contract).
