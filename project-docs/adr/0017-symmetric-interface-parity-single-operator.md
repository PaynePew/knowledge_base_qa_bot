# Symmetric interface parity: Browser, CLI, and MCP as independent equals over a shared on-disk corpus

ADR-0016 scoped the MCP server to a **read-only** corpus surface, justified by an enterprise-KB governance posture (ingest stays a controlled, auditable operator concern). This project's actual posture is a **single local operator**, which dissolves that justification. This ADR supersedes ADR-0016's read-only decision: the Browser (Operator Console), the CLI (`kb`), and the MCP server (`python -m kb_mcp`) become three independent, feature-equivalent interfaces over the same on-disk corpus. Each can drive the full curation lifecycle; none depends on another running.

## Decision

### Three independent interfaces, one shared corpus on disk

The three interfaces are peers — thin adapters over the same `markdown_kb` deep modules (ADR-0016's first decision, retained) that share state only through the filesystem (`raw/`, `docs/`, `wiki/`, `.kb/index.json`). No interface routes through another. ADR-0016's "the agent and the operator meet at `.kb/index.json`" generalises to **"all three interfaces meet at the filesystem."**

### The parity set is the lifecycle, not byte-ingress

Parity covers the four lifecycle operations — **Import, Ingest, Index, Lint** — each a trigger over on-disk state addressed by filename (read operations `ask` / `search` were already shared). What is explicitly NOT a parity target is *byte-ingress* — getting new bytes onto disk — which is inherently interface-shaped.

### Byte-ingress is interface-shaped

- **Browser** — Upload (`POST /upload`): a sandboxed browser cannot hand the server a local path, so it transports bytes into `raw/`.
- **CLI / MCP** (local processes) — pass a filesystem **path**; the server reads the file and converts it. This gives both full Import parity, including binary formats (PDF), with no byte transport.

Format conversion (`.html` / `.txt` / `.pdf` → `.md`) is **always** performed server-side by the Import deep module — no interface may ingest unconverted content. Reading arbitrary local paths is safe *only* under the single-operator posture; it would be a traversal / exfiltration risk in a multi-user deployment.

### Capture — the MCP agent authors a Source from its session

A new **MCP-only** operation: the agent composes a Markdown Source from its conversation and persists it to `docs/` (skipping Import — it is already canonical Markdown), then drives Ingest → Index. Capture exists only on MCP because only the agent holds content that lives nowhere on disk; the Browser and CLI reach `docs/` through a file. Captured Sources MUST carry provenance frontmatter (`origin: mcp-conversation`, `created_at`, `authored_by: agent`) so conversation-derived knowledge is distinguishable from authoritative external Sources — Ingest will otherwise faithfully promote chat content to retrievable "knowledge." This makes the agent a content **author**, the largest governance step here, defensible only under the single-operator posture; provenance is its mitigation. Filename validation reuses Upload's traversal-safe basename check.

### Concurrency — no cross-process lock now; recovery is documented

`indexer._index_lock` is a `threading.Lock` (in-process only), so it does not serialise the three separate processes. We accept this for now: the realistic concurrent case (two *different* Sources) risks only a stale `.kb/index.json` — wiki pages are atomic-written and orphan-deletion is per-Source-scoped, so no curated content is lost — and is repaired by re-running Index. Mitigation is a documented "re-run `kb index` after concurrent writes." A cross-process advisory lock (`filelock`, fail-fast, wrapping only the commit phase) is a **deferred, non-breaking hardening**: it changes no interface contract, so it is added only if the asynchronous MCP agent (an invisible second writer) actually causes trouble.

### Long-running ingest — synchronous, scoped per interface

- **CLI**: synchronous with per-Source progress output (a long block is normal for a CLI; the progress *is* the UX).
- **MCP**: **single-Source** ingest, synchronous, emitting MCP progress notifications to keep the host from timing out. No batch, no async job machinery — the agent curates one Source at a time and loops if needed (keeps the tool surface small, per ADR-0016).
- **Batch** (all of `docs/`) and **bulk / large** ingest live on the CLI and Operator Console, where blocking / streaming is natural.

Whether a single large Source still exceeds the Claude Desktop tool timeout despite progress notifications must be verified before implementation; if it does, an async start/poll pair for MCP ingest is a deferred fallback gated on that finding — not built up front.

## Considered Options

### Keep MCP read-only (ADR-0016 status quo)
Rejected. Its sole justification was an enterprise governance posture; the posture is a single local operator, so the justification no longer holds. The read-only surface also blocked the project's own "persistent agent maintains the KB" vision (the Hot Cache framing in CONTEXT.md).

### Build async job infrastructure (start / poll) for writes now
Rejected for now. Single-Source synchronous ingest with progress notifications covers the common case; async is a deferred fallback gated on an actual timeout finding. Building it up front adds tool surface and statefulness against YAGNI.

### Add a cross-process lock now
Rejected for now. The failure it prevents is recoverable (regenerable index) and rare for one operator; the lock is a non-breaking later addition, so paying its complexity before the risk materialises is premature.

### Route all writes through the Gateway
Rejected. It would make the CLI and MCP depend on the Gateway running — directly contradicting the "three independent systems" goal.

## Consequences

- The whole parity rests on the **single-operator posture**. If that ever changes, three decisions must be revisited together: arbitrary-path reads (Import), agent authorship (Capture), and the absent write lock.
- **Provenance frontmatter is the governance hook** that replaces the read-only restriction — it makes corpus origin *auditable* rather than *forbidden*.
- Three items are deferred-and-gated, each by a concrete trigger: `filelock` (agent write-collision), async MCP ingest (verified host timeout), `.pdf` Import (format demand).
- **Per-Source ingest does not dedupe slugs across calls.** `ingest_sources`' cross-Source slug-collision guard (`used_slugs`, #54 / CODING_STANDARD §12.8) is scoped to a single call, so the MCP single-Source path — and any per-Source CLI loop — gets no cross-call collision resolution: two separately-ingested Sources whose concept pages slug-collide overwrite. Accepted under the single-operator posture, but the ingest result must surface created-vs-overwritten pages so the overwrite is **visible, never silent** (preserving #54's intent); re-ingesting recovers. Resolving collisions against on-disk pages is the deeper fix, deferred.
- Relates to: ADR-0016 (**superseded** — read-only surface), ADR-0002 (independent stacks), ADR-0010 (Gateway mounts both apps), ADR-0011 (Upload separate from Import — Capture is now a third ingress alongside them), ADR-0015 (transport-agnostic error contract — the `WRITE_IN_PROGRESS` / `409 Conflict` shapes follow it).
