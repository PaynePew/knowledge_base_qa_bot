# A separate /upload endpoint stages bytes; Import stays mechanical conversion

Phase 15's Operator Console drop zone needs to get browser file bytes onto the server. The existing `POST /import` ([CONTEXT](../../CONTEXT.md) **Import**) is defined as mechanical `raw/ → docs/` format conversion (no LLM) and reads files already present in the server-side `raw/` directory — it does not accept uploads. Rather than overload it, we add a new `POST /upload` that owns transport/staging only: it writes dropped bytes onto the server (`.html`/`.txt` → `raw/`, `.md` → `docs/`) and `/import` is left untouched.

This keeps the two concerns disjoint: **Upload** moves bytes onto the server; **Import** converts `raw/` files into normalized `docs/` Markdown. The documented pipeline (`raw/foo.html → /import → docs/foo.md → /ingest → …`) and the provenance frontmatter (`imported_from` points at the staged `raw/` file) stay intact.

## Considered Options

- **Make `/import` accept multipart uploads directly.** Rejected: conflates transport with format conversion, breaks the existing `raw/`-glob batch contract and the documented pipeline, and muddies the provenance model that assumes the raw file exists on disk.

## Consequences

- New `upload_*` Wiki Log kinds (`upload_batch_started` / `upload_file` / `upload_rejected` / `upload_error` / `upload_batch_completed`) for audit parity with the `import_*` family.
- Upload is a system boundary: path-traversal-safe filename handling, type allow-list (`.html`/`.txt`/`.md`), and size limits are enforced here (untrusted filenames + bytes).
- `.md` bypasses Import entirely (already canonical Markdown) and lands straight in `docs/` as an Ingest candidate.
