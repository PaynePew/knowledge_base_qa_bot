# raw/ — import inbox

This directory is the local import inbox for `POST /import`.

Drop `.html`, `.txt`, or `.md` files here, then call `POST /import` to convert
them into normalized Markdown docs under `docs/` with provenance frontmatter.

## Quick start

```bash
# Copy the bundled examples into the inbox
cp examples/raw/clean_article.html raw/
cp examples/raw/simple.txt raw/

# Or drop your own .html / .txt / .md files here, then run:
curl -X POST http://localhost:8000/import

# Single-file mode (process one file without touching the rest):
curl -X POST "http://localhost:8000/import?source=clean_article.html"
```

Converted files land in `docs/<basename>.md` with this frontmatter:

```yaml
---
imported_from: raw/clean_article.html
original_format: html
imported_at: '2026-01-01T00:00:00Z'
content_sha256: <sha256-of-raw-bytes>
---
```

## Supported formats

| Extension | Conversion |
|-----------|-----------|
| `.html`   | Markdownify with semantic whitelist |
| `.txt`    | Passthrough (no heading inference) |
| `.md`     | Passthrough (content preserved verbatim) |

## Gitignore behaviour

`raw/` contents are gitignored so user-dropped files stay local.
This `README.md` is the only committed exception — it makes the inbox
discoverable without committing your data.

Sample source files for testing live in `examples/raw/` (tracked).
