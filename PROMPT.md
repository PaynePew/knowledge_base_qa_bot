# Design a Knowledge Base Q&A Bot

## System Requirements

Build a Q&A bot over a small Markdown knowledge base:

- The repo provides sample `.md` documents in `docs/`
- The system builds an index from those documents
- The Markdown KB strategy should write an inspectable `.kb/index.json`
- The Vector RAG strategy should persist its FAISS index in `.kb/faiss_index/`
- Users ask questions through an API
- Answers must be grounded in the indexed documents
- Answers must cite sources using `filename#heading`
- If the knowledge base does not contain the answer, the system should say it cannot confirm

## Choose a Retrieval Strategy

You can solve this with either strategy:

### Strategy A: Markdown KB

```text
Markdown files -> heading sections -> section index -> BM25 keyword search -> raw Markdown context -> LLM answer
```

This is inspired by the Karpathy-style LLM knowledge base pattern: plain Markdown files, explicit indexes, and LLM-readable context instead of embeddings.

### Strategy B: Vector RAG

```text
Markdown files -> chunks -> embeddings -> vector search -> retrieved context -> LLM answer
```

This is the traditional RAG path: semantic retrieval with embeddings and a vector store.

## Design Questions

Answer these before you start coding.

### 1. Which retrieval strategy did you choose, and why?

**Markdown KB (Strategy A).**

| Reason | Detail |
|---|---|
| Scale | 3 docs, ~9 sections — no embedding ROI |
| Content shape | FAQ-style; user queries hit keywords directly (`refund`, `email`, `shipping`) — BM25's strength |
| Inspectability | `.kb/index.json` is plain text — `cat` to debug; FAISS is a binary blob |
| Citation alignment | `filename#heading-slug` falls out of headings naturally, no extra work |
| Zero external dependency at query time | No embedding API call; query latency = one chat completion |
| Aligns with the Karpathy LLM Wiki idea | The wiki is a readable, compounding artifact shared by humans and the LLM |

Vector RAG can be added later as a complement (see Q6), not as a replacement.

### 2. What is the retrieval unit in your design: file, section, or chunk?

**Section** (as the scaffold already encodes).

- **File** is too coarse: `account_help.md` contains three unrelated topics (email, password, deletion); feeding the whole file dilutes context.
- **Chunk** is too fine: splitting "Refund Timeline" across two chunks breaks both retrieval and citation.
- **Section** matches the author's intent, matches the citation format, and matches user query granularity.

`Section.id = "refund_policy.md#refund-timeline"` serves three roles at once: retrieval key, citation anchor, and dedup key.

### 3. How do you decide what goes into the prompt?

Fixed structure, **top-K = 3 sections**:

```
[System] Rules: use only CONTEXT; cite filename#heading;
         if context is insufficient, say "cannot confirm";
         no outside knowledge, no guessing.

[Human]
CONTEXT:

[Source: refund_policy.md#refund-timeline]
Heading: Refund Policy > Refund Timeline
Approved refunds are processed within 5-7 business days. ...

[Source: ...]
...

QUESTION:
How long do refunds take?
```

Principles:
- **Full section content**, no chunking — three sections fit comfortably in context.
- **`[Source: ...]` header before each section** so the LLM knows which one to cite.
- **`heading_path` included** (`Refund Policy > Refund Timeline`) — gives the LLM document structure and improves citation accuracy.
- **CONTEXT before QUESTION** — reduces the model's tendency to lead with world knowledge.
- **Scores are not sent to the LLM** (debug-only) — prevents the model from reasoning "low score, so guess."

### 4. How do you cite sources so users can inspect the original Markdown?

**Format: `filename#heading-slug`**, matching the verification cases in this PROMPT exactly.

- `refund_policy.md#refund-timeline` ← `## Refund Timeline`
- `account_help.md#change-email-address` ← `## Change Email Address`

Implementation: `slugify()` lowercases and replaces non-alphanumerics with `-` (already in the scaffold).
The `sources[]` returned in `ChatResponse` carries three fields per source:
- `source` (the citation id),
- `heading` (the human-readable heading path),
- `content[:240]` (a snippet preview, so the client UI does not need a second fetch).

### 5. What should happen when retrieval finds weak or irrelevant results?

**Three fallback layers**, in order:

| Condition | Behavior |
|---|---|
| `sections == []` (index not built) | Return "The knowledge base has not been indexed yet. Call POST /index first." |
| `ranked == []` (no positive score) | Return "I cannot confirm from the knowledge base." |
| `top_score < threshold` (matches too weak) | Same fallback, **and do not call the LLM** — saves the API call and prevents confabulation |

The threshold is calibrated against the corpus (BM25 scores have no absolute scale). For this dataset, anything below ~1.0 is effectively noise.

**Key principle:** never hand weak context to the LLM and hope it says "cannot confirm" — once context is present, models tend to hallucinate around it. Block the weak case before the LLM call.

The third verification query, "Which restaurants are nearby?", should score near zero against all three docs and trigger the cannot-confirm path directly.

### 6. When would you switch from Markdown KB to Vector RAG?

Not "switch" — **layer on top**. Triggers:

1. **Semantic misses pile up** — users ask "money back" but docs say "refund"; BM25 misses.
2. **Cross-language queries** — Chinese question against English docs; BM25 is monolingual.
3. **Sections grow long** — a single section over ~1,500 words contains multiple sub-topics; semantic search inside the section becomes useful.
4. **Heading structure becomes unreliable** — ingesting transcripts, Slack threads, OCR'd PDFs without clean headings.
5. **Synonym/abbreviation explosion** — `API key` / `token` / `auth credential` mean the same thing but BM25 does not know.

Add a vector layer as **fallback or rerank**, not as replacement. This matches the @Eyaldavid7 tournament finding in the Karpathy gist discussion: **Wiki + RAG beats Wiki-only and RAG-only**.

### 7. When would you switch from Vector RAG back to a Markdown index?

1. **Debugging wrong answers** — embeddings are opaque; a Markdown index is one `cat` away from a diagnosis.
2. **Citation precision complaints** — chunks break at arbitrary boundaries; sections do not.
3. **80%+ of queries hit entity names or exact keywords** — embeddings drift to "similar but wrong"; BM25 wins on exact match.
4. **Cost pressure** — every query embeds; at moderate traffic this dominates the bill.
5. **You want to give the LLM the structure** — Markdown headings are a schema; a vector store has none.
6. **Users need to edit or share the KB** — Markdown is universal; a vector index is locked to one library.

Rule of thumb: **if the LLM can answer by reading the wiki structure directly, use BM25; only reach for vector search when you need to guess the user's intent.**

### 8. If the knowledge base grows from 10 files to 100,000 files, what changes?

| Layer | 10 files | 100,000 files |
|---|---|---|
| **Index storage** | `.kb/index.json` fully in memory | Persistent inverted index (Tantivy / Whoosh / Elasticsearch / Meilisearch); sharded |
| **Search** | Score BM25 over all sections | Inverted index required; O(N) scan is no longer viable |
| **Loading** | Load everything at startup | Lazy load; partition by topic or date |
| **Re-ranking** | Top-3 straight to the LLM | BM25 top-50 → cross-encoder rerank → top-3 → LLM |
| **Index updates** | Full rebuild (sub-second) | Incremental updates; stable `source_id → section_id` mapping is mandatory |
| **`index.md` navigation** | LLM reads the entire index | Blows the context window — must become hierarchical: top-level lists categories, each category has its own sub-index |
| **Embeddings** | Not needed | Hybrid (BM25 + vector rerank, like `qmd`) becomes necessary |
| **Schema (CLAUDE.md)** | 30 lines is enough | Becomes critical — naming conventions, section templates, lint rules, entity-type registry |
| **Lint / health checks** | Manual | Must be automated (cron); contradiction detection, orphan pages, stale claims |
| **Citation** | `filename#heading` | Same format — but doc ids must be stable across renames and restructures |
| **Storage** | `docs/` + `.kb/` on local disk | `docs/` to object storage; index to a dedicated search service |
| **Multi-user / multi-agent writes** | Not a concern | Typed links, conflict resolution, possibly an OS-level memory manager (per @plundrpunk's 12-month report in the Karpathy gist discussion) |

**Critical threshold** (community consensus in the gist comments): around **~200 pages**, `index.md` starts to blow context — this is the first bottleneck to plan for.

## Verification

Before running the server, set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Both strategies use OpenAI for final answer generation. Vector RAG also uses OpenAI embeddings during `/index` and for each `/chat` query.

Your prototype should pass all of these:

```bash
# Health check
curl http://localhost:8000/health
# -> 200, {"status": "ok"}

# Chat before indexing
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do refunds take?"}'
# -> 200, should indicate the knowledge base has not been indexed yet

# Build the index from docs/*.md
curl -X POST http://localhost:8000/index
# -> 200, returns {"files_indexed": N, "sections_indexed": M}

# Markdown KB only: inspect the generated section index
cat .kb/index.json

# Markdown KB only: restart the server, then ask again without POST /index
# -> should load .kb/index.json on startup

# Vector RAG only: inspect the persisted FAISS index metadata
cat .kb/faiss_index/metadata.json

# Vector RAG only: restart the server, then ask again without POST /index
# -> should load .kb/faiss_index/ on startup

# Ask a question answered by the docs
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do refunds take?"}'
# -> 200, answer cites refund_policy.md#refund-timeline

# Ask another grounded question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I change my email address?"}'
# -> 200, answer cites account_help.md#change-email-address

# Ask an out-of-scope question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Which restaurants are nearby?"}'
# -> 200, answer should say it cannot confirm from the knowledge base
```

## Suggested Tech Stack

Python + FastAPI is recommended, but Challenge Track students may use any language or framework.

## Stretch Goals

Pick one or more after the core `/index` and `/chat` flow works.

### Score Threshold and Fallback

Add a retrieval score threshold. If the best sections or chunks are too weak, return an honest cannot-confirm answer instead of forcing a citation.

### Streaming Interface

After `/chat` works, add:

```text
POST /chat/stream
```

Use SSE to stream the answer token by token. A good streaming response should:

- Return selected sources first, so users can see what context the bot is using
- Stream answer tokens as they arrive
- End with a clear `done` event
- Preserve the same grounding and citation rules as `/chat`

Optional UI challenge: build a tiny HTML page that calls `/chat/stream` and renders the answer incrementally.

### Browser UI

Build a tiny browser UI over `/chat` or `/chat/stream`. Show selected sources before the answer so users can inspect grounding.

### Multi-Format Import

Add a small normalization pipeline before indexing:

```text
raw/*.txt or raw/*.html -> docs/*.md -> POST /index -> retrieval index
```

Requirements:

- Keep Markdown as the canonical knowledge format
- Preserve the original source filename
- Convert headings into Markdown headings
- Rebuild the retrieval index after import

Start with `.txt` or `.html`. More complex formats such as PDFs, spreadsheets, and transcripts can be discussed as production extensions.

### Alternative Interfaces

Expose the same retrieval core through another interface:

```text
CLI: kb index / kb ask
MCP: expose index, search, and chat as agent tools
Web UI: simple chat screen over /chat or /chat/stream
```

The goal is to compare interface tradeoffs, not to change the retrieval design.

### Wiki Index Generation

Generate `wiki/index.md` from `.kb/index.json` so humans and agents can browse the available topics.

### Answer Filing

Write useful Q&A results back into `wiki/` after review. Preserve citations back to the source Markdown sections.

### Conversation Memory

Add short conversation memory for follow-up questions. Memory can help interpret the query, but retrieved sources must still control the final answer.

### Paraphrase Comparison

Create paraphrased queries and compare Markdown KB vs Vector RAG. Look for synonym misses, semantic false positives, and citation quality.
