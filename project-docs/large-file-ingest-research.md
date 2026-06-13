# Research — large-file ingest: size + speed (industry approaches)

> **Date:** 2026-06-13 · **Method:** deep-research harness (5 angles, 21 sources fetched,
> 97 claims extracted, 25 adversarially verified → 18 confirmed / 7 killed).
> **Trigger:** planning PDF ingestion (manuals 100KB–2MB); the 256KB byte guard is too low,
> but raising it naively re-hits the speed/timeout wall (see
> [`large-file-ingest-size-limit-findings.md`](./large-file-ingest-size-limit-findings.md)).
> **Companion to** that findings doc (which covers the 256KB guard + the claude-obsidian
> server-side-synthesis contrast).

## Decisions locked (2026-06-13)

- **Fork 1 (long ingest over MCP): do BOTH (a) + (b).** (a) bounded-concurrency + `to_thread`
  for the per-section synthesis; (b) an async submit/poll job (`kb_ingest_start_v1` /
  `kb_ingest_status_v1`, the ADR-0017-anticipated shape). (a) makes reasonable manuals finish
  inside one sync call; (b) is the escape hatch for genuinely huge ones.
- **Fork 2 (size guard): classify-on-outline + replace the flat 256KB byte guard with a
  token/section-aware limit.** Bound the classify input so it no longer needs the whole file,
  then the per-Source ceiling can rise to cover 1–2MB manuals.
- **Fork 3 (PDF→Markdown extraction): DEFERRED.** Research did not deliver a verified
  extraction-tool comparison (see Gaps); needs its own focused pass before we choose a path.

## Verified findings (with sources)

### A. Size — hierarchical / map-reduce merging
- **Standard technique** for docs > one context window: chunk → summarize each chunk →
  recursively concatenate-and-merge chunk summaries until one remains. Common for >100K-token
  texts; implemented in LangChain. _(high; ACL 2025 [arXiv:2502.00977], ICLR 2024 BooookScore
  [arXiv:2310.00785])_
- **Faithfulness risk + fix:** plain recursive merging can amplify hallucination as intermediate
  summaries lose grounding; fix is **context-aware merging** — re-inject source context at merge
  stages via extractive summarization / BM25 retrieval / citations (Extract-Support best).
  _(high; arXiv:2502.00977, ACL Findings 2025.findings-acl.289)_ **Caveat: validated only on
  Llama-3.1 + legal/narrative domains, NOT gpt-4o-mini.** The stronger "merging cannot be
  faithful even with the best LLMs" framing was **refuted (1-2)** — risk is real but not absolute.

### B. Size — chunking practice
- Recursive-character **400–512 tokens, 10–20% overlap (50–100 tokens)** is the common default;
  tune **256–512 for factoid**, **1024+ for analytical**. _(query-type split 3-0; recursive
  default 2-1; Firecrawl/Weaviate tracing to Chroma + NVIDIA benchmarks)_ — treat exact figures
  as starting heuristics to validate on our own data.
- **Structural / heading-aware chunking** (Markdown by `#`/`##`, HTML by tags, code by funcs)
  preserves native structure vs arbitrary offsets. _(high; Weaviate + LangChain
  MarkdownHeaderTextSplitter etc.)_ — **this is exactly our existing per-Section unit.**
- ⭐ **For sparse/BM25 retrievers, chunk overlap gives ~zero benefit** (Jan-2026 analysis) —
  directly relevant: our active prototype is BM25 over Sections.

### C. Speed — Batch APIs
- **OpenAI Batch API and Anthropic Message Batches both give 50% off input AND output tokens**,
  async **<24h (often <1h)**. _(high; both vendors' official docs, 3-0)_
- The 24h is an **expiration ceiling, not a delivery SLA** (can "expire" with partial results
  under load). **Unfit for interactive, timeout-bounded MCP calls** — the decisive fact for the
  Batch-vs-sync fork.
- ⚠️ "10,000 queries per batch" was **refuted (1-2)** — do not rely on that figure.
- prompt-caching specifics (OpenAI auto 50% / Anthropic read-90%/write+25%) **did not survive
  verification** — cite official docs directly if needed, not these numbers.

### D. Long-running jobs over MCP — the key section
- MCP now has a first-class **async Tasks mechanism (SEP-1686)**: a request returns immediately
  with a durable **`taskId`**; client polls **`tasks/get`** and blocks on **`tasks/result`**.
  Lifecycle: submitted → working → input_required → completed/failed/cancelled/unknown.
  **Motivated by exactly our failure mode** (tools "expected to take minutes or more" exceeding
  host/transport timeouts). _(high; official MCP spec 2025-11-25 + SEP-1686 + changelog)_
- **`-32001` confirmed**: Claude Desktop cancels long MCP tool calls with "Request timed out"
  (~60s default / ~4-min Windows cap). _(high; MCP spec issues + claude-code issues #44032 etc.)_
- 🔴 **Caveat (moving target):** Tasks shipped *experimental* in 2025-11-25, and the **2026-07-28
  RC moved it OUT of core into an extension** after production feedback. Wire details (method
  names, `_meta` key, lifecycle) may change → **build our own submit/poll behind an abstraction**,
  don't bind to the raw protocol Tasks yet. (This is why Fork 1 chose a homegrown (b), not Tasks.)

### E. Dedup
- LlamaIndex `IngestionPipeline` does **hash-based per-document skip** (doc_id → document_hash) —
  but that is **re-ingestion dedup, which we already have** via `docs_body_hash`. Cross-chunk
  *semantic* dedup within one document was **not covered** by any surviving claim.

## Gaps (honest — not answered / unverified)

1. **PDF→Markdown extraction tool comparison NOT delivered** — no claim on PyMuPDF/pdfplumber/
   Marker/Docling/Unstructured/LlamaParse/Azure DI/Textract survived. Needs a dedicated pass.
2. Concrete bounded-concurrency numbers (semaphore limits), TPM/RPM backoff specifics, and
   prompt-caching numbers — unverified.
3. Cross-chunk **semantic** dedup within one source — unaddressed.
4. Model-tiering cost numbers — unverified.

## What this means for OUR system

The research **validates half our design**: our per-Section unit IS the recommended structural
chunking, and our `docs_body_hash` skip IS the LlamaIndex-style re-ingestion dedup. The real
problems narrow to three, each with a fix (the three we're implementing):

| Pain point | Now | Fix |
|---|---|---|
| `classify_source` sends the whole file | large doc → context overflow | **classify on outline / first-K sections** → input bounded regardless of doc size |
| concept path = N sequential per-section LLM calls | slow + blows MCP timeout | **bounded async concurrency + `to_thread`** (Fix 1a) |
| MCP host kills minute-long calls (-32001) | even concurrent ingest can exceed timeout | **async submit/poll job** `kb_ingest_start_v1`/`kb_ingest_status_v1` (Fix 1b) |
| entity path concatenates all sections into one call | large entity doc → overflow | route large docs to per-section, or map-reduce (D's hierarchical pattern) |

## Sources (verified subset)

- ACL 2025 hierarchical/context-aware merging — https://arxiv.org/pdf/2502.00977 ·
  https://aclanthology.org/2025.findings-acl.289
- ICLR 2024 BooookScore — https://arxiv.org/abs/2310.00785
- OpenAI Batch — https://developers.openai.com/api/docs/guides/batch
- Anthropic Message Batches — https://www.anthropic.com/news/message-batches-api ·
  https://platform.claude.com/docs/en/docs/build-with-claude/message-batches
- MCP Tasks (SEP-1686) — https://modelcontextprotocol.io/community/seps/1686-tasks ·
  https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks
- MCP -32001 / long tasks — https://mcpcat.io/guides/fixing-mcp-error-32001-request-timeout/ ·
  https://workos.com/blog/mcp-async-tasks-ai-agent-workflows
- Chunking — https://www.firecrawl.dev/blog/best-chunking-strategies-rag ·
  https://weaviate.io/blog/chunking-strategies-for-rag
- LlamaIndex dedup — https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline/
