# Why a Curated Wiki Layer? Industry Rationale and Measured Evidence vs Plain RAG

> Research note, 2026-07-23. Primary-source survey of the rationale for LLM-maintained
> curated-synthesis layers (Karpathy-style wiki, GraphRAG-style derived summaries) over
> immutable raw sources, and the measured evidence that such layers beat plain
> RAG-over-raw-chunks. Every claim carries its source; each is marked **MEASURED**
> (numbers in a paper/benchmark) or **ARGUED** (rationale only, no measurement).
> Companion file: `literature.md` (separate agent, separate scope).

---

## 1. The pattern's origin: Karpathy's LLM Wiki (primary source)

**Source:** gist <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
(`llm-wiki.md`, published 2026-04-04), announcing tweet
<https://x.com/karpathy/status/2039805659525644595> (2026-04-02).

Karpathy's own stated rationale (verbatim quotes from the gist):

- The problem with query-time RAG: with RAG, *"the LLM is rediscovering knowledge from
  scratch on every question. There's no accumulation."* — *"Ask a subtle question that
  requires synthesizing five documents, and the LLM has to find and piece together the
  relevant fragments every time. Nothing is built up."*
- The alternative: *"the LLM incrementally builds and maintains a persistent wiki — a
  structured, interlinked collection of markdown files"*; *"the wiki is a persistent,
  compounding artifact."*
- What compounds: *"The cross-references are already there. The contradictions have
  already been flagged. The synthesis already reflects everything you've read."*
- Why it is feasible now: *"The tedious part of maintaining a knowledge base is not the
  reading or the thinking — it's the bookkeeping… The LLM stays maintained because the
  cost of maintenance is near zero."* Human role: *"curate sources, direct the
  analysis, ask good questions."*
- Prescribed structure: three layers — raw sources (immutable, never modified by the
  LLM), the wiki (LLM-maintained markdown), the schema (CLAUDE.md governing workflows);
  special files `index.md` (catalog, updated on every ingest) and `log.md`
  (append-only chronology); operations ingest / query / lint (lint = health-check for
  contradictions, orphans, stale claims).
- His own caveat: the document is *"intentionally abstract"* — *"pick what's useful,
  ignore what isn't."*

From the announcing tweet (text as reproduced in search-result snippets of the X
status; not re-verified against x.com directly): *"Something I'm finding very useful
recently: using LLMs to build personal knowledge bases for various topics of research
interest… I use an LLM to incrementally 'compile' a wiki, which is just a collection of
.md files in a directory structure."* Note his verb: **compile** — the wiki is a
compilation artifact, the chat is the interface.

**Status: ARGUED.** Karpathy presents zero measurements. The gist is a design pattern
plus personal experience report. Its evidentiary weight is "practitioner rationale from
a credible source", not evidence.

## 2. claude-obsidian (reference implementation)

**Source:** <https://github.com/AgriciDaniel/claude-obsidian> README (fetched
2026-07-23). Explicitly *"Based on Andrej Karpathy's LLM Wiki pattern"* with a link to
the gist.

Claimed value (verbatim from README): *"Knowledge compounds like interest."* — *"Every
source you add gets integrated. Every question you ask pulls from everything."* —
*"The wiki gets richer with every ingest."* — *"Your wiki stays healthy without manual
cleanup."* — *"The next session starts with full recent context, no recap needed."* —
positioning: *"Most Obsidian AI plugins are chat interfaces… claude-obsidian is a
knowledge engine."*

Measurement: exactly one, and it is **not** wiki-vs-RAG. The README reports a 50-query
retrieval benchmark: *"+32 percentage points top-1 accuracy and +41 percent error
reduction"* — but that compares **v1.7 of its own hybrid retrieval against its v1.6
baseline**. It measures the tool's retrieval pipeline improving, not the curated layer
beating RAG-over-raw-chunks. No such comparison is presented anywhere in the README.

**Status: ARGUED** for the wiki-layer value proposition (the one number is an internal
self-comparison; self-reported, methodology not published). Star count ("5.4K") not
independently verified in this pass.

## 3. Microsoft GraphRAG — the strongest measured analogue

**Source:** Edge et al., "From Local to Global: A Graph RAG Approach to
Query-Focused Summarization", arXiv:2404.16130
(<https://arxiv.org/abs/2404.16130>, full text v2 HTML). GraphRAG is exactly a derived
synthesis layer: LLM-extracted entity graph + **pre-written community summaries** built
offline over raw docs — the closest peer-reviewed relative of a curated wiki.

- **Query class it wins — MEASURED:** global sensemaking questions ("What are the main
  themes in the dataset?"), where the paper says plain RAG *"fails"* because no chunk
  contains the answer. On two ~1M-token corpora (podcast transcripts: 1,669 × 600-token
  chunks; news: 3,197 × 600-token chunks; §4.1.1), LLM-judged head-to-head win rates vs
  vector RAG (Fig. 2 / Table 6, §5.1): **comprehensiveness 72–83%** (podcast, p<.001)
  and 72–80% (news); **diversity 75–82%** (podcast) and 62–71% (news, p<.01).
- **Query-time token efficiency — MEASURED:** root-level community summaries (C0)
  needed **9×–43× fewer context tokens** than map-reduce over source text (2.6% of the
  token budget; Table 2), *"for a modest drop in performance… a highly efficient
  method"* (§5.1). This is the strongest quantified form of the wiki argument "pay the
  synthesis cost once at ingest, answer cheaply forever after".
- **What it concedes — MEASURED/DOCUMENTED:** (a) vector RAG **wins directness**
  across all comparisons (§3.3: directness is "effectively in opposition to
  comprehensiveness and diversity"); (b) index construction is expensive — 281 minutes
  of LLM calls for the podcast corpus (§4.1.3).
- **Evaluation caveat (from the paper's own method, §3.3):** wins are LLM-as-judge
  pairwise preferences on comprehensiveness/diversity/empowerment, not ground-truth
  accuracy. Treat "wins global sensemaking" as measured-by-preference, not
  measured-by-correctness.

## 4. RAPTOR — recursive summaries as retrieval units

**Source:** Sarthi et al., arXiv:2401.18059 (<https://arxiv.org/abs/2401.18059>).

Builds a tree of LLM-written recursive summaries over chunks and retrieves from all
abstraction levels, against the limitation that plain RAG methods *"retrieve only
short contiguous chunks… limiting holistic understanding of the overall document
context"* (abstract).

**MEASURED:** *"coupling RAPTOR retrieval with the use of GPT-4, we can improve the
best performance on the QuALITY benchmark by 20% in absolute accuracy"*; state-of-
the-art results also reported on NarrativeQA and QASPER — tasks the abstract
characterizes as requiring "complex, multi-step reasoning" over long documents. Direct
evidence that **LLM-derived summary artifacts, retrieved alongside raw chunks, beat
raw-chunk-only retrieval** on holistic/multi-step questions.

## 5. Dense X Retrieval — synthesis at the retrieval-unit level

**Source:** Chen et al., arXiv:2312.06648 (<https://arxiv.org/abs/2312.06648>, v2 HTML).

LLM-rewritten **propositions** (atomic, self-contained, decontextualized facts) as the
index unit instead of raw passages — curation of the corpus into a derived form.

**MEASURED** (five open-domain QA sets: NQ, TriviaQA, WebQ, SQuAD, EntityQuestions,
over Wikipedia): Recall@5 **+12.0 / +9.3 points** for unsupervised retrievers vs
passage units; 17–25% relative Recall@5 gain on EntityQuestions for supervised ones;
downstream QA **+4.9 to +7.8 EM@100-words** across six retrievers — i.e., better
answers **at a fixed token budget**, the same token-efficiency axis GraphRAG measures.

## 6. STORM, MemGPT, Anthropic memory tool

- **STORM** (Shao et al., arXiv:2402.14207, <https://arxiv.org/abs/2402.14207>):
  LLM-written Wikipedia-style articles from retrieval. **MEASURED:** on FreshWiki,
  **+25% absolute** in articles judged well-organized and **+10%** in coverage vs a
  retrieval-augmented baseline. **Criticism from the same paper:** experienced
  Wikipedia editors flagged **source-bias transfer** into generated articles and
  **over-association of unrelated facts** — direct published evidence that an LLM
  synthesis layer can amplify bias and fabricate connections.
- **MemGPT** (Packer et al., arXiv:2310.08560, <https://arxiv.org/abs/2310.08560>):
  OS-style memory tiers with self-editing persistent memory; demonstrates document
  analysis beyond the context window and multi-session agents that "remember, reflect,
  and evolve". **Status: demonstrated qualitatively in-paper; no wiki-vs-RAG numbers
  extracted in this pass** — treat as ARGUED for our purpose.
- **Anthropic memory tool** (<https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool>):
  first-party rationale for an agent-maintained file store: *"building up knowledge
  over time without keeping everything in the context window"*; *"Rather than loading
  all relevant information up front, an agent records what it learns in memory files
  and reads them back on demand."* Use cases named: maintain project context across
  sessions, apply lessons from past interactions, *"build up a knowledge base over
  time."* **Status: ARGUED** — the docs cite no measurements. Notable: Anthropic ships
  the same three moves as Karpathy's gist (persistent markdown-ish files, agent-run
  upkeep, read-on-demand), which shows industry convergence on the pattern, not proof
  it beats RAG.

## 7. Criticism and negative results (the other side)

- **RAG vs. GraphRAG: A Systematic Evaluation** (Han et al., arXiv:2502.11371,
  <https://arxiv.org/abs/2502.11371>, v3 HTML) — **MEASURED, both directions:**
  - Plain RAG *"excels on detailed single-hop queries"*: 55.28% on NovelQA
    detail-oriented subsets, beating all GraphRAG variants (§4.2).
  - GraphRAG variants win multi-hop: HippoRAG2 70.27% vs RAG 67.02% on MultiHop-RAG
    (§4.1).
  - **Costs (Table 4, §4.6):** graph construction 7,702 s vs RAG's 135 s (~57×) on
    MultiHop-RAG; highest retrieval latency for KG-GraphRAG; larger storage.
  - **Hallucination risk in the summary layer:** Community-GraphRAG (Global) scored
    only 25% on "Null" (unanswerable) queries with iterative retrieval — the derived
    layer made the system *more* willing to answer questions with no answer (§4.3).
  - Verdict: *"complementary strengths rather than a clear dominance"*; combining both
    gained +6.4% on MultiHop-RAG (§4.5).
- **LazyGraphRAG** (Microsoft Research blog,
  <https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/>)
  — Microsoft's **own concession** on the cost of pre-built synthesis: LazyGraphRAG's
  index costs are *"identical to vector RAG and 0.1% of the costs of full GraphRAG"*
  (i.e., full GraphRAG's upfront LLM summarization is ~1000× a vector index), while
  deferring all LLM work to query time achieves *"comparable answer quality to
  GraphRAG Global Search"* at ">700× lower query cost" vs that search mode. Lesson:
  the *owner* of the flagship curated layer found that most of its upfront
  summarization spend can be avoided — eager whole-corpus synthesis is not
  automatically worth it.
- **UnWeaver** ("UnWeaving the knots of GraphRAG — turns out VectorRAG is almost
  enough", arXiv:2603.29875, <https://arxiv.org/abs/2603.29875>) — **MEASURED
  (per abstract):** *"VectorRAG performs better than standard GraphRAG and almost as
  good as current SOTA graph-based solutions, for a fraction of the cost"*; blames
  GraphRAG's "orders of magnitude increased componential complexity".
- **Staleness:** no paper in this pass directly measures derived-layer staleness; it is
  implied by the construction-cost numbers above (any source change re-incurs part of
  the 57×/1000× build cost) and by Karpathy's own inclusion of a "lint" pass for
  *"stale claims"* in the gist — i.e., the pattern's author designs for staleness as an
  expected failure mode. **Status: ARGUED/INFERRED.**

---

## 8. Synthesis

### (a) The industry's claimed value axes for a curated layer

1. **Global / sensemaking / multi-hop questions** whose answer lives in no single chunk
   (GraphRAG abstract; RAPTOR abstract; Karpathy's "synthesizing five documents").
2. **Query-time token efficiency** — pay synthesis cost once at ingest, answer from
   compact artifacts (GraphRAG Table 2; Dense X fixed-budget EM; Anthropic "without
   keeping everything in the context window").
3. **Compounding accumulation** — knowledge, cross-references, and flagged
   contradictions persist and grow instead of being re-derived per query (Karpathy
   gist; claude-obsidian README; Anthropic memory docs; MemGPT).
4. **Dedup / contradiction control and human auditability** — the wiki is a readable,
   diffable, lintable artifact a human can inspect and correct (Karpathy's lint
   operation; claude-obsidian "stays healthy"; our layered design's immutable Sources
   mirror the gist's raw/ layer).
5. **Better retrieval units** — synthesized, self-contained units retrieve better than
   arbitrary raw chunks (Dense X propositions; RAPTOR summaries).

### (b) Which axes are MEASURED vs only ARGUED

| Axis | Status | Evidence |
|---|---|---|
| Global/sensemaking QA | **MEASURED** (LLM-judged preference) | GraphRAG 72–83% comprehensiveness win (2404.16130 Fig.2/T6) |
| Multi-hop QA | **MEASURED** | RAPTOR +20% abs. QuALITY (2401.18059); HippoRAG2 70.27 vs 67.02 (2502.11371 §4.1) |
| Token efficiency at query time | **MEASURED** | GraphRAG 9×–43× fewer tokens (T2); Dense X +4.9–7.8 EM at fixed budget (2312.06648) |
| Better retrieval units via synthesis | **MEASURED** | Dense X Recall@5 +12.0/+9.3; RAPTOR SOTA on NarrativeQA/QASPER |
| Article-level organization/coverage | **MEASURED** (with editor-flagged defects) | STORM +25%/+10% (2402.14207) |
| Compounding cross-session memory | **ARGUED only** | Karpathy gist, claude-obsidian README, Anthropic memory docs — zero head-to-head numbers found |
| Contradiction control / dedup | **ARGUED only** | Karpathy lint op; no measurement found |
| Human auditability | **ARGUED only** | inherent to markdown artifact; no study found |

### (c) Where plain RAG is documented to win

- **Directness of answers** — GraphRAG's own eval (2404.16130 §3.3).
- **Single-hop, detail-oriented factoid lookup** — RAG 55.28% NovelQA detail subset,
  beating all GraphRAG variants (2502.11371 §4.2).
- **Build cost, latency, storage** — 135 s vs 7,702 s construction (2502.11371 T4);
  vector index = 0.1% of full-GraphRAG index cost (LazyGraphRAG blog).
- **Not hallucinating on unanswerable queries** — global summary layer hit 25% on Null
  queries (2502.11371 §4.3): summaries can amplify overconfidence.
- **Freshness** — inferred from build-cost asymmetry, not directly measured (see §7).

### (d) The honest one-paragraph "why wiki" for an interview

> "The measured evidence says the two approaches are complementary, not that one kills
> the other. Plain RAG demonstrably wins single-hop factoid lookup, directness,
> freshness, and build cost — a systematic evaluation (arXiv:2502.11371) puts graph/
> summary construction at ~57× RAG's build time and shows RAG ahead on detail queries.
> But the same literature shows a derived synthesis layer wins exactly where raw chunks
> structurally can't: Microsoft's GraphRAG (arXiv:2404.16130) wins 72–83% of
> comprehensiveness judgments on corpus-level sensemaking questions and answers them
> with 9–43× fewer query-time tokens from pre-built summaries; RAPTOR
> (arXiv:2401.18059) adds +20% absolute on QuALITY by retrieving recursive summaries;
> Dense X (arXiv:2312.06648) shows synthesized retrieval units beat raw chunks at a
> fixed token budget. Our wiki layer is the Karpathy-pattern instantiation of that
> measured idea — 'compile' sources once into a curated, cross-referenced layer so
> multi-document synthesis is paid at ingest, not per query — with two properties the
> papers' graph indexes lack: it's human-auditable markdown, and Sources stay immutable
> so every synthesized claim remains traceable. I'd be equally upfront about the
> honest caveats: the 'knowledge compounds' benefit itself is argued by Karpathy,
> claude-obsidian, and Anthropic's memory-tool docs but not yet benchmarked
> head-to-head; STORM's editors documented bias transfer and fact over-association in
> LLM synthesis; and Microsoft's own LazyGraphRAG showed eager whole-corpus
> summarization can be 1000× costlier than necessary — which is why we keep BM25 over
> raw Sections as the factoid path and treat the wiki as a curated layer on top, not a
> replacement."

---

## Not verified / open doubts

- Karpathy tweet text taken from search-result reproductions of the X status
  (x.com not fetched directly); gist quotes came from a direct fetch but via an
  extraction pass — wording believed faithful, worth a manual read before quoting in
  print.
- claude-obsidian star count and its 50-query benchmark methodology not independently
  verified; the benchmark is v1.7-vs-v1.6 internal, not wiki-vs-RAG.
- GraphRAG "empowerment" metric results and MemGPT's quantitative tables not
  extracted (abstract-level only).
- No paper found (searched, not exhaustively) that benchmarks a *Karpathy-style
  markdown wiki* head-to-head against plain RAG; the measured evidence is all from
  structural analogues (graph summaries, summary trees, propositions).
