# Knowledge Base Q&A Bot

**English** · [繁體中文](#繁體中文)

A grounded Q&A bot over a small Markdown knowledge base. Every answer is built
from cited records — if the sources don't support a claim, the bot says
_"I cannot confirm"_ instead of guessing. It ships **two** retrieval stacks you
can compare side by side: a **Wiki** stack (BM25), whose design follows
[Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
and a **Vector RAG** stack (FAISS), both served from one browser UI.

## The interface

**Reader** — ask a question, pick a stack, read a grounded answer.

![Reader UI](project-docs/screenshots/reader.png)

**Operator Console** — the admin interface: upload documents, run the build
pipeline, and keep your company or personal knowledge base healthy.

![Operator Console](project-docs/screenshots/console.png)

## Quick start

The retrieval indexes ship **pre-built and committed**, so a fresh clone answers
questions immediately — no build step needed on the first run.

```bash
# 1. Clone
git clone https://github.com/PaynePew/knowledge_base_qa_bot.git
cd knowledge_base_qa_bot

# 2. Install (single .venv at the repo root)
uv sync --all-packages

# 3. Add your OpenAI key
cp .env.example .env        # then edit .env and set OPENAI_API_KEY=sk-...

# 4. Launch the Gateway (serves the UI and both stacks on one origin)
uv run uvicorn gateway.app.main:app --port 8000
```

Open <http://localhost:8000/> for the Reader and <http://localhost:8000/console>
for the Console.

> **Keys.** `OPENAI_API_KEY` is **required** — both stacks call OpenAI to write
> the final answer. `ANTHROPIC_API_KEY` is **optional** and only used by the
> evaluation's Claude judge. See [`.env.example`](.env.example).

## Using the Reader

- **Ask anything** in the box (English _or_ Chinese) and press Enter.
- **Toggle Wiki / RAG** to run the same question against either retrieval stack.
- **Sources appear first** — the records the answer is grounded in — then the
  answer streams in below them.
- A **grounding badge** confirms every claim traces back to a cited source; if
  it can't, you get _"I cannot confirm"_ rather than a hallucination.
- **Follow-up questions** continue the same conversation (multi-turn).

Both files and questions can be in **English or Traditional/Simplified Chinese** —
the Wiki stack tokenises CJK text and keeps Chinese answers in Chinese.

## Using the Console

The Console is where you grow and maintain a personal or company knowledge base.
Before the Wiki stack can retrieve a file, it must pass through the five steps of
the **pipeline stepper**, each with its own **Run** button.

**A file's journey — Upload → Import → Ingest → Index:**

| Step       | What it does                                                                      |
| ---------- | --------------------------------------------------------------------------------- |
| **Upload** | Drag-drop files. `.html` / `.txt` land in `raw/`; `.md` goes straight to `docs/`. |
| **Import** | Convert `raw/` sources into clean Markdown in `docs/` (with provenance).          |
| **Ingest** | An LLM synthesises `docs/` Sources into curated `wiki/` pages.                    |
| **Index**  | Build the BM25 search index so the new content is answerable.                     |

The **RAG track** (separate panel): a new file becomes retrievable only after
indexing — click **Rebuild** to rebuild the vector index from `docs/`,
independent of the Wiki chain.

**When to run Lint — and what it gives you:**

Run **Lint** after editing the corpus, or periodically, to audit the knowledge
base's health. It surfaces problems you can't see by eye — orphan pages, broken
`[[wikilinks]]`, contradictions between pages, stale pages, and coverage gaps.

Lint findings feed the **Curation Queue**, where you review the bot's
auto-filed Q&A drafts — read the actual question and answer content — and either
**Promote** a good one to a permanent page or **Discard** it. This is how
high-quality answers get folded back into the knowledge base over time.

The **Resource Browser** at the bottom lets you read through `docs/`, `raw/`,
and `wiki/` without leaving the page.

## Project structure

```
knowledge_base_qa_bot/
├── gateway/          # Gateway app + browser UI (Reader at /, Console at /console) — start here
├── markdown_kb/      # Wiki stack: BM25 over a curated wiki/ layer
├── vector_rag/       # RAG stack: chunk + embed docs/ into FAISS
├── docs/             # Sources — the bot's runtime knowledge base
├── wiki/             # Curated layer written by Ingest (concepts, entities, filed Q&A)
├── raw/              # Local inbox for uploaded .html/.txt before Import
├── .kb/              # Pre-built retrieval indexes (committed demo seed)
├── eval/             # Wiki-vs-RAG paraphrase comparison harness + report
├── project-docs/     # ADRs, roadmap, coding standard, screenshots
├── CONTEXT.md        # Shared vocabulary (glossary)
└── .env.example      # Copy to .env and add your OpenAI key
```

## Evaluation: Wiki vs RAG

**Test set.** 260 queries over one 20-Source / ~51-Gold-Section corpus:
**250 Core paraphrases** (5 LLM-generated rewrite types × 50) plus **10
hand-written structural probes** (2 types × 5). A "hit" requires the retrieved
unit to match the gold section _and_ share content key-tokens, so a
right-document-wrong-content result counts as a miss.

**Cross-platform L2 check.** The deterministic metric's edge-case verdicts are
re-judged by a _different model family_ — **Claude (`claude-sonnet-4-6`)**, not
the OpenAI family that powers RAG's embeddings — so the second opinion shares no
blind spot with the stack it checks. 207 ambiguous items were re-judged.

**Statistical honesty.** On the Core types, no per-type difference reaches
statistical significance after a paired McNemar test with Holm correction. The
structural probes are only **n=5 each** — still too small to support any
statistical inference — so they are reported as descriptive _expected-limit
confirmation_, never averaged into a headline number.

**What the results show:**

- **Synonym misses (Wiki's weak spot).** When a query uses vocabulary absent
  from the source, keyword BM25 can miss where vector similarity matches:
  synonym_swap Wiki 0.90 vs RAG 0.94, with the unseen-jargon probe the extreme
  (Wiki 0.40 vs RAG 1.00).
- **Semantic false positives (RAG's weak spot).** The metric scores a "hit" when
  RAG returns the right document with only weak content overlap; the cross-family
  Claude judge localised exactly these correct-id / weak-content hits, which
  flatter RAG's raw numbers.
- **Citation quality (Wiki's structural edge).** Wiki cites a stable
  `filename#heading` with `sources:` provenance that is grounding-checked at
  ingest; RAG cites raw chunks by similarity score, usually losing section
  boundaries.

**How they should perform, objectively** ([`why-wiki.md`](project-docs/why-wiki.md)).
The two differ in _when_ synthesis happens — Wiki synthesises once at ingest (a
compounding, auditable artifact — this is where its cost is paid); RAG re-derives
every query from raw chunks. The expected verdict is therefore scale-dependent: **under ~1000 pages → Wiki**
(cheap index navigation, zero per-query embedding cost, structured provenance);
**over ~100K pages → RAG** (the index outgrows full-scan navigation); **in
between → hybrid**. This corpus sits squarely in Wiki's regime — which is exactly
what the data shows: a statistical near-tie that Wiki reaches with simpler,
cheaper, more auditable machinery.

![Close on natural paraphrases](eval/paraphrase_comparison/charts/core_hit_rate_at_3.png)

![Probes expose each architecture's limit](eval/paraphrase_comparison/charts/probes_hit_rate_at_3.png)

Full methodology, statistical tests, cost log, and honest limitations are in
[`eval/paraphrase_comparison/report.md`](eval/paraphrase_comparison/report.md).
To regenerate or extend the corpus and re-run the comparison, see the
maintainer runbook at
[`eval/paraphrase_comparison/README.md`](eval/paraphrase_comparison/README.md).

## CLI and MCP command reference

The CLI (`kb`) and the MCP server (`kb_mcp`) drive the same full lifecycle.

### CLI subcommands (`uv run kb <subcommand>`)

| Command | Description |
| --- | --- |
| `kb ask <question>` | Ask a question and print a grounded answer (LLM synthesis + Grounding Check). |
| `kb index` | (Re)build the BM25 Section Index from the wiki corpus and persist it to `.kb/index.json`. |
| `kb import <path>` | Import a local file (`.html`, `.txt`, `.md`) into `docs/` via format conversion. |
| `kb ingest [source]` | Synthesise one named `docs/` Source (or all Sources when omitted) into `wiki/` pages. |
| `kb lint` | Run the Lint Pass health check (orphans, contradictions, stale pages, coverage gaps) and print findings. |
| `kb` (bare) | Enter the interactive REPL with a warm index; supports `:stack <wiki\|rag>` and `quit`. |

### MCP tools (`kb_mcp` server)

| Tool | Description |
| --- | --- |
| `kb_ask_v1` | Ask a question and receive a grounded LLM answer with citations and a Grounding Check result. |
| `kb_search_v1` | Retrieve raw Sections or Chunks from the index with no LLM synthesis; returns BM25 scores for the wiki stack. |
| `kb_read_hot_v1` | Read the working-memory hot cache (`wiki/hot.md`); returns `""` on the first session. |
| `kb_save_hot_v1` | Persist a working-memory summary (composed by the host) to the hot cache. |
| `kb_capture_v1` | Write a Markdown Source directly from conversation to `docs/`; stamps provenance frontmatter automatically. |
| `kb_ingest_v1` | Ingest a single named `docs/` Source into `wiki/` pages synchronously, with progress notifications. |
| `kb_index_v1` | Rebuild the BM25 Section Index from the wiki corpus and return `{files_indexed, sections_indexed}`. |
| `kb_lint_v1` | Run the Lint Pass and return structured findings; supports skipping the LLM-backed C5 contradiction check. |
| `kb_import_v1` | Import a local file by absolute path into `docs/` via the same format-conversion pipeline as `kb import`. |

> **Read surface unchanged.** The interface-parity work (ADR-0017) added write and
> maintenance tools; the read surface is untouched. `kb ask`, `kb_ask_v1`,
> `kb_search_v1`, and the Hot Cache pair (`kb_read_hot_v1` / `kb_save_hot_v1`) are
> exactly as they were before the parity slices.

### Concurrency recovery

Concurrent writes from two interfaces (e.g. `kb_ingest_v1` from MCP while `kb ingest` runs in a terminal) risk leaving `.kb/index.json` in a stale state. The index is fully regenerable: re-run `kb index` (CLI) or call `kb_index_v1` (MCP) to rebuild it from the wiki corpus.

---

## Deep dive

- [`CONTEXT.md`](CONTEXT.md) — the project's shared vocabulary.
- [`PROMPT.md`](PROMPT.md) — the exercise spec and design answers.
- [`project-docs/adr/`](project-docs/adr/) — architectural decisions.
- [`project-docs/roadmap.md`](project-docs/roadmap.md) — the full implementation sequence.

---

# 繁體中文

[English](#knowledge-base-qa-bot) · **繁體中文**

一個建立在小型 Markdown 知識庫之上的「有根據」問答機器人。每個答案都由引用的
來源組成——如果來源無法支持某個說法,機器人會回答 _"I cannot confirm"_,而不是
亂猜。內建**兩套**可並排比較的檢索引擎: **Wiki** 引擎(BM25,
設計參考 [Karpathy 的 LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f))
與 **Vector RAG** 引擎(FAISS),兩者由同一個瀏覽器介面提供服務。

## 介面

**Reader(閱讀端)**——輸入問題、選擇引擎、讀取有根據的答案。

![Reader 介面](project-docs/screenshots/reader.png)

**Operator Console(操作主控台)**——後台介面:上傳文件、執行建置流程、
維護公司或個人知識庫的健康度。

![Operator Console](project-docs/screenshots/console.png)

## 快速開始

檢索索引已**預先建好並提交**,所以剛 clone 下來就能立刻問答——第一次執行
不需要任何建置步驟。

```bash
# 1. Clone
git clone https://github.com/PaynePew/knowledge_base_qa_bot.git
cd knowledge_base_qa_bot

# 2. 安裝(repo 根目錄共用一個 .venv)
uv sync --all-packages

# 3. 設定你的 OpenAI key
cp .env.example .env        # 接著編輯 .env,填入 OPENAI_API_KEY=sk-...

# 4. 啟動 Gateway(同一個來源同時提供 UI 與兩套引擎)
uv run uvicorn gateway.app.main:app --port 8000
```

開啟 <http://localhost:8000/> 進入 Reader,或 <http://localhost:8000/console>
進入 Console。

> **關於 key。** `OPENAI_API_KEY` 是**必填**——兩套引擎都會呼叫 OpenAI 來產生
> 最終答案。`ANTHROPIC_API_KEY` 是**選填**,只有評測用的 Claude judge 會用到。
> 詳見 [`.env.example`](.env.example)。

## 操作 Reader

- 在輸入框**輸入任何問題**(英文**或**中文),按 Enter。
- **切換 Wiki / RAG**,用同一個問題去問兩套不同的檢索引擎。
- **資料來源先行**——也就是答案所依據的記錄——接著答案會在下方逐字串流出現。
- **grounding 標章**會確認每個說法都能追溯到引用來源;若無法,你會看到
  _"I cannot confirm"_,而不是幻覺式的答案。
- **後續追問**會延續同一段對話(多輪)。

檔案與問題都可以是**英文或繁/簡中文**——Wiki 引擎會對中文做 CJK 斷詞,並讓
中文的答案維持中文。

## 操作 Console

Console 是使用者擴充與維護個人或公司知識庫的地方。**流程步驟器(pipeline stepper)**
Wiki要能檢索檔案必須經過五個步驟,各有自己的 **Run** 按鈕。

**一份檔案的旅程 —— Upload → Import → Ingest → Index:**

| 步驟       | 做什麼                                                          |
| ---------- | --------------------------------------------------------------- |
| **Upload** | 拖放檔案。`.html` / `.txt` 進入 `raw/`;`.md` 直接進入 `docs/`。 |
| **Import** | 把 `raw/` 的來源轉成乾淨的 Markdown 放進 `docs/`(含出處)。      |
| **Ingest** | 由 LLM 把 `docs/` 的 Sources 合成為精選的 `wiki/` 頁面。        |
| **Index**  | 建立 BM25 搜尋索引,讓新內容可被問答。                           |

**RAG track**(獨立面板)，新增的檔案要經過**Index**才能檢索。**Rebuild**會從 `docs/` 重建向量索引,與 Wiki 流程互不相干。

**什麼時候該跑 Lint —— 它能幫你什麼:**

在編輯語料之後、或定期執行 **Lint** 來稽核知識庫的健康度。它會找出肉眼看不到的
問題——孤立頁面(orphan)、壞掉的 `[[wikilink]]`、頁面之間的矛盾、過時頁面、
以及涵蓋缺口(coverage gap)。

Lint 的結果會餵進 **Curation Queue(策展佇列)**,你可以在這裡檢視機器人自動
歸檔的 Q&A 草稿——直接看到實際的問題與答案內容——然後把好的**升級(Promote)**
成永久頁面,或**捨棄(Discard)**。這就是高品質答案逐步被收回知識庫的方式。

頁面底部的 **Resource Browser** 讓你不離開頁面就能瀏覽 `docs/`、`raw/`、`wiki/`。

## 專案結構

```
knowledge_base_qa_bot/
├── gateway/          # Gateway 應用 + 瀏覽器 UI(Reader 在 /,Console 在 /console)——從這裡開始
├── markdown_kb/      # Wiki 引擎:在精選的 wiki/ 層上做 BM25
├── vector_rag/       # RAG 引擎:把 docs/ 切塊並嵌入 FAISS
├── docs/             # Sources——機器人執行時的知識庫
├── wiki/             # Ingest 寫出的精選層(concepts、entities、歸檔 Q&A)
├── raw/              # 上傳的 .html/.txt 在 Import 前的本機收件匣
├── .kb/              # 預先建好的檢索索引(提交進 repo 的 demo 種子)
├── eval/             # Wiki-vs-RAG 改寫比較工具 + 報告
├── project-docs/     # ADR、roadmap、coding standard、screenshots
├── CONTEXT.md        # 共用詞彙表(glossary)
└── .env.example      # 複製成 .env 並填入你的 OpenAI key
```

## 評測:Wiki vs RAG

**測試資料量。** 在一份 20-Source / 約 51 個 Gold Section 的語料上,共 260 筆
查詢:**250 筆 Core 改寫**(5 種 LLM 生成的改寫類型 × 50)加上 **10 筆手寫的
結構性探針**(2 種 × 5)。「命中(hit)」必須是檢索結果的來源符合 gold section
**且**內容共享關鍵詞,所以「文件對、內容不對」算未命中。

**跨平台 L2 檢核。** 確定性指標在邊界情況的判定,會由**另一個模型家族**重新評判
——**Claude(`claude-sonnet-4-6`)**,而非驅動 RAG 嵌入的 OpenAI 家族——因此這個
第二意見與它所檢核的引擎沒有共同盲點。共重新評判了 207 筆模稜兩可的項目。

**統計上的誠實。** 在 Core 類型上,經過配對 McNemar 檢定 + Holm 校正後,**沒有
任何一種類型的差異達到統計顯著**。結構性探針每種只有 **n=5**——數量依舊太少,
無法支撐任何統計推論——所以它們僅作為描述性的「**預期極限驗證**」呈現,絕不併入
任何頭條數字。

**測試結果顯示:**

- **同義詞未命中(Wiki 的弱點)。** 當查詢使用了來源中沒有出現過的詞彙,關鍵詞
  BM25 可能漏掉、而向量相似度能命中:synonym_swap Wiki 0.90 vs RAG 0.94,而
  「沒見過的行話」探針是極端例子(Wiki 0.40 vs RAG 1.00)。
- **語意上的偽陽性(RAG 的弱點)。** 當 RAG 回傳了正確文件但內容僅有微弱重疊時,
  指標仍記為「命中」;跨家族的 Claude judge 正好定位出這些「id 對、內容弱」的
  命中,它們會美化 RAG 的原始數字。
- **引用品質(Wiki 的結構性優勢)。** Wiki 引用穩定的 `filename#heading`,並帶有
  在 ingest 時做過 grounding 檢查的 `sources:` 出處;RAG 則以相似度分數引用原始
  chunk,通常會遺失章節邊界。

**客觀上兩者應該如何表現**([`why-wiki.md`](project-docs/why-wiki.md))。兩者的
差別在於**合成發生的時機**——Wiki 在 ingest 時合成一次(形成可稽核、會累積的
產物，Cost花費在這裡發生。);RAG 則在每次查詢時從原始 chunk 重新推導。因此預期的結論取決於規模:
**約 1000 頁以下 → Wiki**(索引導覽便宜、每次查詢零嵌入成本、出處結構化);
**約 10 萬頁以上 → RAG**(索引大到無法全掃導覽);**兩者之間 → 混合**。這份語料
正好落在 Wiki 的範圍——而這正是數據所顯示的:在統計上幾乎打平,但 Wiki 用更簡單、
更便宜、更可稽核的機制就達到了。

![自然改寫上相當接近](eval/paraphrase_comparison/charts/core_hit_rate_at_3.png)

![探針暴露各自架構的極限](eval/paraphrase_comparison/charts/probes_hit_rate_at_3.png)

完整的方法、統計檢定、成本紀錄與誠實的限制說明,都在
[`eval/paraphrase_comparison/report.md`](eval/paraphrase_comparison/report.md)。
若要重新產生或擴充語料並重跑比較,請見維護者操作手冊
[`eval/paraphrase_comparison/README.md`](eval/paraphrase_comparison/README.md)。

## CLI 與 MCP 指令參考

CLI(`kb`)與 MCP 伺服器(`kb_mcp`)共享相同的完整生命週期。

### CLI 子命令（`uv run kb <subcommand>`）

| 命令 | 說明 |
| --- | --- |
| `kb ask <question>` | 詢問問題並輸出有根據的答案（LLM 合成 + Grounding Check）。 |
| `kb index` | 從 wiki 語料重建 BM25 Section Index 並寫入 `.kb/index.json`。 |
| `kb import <path>` | 將本機檔案（`.html`、`.txt`、`.md`）透過格式轉換匯入 `docs/`。 |
| `kb ingest [source]` | 把指定的 `docs/` Source（或省略時所有 Source）合成為 `wiki/` 頁面。 |
| `kb lint` | 執行 Lint Pass 健康度檢查（孤立頁面、矛盾、過時頁面、涵蓋缺口）並印出結果。 |
| `kb`（直接執行）| 進入有暖索引的互動式 REPL；支援 `:stack <wiki\|rag>` 與 `quit`。 |

### MCP 工具（`kb_mcp` 伺服器）

| 工具 | 說明 |
| --- | --- |
| `kb_ask_v1` | 詢問問題並取得帶引用與 Grounding Check 結果的 LLM 有根據答案。 |
| `kb_search_v1` | 從索引取回原始 Section 或 Chunk，不呼叫 LLM；wiki stack 回傳 BM25 分數。 |
| `kb_read_hot_v1` | 讀取工作記憶熱快取（`wiki/hot.md`）；第一次呼叫時回傳 `""`。 |
| `kb_save_hot_v1` | 將由 host 組成的工作記憶摘要寫入熱快取。 |
| `kb_capture_v1` | 直接從對話將 Markdown Source 寫入 `docs/`；自動加上出處 frontmatter。 |
| `kb_ingest_v1` | 同步將單一指定 `docs/` Source 合成為 `wiki/` 頁面，並發出進度通知。 |
| `kb_index_v1` | 從 wiki 語料重建 BM25 Section Index，回傳 `{files_indexed, sections_indexed}`。 |
| `kb_lint_v1` | 執行 Lint Pass 並回傳結構化結果；支援跳過 LLM-backed C5 矛盾檢查。 |
| `kb_import_v1` | 透過與 `kb import` 相同的格式轉換流程，用絕對路徑將本機檔案匯入 `docs/`。 |

> **讀取介面不受影響。** 界面對稱工作（ADR-0017）新增了寫入與維護工具；讀取介面
> 維持原狀。`kb ask`、`kb_ask_v1`、`kb_search_v1` 與熱快取組合
>（`kb_read_hot_v1` / `kb_save_hot_v1`）在對稱工作前後完全不變。

### 並發復原

兩個界面同時寫入（例如 MCP 的 `kb_ingest_v1` 與終端機的 `kb ingest` 同時執行），
最壞情況下會讓 `.kb/index.json` 停在過期狀態。索引可完整重新產生：
重新執行 `kb index`（CLI）或呼叫 `kb_index_v1`（MCP）即可從 wiki 語料重建。

---

## 深入閱讀

- [`CONTEXT.md`](CONTEXT.md) —— 專案的共用詞彙表。
- [`PROMPT.md`](PROMPT.md) —— 題目規格與設計解答。
- [`project-docs/adr/`](project-docs/adr/) —— 架構決策（ADR）。
- [`project-docs/roadmap.md`](project-docs/roadmap.md) —— 完整的實作順序。
