# Memory envelope under concurrent chat + ingest (issue #600)

Characterization report for the 512MB VPS tenant's memory behavior under concurrent
`/chat/stream` load and an Import batch. Produced by the harness at `ops/loadtest/`
(usage in that package's docstrings); measured JSONs are committed alongside this
report at `ops/loadtest/results/*.json`. **This is a characterization, not a code
change** — no app code was touched (env-only integration), per the issue's scope
decision. A follow-up issue implements any recommended caps.

## TL;DR

- The chat concurrency caps (`KB_MAX_INFLIGHT` / `KB_SSE_MAX_CONCURRENT`, both
  default 6) are the knob that bounds peak memory under chat load on this box —
  raising offered concurrency from 6 to 12 did **not** raise peak RSS further
  (192.5 vs 192.5 MB); the extra load was rejected (503) at the door instead of
  being admitted and consuming more memory.
- Lowering those two knobs to 2 measurably lowered the headline (chat+import)
  scenario's peak RSS from 192.0 MB to 179.6 MB — a ~12 MB delta directly
  attributable to fewer concurrent draft+verify LLM chains held in memory at once.
  That is the "which knob bounds it" answer the issue asked for.
- Import's own contribution is small: import-only peaked at 169.5 MB, essentially
  the idle-after-warmup baseline (169.0 MB); running it concurrently with chat load
  added no measurable peak beyond chat load alone.
- All measured peaks stayed in a **169–193 MB** band, well under any 512 MB budget
  reading — but see the Caveats section: this is a Windows dev box, not the 512 MB
  Linux container, and several real prod contributors were not exercised (Transcribe,
  the Hybrid dense stack, sustained/longer-duration load). Treat the absolute
  numbers as approximate; the *deltas* and *knob sensitivity* are the durable signal.

## Methodology

**Harness.** `ops/loadtest/harness.py` (`uv run python -m ops.loadtest.harness run
<scenario>`) spawns the real Gateway (`uvicorn gateway.app.main:app --workers 1`,
matching `Dockerfile`'s prod CMD) as a subprocess, waits for `/healthz`, samples
memory, drives load, tears down, and writes one `<scenario>.json`. Each `run`
invocation is one synchronous command — no agent-side background processes. A
separate `summarize` command merges the committed JSONs into the table below.

**Key-free was not achievable at the process level (a real finding, not a design
choice).** The issue's technical brief expected import-only scenarios to run with
no `OPENAI_API_KEY` at all. In practice, `vector_rag`'s own sub-app lifespan
(`vector_rag/app/main.py`, mounted unconditionally at `/rag`) calls
`load_vector_index()` -> `get_embeddings()`, which raises `RuntimeError` at **boot**
when no key is present — regardless of which stack a scenario intends to exercise.
The Gateway process cannot start at all without *some* key. The harness therefore
runs a local fake OpenAI-compatible upstream (`ops/loadtest/fake_upstream.py`,
FastAPI + uvicorn on a loopback port) for **every** scenario and points the app at
it via `OPENAI_API_KEY=dummy-loadtest-key` + `OPENAI_API_BASE=http://127.0.0.1:<port>/v1`
— env-only, no app code touched. The distinction that survives is at the *load*
level, not the process level: `S2_import` never issues an LLM-shaped request (the
fake upstream sees zero traffic during that scenario), so it still isolates
import's own footprint from the chat code path's. Verified request shapes (both the
plain draft call and the `with_structured_output(GroundingResult)` verifier's
`response_format: json_schema` call) against the real `langchain-openai` client
during implementation — see `fake_upstream.py`'s module docstring.

**Chat load** rotates five queries hand-verified (via a direct `_retrieve_and_gate`
call against the committed BM25 index) to retrieve non-empty, above-threshold
sections, so every request reaches the real draft+verify LLM path instead of the
pre-LLM Cannot-Confirm short-circuit — the fake upstream's *answer* text doesn't
matter, reaching the same code path real traffic reaches does.

**Import load** plants small (~1.2KB, under `KB_LONGFORM_MIN_CHARS`'s 2000-char
floor) synthetic `.txt` files under `raw/` with a run-unique `_loadtest_<id>_`
prefix, submits a batch job via `POST /wiki/import/jobs`, polls to completion, then
always deletes both the planted `raw/` inputs and the `docs/` outputs import
produced from them (`ops/loadtest/import_load.py`, `finally`-guarded). Small
fixtures keep this scenario's Structure Enrichment gate closed, so import makes
zero LLM calls of its own.

**Harness-side env** (`ops/loadtest/config.py::HARNESS_BASE_ENV`), layered under
every scenario so the concurrency knobs under test are the only thing bounding a
run: `KB_RATE_LIMIT_PER_IP=0` (the default 30-req/5-min per-IP limit would
otherwise reject load unrelated to the knob being measured) and
`KB_DAILY_USD_CAP=1000` (the default $3/day cap, at $0.02/`/chat/stream` request,
would flip later requests into degraded no-LLM serving mid-scenario and understate
their cost — irrelevant anyway since the fake upstream spends nothing real).

**Measurement.** `psutil`, added as a dev dependency (`uv add --dev psutil`),
polling the Gateway process tree at ~200ms. **A real environment gotcha found
during implementation**: on this box, `python -m uvicorn ... --workers 1` (no
`--reload`) runs the actual server in exactly **one child process**, while the
`subprocess.Popen`-returned PID stays alive as a thin launcher for the process's
whole lifetime and reports a flat, wrong ~5 MB RSS if read directly — the harness
must sum the process **tree** (parent + children) or it silently reports numbers
30-40x too low. Verified with an isolated diagnostic (a plain Python child holding
a 150 MB buffer read as a flat 5 MB for its entire life when queried directly;
correct once its child was included). `ops/loadtest/sampler.py`'s tree walk exists
because of this, not as defensive padding. On Windows, `memory_info().peak_wset`
(an OS-maintained historical high-water-mark, summed across the same tree, read
once at scenario teardown while the tree is still alive) is reported alongside the
poll-based peak; the two agreed within 0.03 MB on every run here, so `summarize.py`
prefers `peak_wset` when present.

## Results

Regenerate with `uv run python -m ops.loadtest.harness summarize`. Source JSONs:
`ops/loadtest/results/*.json`.

| Scenario | Description | Peak RSS (MB) | Wall clock (s) | Chat sent/ok/err | Import status |
|---|---|---|---|---|---|
| S0_idle | Idle after warmup, zero requests (baseline) | 169.04 | 13.86 | - | - |
| S1_chat_c1 | Chat-only, concurrency=1, 20 requests | 174.78 | 8.27 | 20/20/0 | - |
| S1_chat_c6 | Chat-only, concurrency=6 (== `KB_MAX_INFLIGHT` default), 60 requests | 192.54 | 10.14 | 60/60/0 | - |
| S1_chat_c12 | Chat-only, concurrency=12 (2x default cap), 96 requests | 192.51 | 9.33 | 96/48/48 | - |
| S2_import | Import-only, 6 files, zero LLM-shaped requests | 169.50 | 6.34 | - | completed (6/6) |
| S3_headline | Chat c=6 concurrent with a 6-file import batch | 192.04 | 10.08 | 60/60/0 | completed (6/6) |
| S4_maxinflight2 | Same load as S3, `KB_MAX_INFLIGHT=2` + `KB_SSE_MAX_CONCURRENT=2` | 179.56 | 7.27 | 60/12/48 | completed (6/6) |

All peaks use the OS-tracked `peak_wset` (Windows `memory_info().peak_wset`,
summed over the process tree at scenario teardown).

## Which knob bounds which scenario

- **`KB_MAX_INFLIGHT` / `KB_SSE_MAX_CONCURRENT` (both default 6) bound chat's peak.**
  S1_chat_c6 -> S1_chat_c12 (2x offered concurrency) added **0.0 MB** to peak RSS
  (192.54 -> 192.51) — the extra load 503'd instead of being admitted (48/96
  rejected at c12 vs 0/60 at c6), confirming these are exactly the admission gates
  that stop offered concurrency from turning into resident memory. S4's
  `KB_MAX_INFLIGHT=2` + `KB_SSE_MAX_CONCURRENT=2` rerun of the S3 headline load
  independently confirms the direction: peak dropped 192.04 -> 179.56 MB (-6.5%)
  when fewer requests could be admitted at once (12/60 admitted vs 60/60 at the
  default cap).
- **Import's own footprint is small and additive, not multiplicative.**
  S2_import (169.50 MB) sits almost exactly at the S0 idle baseline (169.04 MB);
  running it concurrently with chat load (S3_headline, 192.04 MB) tracked chat-only
  c6 (192.54 MB) rather than stacking on top of it. `KB_IMPORT_MAX_CONCURRENT_JOBS`
  (default 2) was not stressed here (only ever 1 job in flight) — see Not measured.
- **`KB_TRANSCRIBE_*` knobs were not exercised** — no chat/import scenario in this
  battery reaches the Transcribe code path (small `.txt` fixtures deliberately never
  cross the Structure Enrichment gate, and Transcribe itself needs a scanned-PDF
  fixture — see Not measured / S5).

## Recommended caps for the 512 MB box

Given the measured band (169-193 MB, Windows dev box) sits comfortably under
512 MB even at 2x the default chat concurrency, **no change to the shipped
defaults is indicated by this data alone.** The one actionable signal: `KB_MAX_INFLIGHT`
/ `KB_SSE_MAX_CONCURRENT` already function as the load-bounding knob exactly as
designed (issue #599's intent) — a future tightening pass (if the real Linux
container's baseline turns out far higher than this Windows figure) should turn
`KB_MAX_INFLIGHT` down first, since S4 showed it has a measurable, directionally
correct effect on peak RSS without any code change. `KB_TRANSCRIBE_CONCURRENCY`
(default 16) remains the correct target for a *future* Transcribe-inclusive
characterization (S5) before touching it — issue #456/#459's precedent (16-way
concurrency vs 512 MB -> OOM, fixed via `KB_TRANSCRIBE_CONCURRENCY=3`) is exactly
the failure mode this report's scenario grid does not yet reproduce.

## Not measured (explicit)

- **Transcribe / OCR path (S5, stretch scope).** No committed scanned-PDF fixture
  was confirmed usable within this session's scope; the #456/#459 OOM precedent
  this issue cites as motivation lives specifically in that code path
  (`KB_TRANSCRIBE_CONCURRENCY`, `KB_TRANSCRIBE_PAGE_COUNT_CONCURRENCY`) and remains
  the highest-value follow-up characterization.
- **Hybrid / RAG stacks under load.** All chat load here used `stack=wiki`
  (BM25-only) per the issue's scenario grid; `stack=hybrid`/`stack=rag` load
  through the dense/FAISS path (and the fake upstream's `/v1/embeddings` stub,
  built but never exercised) is unmeasured.
- **Sustained / long-duration load.** Each scenario ran for single-digit seconds
  (kept under the harness's ~5-minute-per-invocation budget); slow memory growth
  (leaks, unbounded caches) over minutes-to-hours is out of this report's reach.
- **`KB_IMPORT_MAX_CONCURRENT_JOBS`, `KB_MAX_ADMIN` saturation.** Only one import
  job ran at a time in every scenario; the admin-path concurrency cap was never
  stressed.
- **True Linux-container numbers.** This box is Windows; the report's own
  methodology section documents a real cross-process-measurement gotcha specific to
  this environment. Absolute MB figures are this box's numbers, not the 512 MB
  Linux tenant's — re-run this harness on (or against) the actual container before
  treating any number here as a hard ceiling.

## Reproducing

```bash
uv run python -m ops.loadtest.harness list
uv run python -m ops.loadtest.harness run S1_chat_c6
uv run python -m ops.loadtest.harness run S3_headline --env KB_MAX_INFLIGHT=2 --env KB_SSE_MAX_CONCURRENT=2 --out-name S4_maxinflight2
uv run python -m ops.loadtest.harness summarize
```

Never point this at the deployed box — it is manual-local only, spends no real
OpenAI tokens (fake upstream), and is not wired into CI (CI runs only
`ops/loadtest/tests/`, fast hermetic unit tests over the config-parsing and
summarize math). After a run, `git status` should show only the committed
`ops/loadtest/results/*.json` — if `wiki/log.md` shows modified (the server's
single log channel has no env override), restore it with
`git checkout -- wiki/log.md` before committing; never commit a mutated
`raw/`, `docs/`, or `.kb/` artifact either.

## S5 addendum (#627) — 2026-07-13

Closes the "Transcribe / OCR path" gap in the "Not measured" section above.
Follow-up to #600 (parent #580); this issue's own 2026-07-13 triage note
narrowed scope: the container-side re-run (Gap 2, below) is carved OUT of
this slice (local docker daemon unavailable, and the deployed tenant must
never be load-tested) — this addendum covers Gap 1 (transcribe-inclusive
characterization) only.

### Methodology delta

**Fixture.** No new fixture was committed (per this issue's scope). A
16-page PDF is assembled **in memory, at harness-run time**
(`ops/loadtest/transcribe_load.py::_build_multipage_pdf`) by repeating the
already-committed `markdown_kb/tests/fixtures/raw_import/image_only.pdf`'s
page 0 via `pypdfium2`'s own page-import API, staged under a run-unique
`raw/` filename, and removed in `finally` — never written to a committed
path. One multi-page **source** (not several 1-page sources) is required to
exercise page-level concurrency at all: `transcribe_jobs._run_batch` is
sequential-over-sources by design (mirrors `import_sources`), with
concurrency living entirely *within* one source's page-worker pool
(`transcriber.transcribe_pdf_bytes_concurrent`).

**Fake-upstream shape, verified not assumed.** The transcriber's page call
sends vision content parts (`HumanMessage(content=[{"type": "text", ...},
{"type": "image_url", ...}])`) with `reasoning_effort` set and no
`response_format`/`tools` key. Probed this directly against a throwaway
local FastAPI stub during implementation (captured request body keys:
`["messages", "model", "reasoning_effort", "stream"]`) — it lands on
`POST /v1/chat/completions`, the same endpoint #600's fake upstream already
implements, and falls into that handler's existing plain-text-stub branch
(no `response_format`/`tools` present). **No fake-upstream code change was
needed.** The stub answer (~71 chars) assembled across 16 pages totals
~1.17KB, comfortably under `KB_LONGFORM_MIN_CHARS` (2000) — same
"keep Structure Enrichment's gate closed" discipline `import_load.py`
documents for its own fixtures, so this scenario's only LLM traffic is the
16 per-page transcription calls it exists to measure.

**Route.** `POST /wiki/transcribe/batch` (single source, `{"sources": [...]
}`) + poll `GET /wiki/transcribe/jobs/{job_id}` — the async batch surface
(issue #459 AC5), admin-gated (`KB_MAX_ADMIN`) same as `/wiki/import/jobs`.
`KB_TRANSCRIBE_ENABLED=true` is layered in as a **scenario-scoped** env
override (`ScenarioSpec.scenario_env`, merged under CLI `--env` so a rerun
can still override it) rather than added to `HARNESS_BASE_ENV` — S0-S4 never
touch the Transcribe path and shouldn't gain a new always-on flag.

### Results

| Scenario | Description | Peak RSS (MB) | Source | Wall clock (s) | Chat sent/ok/err | Transcribe status |
|---|---|---|---|---|---|---|
| S5_transcribe_c16 | Transcribe-only, 16-page single source, `KB_TRANSCRIBE_CONCURRENCY` default (16) | 198.58 | peak_wset (OS) | 21.06 | - | completed (pages 16/16) |
| S5_transcribe_c3 | Same load, `KB_TRANSCRIBE_CONCURRENCY=3` (prod value, #456/#459) | 194.53 | peak_wset (OS) | 8.33 | - | completed (pages 16/16) |
| S5_transcribe_headline | Chat load c=6 (60 req) concurrent with the same 16-page transcribe batch | 213.89 | peak_wset (OS)\* | 11.64 | 60/60/0 | completed (pages 16/16) |

\* `S5_transcribe_headline` is the first scenario across #600+#627 where
`peak_wset_os_mb` (213.89) and `peak_rss_polled_mb` (220.42) disagree by
more than a rounding error (6.5 MB, ~3%) — every S0-S4 scenario and both
transcribe-only S5 runs agreed within 0.03 MB. Plausible cause: this is the
first three-concurrent-load-source scenario (SSE chat workers + Transcribe's
own page-worker pool, both writing memory on independent threads at once),
which the 200ms poll can catch a transient joint spike of that
`peak_wset`'s single post-load read might just miss by a sample or two —
not independently confirmed, flagged here rather than silently picking one
number. `summarize.py`'s existing `peak_rss_mb` preference for
`peak_wset_os_mb` is kept for consistency with S0-S4's rows; either figure
keeps this scenario inside the methodology's overall envelope.

### Which knob bounds S5 — and a real methodology caveat

- **`KB_TRANSCRIBE_CONCURRENCY` (16 -> 3) moved peak RSS by only ~4 MB
  (198.58 -> 194.53, ~2%)** — directionally consistent with #600's
  `KB_MAX_INFLIGHT` finding (lower concurrency -> lower peak) but a much
  smaller effect, despite a *larger* proportional cut (16->3, -81%, vs
  #600's 6->2, -67%).
- **This is very likely an artifact of the fake upstream's near-instant
  response time, not evidence the real knob is weak.** The #456/#459 OOM
  precedent this S5 exists to characterize came from a REAL vision model
  call taking real seconds per page, so `KB_TRANSCRIBE_CONCURRENCY` pages
  were genuinely resident (rendered PNG + in-flight HTTP call) at once for
  that whole duration. Here, the fake upstream answers in milliseconds, so
  even 16 "concurrently permitted" pages barely overlap in wall-clock time —
  there's little sustained concurrent residency for the semaphore to bound.
  **Read this ~2% delta as a floor, not the real effect size**; it does not
  contradict the #456/#459 incident, it just cannot reproduce the
  slow-upstream dynamic that caused it (matching #600's own "answer fidelity
  doesn't matter, reaching the same code path does" framing — true for
  *which* LLM branch executes, not for *how long memory is held* under
  concurrency).
- **Chat + Transcribe compounds, unlike chat + Import (a real, new
  finding).** #600 found `S3_headline` (chat c=6 + import) tracked
  chat-alone almost exactly (192.04 vs 192.54 MB) — import's own footprint
  was additive-but-negligible. Here, `S5_transcribe_headline` (213.89 MB)
  sits **above both** its own components measured alone: transcribe-alone
  (198.58) and #600's chat-alone `S1_chat_c6` (192.54) — roughly +15 MB over
  the higher of the two, not just "tracks the bigger one" the way import
  did. Transcribe holds real per-page image buffers across a page-worker
  pool for the scenario's duration in a way Import's zero-LLM mechanical
  pass never does, so this is the more representative "does a background
  Transcribe batch cost the concurrent chat path anything" signal — and the
  answer is a real, non-trivial yes on this box, even at synthetic-load
  scale.

### Recommended caps (updated)

S0-S4's conclusion stands unchanged (`KB_MAX_INFLIGHT` /
`KB_SSE_MAX_CONCURRENT` bound chat's peak; no shipped-default change
indicated by that data). Adding S5's data:

- **No shipped-default change is indicated by the S5 data alone either** —
  even the worst-observed peak here (`S5_transcribe_headline`, 213.89-220.42
  MB) sits comfortably inside a 512 MB budget, with more headroom to spare
  than #600's own worst case.
- **This is NOT a confident "close as adequate," though — two caveats keep
  it open:**
  1. The concurrency-knob caveat above: this harness's fake-upstream latency
     profile structurally cannot reproduce the sustained-concurrent-residency
     dynamic that caused the real #456/#459 OOM, so a synthetic ~2% delta at
     16 vs 3 is not strong evidence that `KB_TRANSCRIBE_CONCURRENCY=16`
     (the shipped default) is safe under a REAL slow-model workload — only
     that this harness cannot falsify it either way.
  2. Gap 2 (container-side re-run, below) remains unmeasured; every number
     above is this Windows dev box's, not the 512 MB Linux tenant's (same
     caveat #600's report already carries for S0-S4).
- **Conclusion: hold `KB_TRANSCRIBE_CONCURRENCY` at its shipped default (16)
  for now — do not lower it shipped-side on this issue's data — and open a
  dedicated container-run issue at review time** to close both caveats
  together (a real Linux container measurement is also the only way to get a
  real-vision-model-shaped concurrency signal cheaply, by measuring wall
  time per page instead of guessing). If that container run reproduces
  anything close to #456/#459's real OOM shape, that issue is where a
  shipped-default change (mirroring the #474/#482 CapacityLimiter pattern)
  belongs — not here.

### Not measured (updated)

- ~~Transcribe / OCR path (S5)~~ — **closed by this addendum.**
- **True Linux-container numbers (Gap 2) — still open, scope-carved-out of
  this issue** (2026-07-13 triage: local docker daemon is down, and the
  deployed tenant must never be load-tested). Reproduction recipe for
  whoever picks up the dedicated container-run issue:

  ```bash
  # From repo root, on a box with a working docker daemon:
  docker build -t kbqabot-loadtest -f Dockerfile .
  docker run --rm --memory=512m --memory-swap=512m -p 8000:8000 \
    --env-file <(printf 'OPENAI_API_KEY=dummy\nOPENAI_API_BASE=http://host.docker.internal:<fake-upstream-port>/v1\n') \
    kbqabot-loadtest

  # Measurement channel — either works, cross-check like this report's
  # peak_wset vs polled-RSS pair:
  docker stats --no-stream kbqabot-loadtest            # quick, coarse
  # or, more precise (matches this report's "sum the process tree" finding —
  # a container has its own analogous gotcha: cgroup v2 memory.peak is for
  # the WHOLE cgroup, already summed, so no process-tree walk is needed
  # inside the container the way sampler.py needs one on Windows):
  cat /sys/fs/cgroup/memory.peak                        # cgroup v2, single number, bytes
  ```

  The harness's own scenarios (`S1_chat_c6`, `S3_headline`,
  `S5_transcribe_headline` at minimum) should be re-run **against** this
  container rather than the harness spawning `uvicorn` as a local subprocess
  — point the harness's chat/import/transcribe load drivers at the
  container's exposed port instead of `spawn_gateway`'s local subprocess, or
  simply replay the same HTTP call sequences with `curl`/`httpx` by hand.
  Never point this at the deployed box, same rule as the local harness.
- Everything else S0-S4 already left unmeasured (Hybrid/RAG stacks under
  load, sustained/long-duration load, `KB_IMPORT_MAX_CONCURRENT_JOBS` /
  `KB_MAX_ADMIN` saturation) is still unmeasured; S5 did not touch any of it.
