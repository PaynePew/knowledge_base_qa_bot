# Async Ingest: Environment Variables and Flow

Fix 1a + Fix 1b (branch `feat/large-file-ingest`) introduce bounded concurrency
and a submit/poll job pattern for large Sources.  This note documents every
tunable env var and the expected call flow.

## Environment Variables

All variables are read at call-time (no restart required for changes to take
effect).

| Variable | Default | Description |
|---|---|---|
| `KB_INGEST_CONCURRENCY` | `8` | Maximum concurrent in-flight LLM calls inside `aingest_sources` (per-section concept synthesis fan-out). Higher values increase throughput at the cost of more simultaneous API quota. |
| `KB_INGEST_MAX_RETRIES` | `5` | Maximum retry attempts per LLM call (transient errors only). Backed by the tenacity retry decorator on `_call_llm_with_error_handling`. |
| `KB_INGEST_MAX_TOKENS` | *(derived)* | Per-Source SOFT token cap (routing hint). **Absolute override** — when set, wins outright. When unset it is **derived** from the configured model: `context_window(model) * KB_INGEST_TOKEN_FRACTION`. Sources whose estimated token count exceeds the cap route to the async job path instead of running inline in `kb_ingest_v1`. Does NOT reject the Source — only changes how it is scheduled. |
| `KB_INGEST_TOKEN_FRACTION` | `0.5` | Fraction of the model's context window used as the per-Source SOFT cap when `KB_INGEST_MAX_TOKENS` is unset. The 0.5 default leaves headroom for the system prompt, structured output, and the entity path. Ignored when `KB_INGEST_MAX_TOKENS` is set. |
| `KB_INGEST_MAX_SECTION_TOKENS` | `6000` | Per-section HARD token cap. A *quality* cap (not a capacity cap): a section over ~6 K tokens is almost always a malformed unsplit blob — so this is a fixed constant and deliberately does **not** scale with the model window. A Source with ANY section exceeding it is rejected immediately (before any LLM call) with a clear `reason`. The batch continues for other Sources. |

## Derived per-Source token budget

The per-Source SOFT cap is no longer a frozen literal. The ingest model is
resolved at call time (`OPENAI_INGEST_MODEL → OPENAI_MODEL → gpt-4o-mini`,
`templates._ingest_model_name`), and the budget is derived from *that* model's
context window so swapping the model re-scales the cap automatically:

```
budget = ingest_model_context_window(model) * KB_INGEST_TOKEN_FRACTION
```

Precedence (two-tier knob):

1. `KB_INGEST_MAX_TOKENS` — absolute override, wins outright (escape hatch).
2. `context_window(model) * KB_INGEST_TOKEN_FRACTION` (default `0.5`).

Known model windows live in `templates._MODEL_CONTEXT_WINDOWS`
(`gpt-4o-mini`/`gpt-4o`/`gpt-4-turbo` = 128 K, `gpt-4.1`/`gpt-4.1-mini` = 1 M,
`gpt-3.5-turbo` = 16 385). An **unknown** model falls back to
`_FALLBACK_CONTEXT_WINDOW` (32 K) — deliberately pessimistic, so an unrecognised
model under-fills rather than overflows; set `KB_INGEST_MAX_TOKENS` to override.
For the default gpt-4o-mini (128 K) the 0.5 fraction reproduces the historical
`64000`, so this change is a behaviour-preserving refactor for the default model.

Token estimation uses `len(content) // 3` (integer floor-division): this is
CJK-pessimistic (over-counts ASCII, under-counts CJK), which is the safe
direction for a guard.

## Async Ingest Flow

For Sources that exceed `KB_INGEST_MAX_TOKENS`, use the submit/poll pattern
to avoid the MCP host tool-call timeout (-32001):

```
MCP host                         kb_mcp server
--------                         -------------
kb_ingest_v1(source)
  → (auto-routed if large)
  ← {status: "routed_async",     submit background job
      job_id: "abc123",           return immediately
      note: "...poll..."}

kb_ingest_status_v1(job_id)
  ← {status: "working",          pipeline still running
      progress: [0, 1]}

kb_ingest_status_v1(job_id)
  ← {status: "completed",        pipeline done
      progress: [1, 1],
      result: {pages_created, …}}
```

For Sources you know are large (e.g. multi-chapter PDF manuals), you can also
call `kb_ingest_start_v1(source)` directly to skip the routing check and
always get an async job:

```
kb_ingest_start_v1(source)
  ← {job_id: "abc123",           job submitted immediately
      status: "submitted"}

kb_ingest_status_v1(job_id)     poll until "completed" or "failed"
  ← {status: "completed",
      progress: [1, 1],
      result: {pages_created, …}}
```

### Status values

| status | Meaning |
|---|---|
| `submitted` | Job registered, background Task scheduled but not yet running. |
| `working` | Background Task has started `aingest_sources`. |
| `completed` | Pipeline finished; `result` carries the ingest outcome dict. |
| `failed` | Pipeline raised an exception; `error` carries `{code, message}`. |
| `unknown` | `job_id` not found in the registry (wrong id, or server restarted). |

### Error codes in failed jobs

| code | Retryable | Cause |
|---|---|---|
| `LLM_UNAVAILABLE` | Yes | Transient LLM service error (timeout, rate limit). |
| `LLM_ERROR` | No | Auth failure, bad API key, or unexpected API error. |
| `INGEST_ERROR` | No | Unexpected exception in the ingest pipeline. |

## GC Safety Note

`asyncio.create_task()` keeps only a WEAK reference to the Task it creates.
Fix 1b stores a strong reference (`_TASKS` module-level set + `Job._task`)
so the GC cannot cancel a task mid-run.  Each task removes itself from
`_TASKS` via `add_done_callback` when it completes, so the set does not grow
unbounded across many jobs.
