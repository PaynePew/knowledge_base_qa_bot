# Deploy: container build + local smoke

The Gateway ships as a single container image, pulled onto the VPS from GHCR and
run as tenant `ask-wiki-rag` (issue #270, deploy S2). This note covers the
**local** build/run smoke you do before pushing the image — no VPS needed.

## What the image contains

- The full uv workspace resolved with `uv sync --frozen --no-dev` (third-party
  deps + the in-tree members). Dev-only deps — eval's deepeval / matplotlib /
  anthropic and everyone's pytest / ruff — are **not** baked.
- The **baked seed**: `.kb/index.json`, `.kb/faiss_index/*`, and the curated
  `wiki/`. A freshly pulled container answers `/chat` immediately — no ingest or
  index at boot, and **no OpenAI key needed at build time** (FAISS is not rebuilt
  during the build). The key is only needed at *run* time to write answers.
- No host ports are baked in. The container listens on `8000`; the host maps a
  published port at run time.

## Build

```bash
# From the repo root (the build context must include the baked seed).
docker build -t ask-wiki-rag:local .
```

The `.dockerignore` deliberately **keeps** `.kb/ wiki/ docs/` in the build
context (it is NOT a mirror of `.gitignore`, which lists `.kb/`). If those were
excluded, the image would ship an empty index and `/chat` would fail.

## Run

```bash
# Provide a dedicated OpenAI key + guardrails via an env file.
cp .env.prod.example .env.prod   # then fill in OPENAI_API_KEY
docker run --rm --env-file .env.prod -p 8000:8000 ask-wiki-rag:local
```

`.env.prod` is gitignored and dockerignored — secrets never get committed or
baked into the image.

## Smoke

```bash
# Liveness: top-level /healthz returns 200 {"status":"ok"} (added in #269).
# This is the route the VPS health check / CD smoke step targets.
curl -fsS http://localhost:8000/healthz
# Readiness / load-shed: 200 normally, 503 when the read pool is saturated.
curl -fsS http://localhost:8000/healthz/shed
# Reader UI (HTML 200):
curl -fsS -o /dev/null -w '%{http_code}\n' http://localhost:8000/
```

> The mounted sub-apps still expose `/wiki/health` and `/rag/health`, but the
> top-level `/healthz` (liveness) + `/healthz/shed` (readiness) added in #269 are
> the canonical probes for the deploy.

A one-shot grounded answer (needs `OPENAI_API_KEY` set):

```bash
curl -fsS -X POST 'http://localhost:8000/chat/stream?stack=wiki' \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the refund policy?"}'
```

## Single worker

The CMD runs `uvicorn ... --workers 1`. The app relies on in-process singletons
and an append-only Wiki Log (CODING_STANDARD single-process assumption), so do
**not** raise the worker count — scale horizontally (more containers behind a
proxy) instead if needed.
