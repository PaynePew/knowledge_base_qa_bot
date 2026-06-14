# Prod deploy — `ask-wiki-rag` VPS tenant

How this Q&A bot runs in production: one Docker Compose tenant behind the
shared `edge` Caddy on a single VPS. See [`docker-compose.prod.yml`](../../docker-compose.prod.yml)
at the repo root (issue #271).

## Box layout

On the VPS, this tenant lives in its own directory:

```
/opt/ask-wiki-rag/
  docker-compose.prod.yml   # copied from the repo root
  .env.prod                 # secrets — NEVER committed (OPENAI_API_KEY, ...)
```

`.env.prod` mirrors the keys documented in [`.env.example`](../../.env.example).
It is created by hand on the box and stays out of git.

## Networking — no host ports, join `edge`

This tenant exposes **no host ports**. The shared `edge` Caddy owns `:80`/`:443`
and reverse-proxies to the container as `ask-wiki-rag:8000` over the external
`edge` Docker network. The compose file joins both `default` (intra-project) and
`edge` (cross-tenant ingress); the `edge` network is declared `external: true`
because Caddy's stack creates and owns it.

The container serves the Gateway ASGI app on port `8000`
(`uvicorn gateway.app.main:app --port 8000`), reachable only via `edge`.

## Bulkhead — mem/cpu caps are mandatory

Both `mem_limit: 512m` and `cpus: "0.5"` are set so one tenant cannot starve
the box. The app runs a single worker (the prototype's single-process model —
see `CODING_STANDARD.md` §2.6), so these caps bound the whole tenant.

## State — baked, read-only, ephemeral

The KB is baked into the image and treated as read-only at runtime. Writes
(runtime traces, ingest artifacts) are ephemeral inside the container — there is
**no named volume and no backup**. A redeploy pulls a fresh
`ghcr.io/paynepew/knowledge_base_qa_bot:latest` and starts clean.

## Deploy / redeploy

From `/opt/ask-wiki-rag/` on the box (these are manual ops steps; run them
yourself on the VPS, not in CI):

```sh
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

`restart: unless-stopped` brings the tenant back after a host reboot or crash
unless it was explicitly stopped.

## Validate the compose file (no daemon needed)

The compose file is syntax-validated in CI / locally without a running Docker
daemon by parsing it as YAML:

```sh
python -c "import yaml; yaml.safe_load(open('docker-compose.prod.yml', encoding='utf-8'))"
```

On a box with Docker available, `docker compose -f docker-compose.prod.yml config`
additionally resolves the merged config.
