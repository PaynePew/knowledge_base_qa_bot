# Operations runbook: logs and errors on the VPS

Where to look when something breaks on the live tenant, and the exact commands
to run. Everything here is verified against the tracked files that define the
deploy (`.github/workflows/deploy.yml`, `docker-compose.prod.yml`,
`deploy/k8s/README.md`, `project-docs/log-kinds.md`).

## Deployment model

CI (`.github/workflows/deploy.yml`) bakes this repo into the image
`ghcr.io/paynepew/knowledge_base_qa_bot` on every push to `main` (gated on the
same pytest suite `ci.yml` runs) and pushes `:latest` + `:<sha>`. **The box
never holds a checkout of this repo.** The only two files that live on it, at
`/opt/ask-wiki-rag`, are the tracked `docker-compose.prod.yml` (re-shipped by
`scp` on every deploy) and a hand-maintained `.env.prod` (secrets, never
committed). A deploy is `docker image prune -af` (reclaim disk) →
`docker compose ... pull` → `docker compose ... up -d`, then a retried
`/healthz` smoke over SSH. The `reset` workflow (`.github/workflows/reset.yml`,
cron every ~2 days) force-recreates the same service from whatever image is
already on the box — no pull, no rebuild, zero OpenAI tokens by default.

Practical consequence: there is no `git pull` or `cd` into a checkout to
"fix something on the box" — the fix always ships through `main` and the next
deploy (or `workflow_dispatch`).

## Container logs (uvicorn access log, unhandled tracebacks)

The compose project is named `ask-wiki-rag` and the file is
`docker-compose.prod.yml` — **neither is auto-detected** from a bare
`docker compose` invocation on this box, so both flags are required on every
command, and the subcommand is `logs` (plural), not `log`:

```bash
cd /opt/ask-wiki-rag
docker compose -p ask-wiki-rag -f docker-compose.prod.yml logs -f
docker compose -p ask-wiki-rag -f docker-compose.prod.yml logs -f --tail 200
```

Fallback if the compose project context is unavailable (e.g. you only know the
container name from `docker ps`):

```bash
docker logs -f <container>
```

## Wiki Log channels (application events)

Container logs capture the ASGI/uvicorn layer. Domain events — a successful
`/chat`, a Cannot-Confirm fallback, an ingest failure, a budget shed — go to a
separate **Wiki Log**: one append-only `log.md` per package, written via each
package's own `log_event(kind, summary)`:

| File | Package |
| --- | --- |
| `gateway/log.md` | Gateway (query rewriting, budget/rate-limit/shed events) |
| `wiki/log.md` | Wiki stack (A) + the shared Grounding Check |
| `vector_rag/log.md` | RAG stack (B) |
| `hybrid_kb/log.md` | Hybrid stack (C) |

These files live **inside the running container**, not on the box's
filesystem, so read them with `exec`:

```bash
docker compose -p ask-wiki-rag -f docker-compose.prod.yml exec ask-wiki-rag cat gateway/log.md
docker compose -p ask-wiki-rag -f docker-compose.prod.yml exec ask-wiki-rag cat wiki/log.md
docker compose -p ask-wiki-rag -f docker-compose.prod.yml exec ask-wiki-rag cat vector_rag/log.md
docker compose -p ask-wiki-rag -f docker-compose.prod.yml exec ask-wiki-rag cat hybrid_kb/log.md
```

For the full `kind=` enumeration (every event a channel can emit, when it
fires, and its `summary` template), see
[`project-docs/log-kinds.md`](log-kinds.md) — this runbook does not duplicate
that reference.

## Ephemerality warning

The service has **no volume** (`docker-compose.prod.yml`: no `volumes:` entry,
deliberate per ADR-0021). Every deploy and every scheduled `reset` recreates
the container from the baked image, which wipes:

- All four Wiki Log files above.
- Any other runtime write the container made (e.g. filed Q&A drafts, uploaded
  files still sitting in `raw/`).

If you need a log line to survive investigation, copy it out (`docker compose
exec ... cat <path> > local-file`) before the next deploy or reset fires.

## Health endpoints

All three are unauthenticated (`gateway/app/main.py`), even when
`KB_ADMIN_TOKEN` is set.

| Endpoint | Semantics |
| --- | --- |
| `GET /healthz` | Liveness — **always 200**, even under budget exhaustion or read-saturation. A restart-policy check should hit this one: a non-200 here would kill an otherwise-healthy, merely-busy worker. |
| `GET /healthz/shed` | Readiness / load-shed — 200 normally, **503** when the read semaphore is fully held (edge drains this box from the read pool until it recovers). Reflects read saturation only; admin-load never flips it. |
| `GET /healthz/budget` | Read-only daily USD budget ledger snapshot (`{day, spent_estimate, cap, remaining}`) for the current UTC day. Never charges the ledger. |

## k8s variant

The local `kind` deployment (`deploy/k8s/README.md`) runs the same image with
`kubectl` instead of `docker compose`:

```bash
kubectl -n ask-wiki-rag logs -f deploy/ask-wiki-rag
```

Same ephemerality caveat applies — no PVC, `strategy: Recreate`, so a pod
replacement (deploy or a manual kill) wipes runtime state exactly like the VPS
container recreate. See that file for cluster setup, troubleshooting, and the
self-healing demo.

## Known gaps

- **No error tracker.** Unhandled exceptions surface only in the container's
  stdout/stderr (`docker compose logs`) and are not currently written to a
  Wiki Log channel — tracked in
  [#648](https://github.com/PaynePew/knowledge_base_qa_bot/issues/648)
  (`unhandled_error` kind).
- **No metrics endpoint.** `/healthz*` are boolean/gauge probes, not a
  Prometheus-style scrape target.
- **No log rotation.** Container logs grow unbounded until the next
  deploy/reset recreates the container — tracked in
  [#647](https://github.com/PaynePew/knowledge_base_qa_bot/issues/647)
  (`json-file` rotation in `docker-compose.prod.yml`).
