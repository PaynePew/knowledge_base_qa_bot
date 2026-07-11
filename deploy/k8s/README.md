# Local k8s (kind) deployment

Runs the same baked-seed gateway image the VPS tenant runs, on a local
[kind](https://kind.sigs.k8s.io/) cluster. Purpose: prove the compose-based
deploy maps cleanly onto Kubernetes primitives, on one laptop, in one sitting.

Compose → k8s mapping (each manifest carries the detailed comments):

| docker-compose.prod.yml | here |
|---|---|
| `name: ask-wiki-rag` (tenant isolation) | `namespace.yaml` |
| `image: ghcr.io/...` (image-pull CD) | `deployment.yaml` `image:` (same public image) |
| `mem_limit: 512m` + `cpus: "0.5"` (bulkhead) | `resources.limits` |
| `env_file: .env.prod` (secrets) | `Secret` (created imperatively, never committed) |
| `environment:` flags (tracked) | `configmap.yaml` (two deliberate local deviations, see file) |
| `restart: unless-stopped` | Deployment self-healing + probes |
| shared `edge` Caddy reverse proxy | NodePort + kind `extraPortMappings` (Ingress = stretch) |
| single service, no volumes (ADR-0021 ephemeral) | `replicas: 1`, `strategy: Recreate`, no PVC |

## Runbook

**Live-verified end-to-end 2026-07-11** on Windows 11 + Docker Desktop (WSL2,
cgroup v1) + kind v0.32.0 + kubectl v1.34.1: cluster create ~1 min, pod
Ready in 48 s (including the GHCR pull), `/healthz` 200 via
`localhost:8080`, grounded `/chat/stream` answer with citation identical in
shape to the VPS tenant, and the self-healing demo below (replacement pod
Ready seconds after a kill, healthz 200 again).

Prereqs: Docker Desktop running; `kubectl` + `kind` installed (kind lands via
WinGet at `$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Kubernetes.kind_*\kind.exe`
— **open a fresh shell** so PATH picks it up, or invoke by full path).

```powershell
# 1. Cluster (~1-2 min; downloads the node image on first run)
kind create cluster --name ask-wiki-rag --config kind-cluster.yaml
kubectl cluster-info --context kind-ask-wiki-rag

# 2. Namespace + config
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml

# 3. Secret — imperative, so no key ever touches a file
kubectl -n ask-wiki-rag create secret generic ask-wiki-rag-secrets `
  --from-literal=OPENAI_API_KEY=sk-REPLACE

# 4. Workload + service. The image is PUBLIC on GHCR (verified with an
#    anonymous `docker manifest inspect`), so kind pulls it directly.
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 4b. (only for testing uncommitted changes) build + side-load instead:
#   docker build -t ghcr.io/paynepew/knowledge_base_qa_bot:latest .
#   kind load docker-image ghcr.io/paynepew/knowledge_base_qa_bot:latest --name ask-wiki-rag
#   kubectl -n ask-wiki-rag rollout restart deployment/ask-wiki-rag

# 5. Watch it come up (startupProbe budgets 90s for the import-heavy boot)
kubectl -n ask-wiki-rag get pods -w

# 6. Smoke — same checks the VPS deploy runs
curl.exe http://localhost:8080/healthz          # {"status":"ok"}
# grounded chat (spends real OpenAI tokens, ~a cent):
#   POST /chat/stream, body field is `query`, answer arrives as SSE token
#   frames, the final frame carries {"grounding":{...}}.
curl.exe -N -X POST http://localhost:8080/chat/stream `
  -H "Content-Type: application/json" `
  -d '{\"query\": \"How long do refunds take?\"}'
# Console UI in a browser: http://localhost:8080/console

# 8. Teardown (removes everything, including the pulled node image cache)
kind delete cluster --name ask-wiki-rag
```

## Self-healing demo (30 s, worth doing once)

```powershell
kubectl -n ask-wiki-rag delete pod -l app=ask-wiki-rag   # kill the only pod
kubectl -n ask-wiki-rag get pods -w                       # watch the replacement
```

The replacement pod serves the committed seed again — the k8s analog of
`reset.yml` (ephemeral-by-design, ADR-0021).

## Troubleshooting

- **`kind create cluster` fails at "Starting control-plane"** with kubeadm's
  API POSTs looping on empty responses — check `docker info --format
  '{{.CgroupVersion}}'`. On cgroup **v1** the default node image
  (v1.36.1) crash-loops; `kind-cluster.yaml` pins `kindest/node:v1.33.1`
  for exactly this (observed + fixed 2026-07-11, see comment there).
- `ImagePullBackOff` — check `kubectl -n ask-wiki-rag describe pod ...`. If
  GHCR rate-limits or the package went private, fall back to §4b side-load.
- `OOMKilled` (`describe pod` shows exit 137) — the 512Mi bulkhead is sized for
  chat + 3-page transcribe concurrency. Don't raise the limit alone; see the
  comment pairing in `deployment.yaml`.
- Stuck `Pending` — laptop Docker has < the requested 256Mi/100m free, or the
  NodePort 30080 collided (another cluster?): `kubectl get events -A`.
- `localhost:8080` refused but pod Ready — the cluster was created WITHOUT
  `--config kind-cluster.yaml` (extraPortMappings missing). Recreate it, or
  bypass with `kubectl -n ask-wiki-rag port-forward svc/ask-wiki-rag 8080:80`.
- Probes failing on a slow machine — bump `startupProbe.failureThreshold`
  before touching anything else; boot is import-bound, not broken.

## Stretch (only if the timebox has slack, in this order)

1. `kubectl -n ask-wiki-rag logs -f deploy/ask-wiki-rag` while running a chat —
   see the gateway log lines you know from the VPS.
2. ingress-nginx on kind + an Ingress manifest — the honest analog of the
   shared `edge` Caddy.
3. Kustomize overlay splitting local-vs-prod config (the two ConfigMap
   deviations become a patch instead of a comment).

## What this deliberately does NOT claim

No HA (replicas pinned to 1 by the single-process design — the manifest comment
is the interview answer), no PVC (ephemeral is a product decision, ADR-0021),
no ingress/TLS (edge Caddy owns that on the VPS), no HPA (the overload gate is
in-process 503 shedding, issue #269 — autoscaling on top of a per-pod budget
ledger would multiply the budget, not the capacity).
