# syntax=docker/dockerfile:1
#
# Container build for the knowledge_base_qa_bot Gateway (issue #270, deploy S2).
#
# VPS deploy as tenant `ask-wiki-rag` via image-pull from GHCR. The image bakes
# the curated seed (`.kb/index.json`, `.kb/faiss_index/*`, `wiki/`) so a freshly
# pulled container can answer `/chat` immediately — no ingest/index at boot and
# no OpenAI key needed at *build* time (FAISS is NOT rebuilt here).
#
# uv workspace has SIX members (markdown_kb, vector_rag, gateway, kb_cli, kb_mcp,
# eval/paraphrase_comparison). `uv sync --frozen` resolves the whole workspace,
# so every member manifest must be present before the deps-only sync or it fails
# (`uv sync --package gateway` can't be scoped — gateway/pyproject doesn't declare
# vector-rag though main.py imports it). `--no-dev` keeps eval's
# deepeval/matplotlib/anthropic and everyone's pytest/ruff out of the image.

# --- builder: resolve + install the workspace into a self-contained .venv ------
FROM python:3.11-slim AS builder

# Pin uv from its official image (no curl|sh, reproducible).
COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /uvx /bin/

# uv settings: compile bytecode for faster cold starts, copy (not symlink) the
# wheel cache so the .venv is portable into the runtime stage, and target the
# in-tree .venv.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Step 1 — deps only. COPY the lockfile + python pin + the root manifest, then
# ALL SIX member manifests (workspace resolution reads every member's pyproject).
# `--no-install-project` installs only third-party deps, so this layer is cached
# across source-only edits.
COPY pyproject.toml uv.lock .python-version ./
COPY markdown_kb/pyproject.toml markdown_kb/pyproject.toml
COPY vector_rag/pyproject.toml vector_rag/pyproject.toml
COPY gateway/pyproject.toml gateway/pyproject.toml
COPY kb_cli/pyproject.toml kb_cli/pyproject.toml
COPY kb_mcp/pyproject.toml kb_mcp/pyproject.toml
COPY eval/paraphrase_comparison/pyproject.toml eval/paraphrase_comparison/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Step 2 — project source. `.dockerignore` keeps the baked seed (.kb/ wiki/ docs/)
# and drops eval/ source, but eval's manifest was COPYed above so the workspace
# still resolves. The second sync builds the in-tree package members (kb_cli,
# kb_mcp) now that their source is present.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime: slim image carrying only the built venv + app source -------------
FROM python:3.11-slim AS runtime

# Copy uv so the CMD's `uv run` works (it just activates the already-built venv).
COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /uvx /bin/

# PYTHONPATH=/app puts the repo root on sys.path so the `markdown_kb` /
# `vector_rag` namespace packages resolve the same way the test harness wires
# them (CLAUDE.md memory: entry points need repo root on sys.path).
ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Bring the fully-built workspace (venv + source + baked seed) from the builder.
COPY --from=builder /app /app

# Container-internal port only — no host ports are baked into the image; the
# host maps a published port at `docker run -p` / compose time.
EXPOSE 8000

# Single worker per CODING_STANDARD's single-process assumption (in-process
# singletons, append-only Wiki Log). `--no-dev` keeps the dev group out of the
# resolution `uv run` would otherwise re-check.
CMD ["uv", "run", "--no-dev", "uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
