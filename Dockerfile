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
# eval/paraphrase_comparison). The deployable `gateway` imports markdown_kb AND
# vector_rag, but gateway/pyproject doesn't declare vector-rag — so the build
# can't be scoped with `uv sync --package gateway`. We sync the whole workspace
# with `--all-packages` (see the RUN below for why that flag is load-bearing).
# `--no-dev` keeps eval's deepeval/matplotlib/anthropic and everyone's
# pytest/ruff out of the image.

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

# Install the whole workspace into /app/.venv in ONE sync after copying the full
# source. `--all-packages` is REQUIRED: a plain `uv sync` installs only the ROOT
# project's deps (python-dotenv) and leaves every member dep (uvicorn, fastapi,
# langchain, faiss-cpu, ...) UNINSTALLED — the container would then exit with
# "Failed to spawn: uvicorn" on boot. We do NOT split out a deps-only
# `--no-install-project` pre-source layer: `--all-packages` builds the
# package=true members (kb_cli, kb_mcp), which need their source present, so that
# combination fails before `COPY . .`. The uv cache mount keeps wheels warm so a
# single post-source sync still rebuilds fast.
# `.dockerignore` keeps the baked seed (.kb/ wiki/ docs/) and drops eval/ source;
# eval's manifest survives (re-included) so workspace resolution still succeeds.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-packages

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
# singletons, append-only Wiki Log). Invoke the venv's uvicorn directly (it is on
# PATH via /app/.venv/bin) rather than `uv run`, which would re-resolve the env at
# boot — slower and fragile in the runtime stage.
CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
