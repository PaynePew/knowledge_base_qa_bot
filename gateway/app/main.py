"""Shallow module per Ousterhout. Public surface: ``app`` (the FastAPI Gateway instance).

Phase 9 Gateway entrypoint (ADR-0010).

The Gateway is a thin parent ASGI app that:
  - Mounts the markdown_kb sub-app at ``/wiki`` (ADR-0010).
  - Serves the reader browser UI at ``/`` — a vanilla single-file HTML page
    (CODING_STANDARD §12.1). Shipped from ``gateway/static/index.html``.
  - Serves the Operator Console at ``/console`` — a second vanilla single-file
    HTML page (Phase 15 S1 / issue #169). Shipped from
    ``gateway/static/console.html``.
  - Serves ``/static/shared.css`` (shared design tokens, Phase 15 S1).
  - Exposes ``POST /chat/stream?stack=wiki|rag`` that dispatches in-process to
    the selected stack's stream_query() and emits SSE events per ADR-0009.
  - Exposes ``POST /upload`` (multipart) that delegates to
    ``markdown_kb.app.upload.upload_files`` (ADR-0011).
  - Installs ``ProdMiddleware`` (issue #269): read/admin concurrency caps, a
    daily USD budget guard, an optional admin-token kill-switch, and a
    graceful OpenAI-quota→503 mapping for non-streaming heavy paths.
  - Exposes ``GET /healthz`` (always 200 liveness) and ``GET /healthz/shed``
    (200 normally, 503 when the read semaphore is saturated — edge-active
    health check).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Env must be loaded BEFORE importing app modules — retrieval reads
# KB_SCORE_THRESHOLD at import time.
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# Import sub-apps AFTER env is loaded so retrieval singletons see OPENAI_API_KEY.
from markdown_kb.app.main import app as _wiki_app  # noqa: E402
from vector_rag.app.main import app as _rag_app  # noqa: E402

from .middleware import ProdMiddleware, read_saturated  # noqa: E402
from .routes import router  # noqa: E402

# Path constants — resolved relative to this file so they work regardless of cwd.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_UI_PATH = _STATIC_DIR / "index.html"
_CONSOLE_PATH = _STATIC_DIR / "console.html"
_FAVICON_PATH = _STATIC_DIR / "favicon.svg"

# The favicon never changes between deploys, but StaticFiles only sends
# ETag/Last-Modified — so the browser re-validates it on every page load (a 304
# round-trip per request). A max-age lets the browser serve it from cache without
# asking, so those repeated conditional requests disappear. Bust on redesign with
# a hard refresh (the mark is tiny; a day of staleness is harmless).
_FAVICON_CACHE = "public, max-age=86400"

app = FastAPI(title="KB Gateway", version="0.1.0")
# Production overload + cost-protection guard (issue #269).  Added BEFORE the
# routes/mounts so it wraps every request to the parent app and both sub-apps.
app.add_middleware(ProdMiddleware)
app.include_router(router)
app.mount("/wiki", _wiki_app)
app.mount("/rag", _rag_app)
# Serve shared.css (and any future static assets) at /static/<filename>.
# Must be mounted AFTER include_router so /static doesn't shadow API routes.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/healthz", include_in_schema=False)
def healthz() -> JSONResponse:
    """Liveness probe — ALWAYS 200 (issue #269 AC1).

    Stays 200 even when the daily budget is exhausted or the read semaphore is
    saturated: a non-200 here would make the box's orchestrator (Docker
    restart-policy) kill an otherwise-healthy, merely-busy worker, turning a
    transient overload into a restart loop.  Readiness/shedding lives in the
    separate ``/healthz/shed`` probe.
    """
    return JSONResponse({"status": "ok"})


@app.get("/healthz/shed", include_in_schema=False)
def healthz_shed() -> Response:
    """Readiness / load-shed probe — 200 normally, 503 when read-saturated.

    Reflects ONLY the read semaphore's saturation (issue #269 AC2): the edge
    load balancer drains this box from the read pool while it is full, then
    re-adds it when a slot frees.  Admin-semaphore saturation does NOT flip
    this — index/maintenance load must not shed reader traffic at the edge.
    """
    if read_saturated():
        return JSONResponse({"status": "shedding"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui() -> str:
    """Serve the Gateway reader UI (CODING_STANDARD §12.1, §12.2, §12.4).

    The UI is a vanilla single HTML/CSS/JS file (no framework, no build step)
    stored at ``gateway/static/index.html``.  It:
    - Uses fetch() + ReadableStream (§12.2 — POST streaming; EventSource is GET-only).
    - Inserts all server/LLM-derived content via textContent, never innerHTML (§12.4 XSS).
    - Renders sources first; answer area only appears after the sources event (§12.3).
    - Stack toggle maps to the ``stack`` query param; switching is a fresh request (§12.3).
    - done.grounding.passed drives the grounding badge; done.filed the filed indicator (§12.3).
    - Shared design tokens linked from ``/static/shared.css`` (Phase 15 S1).
    - Console button in masthead navigates to ``GET /console`` (Phase 15 S1).
    """
    return _UI_PATH.read_text(encoding="utf-8")


@app.get("/favicon.svg", include_in_schema=False)
def serve_favicon_svg() -> FileResponse:
    """Serve the site favicon with a cacheable ``Cache-Control`` header.

    Both HTML pages declare ``<link rel="icon" href="/favicon.svg">`` and fetch it
    here rather than via the ``/static`` mount: StaticFiles only sets
    ETag/Last-Modified, so the browser re-validates the favicon on every load (a
    304 round-trip each time). Serving it from a dedicated route lets us attach a
    ``max-age`` so the browser caches it outright and stops re-requesting it.
    """
    return FileResponse(
        _FAVICON_PATH,
        media_type="image/svg+xml",
        headers={"Cache-Control": _FAVICON_CACHE},
    )


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon() -> FileResponse:
    """Serve the site favicon for the literal ``/favicon.ico`` request.

    The HTML pages declare ``<link rel="icon" href="/favicon.svg">`` so modern
    browsers fetch the SVG directly, but bare ``/favicon.ico`` hits (older
    clients, crawlers, prefetchers) would otherwise 404. This route returns the
    same single-color SVG mark with the correct media type — and the same
    ``Cache-Control`` — so those requests resolve instead of logging a 404.
    """
    return FileResponse(
        _FAVICON_PATH,
        media_type="image/svg+xml",
        headers={"Cache-Control": _FAVICON_CACHE},
    )


@app.get("/console", response_class=HTMLResponse, include_in_schema=False)
def serve_console() -> str:
    """Serve the Operator Console UI (Phase 15 S1, issue #169, ADR-0010).

    A second vanilla single-file HTML page — the curator-facing write/maintain
    surface, served by the Gateway alongside the reader UI.  Per ADR-0010 the
    Gateway is the composition layer; new curator-facing pages live here, not
    on the sub-apps.

    The Console:
    - Links ``/static/shared.css`` for shared design tokens.
    - Provides a ``console`` breadcrumb in the masthead with a ``reader`` link back.
    - Wires the drag-drop Upload drop zone to ``POST /upload`` (ADR-0011).
    - Inserts all server-derived content via textContent (CODING_STANDARD §12.4).
    """
    return _CONSOLE_PATH.read_text(encoding="utf-8")
