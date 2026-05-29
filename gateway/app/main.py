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
"""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Env must be loaded BEFORE importing app modules — retrieval reads
# KB_SCORE_THRESHOLD at import time.
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# Import sub-apps AFTER env is loaded so retrieval singletons see OPENAI_API_KEY.
from markdown_kb.app.main import app as _wiki_app  # noqa: E402
from vector_rag.app.main import app as _rag_app  # noqa: E402

from .routes import router  # noqa: E402

# Path constants — resolved relative to this file so they work regardless of cwd.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_UI_PATH = _STATIC_DIR / "index.html"
_CONSOLE_PATH = _STATIC_DIR / "console.html"

app = FastAPI(title="KB Gateway", version="0.1.0")
app.include_router(router)
app.mount("/wiki", _wiki_app)
app.mount("/rag", _rag_app)
# Serve shared.css (and any future static assets) at /static/<filename>.
# Must be mounted AFTER include_router so /static doesn't shadow API routes.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


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
