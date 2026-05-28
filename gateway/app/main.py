"""Shallow module per Ousterhout. Public surface: ``app`` (the FastAPI Gateway instance).

Phase 9 Gateway entrypoint (ADR-0010).

The Gateway is a thin parent ASGI app that:
  - Mounts the markdown_kb sub-app at ``/wiki`` (ADR-0010).
  - Serves the chosen browser UI (Phase 9 Slice 6 / issue #122) at ``/`` —
    a vanilla single-file HTML page with no framework and no build step
    (CODING_STANDARD §12.1). Wired to the real Gateway SSE stream via
    fetch() + ReadableStream (§12.2). Shipped from ``gateway/static/index.html``.
  - Exposes ``POST /chat/stream?stack=wiki|rag`` that dispatches in-process to
    the selected stack's stream_query() and emits SSE events per ADR-0009.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Env must be loaded BEFORE importing app modules — retrieval reads
# KB_SCORE_THRESHOLD at import time.
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

# Import sub-apps AFTER env is loaded so retrieval singletons see OPENAI_API_KEY.
from markdown_kb.app.main import app as _wiki_app  # noqa: E402
from vector_rag.app.main import app as _rag_app  # noqa: E402

from .routes import router  # noqa: E402

# Path to the production UI HTML file (gateway/static/index.html).
# Resolved relative to this file so it works regardless of cwd.
_UI_PATH = Path(__file__).resolve().parent.parent / "static" / "index.html"

app = FastAPI(title="KB Gateway", version="0.1.0")
app.include_router(router)
app.mount("/wiki", _wiki_app)
app.mount("/rag", _rag_app)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui() -> str:
    """Serve the Gateway browser UI (CODING_STANDARD §12.1, §12.2, §12.4).

    The UI is a vanilla single HTML/CSS/JS file (no framework, no build step)
    stored at ``gateway/static/index.html``.  It:
    - Uses fetch() + ReadableStream (§12.2 — POST streaming; EventSource is GET-only).
    - Inserts all server/LLM-derived content via textContent, never innerHTML (§12.4 XSS).
    - Renders sources first; answer area only appears after the sources event (§12.3).
    - Stack toggle maps to the ``stack`` query param; switching is a fresh request (§12.3).
    - done.grounding.passed drives the grounding badge; done.filed the filed indicator (§12.3).
    """
    return _UI_PATH.read_text(encoding="utf-8")
