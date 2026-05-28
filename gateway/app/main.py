"""Shallow module per Ousterhout. Public surface: ``app`` (the FastAPI Gateway instance).

Phase 9 Gateway entrypoint (ADR-0010).

The Gateway is a thin parent ASGI app that:
  - Mounts the markdown_kb sub-app at ``/wiki`` (ADR-0010).
  - Serves a bare debug HTML page at ``/`` (CODING_STANDARD §12.1 — vanilla
    single-file, no framework, no build step).
  - Exposes ``POST /chat/stream?stack=wiki`` that dispatches in-process to
    the Wiki stack's stream_query() and emits SSE events per ADR-0009.

Phase 9 walking skeleton: Wiki stack only.  The ``stack=rag`` path and the
toggle UI arrive in later slices (#118 onward per PRD #116).
"""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

# Env must be loaded BEFORE importing app modules — retrieval reads
# KB_SCORE_THRESHOLD at import time.
load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

# Import and mount the markdown_kb sub-app AFTER env is loaded.
from markdown_kb.app.main import app as _wiki_app  # noqa: E402

from .routes import router  # noqa: E402

app = FastAPI(title="KB Gateway", version="0.1.0")
app.include_router(router)
app.mount("/wiki", _wiki_app)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def debug_ui() -> str:
    """Serve the bare debug HTML page (CODING_STANDARD §12.1, §12.2, §12.4).

    A single inline HTML file that:
    - Uses fetch() + ReadableStream (§12.2 — POST streaming; EventSource is GET-only).
    - Inserts all server/LLM-derived content via textContent, never innerHTML (§12.4 XSS).
    - Renders sources first, then token events, then done (§12.3 sources-first invariant).
    - Has no styling, no framework, no build step (§12.1 — debug page only).
    """
    return DEBUG_HTML


# ---------------------------------------------------------------------------
# Debug HTML (bare — no styling, §12.1)
# ---------------------------------------------------------------------------

DEBUG_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KB Gateway Debug</title>
</head>
<body>
<h1>KB Gateway — Debug UI</h1>
<form id="qform">
  <label for="q">Query:</label>
  <input id="q" type="text" size="60" value="What is the refund policy?">
  <button type="submit">Ask (stack=wiki)</button>
</form>

<h2>Sources</h2>
<pre id="sources">(waiting)</pre>

<h2>Answer</h2>
<pre id="answer">(waiting)</pre>

<h2>Done</h2>
<pre id="done">(waiting)</pre>

<h2>Raw SSE events</h2>
<pre id="raw"></pre>

<script>
// SSE client per CODING_STANDARD §12.2:
//   - Uses fetch() + ReadableStream (required: POST streaming; §12.2).
//   - All server/LLM-derived content set via textContent (not innerHTML) per §12.4.
//   - Sources render first; answer area only populated after sources event (§12.3).

const qform = document.getElementById("qform");
const sourcesEl = document.getElementById("sources");
const answerEl = document.getElementById("answer");
const doneEl = document.getElementById("done");
const rawEl = document.getElementById("raw");

qform.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = document.getElementById("q").value.trim();
  if (!query) return;

  sourcesEl.textContent = "(requesting...)";
  answerEl.textContent = "(waiting for sources first...)";
  doneEl.textContent = "";
  rawEl.textContent = "";

  let sourcesReceived = false;
  let answerTokens = [];

  const resp = await fetch("/chat/stream?stack=wiki", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!resp.ok) {
    sourcesEl.textContent = "Error: " + resp.status + " " + resp.statusText;
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // Parse SSE frames: frames are separated by double newline.
    const frames = buf.split("\\n\\n");
    buf = frames.pop();  // incomplete last frame stays in buffer

    for (const frame of frames) {
      if (!frame.trim()) continue;
      const lines = frame.split("\\n");
      let eventType = "message";
      let dataStr = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) eventType = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataStr = line.slice(6);
      }
      rawEl.textContent += frame + "\\n\\n";

      let data;
      try { data = JSON.parse(dataStr); } catch { continue; }

      if (eventType === "sources") {
        sourcesReceived = true;
        // textContent only — no innerHTML per §12.4.
        sourcesEl.textContent = JSON.stringify(data.sources, null, 2);
        answerEl.textContent = "(receiving answer...)";
      } else if (eventType === "token" && sourcesReceived) {
        // Append token text (sources-first invariant: only render after sources).
        answerTokens.push(data.text);
        answerEl.textContent = answerTokens.join(" ");
      } else if (eventType === "done") {
        doneEl.textContent = JSON.stringify(data, null, 2);
      }
    }
  }
});
</script>
</body>
</html>
"""
