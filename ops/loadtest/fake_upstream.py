"""Local fake OpenAI-compatible upstream, started for every scenario.

The gateway's ``/chat/stream`` route always constructs a real ``ChatOpenAI``
(no stub-LLM env flag exists — see issue #600's technical brief), and
``vector_rag``'s sub-app lifespan requires *some* ``OPENAI_API_KEY`` to be
present just to boot (see ``scenarios.py``'s module docstring) — so every
scenario, including import-only ones, runs against this local double of the
two endpoints the app actually calls rather than a real OpenAI endpoint.
Pointed at via ``OPENAI_API_BASE`` env only, no app code touched.

Shape verified against the real request payloads (issue #600 implementation
pass, ``langchain-openai`` installed in this repo's venv):

- The plain draft call (``markdown_kb/app/retrieval.py::get_llm().invoke(...)``)
  sends a normal chat-completions body with no ``response_format`` — answered
  with a fixed plain-text assistant message.
- The grounding verifier (``markdown_kb/app/grounding.py``,
  ``llm.with_structured_output(GroundingResult)``) sends ``response_format``
  of type ``json_schema`` (langchain-openai's current default method) — answered
  with a JSON string that parses into ``GroundingResult`` (reasoning + empty
  claims + ``passed: true``; the grounding model has no cross-field validator
  requiring non-empty claims, confirmed against ``markdown_kb/app/grounding.py``).
- ``POST /v1/embeddings`` is a defensive stub (not expected to be hit by the
  in-scope wiki-only scenarios — warmup only *constructs* embedding clients
  when a key is present, never calls the API unless ``KB_WARMUP_PING`` is
  truthy, which the harness never sets) so a surprise embedding call degrades
  gracefully instead of hanging on a real request to api.openai.com.

Never talks to a real OpenAI endpoint and never spends a real token.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI, Request

_STUB_ANSWER = "This is a stub grounded answer produced by the load-test fake upstream."

_STUB_GROUNDING_RESULT = {
    "reasoning": "stub verifier: treating every draft as fully supported.",
    "claims": [],
    "unsupported_claims": [],
    "passed": True,
}


def _make_app() -> FastAPI:
    app = FastAPI(title="loadtest-fake-openai")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict:
        body = await request.json()
        is_structured = "response_format" in body or "tools" in body
        content = json.dumps(_STUB_GROUNDING_RESULT) if is_structured else _STUB_ANSWER
        return {
            "id": "chatcmpl-loadtest-stub",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "gpt-4o-mini"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> dict:
        body = await request.json()
        raw_input = body.get("input", [])
        n = len(raw_input) if isinstance(raw_input, list) else 1
        dim = int(body.get("dimensions") or 1536)
        vector = [0.0] * dim
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vector}
                for i in range(n)
            ],
            "model": body.get("model", "text-embedding-3-small"),
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    @app.get("/v1/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


@contextmanager
def run_fake_upstream(port: int, host: str = "127.0.0.1") -> Iterator[str]:
    """Run the fake upstream on a background thread for the ``with`` block's lifetime.

    Yields the base URL (``http://<host>:<port>/v1``) to point ``OPENAI_API_BASE``
    at. This is an in-process helper thread scoped to one synchronous harness
    invocation (starts and stops within a single ``run`` command) — not a
    detached background process.
    """
    config = uvicorn.Config(_make_app(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, name="loadtest-fake-upstream", daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("fake upstream did not start within 10s")
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
