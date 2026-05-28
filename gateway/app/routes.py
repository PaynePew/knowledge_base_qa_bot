"""Shallow module per Ousterhout. Public surface: ``router``.

Gateway HTTP wiring for ``POST /chat/stream``.

Phase 9 Slice 1 — Wiki SSE happy-path tracer bullet (ADR-0009, ADR-0010).
All streaming complexity lives in markdown_kb.app.retrieval.stream_query()
and markdown_kb.app.sse.events_for_result(); this module is a shallow
dispatcher (CODING_STANDARD §2.3).

``POST /chat/stream?stack=wiki`` dispatches to the Wiki Retrieval Stack.
The ``stack=rag`` path arrives in a later slice.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from markdown_kb.app.retrieval import stream_query
from markdown_kb.app.schemas import ChatRequest
from markdown_kb.app.sse import encode_event, events_for_result

router = APIRouter()


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, stack: str = "wiki") -> StreamingResponse:
    """Stream a grounded answer as SSE events.

    Dispatches to the selected Retrieval Stack (Phase 9 Slice 1: Wiki only).

    SSE event order (ADR-0009 verify-then-stream):
      1. ``sources`` — emitted immediately after retrieval, before any LLM call.
         Carries the list of retrieved sources (citation id, heading, content
         snippet, derived_from). This is the genuine latency win — BM25 is
         ~instant; the user sees grounding context while the LLM drafts (~4-7s).
      2. ``token``(s) — words of the verified answer only; no unverified draft.
      3. ``done`` — grounding outcome (passed, reason) and optional filing status.

    Args:
        req: ChatRequest body with ``query`` field.
        stack: Query param selecting the Retrieval Stack (default ``wiki``).
            Only ``wiki`` is implemented in this slice; ``rag`` returns 501.

    Returns:
        StreamingResponse with media_type ``text/event-stream``.

    Raises:
        HTTPException 400: unrecognised stack value.
        HTTPException 501: stack is recognised but not yet implemented.
    """
    if stack not in ("wiki", "rag"):
        raise HTTPException(
            status_code=400, detail=f"Unknown stack={stack!r}. Use 'wiki' or 'rag'."
        )

    if stack == "rag":
        raise HTTPException(
            status_code=501, detail="stack=rag not yet implemented (Phase 9 Slice 2)."
        )

    def _sse_generator():
        """Consume stream_query() and yield SSE frames.

        stream_query() yields exactly two dicts:
          1. sources_ready partial (before LLM) — emit sources event.
          2. full result (after LLM + verify) — emit token(s) + done events.

        This separation ensures the sources event is always emitted BEFORE
        any LLM call starts (ADR-0009 §"sources-first is real, not post-hoc").
        """
        gen = stream_query(req.query)

        # First yield: sources_ready partial — emit sources event only.
        partial = next(gen)
        source_list = [
            {
                "source": s["source"],
                "heading": s["heading"],
                "content": s["content"],
                "derived_from": s.get("derived_from"),
            }
            for s in partial.get("sources", [])
        ]
        yield encode_event("sources", {"sources": source_list})

        # Second yield: full result — emit token(s) + done.
        full_result = next(gen)
        # events_for_result emits sources + token(s) + done; we skip the
        # sources frame here (already emitted above) and forward the rest.
        all_frames = events_for_result(full_result)
        # Skip the first frame (sources) — it was already sent.
        yield from all_frames[1:]

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
