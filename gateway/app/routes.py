"""Shallow module per Ousterhout. Public surface: ``router``.

Gateway HTTP wiring for ``POST /chat/stream``.

Phase 9 Slice 1 — Wiki SSE happy-path tracer bullet (ADR-0009, ADR-0010).
Phase 9 Slice 2 (issue #120) — RAG dispatch added; ``stack=rag`` now routes
to ``vector_rag.app.retrieval.stream_query``.

All streaming complexity lives in the per-stack ``stream_query()`` functions and
the shared ``markdown_kb.app.sse.events_for_result()`` serializer; this module is
a shallow dispatcher (CODING_STANDARD §2.3).  RAG sources carry ONLY citation id
+ heading + content (no score, no derived_from — issue #120 spec); the shared
serializer renders them without modification.

The dispatch mapping (``_STACK_STREAM_FN``) is the only place that knows which
stream function to call for each stack — the generator body is identical for
both (ADR-0010: gateway is the composition layer).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from markdown_kb.app.retrieval import stream_query as _wiki_stream_query
from markdown_kb.app.schemas import ChatRequest
from markdown_kb.app.sse import encode_event, events_for_result
from vector_rag.app.retrieval import stream_query as _rag_stream_query

router = APIRouter()

# Per-stack dispatch mapping.  Adding a new stack = one entry here.
# Kept near the top of the handler (PARALLEL-WORK NOTE: #119 edits the
# generator body below — this dict is the only change this slice makes to
# the dispatch logic, so merge conflicts are minimised).
_STACK_STREAM_FN: dict[str, Callable[[str], Iterator[dict]]] = {
    "wiki": _wiki_stream_query,
    "rag": _rag_stream_query,
}


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, stack: str = "wiki") -> StreamingResponse:
    """Stream a grounded answer as SSE events.

    Dispatches to the selected Retrieval Stack via ``_STACK_STREAM_FN``.

    SSE event order (ADR-0009 verify-then-stream):
      1. ``sources`` — emitted immediately after retrieval, before any LLM call.
         Carries the list of retrieved sources.  For ``stack=wiki``: citation id,
         heading, content snippet, derived_from.  For ``stack=rag``: citation id,
         heading, content snippet only (no score, no derived_from).
      2. ``token``(s) — words of the verified answer only; no unverified draft.
      3. ``done`` — grounding outcome (passed, reason) and optional filing status.
         ``done.filed`` is always null for ``stack=rag`` (RAG never files).

    Args:
        req: ChatRequest body with ``query`` field.
        stack: Query param selecting the Retrieval Stack (default ``wiki``).
            ``wiki`` — Wiki BM25 stack (markdown_kb).
            ``rag``  — Vector RAG stack (vector_rag).

    Returns:
        StreamingResponse with media_type ``text/event-stream``.

    Raises:
        HTTPException 400: unrecognised stack value.
    """
    if stack not in _STACK_STREAM_FN:
        raise HTTPException(
            status_code=400, detail=f"Unknown stack={stack!r}. Use 'wiki' or 'rag'."
        )

    stream_fn = _STACK_STREAM_FN[stack]

    def _sse_generator():
        """Consume stream_fn() and yield SSE frames.

        stream_fn() (for any stack) yields exactly two dicts:
          1. sources_ready partial (before LLM) — emit sources event.
          2. full result (after LLM + verify) — emit token(s) + done events.

        This separation ensures the sources event is always emitted BEFORE
        any LLM call starts (ADR-0009 §"sources-first is real, not post-hoc").

        PARALLEL-WORK NOTE (slice #119): #119 adds status/error events to this
        generator body.  This slice keeps the body structurally identical to the
        Slice 1 version so #119's edits apply cleanly — only the ``stream_fn``
        binding above is new.
        """
        gen = stream_fn(req.query)

        # First yield: sources_ready partial — emit sources event only.
        partial = next(gen)
        # Pass through only the keys that each stack's source dict carries.
        # Wiki sources carry: source, heading, content, derived_from.
        # RAG sources carry: source, heading, content (NO derived_from, NO score).
        # Using dict comprehension over allowed keys + conditional derived_from
        # preserves the per-stack contract without coupling this layer to
        # individual field names beyond the mandatory three.
        source_list = []
        for s in partial.get("sources", []):
            entry: dict = {
                "source": s["source"],
                "heading": s["heading"],
                "content": s["content"],
            }
            if "derived_from" in s:
                entry["derived_from"] = s["derived_from"]
            source_list.append(entry)
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
