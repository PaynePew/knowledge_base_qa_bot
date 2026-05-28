"""Shallow module per Ousterhout. Public surface: ``router``.

Gateway HTTP wiring for ``POST /chat/stream``.

Phase 9 Slice 1 — Wiki SSE happy-path tracer bullet (ADR-0009, ADR-0010).
Phase 9 Slice 2 (issue #119) — Full SSE event contract: status event (liveness
during the draft+verify gap), terminal error event (post-sources LLM/infra
failure), and uniform Cannot Confirm for all five CC reasons.
Phase 9 Slice 3 (issue #120) — RAG dispatch added; ``stack=rag`` routes to
``vector_rag.app.retrieval.stream_query``.

All streaming complexity lives in the per-stack ``stream_query()`` functions and
the shared ``markdown_kb.app.sse.events_for_result()`` serializer; this module is
a shallow dispatcher (CODING_STANDARD §2.3).  The dispatch mapping
(``_STACK_STREAM_FN``) is the only place that knows which stream function to call
per stack — the generator body is identical for both (ADR-0010: gateway is the
composition layer).  RAG sources carry ONLY citation id + heading + content (no
score, no derived_from — issue #120); the shared serializer renders them as-is.

SSE event contract (ADR-0009):
  sources            — immediately after retrieval (real latency win)
  status{phase}      — liveness between sources and first token; LLM path only
  token(s)           — verified answer or CANNOT_CONFIRM_PHRASE; one per word
  done{grounding:{passed,reason},filed,stack} — terminal success event (PRD #116
                       shape; filed null for stack=rag)
  error{detail,retryable}   — terminal failure when the LLM errors AFTER sources
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

# Per-stack dispatch mapping.  Adding a new stack = one entry here; the
# generator body below is identical for every stack (ADR-0010).
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
         Carries the retrieved sources. For ``stack=wiki``: citation id, heading,
         content snippet, derived_from. For ``stack=rag``: citation id, heading,
         content snippet only (no score, no derived_from). This is the genuine
         latency win — retrieval is ~instant; the user sees grounding context
         while the LLM drafts (~4-7s).
      2. ``status`` — liveness signal emitted after sources, before the first
         token, on the LLM path only (early-exit CC paths skip this).  Payload:
         ``{phase: "verifying"}`` — tells the UI the draft+verify step is running.
      3. ``token``(s) — words of the verified answer only (or CANNOT_CONFIRM_PHRASE
         on any CC path); no unverified draft ever reaches the stream.
      4. ``done`` — grounding outcome (passed, reason) and optional filing status.
         ``done.filed`` is always null for ``stack=rag`` (RAG never files).
         OR
         ``error`` — terminal failure event when the LLM/infra errors AFTER the
         sources event has been committed (HTTP 200 already sent).  Payload:
         ``{detail, retryable}``.  No ``done`` event follows an ``error``.

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

        Event emission (Slices 2+3):
          - status{phase:"verifying"} emitted after sources, before the LLM
            call, on the LLM path only (early_exit=False).
          - On the LLM path, _draft_and_verify raises HTTPException on LLM/infra
            error.  We catch it here and emit a terminal error event instead of
            propagating (HTTP 200 is already committed; no done follows).
          - Cannot Confirm paths (early_exit=True) always stream
            CANNOT_CONFIRM_PHRASE — stream_query normalises index_missing to
            CANNOT_CONFIRM_PHRASE so the token stream is uniform across all 5
            CC reasons (done.reason still carries the specific gate).
          - ``stream_fn`` is selected per stack via _STACK_STREAM_FN; the body
            is identical for wiki and rag (rag never files → done.filed null).
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

        # Emit status{phase:"verifying"} on the LLM path only (early_exit=False).
        # Early-exit CC paths skip the LLM entirely — emitting "verifying" there
        # would be a lie; the answer is already known from the pre-LLM gate.
        if not partial.get("early_exit", False):
            yield encode_event("status", {"phase": "verifying"})

        # Second yield: full result — emit token(s) + done.
        # On LLM/infra error (HTTPException raised by _draft_and_verify), HTTP
        # 200 is already committed, so we emit a terminal error event instead of
        # propagating the exception (which would silently truncate the stream).
        try:
            full_result = next(gen)
        except HTTPException as exc:
            retryable = exc.status_code == 503
            yield encode_event("error", {"detail": exc.detail, "retryable": retryable})
            return
        except Exception:  # noqa: BLE001 — intentional last-resort catch
            # Defense in depth (ADR-0009): HTTP 200 is already committed once the
            # sources event is sent, so ANY error during draft/verify — including
            # unmapped or unexpected ones (e.g. an LLM client misconfiguration that
            # raises a base openai.OpenAIError rather than a mapped HTTPException) —
            # must surface as a terminal SSE error event, never a silently
            # truncated stream that hangs the client. Detail is generic to avoid
            # leaking internals; mapped transient/auth errors keep their curated
            # detail in the HTTPException branch above.
            yield encode_event(
                "error",
                {"detail": "Internal error while generating the answer.", "retryable": False},
            )
            return

        # events_for_result emits sources + token(s) + done; we skip the
        # sources frame here (already emitted above) and forward the rest.
        # Pass `stack` so done.stack reflects the dispatched retrieval stack
        # (the serializer is stack-agnostic; only the Gateway knows the stack).
        all_frames = events_for_result(full_result, stack=stack)
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
