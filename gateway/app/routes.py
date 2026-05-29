"""Shallow module per Ousterhout. Public surface: ``router``.

Gateway HTTP wiring for ``POST /chat/stream``.

Phase 9 Slice 1 — Wiki SSE happy-path tracer bullet (ADR-0009, ADR-0010).
Phase 9 Slice 2 (issue #119) — Full SSE event contract: status event (liveness
during the draft+verify gap), terminal error event (post-sources LLM/infra
failure), and uniform Cannot Confirm for all five CC reasons.
Phase 9 Slice 3 (issue #120) — RAG dispatch added; ``stack=rag`` routes to
``vector_rag.app.retrieval.stream_query``.
Phase 11 Slice 1 (issue #159) — Conversation Memory tracer bullet: session
lifecycle, Query Rewriting (Wiki turn 2+), Conversation Store write-on-done,
``done.session`` field injection.  Sub-apps unchanged (ADR-0010).
Phase 11 Slice 4 (issue #162) — status:{phase:"rewriting"} SSE event emitted
inside the generator on turn 2+ (before sources); rewrite_query moved into
_sse_generator so errors surface as terminal SSE error events.

All streaming complexity lives in the per-stack ``stream_query()`` functions and
the shared ``markdown_kb.app.sse.events_for_result()`` serializer; this module is
a shallow dispatcher (CODING_STANDARD §2.3).  The dispatch mapping
(``_STACK_STREAM_FN``) is the only place that knows which stream function to call
per stack — the generator body is identical for both (ADR-0010: gateway is the
composition layer).  RAG sources carry ONLY citation id + heading + content (no
score, no derived_from — issue #120); the shared serializer renders them as-is.

SSE event contract (ADR-0009, extended by Phase 11):
  status{phase:"rewriting"}  — emitted on turn 2+ BEFORE sources (Phase 11 Slice 4)
  sources            — immediately after retrieval (real latency win)
  status{phase:"verifying"}  — liveness between sources and first token; LLM path only
  token(s)           — verified answer or CANNOT_CONFIRM_PHRASE; one per word
  done{grounding:{passed,reason},filed,stack,session} — terminal success event
                       (PRD #116 shape + Phase 11 session field; filed null for
                       stack=rag)
  error{detail,retryable}   — terminal failure when the LLM or rewriter errors
                              AFTER HTTP 200 is committed (sources or rewriting
                              event already sent)
"""

from __future__ import annotations

import datetime
import json
import uuid
from collections.abc import Callable, Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from markdown_kb.app.retrieval import stream_query as _wiki_stream_query
from markdown_kb.app.schemas import ChatRequest
from markdown_kb.app.sse import encode_event, events_for_result
from vector_rag.app.retrieval import stream_query as _rag_stream_query

from . import conversation_store as _conv_store_module
from .query_rewriting import rewrite_query

router = APIRouter()

# Per-stack dispatch mapping.  Adding a new stack = one entry here; the
# generator body below is identical for every stack (ADR-0010).
_STACK_STREAM_FN: dict[str, Callable[[str], Iterator[dict]]] = {
    "wiki": _wiki_stream_query,
    "rag": _rag_stream_query,
}


@router.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    stack: str = "wiki",
    session: str | None = None,
) -> StreamingResponse:
    """Stream a grounded answer as SSE events.

    Dispatches to the selected Retrieval Stack via ``_STACK_STREAM_FN``.

    Phase 11 Slice 1 multi-turn additions:
    - ``session`` query param identifies the conversation session.  Absent on
      the first request; minted as a UUID by the Gateway; returned in
      ``done.session``; echoed by the client on subsequent requests.
    - Turn 1 (no prior history): query dispatched unchanged (passthrough — no
      rewrite LLM call, preserving Phase 9 sources-first latency win).
    - Turn 2+: ``rewrite_query`` reformulates the raw follow-up inside the
      generator (Phase 11 Slice 4), after emitting ``status:rewriting``.
    - On ``done``: the turn ``{question: rewritten, answer, stack,
      grounding_reason, ts}`` is appended to the Conversation Store.
    - On ``error`` (before ``done``): nothing is written to the store.

    Phase 11 Slice 4 addition:
    - Turn 2+ emits ``status:{phase:"rewriting"}`` FIRST (before sources),
      then performs the rewrite, then calls stream_fn.  This moves rewrite_query
      into the SSE generator so a rewrite error surfaces as a terminal SSE
      ``error`` event (HTTP 200 already committed once the status event is sent).

    SSE event order (ADR-0009 verify-then-stream, Phase 11 Slice 4 extension):
      1. [Turn 2+ only] ``status:{phase:"rewriting"}`` — emitted before any
         retrieval starts, signals that query rewriting is underway.
      2. ``sources`` — emitted immediately after retrieval, before any LLM call.
         Carries the retrieved sources. For ``stack=wiki``: citation id, heading,
         content snippet, derived_from. For ``stack=rag``: citation id, heading,
         content snippet only (no score, no derived_from). This is the genuine
         latency win — retrieval is ~instant; the user sees grounding context
         while the LLM drafts (~4-7s).
      3. ``status:{phase:"verifying"}`` — liveness signal emitted after sources,
         before the first token, on the LLM path only (early-exit CC paths skip
         this).
      4. ``token``(s) — words of the verified answer only (or CANNOT_CONFIRM_PHRASE
         on any CC path); no unverified draft ever reaches the stream.
      5. ``done`` — grounding outcome (passed, reason), optional filing status,
         and ``session`` id.  ``done.filed`` is always null for ``stack=rag``
         (RAG never files).
         OR
         ``error`` — terminal failure event when the rewriter or LLM/infra
         errors AFTER HTTP 200 is committed (status:rewriting or sources event
         already sent).  Payload: ``{detail, retryable}``.  No ``done`` event
         follows an ``error``.

    Args:
        req: ChatRequest body with ``query`` field.
        stack: Query param selecting the Retrieval Stack (default ``wiki``).
            ``wiki`` — Wiki BM25 stack (markdown_kb).
            ``rag``  — Vector RAG stack (vector_rag).
        session: Optional session id.  Absent → Gateway mints a new UUID.
            Present → Gateway continues the existing conversation.

    Returns:
        StreamingResponse with media_type ``text/event-stream``.

    Raises:
        HTTPException 400: unrecognised stack value.
    """
    if stack not in _STACK_STREAM_FN:
        raise HTTPException(
            status_code=400, detail=f"Unknown stack={stack!r}. Use 'wiki' or 'rag'."
        )

    # Sweep idle sessions at request entry (CODING_STANDARD §2.6: the TTL sweep
    # iterates over a snapshot of keys so it never mutates during iteration).
    # This is the only production call site for evict_expired(); wiring it here
    # keeps TTL eviction lazy (triggered by incoming traffic) without a background
    # thread — sufficient for the single-process prototype model.
    _conv_store_module.store.evict_expired()

    # Session lifecycle: mint a new UUID when no session is supplied.
    session_id: str = session if session else str(uuid.uuid4())

    # Look up conversation history BEFORE the response starts (synchronous,
    # no I/O — plain dict lookup in the in-memory store).
    history = _conv_store_module.store.get_history(session_id)

    stream_fn = _STACK_STREAM_FN[stack]

    def _sse_generator():
        """Consume stream_fn() and yield SSE frames.

        Phase 11 Slice 4 event order (turn 2+):
          status:rewriting → [rewrite_query] → sources → status:verifying →
          token(s) → done

        Phase 9 event order (turn 1 / passthrough):
          sources → status:verifying → token(s) → done

        stream_fn() (for any stack) yields exactly two dicts:
          1. sources_ready partial (before LLM) — emit sources event.
          2. full result (after LLM + verify) — emit token(s) + done events.

        This separation ensures the sources event is always emitted BEFORE
        any LLM call starts (ADR-0009 §"sources-first is real, not post-hoc").

        Phase 11 multi-turn additions:
          - Turn 2+ (history non-empty): emit status:rewriting first, then call
            rewrite_query inside the generator.  HTTP 200 is committed with the
            status:rewriting frame, so any rewrite error must surface as a
            terminal SSE error event (not an unhandled 500).
          - Turn 1 (empty history): passthrough — no status:rewriting, no
            rewrite LLM call (preserves Phase 9 sources-first latency win).
          - Dispatch uses ``self_contained_query`` (rewritten or passthrough),
            never ``req.query`` directly.
          - On ``done``: append turn to Conversation Store; inject ``session``
            into the ``done`` payload.
          - On ``error`` (before ``done``): forward the error, do NOT write.
        """
        # Phase 11 Slice 4: turn 2+ emits status:rewriting BEFORE retrieval.
        # Rewrite happens inside the generator so errors surface as SSE error
        # events (HTTP 200 is committed once the first byte is sent).
        if history:
            yield encode_event("status", {"phase": "rewriting"})
            try:
                self_contained_query = rewrite_query(req.query, history=history)
            except Exception:  # noqa: BLE001
                # Rewrite error after status:rewriting is committed (HTTP 200
                # already sent) — surface as terminal SSE error event and do NOT
                # write to the store (consistent with error-before-done invariant).
                yield encode_event(
                    "error",
                    {
                        "detail": "Internal error during query rewriting.",
                        "retryable": False,
                    },
                )
                return
        else:
            # Turn 1: passthrough (no rewrite, no status:rewriting event).
            self_contained_query = req.query

        gen = stream_fn(self_contained_query)

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
            return  # error-before-done: do NOT write to the store (AC)
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
            return  # error-before-done: do NOT write to the store (AC)

        # events_for_result emits sources + token(s) + done; we skip the
        # sources frame here (already emitted above) and forward token frames.
        # We rebuild the done frame ourselves to inject ``session`` (Phase 11).
        all_frames = events_for_result(full_result, stack=stack)
        # all_frames layout: [sources_frame, token_frame..., done_frame]
        # Yield token frames (indices 1..n-1); rebuild done (index -1).
        token_frames = all_frames[1:-1]
        yield from token_frames

        # --- Phase 11: inject session into done + append turn to store ---
        # Parse the existing done frame to get the base payload, then add session.
        done_frame_raw = all_frames[-1]  # e.g. "event: done\ndata: {...}\n\n"
        done_data_line = next(
            line for line in done_frame_raw.split("\n") if line.startswith("data: ")
        )
        done_payload: dict = json.loads(done_data_line[len("data: ") :])
        done_payload["session"] = session_id
        yield encode_event("done", done_payload)

        # Append turn to the Conversation Store (only reached on normal done —
        # error-before-done returns early above without executing this block).
        outcome = full_result["grounding_outcome"]
        ts = datetime.datetime.now(datetime.UTC).isoformat()
        _conv_store_module.store.append_turn(
            session_id,
            {
                "question": self_contained_query,  # rewritten or passthrough
                "answer": full_result.get("answer", ""),
                "stack": stack,
                "grounding_reason": outcome.reason,
                "ts": ts,
            },
        )

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
