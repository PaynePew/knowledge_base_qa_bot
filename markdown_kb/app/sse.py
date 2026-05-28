"""Deep module per Ousterhout. Public surface: ``events_for_result``, ``encode_event``.

SSE serializer for the /chat/stream endpoint (Phase 9, ADR-0009/ADR-0010).

Converts a ``stream_query`` result dict (or an in-progress partial) into an
ordered list of SSE event strings, enforcing the verify-then-stream protocol:

    sources  — emitted immediately after retrieval (before any LLM call)
    token(s) — verified-answer text, chunked by whitespace words
    done     — terminal event carrying grounding outcome + optional filed

ADR-0009: no unverified text is ever streamed. The ``sources`` event is the
real latency win (~instant BM25, before 4-7s LLM). Token delivery is a
cosmetic replay of already-verified text.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# SSE wire format helper
# ---------------------------------------------------------------------------


def encode_event(event_type: str, data: dict[str, Any]) -> str:
    """Encode one SSE frame as ``event: <type>\\ndata: <json>\\n\\n``.

    Pure function: no side effects, no I/O.  The double newline terminates
    the SSE frame per RFC 8895 §9.

    Args:
        event_type: The SSE ``event:`` field name (``sources``, ``token``,
            ``done``, ``error``).
        data: JSON-serialisable payload dict.

    Returns:
        A complete SSE frame string ready to write to the response stream.
    """
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Result → event list
# ---------------------------------------------------------------------------


def events_for_result(result: dict[str, Any], *, stack: str | None = None) -> list[str]:
    """Convert a ``query()`` result dict into an ordered list of SSE frames.

    Pure function.  Mirrors the verify-then-stream protocol (ADR-0009):

    1. ``sources`` — one frame with the list of source dicts (citation id,
       heading, content snippet, derived_from).  Always present even when
       the answer is Cannot Confirm (sources is always populated whenever
       retrieval ran; empty list on index-missing path).
    2. ``token`` — one frame per word (each non-whitespace run plus its
       adjacent whitespace), so concatenating all token texts reconstructs
       the verified answer exactly (whitespace is never collapsed).
    3. ``done`` — terminal frame with PRD-locked shape:
       ``{grounding: {passed, reason}, filed, stack}``.
       ``grounding`` nests the outcome fields (PRD #116 §"SSE event contract").
       ``filed`` is populated when Answer Filing ran; ``None`` otherwise.
       ``stack`` is the dispatched retrieval stack name; ``None`` when not
       known to this serializer (it is passed in by the Gateway, which is the
       only caller that knows the stack).

    Args:
        result: A dict with keys ``answer``, ``sources``,
            ``grounding_outcome`` (a ``GroundingOutcome`` instance), and
            optionally ``filed`` (a ``FiledStatus`` instance or ``None``).
        stack: Optional stack identifier (e.g. ``"wiki"`` or ``"rag"``).
            Passed by the Gateway when it calls this serializer so the
            ``done`` frame carries the dispatched stack name.  Defaults to
            ``None``; callers that do not know the stack omit this arg.

    Returns:
        Ordered list of SSE frame strings.
    """
    frames: list[str] = []

    # 1. sources event — emitted first (real latency win: retrieval ran,
    # LLM has not yet been called; ADR-0009 §"The genuine, non-cosmetic
    # streaming value is sources-first").
    source_list = [
        {
            "source": s["source"],
            "heading": s["heading"],
            "content": s["content"],
            "derived_from": s.get("derived_from"),
        }
        for s in result.get("sources", [])
    ]
    frames.append(encode_event("sources", {"sources": source_list}))

    # 2. token events — verified answer only (ADR-0009: "stream only the
    # verified answer; the answer's token-by-token delivery is a cosmetic
    # replay of already-verified text, not real-time generation").
    answer: str = result.get("answer", "")
    # Chunk into words carrying their surrounding whitespace so the
    # concatenation of all token texts reconstructs the verified answer
    # EXACTLY (ADR-0009: streamed text must equal verified text — newlines,
    # tabs, and repeated spaces are preserved, never collapsed). Each chunk
    # is one non-whitespace run plus adjacent whitespace; the client appends
    # token.text directly with no separator. An empty answer emits nothing.
    for chunk in re.findall(r"\s*\S+\s*", answer):
        frames.append(encode_event("token", {"text": chunk}))

    # 3. done event — PRD-locked shape (PRD #116 §"SSE event contract"):
    #    {grounding: {passed, reason}, filed, stack}
    # `grounding` nests the outcome fields so the client reads
    # done.grounding.passed / done.grounding.reason (UI mockup contract).
    outcome = result["grounding_outcome"]
    done_payload: dict[str, Any] = {
        "grounding": {
            "passed": outcome.passed,
            "reason": outcome.reason,
        },
    }
    # filed is present on query() results that went through the Answer
    # Filing side-effect (Phase 6). Not always present in direct
    # stream_query calls — default to None so event shape is stable.
    filed = result.get("filed")
    if filed is not None:
        done_payload["filed"] = {
            "slug": filed.slug,
            "status": filed.status,
            "op": filed.op,
            "count": filed.count,
        }
    else:
        done_payload["filed"] = None
    # stack is injected by the Gateway (the only caller that knows which
    # stack was dispatched). The serializer itself is stack-agnostic.
    done_payload["stack"] = stack
    frames.append(encode_event("done", done_payload))

    return frames
