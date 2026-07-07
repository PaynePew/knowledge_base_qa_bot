"""Shallow module per Ousterhout. Public surface: ``router``.

Gateway HTTP wiring for ``POST /chat/stream`` and ``POST /upload``.

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
Phase 15 S1 (issue #169) — ``POST /upload`` (multipart) added; delegates to
``markdown_kb.app.upload.upload_files`` (deep module).  Upload is a Gateway
concern per ADR-0010 (gateway is the composition layer that owns the Console
and all Console-adjacent system boundaries) and ADR-0011 (Upload only stages
bytes; Import stays unchanged).
Issue #533 (ADR-0036 §6) — ``POST /upload`` gains an optional
``overwrite_relpath`` form field, forwarded verbatim to ``upload_files``;
the destination-aware overwrite guard lives entirely in the deep module.

All streaming complexity lives in the per-stack ``stream_query()`` functions and
the shared ``markdown_kb.app.sse.events_for_result()`` serializer; this module is
a shallow dispatcher (CODING_STANDARD §2.3).  The dispatch mapping
(``_STACK_STREAM_FN``) is the only place that knows which stream function to call
per stack — the generator body is identical for both (ADR-0010: gateway is the
composition layer).  RAG sources carry citation id + heading + content plus an
OPTIONAL ``path`` for the clickable citation (#266/#307) — but NO score, NO
derived_from: issue #120 excludes the ranking/wiki-layer signals, and ``path`` is
orthogonal (it is the UI-resolution target the gateway forwards generically). The
shared serializer renders them as-is.

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

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

# Hybrid Retrieval (Stack C) — Phase 13 S4 (issue #314). Additive third stack;
# ``hybrid_kb.query.stream_query`` is the same two-dict generator contract the
# Wiki/RAG stream functions expose, so the generator body below is unchanged
# (ADR-0010: the gateway is the composition layer; adding a stack is one map
# entry). ADR-0018: Hybrid fuses BM25 + dense over the wiki Section corpus.
from hybrid_kb.app.dense_index import build_index as _hybrid_build_index
from hybrid_kb.app.query import stream_query as _hybrid_stream_query
from markdown_kb.app.errors import LLMError
from markdown_kb.app.read import FileNotFound as _ReadFileNotFound
from markdown_kb.app.read import NotAFile as _ReadNotAFile
from markdown_kb.app.read import PathRejected as _ReadPathRejected
from markdown_kb.app.read import TreeEntry as _TreeEntry
from markdown_kb.app.read import list_tree as _list_tree
from markdown_kb.app.read import read_file as _read_file
from markdown_kb.app.retrieval import stream_query as _wiki_stream_query
from markdown_kb.app.schemas import ChatRequest
from markdown_kb.app.sse import encode_event, events_for_result
from markdown_kb.app.upload import upload_files as _upload_files
from pydantic import BaseModel
from vector_rag.app.retrieval import stream_query as _rag_stream_query

from . import conversation_store as _conv_store_module
from .logger import log_event as _gateway_log_event
from .query_rewriting import rewrite_query

router = APIRouter()


# ---------------------------------------------------------------------------
# Upload response schema (Phase 15 S1, issue #169)
# ---------------------------------------------------------------------------


class UploadFileResultSchema(BaseModel):
    """Per-file result returned by POST /upload."""

    filename: str
    status: str  # "written" | "rejected" | "error"
    target_dir: str = ""
    reason: str = ""


class UploadBatchResultSchema(BaseModel):
    """Response body for POST /upload."""

    results: list[UploadFileResultSchema]


class HybridIndexResponseSchema(BaseModel):
    """Response body for POST /hybrid/index (ADR-0022, issue #348)."""

    sections_indexed: int


# Per-stack dispatch mapping.  Adding a new stack = one entry here; the
# generator body below is identical for every stack (ADR-0010).
_STACK_STREAM_FN: dict[str, Callable[[str], Iterator[dict]]] = {
    "wiki": _wiki_stream_query,
    "rag": _rag_stream_query,
    "hybrid": _hybrid_stream_query,
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
         content snippet, derived_from, plus a resolvable wiki-page ``path``. For
         ``stack=rag``: citation id, heading, content snippet, plus an OPTIONAL
         docs/-relative ``path`` for the clickable citation (#307; present when
         the chunk carries source-path metadata) — but no score, no derived_from.
         This is the genuine latency win — retrieval is ~instant; the user sees
         grounding context while the LLM drafts (~4-7s).
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
            ``wiki``   — Wiki BM25 stack (markdown_kb).
            ``rag``    — Vector RAG stack (vector_rag).
            ``hybrid`` — Hybrid stack: BM25 + dense fused over the wiki Section
                corpus (hybrid_kb, ADR-0018). Never files (``done.filed`` null,
                like rag); citations carry a resolvable wiki-page ``path``.
        session: Optional session id.  Absent → Gateway mints a new UUID.
            Present → Gateway continues the existing conversation.

    Returns:
        StreamingResponse with media_type ``text/event-stream``.

    Raises:
        HTTPException 400: unrecognised stack value.
    """
    if stack not in _STACK_STREAM_FN:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown stack={stack!r}. Use 'wiki', 'rag', or 'hybrid'.",
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
                # Emit chat_rewrite log entry only when a rewrite actually happened
                # (turn 2+).  raw and rewritten are bounded to 60 chars per §5.3.
                _gateway_log_event(
                    "chat_rewrite",
                    f"session={session_id} "
                    f'raw="{req.query[:60].replace(chr(34), chr(39))}" '
                    f'rewritten="{self_contained_query[:60].replace(chr(34), chr(39))}"',
                )
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
        # Wiki sources carry: source, heading, content, derived_from, path.
        # RAG sources carry: source, heading, content, and an OPTIONAL path
        # (#307) — NO derived_from, NO score. Using a fixed mandatory trio plus
        # conditional derived_from / path forwarding preserves each stack's
        # contract without coupling this layer to individual field names.
        source_list = []
        for s in partial.get("sources", []):
            entry: dict = {
                "source": s["source"],
                "heading": s["heading"],
                "content": s["content"],
            }
            if "derived_from" in s:
                entry["derived_from"] = s["derived_from"]
            # Forward the resolvable source path so the reader UI can link the
            # citation: the wiki-page path (#266) and the docs/ path on the RAG
            # stack (#307). Generic by design — both stacks emit ``path`` only
            # when the source is resolvable, so this single check serves both.
            if "path" in s:
                entry["path"] = s["path"]
            source_list.append(entry)
        yield encode_event("sources", {"sources": source_list})

        # Emit status{phase:"verifying"} on the LLM path only (early_exit=False).
        # Early-exit CC paths skip the LLM entirely — emitting "verifying" there
        # would be a lie; the answer is already known from the pre-LLM gate.
        if not partial.get("early_exit", False):
            yield encode_event("status", {"phase": "verifying"})

        # Second yield: full result — emit token(s) + done.
        # On LLM/infra error (LLMError raised by _draft_and_verify), HTTP 200 is
        # already committed, so we emit a terminal error event instead of
        # propagating the exception (which would silently truncate the stream).
        # ADR-0015: the wrapper raises LLMError (transport-agnostic); the SSE
        # adapter reads .retryable directly instead of deriving it from a 503
        # status code (cleaner — no round-trip through HTTP semantics).
        try:
            full_result = next(gen)
        except LLMError as exc:
            yield encode_event("error", {"detail": exc.message, "retryable": exc.retryable})
            return  # error-before-done: do NOT write to the store (AC)
        except Exception:  # noqa: BLE001 — intentional last-resort catch
            # Defense in depth (ADR-0009): HTTP 200 is already committed once the
            # sources event is sent, so ANY error during draft/verify — including
            # unmapped or unexpected ones (e.g. an LLM client misconfiguration that
            # raises a non-LLMError exception) — must surface as a terminal SSE
            # error event, never a silently truncated stream that hangs the client.
            # Detail is generic to avoid leaking internals.
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


# ---------------------------------------------------------------------------
# POST /upload — Phase 15 S1 (issue #169, ADR-0011)
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=UploadBatchResultSchema)
async def upload(
    files: list[UploadFile],
    overwrite_relpath: str | None = Form(None),
) -> UploadBatchResultSchema:
    """Stage a batch of uploaded files onto the server.

    Accepts multipart/form-data with one or more ``files`` fields.  Delegates
    all validation and routing logic to ``markdown_kb.app.upload.upload_files``
    (deep module — CODING_STANDARD §2.3).

    Per ADR-0011: Upload only stages bytes; Import (``POST /wiki/import``) is
    unchanged and still converts ``raw/`` → ``docs/``.

    Routing:
      ``.html`` / ``.txt`` / ``.pdf``  →  ``raw/``   (then Import converts to ``docs/``)
      ``.md``                          →  ``docs/``  (already canonical Markdown)
      Other extensions                 →  rejected with reason

    Validation (system boundary — all untrusted-input checks live here):
      - Traversal-safe filename (no ``..``, no path separators, no bidi chars)
      - Type allow-list (``.html`` / ``.txt`` / ``.md`` / ``.pdf``)
      - Size limit (10 MB per file)

    Always returns HTTP 200.  Per-file failures (rejections, errors) are
    recorded in the ``results`` list; the batch never aborts on a single
    file failure (continue-on-error semantics mirror ``/import``).

    Wiki Log events emitted (``project-docs/log-kinds.md`` Phase 15 section):
      ``upload_batch_started`` / ``upload_file`` / ``upload_rejected`` /
      ``upload_error`` / ``upload_batch_completed``.

    Args:
        files: One or more ``UploadFile`` items from the multipart body.
        overwrite_relpath: Optional form field (issue #533, ADR-0036 §6). When
            present, routes the (``.md``) upload to overwrite that resolved,
            existing Source path under ``docs/`` in place instead of the
            default root write — see ``markdown_kb.app.upload`` for the guard
            (overwrite-only of an existing, uniquely-resolvable Source; never
            falls back to a root write).

    Returns:
        ``UploadBatchResultSchema`` with one ``UploadFileResultSchema`` per
        input file, in the same order.
    """
    # Read all file bytes first (UploadFile is async).
    file_pairs: list[tuple[str, bytes]] = []
    for uf in files:
        content = await uf.read()
        file_pairs.append((uf.filename or "", content))

    batch = _upload_files(file_pairs, overwrite_relpath=overwrite_relpath)

    return UploadBatchResultSchema(
        results=[
            UploadFileResultSchema(
                filename=r.filename,
                status=r.status,
                target_dir=r.target_dir,
                reason=r.reason,
            )
            for r in batch.results
        ]
    )


# ---------------------------------------------------------------------------
# POST /hybrid/index — ADR-0022 / issue #348
# Operator-triggered full re-embed of the Hybrid stack's dense arm.
# ---------------------------------------------------------------------------


@router.post("/hybrid/index", response_model=HybridIndexResponseSchema)
def hybrid_index() -> HybridIndexResponseSchema:
    """Trigger a full re-embed of the Hybrid stack's dense arm.

    Calls ``hybrid_kb.app.dense_index.build_index()`` in-process, which
    re-embeds every wiki Section and persists the FAISS seed under
    ``.kb/hybrid_dense/``.  ``hybrid_kb`` stays **library-only** (no
    FastAPI app on the hybrid side) — the route lives here in the Gateway,
    the composition layer (ADR-0010).

    A full re-embed is trivially correct: it restores the ADR-0018 1:1
    dense↔BM25 id alignment by construction and is symmetric with
    ADR-0020's full BM25 rebuild.

    Cost: ~$0.50 (whole-corpus re-embed, symmetric with ``/rag/index``).
    Gating: ``ADMIN_PATHS`` → admin semaphore + per-UTC-day cost cap +
    optional admin-token kill-switch (ADR-0021 / ADR-0022 Consequences).

    Returns:
        ``HybridIndexResponseSchema`` with ``sections_indexed`` count.
    """
    n = _hybrid_build_index()
    return HybridIndexResponseSchema(sections_indexed=n)


# ---------------------------------------------------------------------------
# GET /read/tree — Phase 15 S5 (issue #171)
# GET /read/file — Phase 15 S5 (issue #171)
# ---------------------------------------------------------------------------


class TreeEntrySchema(BaseModel):
    """One entry in a directory listing returned by GET /read/tree."""

    name: str
    relpath: str
    is_dir: bool
    size: int = 0


class TreeListSchema(BaseModel):
    """Response body for GET /read/tree."""

    entries: list[TreeEntrySchema]


class FileContentSchema(BaseModel):
    """Response body for GET /read/file."""

    relpath: str
    content: str


def _tree_entry_to_schema(e: _TreeEntry) -> TreeEntrySchema:
    return TreeEntrySchema(name=e.name, relpath=e.relpath, is_dir=e.is_dir, size=e.size)


@router.get("/read/tree", response_model=TreeListSchema)
def read_tree(
    path: str = Query(
        default="", description="Relative path within the whitelist tree. Empty = list roots."
    ),
) -> TreeListSchema:
    """List one level of the whitelisted corpus tree.

    Whitelisted roots: ``docs/``, ``raw/``, ``wiki/``.  ``.kb/`` is excluded.

    Args:
        path: Relative path string (e.g. ``''``, ``'docs'``, ``'wiki/sub'``).
            Empty string returns the three root entries.

    Returns:
        ``TreeListSchema`` with a list of ``TreeEntrySchema`` entries, sorted
        directories first then files (alphabetical within each group).

    Raises:
        HTTP 400: ``path`` contains ``..``, is absolute, or otherwise rejected.
        HTTP 404: the resolved path does not exist.
        HTTP 400: the resolved path is a file (call ``GET /read/file`` instead).
    """
    try:
        entries = _list_tree(path)
    except _ReadPathRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except _ReadFileNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except _ReadNotAFile as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Path is a file; use GET /read/file to read it: {exc}",
        ) from exc

    return TreeListSchema(entries=[_tree_entry_to_schema(e) for e in entries])


@router.get("/read/file", response_model=FileContentSchema)
def read_file_endpoint(
    path: str = Query(description="Relative path of the file to read (e.g. 'docs/policy.md')."),
) -> FileContentSchema:
    """Return the raw UTF-8 text of a file inside the whitelisted roots.

    Whitelisted roots: ``docs/``, ``raw/``, ``wiki/`` (including the
    ``index.md`` / ``log.md`` / ``lint-report.md`` runtime artifacts).
    ``.kb/`` is excluded.

    Args:
        path: Relative path string (e.g. ``'docs/policy.md'``,
            ``'wiki/log.md'``).

    Returns:
        ``FileContentSchema`` with ``relpath`` and ``content`` (raw text).

    Raises:
        HTTP 400: ``path`` contains ``..``, is absolute, or otherwise rejected.
        HTTP 404: the resolved path does not exist.
        HTTP 400: the resolved path is a directory (call ``GET /read/tree``).
    """
    try:
        content = _read_file(path)
    except _ReadPathRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except _ReadFileNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except _ReadNotAFile as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Path is a directory; use GET /read/tree to list it: {exc}",
        ) from exc

    return FileContentSchema(relpath=path, content=content)
