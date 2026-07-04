"""Shallow module per Ousterhout. Public surface: ``warm_hybrid_indexes``,
``warm_openai_clients``, ``warmup_ping_enabled``.

Gateway startup warmup (issue #439) — fixes the post-deploy ``/chat`` cold
start. Two independent problems, two independent fixes, both driven from
``gateway/app/main.py``'s ``lifespan``:

1. **Hybrid dense index never warmed at startup.** ``markdown_kb`` and
   ``vector_rag`` rehydrate their index from their OWN sub-app lifespan
   (entered by the Gateway per issue #398), but ``hybrid_kb`` ships **no**
   FastAPI sub-app (it is library-only — see ``gateway/app/routes.py``'s
   ``/hybrid/index`` route) so nothing warmed its two-arm index. The FIRST
   ``stack=hybrid`` query after a cold boot paid the dense-index FAISS
   deserialization inline. ``warm_hybrid_indexes`` closes this by calling the
   public warmup seam ``hybrid_kb.app.query.ensure_indexes_loaded`` —
   unconditionally, every boot, and it is TOKEN-FREE (a pure disk load of the
   committed seed; no OpenAI call is made to warm an index, only to
   *construct* the embeddings client the loaded FAISS store needs for future
   queries — the exact same shape ``vector_rag``'s own lifespan already has).
2. **Per-stack OpenAI clients pay cold-connection cost on first real use.**
   Each stack's lazy-singleton LLM / embeddings getter constructs a real
   client (TLS handshake, connection pool) the first time it is invoked.
   ``warm_openai_clients`` fires one tiny ping per distinct client so that
   cost lands at boot instead of on a user's first question — but this DOES
   spend a few real OpenAI tokens per boot, so it is opt-in behind
   ``KB_WARMUP_PING`` (unset/false = the pre-issue-#439 zero-OpenAI-calls-at-
   boot behaviour, unchanged; see ``.github/workflows/reset.yml``).

Both warm functions are **best-effort**: any failure (missing
``OPENAI_API_KEY``, a transient OpenAI error, a corrupt seed) is caught and
logged, never raised, so a warmup problem never blocks Gateway startup — the
existing per-request lazy-load fallback (``hybrid_kb.query._retrieve_and_gate``,
issue #326) and lazy-singleton getters still apply on the first real request
either way.
"""

from __future__ import annotations

import os

from hybrid_kb.app.dense_index import warm_embeddings_client as _warm_hybrid_embeddings
from hybrid_kb.app.query import ensure_indexes_loaded as _ensure_hybrid_indexes_loaded
from hybrid_kb.app.query import warm_llm_client as _warm_hybrid_llm
from markdown_kb.app.retrieval import warm_llm_client as _warm_wiki_llm
from vector_rag.app.indexer import warm_embeddings_client as _warm_rag_embeddings
from vector_rag.app.retrieval import warm_llm_client as _warm_rag_llm

from .logger import log_event

_TRUTHY = ("1", "true", "yes")

# Per-distinct-client ping functions (issue #439 AC2). Each is a lazy-singleton
# getter's own best-effort ping (defined in its owning LLM-facing module so no
# LangChain type crosses into this module — CODING_STANDARD §2.4). Wiki/BM25
# has no embeddings client; RAG and Hybrid each have one LLM + one embeddings
# client, for 3 LLM + 2 embeddings = 5 distinct clients total.
_WARM_CLIENT_FNS = (
    _warm_wiki_llm,
    _warm_rag_llm,
    _warm_rag_embeddings,
    _warm_hybrid_llm,
    _warm_hybrid_embeddings,
)


def warmup_ping_enabled() -> bool:
    """True when ``KB_WARMUP_PING`` is set to a truthy value, read at call time.

    Read at call time (not cached) so a test can flip it with
    ``monkeypatch.setenv`` before entering the Gateway's ``TestClient``
    context, mirroring the established pattern (``hybrid_kb.app.rerank.
    is_enabled``, ``markdown_kb.app.transcriber.transcribe_available``).
    Unset/false is the default — boot makes zero OpenAI calls, preserving the
    reset workflow's pre-issue-#439 semantics.
    """
    return os.getenv("KB_WARMUP_PING", "").strip().lower() in _TRUTHY


def warm_hybrid_indexes() -> None:
    """Load the Hybrid (Stack C) two-arm index at startup — unconditional, token-free.

    Calls the public warmup seam so a fresh Gateway process answers
    ``stack=hybrid`` straight from the committed seeds, closing the parity gap
    with the Wiki/RAG sub-apps (issue #398). Best-effort: any failure (e.g. a
    missing ``OPENAI_API_KEY`` needed to construct the embeddings client the
    loaded FAISS store carries, or a corrupt committed seed) is caught and
    logged rather than raised, so it can never fail Gateway startup — the
    per-request lazy-load fallback still applies on the first real
    ``stack=hybrid`` query.
    """
    try:
        _ensure_hybrid_indexes_loaded()
    except Exception as exc:
        log_event(
            "startup_warmup", f"target=hybrid_dense_index status=failed exc={type(exc).__name__}"
        )


def warm_openai_clients() -> None:
    """Fire one tiny ping per distinct OpenAI client (issue #439 AC2).

    No-op unless ``KB_WARMUP_PING`` is truthy (see ``warmup_ping_enabled``).
    Each ping function is itself best-effort (catches and logs to its own
    package's log channel per CODING_STANDARD §5.1) so a single client's
    failure never stops the remaining pings from firing.
    """
    if not warmup_ping_enabled():
        return
    for warm in _WARM_CLIENT_FNS:
        warm()
