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
   public warmup seam ``hybrid_kb.app.query.ensure_indexes_loaded`` — every
   boot where an ``OPENAI_API_KEY`` is present, and it is TOKEN-FREE (a pure
   disk load of the committed seed; no OpenAI call is made to warm an index,
   only to *construct* the embeddings client the loaded FAISS store needs for
   future queries — the exact same shape ``vector_rag``'s own lifespan
   already has).
2. **Per-stack OpenAI clients pay cold-connection cost on first real use.**
   Each stack's lazy-singleton LLM / embeddings getter constructs a real
   client (TLS handshake, connection pool) the first time it is invoked.
   ``warm_openai_clients`` fires one tiny ping per distinct client so that
   cost lands at boot instead of on a user's first question — but this DOES
   spend a few real OpenAI tokens per boot, so it is opt-in behind
   ``KB_WARMUP_PING`` (unset/false = the pre-issue-#439 zero-OpenAI-calls-at-
   boot behaviour, unchanged; see ``.github/workflows/reset.yml``).

**Failure-timing symmetry (issue #457).** ``warm_hybrid_indexes`` checks for
``OPENAI_API_KEY`` up front: absent, it SKIPS the warmup (logs
``status=skipped``) so a keyless boot (e.g. the ``reset.yml`` CI environment,
which never sets a real key) stays green exactly as before. Once a key is
present, it no longer swallows failures — a corrupt or missing committed
dense seed now PROPAGATES out of ``load_dense_index`` through the Gateway
``lifespan``, matching the fail-fast contract ``markdown_kb``'s and
``vector_rag``'s own sub-app lifespans already have (neither catches its
``load_index_json`` / ``load_vector_index`` call either). This closes the
asymmetry PR #448 left as a non-blocking finding: before, a corrupt hybrid
seed booted "healthy" and 500'd on the first real ``stack=hybrid`` query;
now the deploy/reset healthz gate goes red at boot instead, same as the
other two stacks. ``warm_openai_clients`` is unaffected and stays
**best-effort**: each client ping's own lazy-singleton getter catches and
logs its failure (missing key, transient OpenAI error) so one dead client
never blocks the remaining pings or Gateway startup — the existing
per-request lazy-load fallback (``hybrid_kb.query._retrieve_and_gate``,
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
    """Load the Hybrid (Stack C) two-arm index at startup, token-free.

    Calls the public warmup seam so a fresh Gateway process answers
    ``stack=hybrid`` straight from the committed seeds, closing the parity gap
    with the Wiki/RAG sub-apps (issue #398).

    Failure-timing symmetry (issue #457): a missing ``OPENAI_API_KEY`` is
    checked up front and treated as a SKIP, not a failure — logged as
    ``status=skipped`` so a keyless boot stays green (mirrors the pre-#457
    behaviour for this one case). Once a key is present, any OTHER failure
    (most notably a corrupt or missing committed dense seed) is left to
    PROPAGATE out of ``_ensure_hybrid_indexes_loaded`` and out of the Gateway
    ``lifespan`` — exactly like ``markdown_kb``'s and ``vector_rag``'s own
    ``load_index_json`` / ``load_vector_index`` lifespan calls, neither of
    which catches its loader's exceptions either. This turns a broken hybrid
    seed into a red deploy/reset healthz gate at boot instead of a 500 on the
    first real ``stack=hybrid`` query.
    """
    if not os.getenv("OPENAI_API_KEY"):
        log_event(
            "startup_warmup", "target=hybrid_dense_index status=skipped reason=no_openai_api_key"
        )
        return
    _ensure_hybrid_indexes_loaded()


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
