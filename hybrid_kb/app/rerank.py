"""Deep module per Ousterhout. Public surface: ``rerank``, ``get_cross_encoder``, ``is_enabled``, ``RERANK_ENABLED_ENV``, ``RERANK_MODEL``, ``DEFAULT_RERANK_DEPTH``.

Hybrid Retrieval (Stack C) — cross-encoder **reranker** (ADR-0019 / #310).

The *precision* step ADR-0018 deferred. RRF is a recall-union step that rescues a
relevant Section one method ranked low; the reranker is the orthogonal precision
step that re-scores the fused candidate pool with a **cross-encoder** — query and
Section scored *jointly* in one model pass, unlike the dense arm's separate
bi-encoder similarity — and reorders it before the final ``top_k`` cut. The two
compose; the reranker does NOT replace RRF.

Two hard constraints shape this module (ADR-0019):

  * **Default-OFF.** ``KB_HYBRID_RERANK`` gates the whole stage; it is read THROUGH
    this module at call time (``is_enabled``) so a test / the eval can toggle it
    via the environment, mirroring the ``KB_SCORE_THRESHOLD`` /
    ``KB_RAG_DISTANCE_THRESHOLD`` pattern. With the flag off, nothing here runs.

  * **Never loaded on the VPS tenant.** The cross-encoder (``bge-reranker-v2-m3``,
    ~2.3 GB, multilingual incl. Chinese) and its ``sentence-transformers`` / torch
    dependency are an OPTIONAL group (``uv sync --group rerank``) that the slim
    production image never installs. The import therefore lives LAZILY inside
    ``get_cross_encoder`` (never at module import), so importing ``hybrid_kb`` with
    the flag off needs no torch, and enabling the flag without the dep fails fast
    with an actionable message rather than a raw ImportError.

The reranker is pure reordering: it never mints, mutates, or drops Section
content (1:1 id alignment + ``CitableContent`` preserved), its score is never
surfaced as a citation score and never feeds the Cannot-Confirm OR-gate (which
stays on each arm's native pre-fusion score, upstream of fusion) — ADR-0018/0019.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from markdown_kb.app.indexer import Section

from . import dense_index

if TYPE_CHECKING:  # import only for the type-checker — never at runtime (optional dep)
    from sentence_transformers import CrossEncoder

__all__ = [
    "rerank",
    "get_cross_encoder",
    "is_enabled",
    "RERANK_ENABLED_ENV",
    "RERANK_MODEL",
    "DEFAULT_RERANK_DEPTH",
]

# ---------------------------------------------------------------------------
# Configuration (ADR-0019)
# ---------------------------------------------------------------------------
# The master switch. Default-off everywhere; read through this module at call
# time so tests / the eval toggle it via the environment (the established
# threshold-env pattern), never a def-time-bound constant.
RERANK_ENABLED_ENV = "KB_HYBRID_RERANK"
_TRUTHY = {"1", "true", "yes", "on"}

# The cross-encoder used for the eval-only ceiling measurement. Multilingual
# (incl. Chinese), the strongest open reranker — overridable for experiments,
# but the default is the model ADR-0019 measures against.
RERANK_MODEL = os.getenv("KB_HYBRID_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# How deep into the RRF-fused pool the reranker re-scores before the final cut.
# Deeper = more buried gold recovered, more model work. At this corpus (~51
# Sections) 20 reliably keeps gold in the rerank window (ADR-0019). Tunable by
# the caller (``retrieve_and_gate(rerank_depth=...)``).
DEFAULT_RERANK_DEPTH = 20


# ---------------------------------------------------------------------------
# Cross-encoder singleton (lazy — the optional-dependency seam)
# ---------------------------------------------------------------------------
# Single-process prototype model (CODING_STANDARD §2.7): a module-level global,
# swappable by tests via monkeypatch (the established ``get_*`` leaf pattern).
_cross_encoder = None


def is_enabled() -> bool:
    """True when ``KB_HYBRID_RERANK`` is set to a truthy value, read at call time.

    Resolved THROUGH the environment on every call (not cached) so a test or the
    eval can flip the reranker on/off without reimporting — the same call-time
    resolution the per-language BM25 thresholds and the dense distance ceiling
    use (ADR-0018 / ADR-0019).
    """
    return os.getenv(RERANK_ENABLED_ENV, "").strip().lower() in _TRUTHY


def get_cross_encoder() -> CrossEncoder:
    """Return the lazily-constructed cross-encoder (the single mockable seam).

    Imports ``sentence-transformers`` only HERE — never at module import — so the
    slim production image (which omits the optional ``rerank`` group) can import
    ``hybrid_kb`` with the flag off and never pull in torch. When the flag is on
    but the dependency is absent, raise a RuntimeError that names the fix rather
    than letting a raw ImportError surface deep in a request (CODING_STANDARD
    §4.1 fail-fast with an actionable message).

    Hermetic tests monkeypatch THIS function to a deterministic fake, so the real
    model is loaded only by the opt-in ``-m live`` smoke.
    """
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                f"Reranker is enabled ({RERANK_ENABLED_ENV}) but the optional "
                "'rerank' dependency is not installed. Install it with: "
                "uv sync --group rerank"
            ) from exc
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Rerank — re-score (query, Section) pairs with the cross-encoder, reorder, cut
# ---------------------------------------------------------------------------
def rerank(query: str, candidates: Sequence[Section], top_n: int) -> list[Section]:
    """Re-score the fused candidate pool with the cross-encoder and return top_n.

    Scores each ``(query, Section-passage)`` pair jointly (the cross-encoder
    contract — full query↔document attention), then returns the ``top_n``
    Sections in descending relevance. The sort is stable, so equal scores keep
    the input (RRF) order — deterministic tie-breaking.

    Pure reordering: the returned Sections are the SAME objects the caller passed
    (no content minted, mutated, or dropped), preserving 1:1 id alignment and the
    ``CitableContent`` contract (ADR-0018/0019). Empty pool → empty list (no model
    call). The reranker's score is intentionally not returned — it is not a
    calibrated citation magnitude and never feeds the OR-gate (ADR-0019).
    """
    if not candidates:
        return []
    encoder = get_cross_encoder()
    # Score the SAME passage text the dense arm embeds (``dense_index._embed_text``,
    # reused — NOT re-implemented) so the reranker re-scores exactly the signal RRF
    # fused: heading-path breadcrumb + body, never empty (ADR-0019).
    pairs = [[query, dense_index._embed_text(section)] for section in candidates]
    scores = encoder.predict(pairs)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return [section for section, _score in ranked[:top_n]]
