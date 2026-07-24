"""Deep module per Ousterhout. Public surface: ``stack_a_retrieval``,
``stack_b_retrieval``, ``stack_c_retrieval``, ``dense_over_wiki_retrieval``,
``index_wiki_corpus``, ``index_dense_over_wiki``, ``index_docs_corpus``,
``index_stack_c``, ``ARM_REGISTRY``, ``FIXTURES``.

Retrieval-arm adapters for the corpus v3 fair experiment (PRD #654, ADR-0045).
Each arm is exposed as a plain callable ``(query: str, k: int) ->
list[RetrievedItem]`` — NO HTTP — reusing the stack-adapter seam pattern from
``eval.paraphrase_comparison.stacks`` so the corpus v3 harness (issue #662)
can drive every arm in one process, offline.

Issue #655 shipped the missing 2x2 cell named in ADR-0045 Prerequisite 1:
**dense-over-wiki standalone** — the hybrid stack's dense arm
(``hybrid_kb.app.dense_index``) evaluated WITHOUT Reciprocal Rank Fusion. Issue
#662 closes the remaining registry gap ADR-0045 names as "run all four arms
(A, B, C, dense-over-wiki)" by adding the other three:

- **Stack A** (wiki + BM25, ``markdown_kb``) over the corpus v3 wiki fixtures.
- **Stack B** (dense-over-docs, ``vector_rag``) over the corpus v3 raw corpus.
- **Stack C** (Hybrid: BM25 + dense-over-wiki, RRF-fused, ``hybrid_kb``).

Unlike ``eval.paraphrase_comparison.stacks``'s v2 adapters, none of these
resolve a retrieved item's id through a wiki-slug-to-docs-gold map at the
adapter layer — that resolution is exactly the v2 harness tilt ADR-0045
Prerequisite 3 removes (``eval.corpus_v3.gold.resolve_gold_sections``, applied
downstream by the metric layer). Every arm here stays a dumb id passthrough:
whichever corpus-neutral Section id the retrieving stack natively returns.

``ARM_REGISTRY`` is the registration point every arm plugs into by name
(``"wiki"``, ``"rag"``, ``"hybrid"``, ``"dense_over_wiki"``) so the runner
(``run_verdict.py``) drives all four without hard-coding call sites.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import hybrid_kb.app.dense_index as hk_dense
import markdown_kb.app.indexer as mk_indexer
import vector_rag.app.indexer as vr_indexer
from hybrid_kb.app.retrieval import (
    DEFAULT_CANDIDATE_DEPTH,
    RRF_K,
    reciprocal_rank_fusion,
)
from markdown_kb.app.indexer import Section

from .models import RetrievedItem

# ---------------------------------------------------------------------------
# Fixture locations (committed under the eval package)
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
FIXTURES = {
    "wiki": _PKG_ROOT / "wiki",
    "corpus": _PKG_ROOT / "corpus",
}


# ---------------------------------------------------------------------------
# Shared corpus intake — populates markdown_kb's Section list the dense arm
# embeds from (the ADR-0018 same-corpus id-alignment invariant).
# ---------------------------------------------------------------------------
def index_wiki_corpus() -> tuple[int, int]:
    """Build markdown_kb's Section list over the committed corpus v3 wiki fixtures.

    Points ``SOURCE_DIRS`` at BOTH wiki subdirs (``concepts`` AND ``entities``,
    when the latter exists) and runs the production build path, populating
    ``mk_indexer.sections`` — the Section list Stack A's BM25 retrieves over
    AND :func:`index_dense_over_wiki` embeds from (ADR-0018 same-corpus
    invariant: the dense arm must see exactly what BM25 sees, entity pages
    included, or the two id spaces drift apart). Caller is responsible for
    redirecting ``INDEX_PATH`` / ``WIKI_DIR`` to tmp (production isolation).
    """
    wiki = FIXTURES["wiki"]
    source_dirs = [wiki / "concepts"]
    entities = wiki / "entities"
    if entities.is_dir():
        source_dirs.append(entities)
    mk_indexer.SOURCE_DIRS = source_dirs
    return mk_indexer.build_index()


def index_dense_over_wiki() -> int:
    """Build the dense-over-wiki index from the corpus v3 wiki Section list.

    Assumes :func:`index_wiki_corpus` already populated ``mk_indexer.sections``.
    Embeds that EXACT Section list (not a re-scan of ``wiki/``), matching
    ``hybrid_kb``'s own same-corpus invariant (ADR-0018). Caller redirects
    ``hk_dense.DENSE_INDEX_DIR`` to tmp, and an offline run swaps
    ``hk_dense.get_embeddings`` for a deterministic fake. Returns the number of
    dense Sections indexed.
    """
    return hk_dense.build_index(sections=list(mk_indexer.sections))


# ---------------------------------------------------------------------------
# Dense-over-wiki standalone arm (ADR-0045 Prerequisite 1)
# ---------------------------------------------------------------------------
def dense_over_wiki_retrieval(query: str, k: int = 3) -> list[RetrievedItem]:
    """Retrieve via ``hybrid_kb``'s dense index over the wiki, WITHOUT RRF fusion.

    Assumes both :func:`index_wiki_corpus` and :func:`index_dense_over_wiki`
    have run. Each dense hit's native wiki Section id is carried through
    unresolved (Prerequisite 3's gold-label mapping is out of scope here).
    """
    return [
        RetrievedItem(
            source_section_id=section.id,
            content=section.content,
            heading_path=list(section.heading_path),
        )
        for section in hk_dense.search(query, k=k)
    ]


# ---------------------------------------------------------------------------
# Stack A — Wiki + BM25 (markdown_kb)
# ---------------------------------------------------------------------------
def stack_a_retrieval(query: str, k: int = 3) -> list[RetrievedItem]:
    """Retrieve via markdown_kb BM25 over the corpus v3 wiki (Stack A).

    Assumes :func:`index_wiki_corpus` has run. The native wiki Section id is
    carried through unresolved — see module docstring on why this arm does
    NOT bake in a wiki-slug-to-docs-gold mapping the way
    ``eval.paraphrase_comparison.stacks.stack_a_retrieval`` does.
    """
    return [
        RetrievedItem(
            source_section_id=section.id,
            content=section.content,
            heading_path=list(section.heading_path),
        )
        for section, _score in mk_indexer.search(query, k=k)
    ]


# ---------------------------------------------------------------------------
# Stack B — Vector RAG, dense-over-docs (vector_rag)
# ---------------------------------------------------------------------------
def index_docs_corpus() -> tuple[int, int]:
    """Build vector_rag's FAISS index over the corpus v3 raw corpus fixtures.

    Points ``DOCS_DIR`` at ``FIXTURES["corpus"]`` and runs the production
    build path. Caller redirects ``FAISS_INDEX_DIR`` to tmp (production
    isolation); an offline run swaps ``vr_indexer._build_faiss`` for a
    deterministic fake (see tests/conftest).
    """
    corpus = FIXTURES["corpus"]
    vr_indexer.DOCS_DIR = corpus
    return vr_indexer.build_index(corpus)


def stack_b_retrieval(query: str, k: int = 3) -> list[RetrievedItem]:
    """Retrieve via vector_rag FAISS over the corpus v3 raw corpus (Stack B).

    A Chunk's ``source`` is already a docs-native Section id, so the mapping
    to ``RetrievedItem`` is direct — no gold-mapping resolution needed here
    either way, since Stack B never had the v2 tilt (its ids were always
    docs-native).
    """
    return [
        RetrievedItem(
            source_section_id=chunk.source,
            content=chunk.content,
            heading_path=list(chunk.heading_path),
        )
        for chunk in vr_indexer.search(query, k=k)
    ]


# ---------------------------------------------------------------------------
# Stack C — Hybrid (BM25 + dense-over-wiki, RRF fused) (hybrid_kb)
# ---------------------------------------------------------------------------
def index_stack_c() -> int:
    """Alias for :func:`index_dense_over_wiki` under Stack C's own name.

    Stack C's dense arm is built from the SAME Section list BM25 indexes
    (``index_wiki_corpus`` must run first) — the ADR-0018 same-corpus
    invariant, mirrored from ``eval.paraphrase_comparison.stacks.index_stack_c``.
    """
    return index_dense_over_wiki()


def _fused_wiki_sections(
    query: str, *, candidate_depth: int, top_k: int
) -> list[Section]:
    """Overfetch both arms and RRF-fuse to ``top_k`` wiki Sections — Stack C's pool."""
    bm25_ranked: list[Section] = [
        section for section, _score in mk_indexer.search(query, k=candidate_depth)
    ]
    dense_ranked: list[Section] = hk_dense.search(query, k=candidate_depth)
    fused = reciprocal_rank_fusion(bm25_ranked, dense_ranked, k=RRF_K, top_k=top_k)
    return [section for section, _score in fused]


def stack_c_retrieval(
    query: str,
    k: int = 3,
    *,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> list[RetrievedItem]:
    """Retrieve via Hybrid: BM25 + dense over the same wiki, RRF-fused (Stack C).

    Assumes both :func:`index_wiki_corpus` and :func:`index_stack_c` have run.
    Reuses ``reciprocal_rank_fusion`` from ``hybrid_kb.app.retrieval`` —
    fusion is NOT reimplemented here (mirrors
    ``eval.paraphrase_comparison.stacks.stack_c_retrieval``).
    """
    fused = _fused_wiki_sections(query, candidate_depth=candidate_depth, top_k=k)
    return [
        RetrievedItem(
            source_section_id=section.id,
            content=section.content,
            heading_path=list(section.heading_path),
        )
        for section in fused
    ]


# ---------------------------------------------------------------------------
# Adapter registry — the scaffold's registration point (issue #655: "the
# scaffold owns the package's registration points ... so later slices plug in
# without conflicting"). Keyed by arm name so a later slice adds an entry
# rather than editing an existing one.
# ---------------------------------------------------------------------------
ARM_REGISTRY: dict[str, Callable[..., list[RetrievedItem]]] = {
    "wiki": stack_a_retrieval,
    "rag": stack_b_retrieval,
    "hybrid": stack_c_retrieval,
    "dense_over_wiki": dense_over_wiki_retrieval,
}
