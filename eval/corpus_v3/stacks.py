"""Deep module per Ousterhout. Public surface: ``dense_over_wiki_retrieval``,
``index_wiki_corpus``, ``index_dense_over_wiki``, ``ARM_REGISTRY``, ``FIXTURES``.

Retrieval-arm adapters for the corpus v3 fair experiment (PRD #654, ADR-0045).
Each arm is exposed as a plain callable ``(query: str, k: int) ->
list[RetrievedItem]`` — NO HTTP — reusing the stack-adapter seam pattern from
``eval.paraphrase_comparison.stacks`` so the corpus v3 harness (a later slice)
can drive every arm in one process, offline.

This slice ships the missing 2x2 cell named in ADR-0045 Prerequisite 1:
**dense-over-wiki standalone** — the hybrid stack's dense arm
(``hybrid_kb.app.dense_index``) evaluated WITHOUT Reciprocal Rank Fusion. The
v2 eval only measured Stack A (wiki + BM25) and Stack C (wiki + BM25 + dense,
fused); it never isolated "wiki corpus, dense algorithm" from "wiki corpus,
BM25 algorithm", so a wiki loss could not be attributed to the corpus or the
algorithm. This arm closes that cell.

``ARM_REGISTRY`` is the registration point later slices (a wiki-BM25 arm, a
docs-dense arm, a hybrid arm — all re-run over the corpus v3 fixtures) add to
without conflicting with this slice's dense-over-wiki entry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import hybrid_kb.app.dense_index as hk_dense
import markdown_kb.app.indexer as mk_indexer

from .models import RetrievedItem

# ---------------------------------------------------------------------------
# Fixture locations (committed under the eval package; shared layout for
# later arms/slices to add to — e.g. a ``corpus`` dir for a raw-docs arm)
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
FIXTURES = {
    "wiki": _PKG_ROOT / "wiki",
}


# ---------------------------------------------------------------------------
# Shared corpus intake — populates markdown_kb's Section list the dense arm
# embeds from (the ADR-0018 same-corpus id-alignment invariant).
# ---------------------------------------------------------------------------
def index_wiki_corpus() -> tuple[int, int]:
    """Build markdown_kb's Section list over the committed corpus v3 wiki fixtures.

    Points ``SOURCE_DIRS`` at the fixture wiki subdirs and runs the production
    build path, populating ``mk_indexer.sections`` — the Section list
    :func:`index_dense_over_wiki` embeds from. Caller is responsible for
    redirecting ``INDEX_PATH`` / ``WIKI_DIR`` to tmp (production isolation).
    """
    wiki = FIXTURES["wiki"]
    mk_indexer.SOURCE_DIRS = [wiki / "concepts"]
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
# Adapter registry — the scaffold's registration point (issue #655: "the
# scaffold owns the package's registration points ... so later slices plug in
# without conflicting"). Keyed by arm name so a later slice adds an entry
# rather than editing an existing one.
# ---------------------------------------------------------------------------
ARM_REGISTRY: dict[str, Callable[..., list[RetrievedItem]]] = {
    "dense_over_wiki": dense_over_wiki_retrieval,
}
