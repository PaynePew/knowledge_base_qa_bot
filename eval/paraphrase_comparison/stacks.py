"""Deep module per Ousterhout. Public surface: ``stack_a_retrieval``, ``stack_b_retrieval``, ``stack_c_retrieval``, ``FIXTURES``.

In-process Retrieval Stack adapters for the Phase 8 comparison (CONTEXT.md
§ Phase 8 > Retrieval Stack, PRD #100) extended to a third arm in Phase 13
(ADR-0018, PRD #309 / #316). Each Stack's retrieval is exposed as a plain
callable ``(query: str, k: int) -> list[RetrievedItem]`` — NO HTTP — so the
DeepEval runner drives all three arms in one process.

Stack A (Wiki + BM25, markdown_kb): indexes the committed ``/ingest`` wiki
output under ``wiki/{entities,concepts}/`` and retrieves Sections via BM25
(ADR-0006 W1: wiki is the sole query surface). A retrieved wiki Section is
resolved back to its docs Gold Section id via the page's ``sources``
frontmatter so the C5c metric can compare against the docs-granular gold.

Stack B (Vector RAG, vector_rag): indexes the raw corpus into FAISS and
retrieves Chunks whose ``source`` is already a docs Section id.

Stack C (Hybrid, hybrid_kb): runs BM25 AND a dense-over-wiki Section index over
the SAME curated wiki corpus and fuses the two ranked lists with Reciprocal Rank
Fusion (RRF, reused from ``hybrid_kb.app.retrieval`` — NOT reimplemented). The
dense arm is built from the SAME Section list BM25 indexes (the ADR-0018
same-corpus invariant), so a fused wiki Section resolves to its docs Gold Section
id exactly like Stack A. For the hit-rate eval the arm returns the fused RANKED
list over the overfetched pool — NOT the per-arm OR-gate verdict — because the
eval measures retrieval quality, not the Cannot Confirm gate.

Production isolation (PRD #100 acceptance): ``index_stack_a`` / ``index_stack_b``
/ ``index_stack_c`` repoint markdown_kb ``SOURCE_DIRS`` and vector_rag
``DOCS_DIR`` to the eval fixtures, and the caller —
``runner._isolate_production_paths`` (mirrored by the test suite's autouse
conftest fixture) — redirects markdown_kb ``INDEX_PATH`` / ``WIKI_DIR`` / log
path, vector_rag ``FAISS_INDEX_DIR`` / log path, AND hybrid_kb's dense
``DENSE_INDEX_DIR`` / log path to a tmp directory, so the builds' atomic-write
and ``write_wiki_index`` side effects land in tmp and production ``wiki/`` /
``docs/`` / ``.kb/`` are never read or written.
"""

from __future__ import annotations

import re
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
    "corpus": _PKG_ROOT / "corpus",
    "wiki": _PKG_ROOT / "wiki",
}

# Frontmatter scan: wiki pages open with a sentinel HTML comment then a YAML
# frontmatter block. markdown_kb.parse_markdown only reads frontmatter when the
# file STARTS with '---', so it never populates Section.metadata for these
# pages; the Stack A adapter parses the `sources:` list directly here to bridge
# wiki-slug ids to docs Gold Section ids.
_SOURCES_BLOCK_RE = re.compile(
    r"^sources:\s*\n((?:[ \t]*-[ \t]*\S+.*\n)+)", re.MULTILINE
)
_SOURCE_ITEM_RE = re.compile(r"^[ \t]*-[ \t]*(\S+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Stack A — Wiki + BM25 (markdown_kb)
# ---------------------------------------------------------------------------
def _wiki_section_to_item(
    section: Section, file_to_gold: dict[str, str]
) -> RetrievedItem:
    """Normalise a wiki Section to a docs-Gold-Section-granular ``RetrievedItem``.

    Maps the Section's wiki-slug ``file`` back to the docs Gold Section id via the
    page's ``sources`` frontmatter (falling back to the Section's own id when the
    page is not a 1:1 concept page). Shared by the Wiki (Stack A) and Hybrid
    (Stack C) arms so both resolve a wiki hit to the docs gold the same way.
    """
    return RetrievedItem(
        source_section_id=file_to_gold.get(section.file, section.id),
        content=section.content,
        heading_path=list(section.heading_path),
    )


def stack_a_retrieval(query: str, k: int = 3) -> list[RetrievedItem]:
    """Retrieve via markdown_kb BM25 over the indexed wiki, normalised to docs ids.

    Assumes the index has been built against the eval wiki fixtures (see
    ``index_stack_a``). Each BM25 hit Section's wiki-slug file is mapped back to
    the docs Gold Section id via the wiki page's ``sources`` frontmatter.
    """
    file_to_gold = _wiki_slug_to_gold_section()
    return [
        _wiki_section_to_item(section, file_to_gold)
        for section, _score in mk_indexer.search(query, k=k)
    ]


def index_stack_a() -> tuple[int, int]:
    """Build markdown_kb's Section Index over the eval wiki fixtures.

    Points ``SOURCE_DIRS`` at the eval wiki subdirs and runs the production
    (slug-id) build path so wiki Section ids match the production convention.
    Caller is responsible for redirecting ``INDEX_PATH`` / ``WIKI_DIR`` to tmp
    (production isolation).
    """
    wiki = FIXTURES["wiki"]
    mk_indexer.SOURCE_DIRS = [wiki / "entities", wiki / "concepts"]
    # Default docs_dir triggers the slug-id production path over SOURCE_DIRS.
    return mk_indexer.build_index()


def _wiki_slug_to_gold_section() -> dict[str, str]:
    """Map each wiki page's bare slug to the docs Gold Section id it synthesises.

    Reads the ``sources:`` frontmatter of every concept page (1:1 with a docs
    Section, so a single source). Entity pages (1:N) are not Paraphrase targets
    and are skipped from the map; a BM25 hit on one falls back to its own id.
    """
    mapping: dict[str, str] = {}
    concepts = FIXTURES["wiki"] / "concepts"
    for page in sorted(concepts.glob("*.md")):
        raw = page.read_text(encoding="utf-8")
        block = _SOURCES_BLOCK_RE.search(raw)
        if not block:
            continue
        sources = _SOURCE_ITEM_RE.findall(block.group(1))
        if sources:
            mapping[page.stem] = sources[0]
    return mapping


# ---------------------------------------------------------------------------
# Stack B — Vector RAG (vector_rag)
# ---------------------------------------------------------------------------
def stack_b_retrieval(query: str, k: int = 3) -> list[RetrievedItem]:
    """Retrieve via vector_rag FAISS over the raw corpus.

    A Chunk's ``source`` is already a docs Gold Section id, so the mapping to
    ``RetrievedItem`` is direct.
    """
    return [
        RetrievedItem(
            source_section_id=chunk.source,
            content=chunk.content,
            heading_path=list(chunk.heading_path),
        )
        for chunk in vr_indexer.search(query, k=k)
    ]


def index_stack_b() -> tuple[int, int]:
    """Build vector_rag's FAISS index over the eval raw corpus.

    Points ``DOCS_DIR`` at the eval corpus and builds. Embedding the corpus
    requires OPENAI_API_KEY; offline tests swap ``vr_indexer._build_faiss`` for
    a deterministic fake (see tests/conftest).
    """
    corpus = FIXTURES["corpus"]
    vr_indexer.DOCS_DIR = corpus
    return vr_indexer.build_index(corpus)


# ---------------------------------------------------------------------------
# Stack C — Hybrid (BM25 + dense-over-wiki, RRF fused) (hybrid_kb)
# ---------------------------------------------------------------------------
def index_stack_c() -> int:
    """Build Stack C's dense-over-wiki arm from BM25's Section list (1:1 ids).

    Assumes ``index_stack_a`` already built markdown_kb's BM25 index over the eval
    wiki fixtures, populating ``mk_indexer.sections``. The dense index is built
    from that EXACT Section list — not by re-scanning ``wiki/`` — so dense ids
    align 1:1 with the BM25 Section ids (the ADR-0018 same-corpus invariant) and
    fusion is true same-corpus fusion. ``hybrid_kb.dense_index`` is not modified;
    only its public ``build_index`` is called (additive per ADR-0002 / ADR-0018).

    Caller redirects ``hk_dense.DENSE_INDEX_DIR`` to tmp (production isolation),
    and an offline run swaps ``hk_dense.get_embeddings`` for a deterministic fake.
    Returns the number of dense Sections indexed.
    """
    return hk_dense.build_index(sections=list(mk_indexer.sections))


def stack_c_retrieval(
    query: str,
    k: int = 3,
    *,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> list[RetrievedItem]:
    """Retrieve via Hybrid: BM25 + dense over the same wiki, RRF-fused, docs ids.

    Both arms overfetch a deep candidate pool (``candidate_depth``, default 50)
    DECOUPLED from the final cutoff ``k``; the two ranked Section lists are fused
    with Reciprocal Rank Fusion (``reciprocal_rank_fusion`` reused from
    ``hybrid_kb.app.retrieval`` — fusion is NOT reimplemented here) and truncated
    to the top-``k``. Each fused wiki Section is resolved to its docs Gold Section
    id exactly like Stack A.

    For the hit-rate eval this returns the fused RANKED list over the pool, NOT
    the per-arm OR-gate verdict: the comparison measures retrieval quality across
    a cutoff sweep, and Cannot Confirm gating is a downstream concern the eval
    deliberately does not apply (ADR-0018 / #316 guardrail). Assumes both arms are
    indexed (``index_stack_a`` + ``index_stack_c``).
    """
    file_to_gold = _wiki_slug_to_gold_section()
    bm25_ranked: list[Section] = [
        section for section, _score in mk_indexer.search(query, k=candidate_depth)
    ]
    dense_ranked: list[Section] = hk_dense.search(query, k=candidate_depth)
    fused = reciprocal_rank_fusion(bm25_ranked, dense_ranked, k=RRF_K, top_k=k)
    return [_wiki_section_to_item(section, file_to_gold) for section, _score in fused]
