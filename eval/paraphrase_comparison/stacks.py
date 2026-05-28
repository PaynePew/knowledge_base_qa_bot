"""Deep module per Ousterhout. Public surface: ``stack_a_retrieval``, ``stack_b_retrieval``, ``FIXTURES``.

In-process Retrieval Stack adapters for the Phase 8 comparison (CONTEXT.md
§ Phase 8 > Retrieval Stack, PRD #100). Each Stack's retrieval is exposed as a
plain callable ``(query: str, k: int) -> list[RetrievedItem]`` — NO HTTP — so
the DeepEval runner drives both arms in one process.

Stack A (Wiki + BM25, markdown_kb): indexes the committed ``/ingest`` wiki
output under ``wiki/{entities,concepts}/`` and retrieves Sections via BM25
(ADR-0006 W1: wiki is the sole query surface). A retrieved wiki Section is
resolved back to its docs Gold Section id via the page's ``sources``
frontmatter so the C5c metric can compare against the docs-granular gold.

Stack B (Vector RAG, vector_rag): indexes the raw corpus into FAISS and
retrieves Chunks whose ``source`` is already a docs Section id.

Production isolation (PRD #100 acceptance): ``index_stack_a`` / ``index_stack_b``
repoint markdown_kb ``SOURCE_DIRS`` and vector_rag ``DOCS_DIR`` to the eval
fixtures, and the caller — ``runner._isolate_markdown_kb`` (mirrored by the test
suite's autouse conftest fixture) — redirects markdown_kb ``INDEX_PATH`` /
``WIKI_DIR`` / log path to a tmp directory, so the build's atomic-write and
``write_wiki_index`` side effects land in tmp and production ``wiki/`` /
``docs/`` / ``.kb/`` are never read or written.
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown_kb.app.indexer as mk_indexer
import vector_rag.app.indexer as vr_indexer

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
def stack_a_retrieval(query: str, k: int = 3) -> list[RetrievedItem]:
    """Retrieve via markdown_kb BM25 over the indexed wiki, normalised to docs ids.

    Assumes the index has been built against the eval wiki fixtures (see
    ``index_stack_a``). Each BM25 hit Section's wiki-slug file is mapped back to
    the docs Gold Section id via the wiki page's ``sources`` frontmatter.
    """
    file_to_gold = _wiki_slug_to_gold_section()
    items: list[RetrievedItem] = []
    for section, _score in mk_indexer.search(query, k=k):
        gold_id = file_to_gold.get(section.file, section.id)
        items.append(
            RetrievedItem(
                source_section_id=gold_id,
                content=section.content,
                heading_path=list(section.heading_path),
            )
        )
    return items


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
