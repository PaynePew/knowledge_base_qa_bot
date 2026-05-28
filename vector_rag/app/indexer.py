"""Deep module per Ousterhout. Public surface: ``Chunk``, ``build_index``, ``search``, ``DOCS_DIR``.

Vector RAG (Stack B) retrieval core — heading-aware sectioning, recursive
character chunking, and in-memory FAISS search.

Stack B is the Vector RAG arm of the Phase 8 retrieval comparison (CONTEXT.md
§ Phase 8 vocabulary, PRD #100). The corpus is the raw ``docs/`` Source layer
(NOT the curated wiki layer — that is Stack A's surface). The pipeline:

    raw Source markdown
      → markdown_kb.parse_markdown (heading-aware Sections; ADR-0002 blessed
        cross-app reuse so Chunk.source uses the identical slug convention as
        the docs Section ids Stack A is scored against)
      → RecursiveCharacterTextSplitter within each Section body (chunk_size=500,
        chunk_overlap=50) → one or more Chunks per Section
      → OpenAIEmbeddings → in-memory FAISS index
      → search(query, k) → ranked Chunks

A ``Chunk`` is a character-bounded slice within a Section that carries its
parent Section's id as ``source`` so Citations stay Section-granular
(CONTEXT.md § Phase 8 > Chunk). ``Chunk`` satisfies the
``markdown_kb.app.grounding.CitableContent`` Protocol (``id`` / ``heading_path``
/ ``content``) so grounding.verify() can consume it unchanged (ADR-0004 Q9).

LangChain's ``Document`` is an implementation detail of the indexing pipeline
and never leaves this module — ``search`` returns ``Chunk``, not ``Document``
(CODING_STANDARD §2.4 no-LangChain-leak; vector_rag is an LLM-facing module so
it may import LangChain internally).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ADR-0002: reusing markdown_kb's parser is an intentional recorded cross-app
# coupling so a Chunk's parent-Section id uses the identical slug convention as
# the docs Section ids Stack A is scored against. Importing the leaf functions
# only — no markdown_kb state is mutated from here.
from markdown_kb.app.indexer import parse_markdown

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Recursive character splitter: prefer Markdown structure (paragraph, line)
# before falling back to sentence and word boundaries, so a fact is cut at the
# coarsest available boundary (CONTEXT.md § Phase 8 > Chunk).
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """A character-bounded slice within a Section (CONTEXT.md § Phase 8 > Chunk).

    ``source`` is the parent Section's id (``{source-filename}#{heading-slug}``)
    so Citations stay Section-granular. ``id``, ``heading_path`` and ``content``
    satisfy the ``CitableContent`` Protocol so grounding.verify() consumes a
    Chunk with no changes (ADR-0004 Q9).
    """

    id: str
    source: str
    heading_path: list[str]
    content: str


# ---------------------------------------------------------------------------
# In-memory index state
# ---------------------------------------------------------------------------
# Single-process prototype model (CODING_STANDARD §2.7): module-level globals,
# swappable by tests via monkeypatch. FAISS is held in memory; persistence is a
# later slice.
vectorstore: FAISS | None = None
_embeddings: OpenAIEmbeddings | None = None
files_indexed = 0
chunks_indexed = 0


# ---------------------------------------------------------------------------
# Embeddings (lazy singleton — CODING_STANDARD §2.7 / §10 lazy-singleton)
# ---------------------------------------------------------------------------
def get_embeddings() -> OpenAIEmbeddings:
    """Return the lazily-constructed OpenAI embeddings client.

    Raises RuntimeError when OPENAI_API_KEY is absent so a missing key fails
    fast with a clear message rather than deep inside the FAISS call. Tests
    swap the FAISS factory (``_build_faiss``) so this never runs offline.
    """
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------
def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    """Build an in-memory FAISS index over the raw Source corpus.

    Sections come from ``markdown_kb.parse_markdown`` so a Chunk's ``source`` is
    a single docs Section id under the canonical slug convention. Each Section
    body is recursively character-split (500/50); a multi-sub-fact Section
    therefore yields multiple Chunks, every one tagged with that Section's id.

    Returns ``(files_indexed, chunks_indexed)``.
    """
    global vectorstore, files_indexed, chunks_indexed

    documents = _load_documents(docs_dir)

    file_count = len({d.metadata["file"] for d in documents}) if documents else 0

    if not documents:
        vectorstore = None
        files_indexed = 0
        chunks_indexed = 0
        return 0, 0

    vectorstore = _build_faiss(documents)
    files_indexed = file_count
    chunks_indexed = len(documents)
    return files_indexed, chunks_indexed


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(query: str, k: int = 3) -> list[Chunk]:
    """Return the top-``k`` Chunks for ``query`` by vector similarity.

    Returns domain ``Chunk`` objects (never LangChain ``Document``) so no
    LangChain type leaks past this module (CODING_STANDARD §2.4). An empty list
    is returned when the index has not been built.
    """
    if vectorstore is None:
        return []
    scored = vectorstore.similarity_search_with_score(query, k=k)
    return [_chunk_from_document(doc) for doc, _score in scored]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_faiss(documents: list[Document]) -> FAISS:
    """Construct the FAISS vectorstore from chunk Documents (real-embedding path)."""
    return FAISS.from_documents(documents, get_embeddings())


def _load_documents(docs_dir: Path) -> list[Document]:
    """Section-then-char-split the corpus into chunk-carrying LangChain Documents.

    Each Document's ``page_content`` is one Chunk's text; its metadata carries
    ``source`` (parent Section id), ``heading_path`` (joined breadcrumb) and
    ``file`` (for the files-indexed count). Empty-body Sections (heading-only
    leaves) are skipped — a vector chunk needs body text to embed meaningfully.
    """
    documents: list[Document] = []
    for md_file in sorted(docs_dir.glob("**/*.md")):
        for section in parse_markdown(md_file):
            if not section.content.strip():
                continue
            for piece in _splitter.split_text(section.content):
                documents.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source": section.id,
                            "heading_path": list(section.heading_path),
                            "file": section.file,
                        },
                    )
                )
    return documents


def _chunk_from_document(doc: Document) -> Chunk:
    """Map a retrieved LangChain Document back to a domain Chunk.

    Confines the LangChain ``Document`` type to this module (CODING_STANDARD
    §2.4). ``id`` mirrors ``source`` because, at Section-granular citation,
    a Chunk identifies itself by its parent Section id.
    """
    source = doc.metadata["source"]
    return Chunk(
        id=source,
        source=source,
        heading_path=list(doc.metadata.get("heading_path", [])),
        content=doc.page_content,
    )
