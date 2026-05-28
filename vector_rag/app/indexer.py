"""Deep module per Ousterhout. Public surface: ``Chunk``, ``build_index``, ``search``, ``save_vector_index``, ``load_vector_index``, ``DOCS_DIR``, ``FAISS_INDEX_DIR``.

Vector RAG (Stack B) retrieval core — heading-aware sectioning, recursive
character chunking, FAISS search, and on-disk index persistence.

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

import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import openai
from fastapi import HTTPException
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ADR-0002: reusing markdown_kb's parser is an intentional recorded cross-app
# coupling so a Chunk's parent-Section id uses the identical slug convention as
# the docs Section ids Stack A is scored against. Importing the leaf functions
# only — no markdown_kb state is mutated from here.
from markdown_kb.app.indexer import parse_markdown

from .logger import log_event

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# Persisted FAISS index lives under .kb/faiss_index/ (PROMPT.md verification
# contract). The directory holds FAISS's own index.faiss + index.pkl plus our
# metadata.json. .kb/ is gitignored — rebuilt by POST /index.
FAISS_INDEX_DIR = Path(__file__).resolve().parents[2] / ".kb" / "faiss_index"
METADATA_FILENAME = "metadata.json"

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
# swappable by tests via monkeypatch. The FAISS index is held in memory and
# persisted to FAISS_INDEX_DIR on build (see save_vector_index).
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
    """Build a FAISS index over the raw Source corpus and persist it to disk.

    Sections come from ``markdown_kb.parse_markdown`` so a Chunk's ``source`` is
    a single docs Section id under the canonical slug convention. Each Section
    body is recursively character-split (500/50); a multi-sub-fact Section
    therefore yields multiple Chunks, every one tagged with that Section's id.

    On a successful build the index is persisted via :func:`save_vector_index`
    so a server restart reloads it without re-embedding (PROMPT.md contract).
    An empty corpus clears any in-memory index but leaves a previously
    persisted one untouched.

    Returns ``(files_indexed, chunks_indexed)``.
    """
    global vectorstore, files_indexed, chunks_indexed

    documents = _load_documents(docs_dir)

    file_count = len({d.metadata["file"] for d in documents}) if documents else 0

    if not documents:
        vectorstore = None
        files_indexed = 0
        chunks_indexed = 0
        log_event("index_built", "files=0 chunks=0")
        return 0, 0

    vectorstore = _embed_with_error_handling(documents)
    files_indexed = file_count
    chunks_indexed = len(documents)
    save_vector_index()
    log_event("index_built", f"files={files_indexed} chunks={chunks_indexed}")
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
# Persistence (PROMPT.md contract: .kb/faiss_index/ + metadata.json)
# ---------------------------------------------------------------------------
def save_vector_index(index_dir: Path | None = None) -> None:
    """Persist the in-memory FAISS index + ``metadata.json`` atomically.

    ``index_dir`` defaults to the module-level ``FAISS_INDEX_DIR``, resolved at
    call time so tests monkeypatching it (and the startup lifespan) see the
    current value rather than a def-time-bound default.

    Writes the whole index directory (FAISS's ``index.faiss`` / ``index.pkl``
    plus our ``metadata.json``) to a sibling tmp directory, then swaps it into
    place with :func:`os.replace` (CODING_STANDARD §2.6 atomic write). A crash
    mid-write therefore never leaves a half-written index for the next load.

    ``metadata.json`` carries ``files_indexed`` / ``chunks_indexed`` and the
    ``embedding_model`` so a reload can report the same counts and detect an
    embedding-model mismatch without re-embedding the corpus.

    No-op when there is no index to persist (``vectorstore is None``).
    """
    if vectorstore is None:
        return

    if index_dir is None:
        index_dir = FAISS_INDEX_DIR

    index_dir.parent.mkdir(parents=True, exist_ok=True)

    # Build the complete directory in a sibling tmp dir, then os.replace it over
    # the target as one atomic step. FAISS.save_local writes two files; writing
    # them straight into index_dir would expose a torn state to a concurrent load.
    tmp_dir = Path(tempfile.mkdtemp(dir=index_dir.parent, prefix="faiss_index_"))
    try:
        vectorstore.save_local(str(tmp_dir))
        metadata = {
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "embedding_model": EMBEDDING_MODEL,
        }
        (tmp_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        # os.replace onto an existing directory fails on Windows, so clear the
        # old index dir first. The window between rmtree and replace is the only
        # non-atomic moment; it is acceptable for the single-process prototype
        # model (CODING_STANDARD §2.6 — multi-worker is post-prototype).
        if index_dir.exists():
            shutil.rmtree(index_dir)
        os.replace(tmp_dir, index_dir)
    except Exception:
        with contextlib.suppress(OSError):
            shutil.rmtree(tmp_dir)
        raise


def load_vector_index(index_dir: Path | None = None) -> tuple[int, int]:
    """Load a persisted FAISS index into the module-level state.

    ``index_dir`` defaults to the module-level ``FAISS_INDEX_DIR``, resolved at
    call time (mirrors :func:`save_vector_index`).

    Returns ``(files_indexed, chunks_indexed)``. Returns ``(0, 0)`` and leaves
    the in-memory index unset when no persisted index exists.

    Fail-fast on corruption (CODING_STANDARD §4.1): a present-but-unreadable
    index (missing ``metadata.json``, unparseable JSON, or a FAISS payload that
    will not deserialize) raises rather than silently serving an empty index.
    A successful load emits an ``index_loaded`` log entry.
    """
    global vectorstore, files_indexed, chunks_indexed

    if index_dir is None:
        index_dir = FAISS_INDEX_DIR

    if not index_dir.exists():
        return 0, 0

    metadata_path = index_dir / METADATA_FILENAME
    if not metadata_path.exists():
        # Present-but-incomplete index — fail fast rather than serve empty.
        raise RuntimeError(
            f"persisted FAISS index at {index_dir} is missing {METADATA_FILENAME}"
        )

    # Let json.JSONDecodeError propagate — a corrupt metadata file is fail-fast.
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # FAISS.load_local deserializes a pickle (our own, written by save_local);
    # allow_dangerous_deserialization is required by the API. Any deserialization
    # failure propagates as fail-fast per §4.1.
    loaded = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    vectorstore = loaded
    files_indexed = int(metadata.get("files_indexed", 0))
    chunks_indexed = int(metadata.get("chunks_indexed", 0))

    log_event("index_loaded", f"files={files_indexed} chunks={chunks_indexed}")
    return files_indexed, chunks_indexed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_faiss(documents: list[Document]) -> FAISS:
    """Construct the FAISS vectorstore from chunk Documents (real-embedding path)."""
    return FAISS.from_documents(documents, get_embeddings())


def _embed_with_error_handling(documents: list[Document]) -> FAISS:
    """Embed + build FAISS, mapping OpenAI exceptions to HTTP status.

    The embedding call is an LLM-facing operation, so it follows the same
    OpenAI exception → HTTP status mapping as /chat (CODING_STANDARD §4.2):
      - APITimeoutError, RateLimitError → HTTP 503 (openai_transient)
      - AuthenticationError            → HTTP 500 (openai_auth)
      - Any other APIError             → HTTP 500 (openai_api)

    Each branch emits a ``chat_error`` log entry (the repo's shared LLM-error
    kind). Use ``raise HTTPException(...) from exc`` to preserve the chain.
    """
    try:
        return _build_faiss(documents)
    except (openai.APITimeoutError, openai.RateLimitError) as exc:
        log_event(
            "chat_error", f"op=index kind=openai_transient exc={type(exc).__name__}"
        )
        raise HTTPException(
            status_code=503,
            detail="Embedding service temporarily unavailable, please retry.",
        ) from exc
    except openai.AuthenticationError as exc:
        log_event("chat_error", f"op=index kind=openai_auth exc={type(exc).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Embedding service auth failed (check OPENAI_API_KEY).",
        ) from exc
    except openai.APIError as exc:
        log_event("chat_error", f"op=index kind=openai_api exc={type(exc).__name__}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding service error: {exc!s}",
        ) from exc


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
