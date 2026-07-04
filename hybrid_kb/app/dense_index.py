"""Deep module per Ousterhout. Public surface: ``Section`` (re-export), ``build_index``, ``search``, ``search_with_distance``, ``save_dense_index``, ``load_dense_index``, ``warm_embeddings_client``, ``DENSE_INDEX_DIR``, ``EMBEDDING_MODEL``.

Hybrid Retrieval (Stack C) — dense-over-wiki Section index (slice S1, ADR-0018).

This is the dense arm of the Phase 13 Hybrid stack. It is the **same-corpus**
counterpart to BM25 (``markdown_kb``): one embedding per **wiki Section**, built
from ``markdown_kb``'s already-filtered Section list (entities + concepts +
``status: live`` qa), so each dense entry's id equals the corresponding BM25
Section id **1:1** (ADR-0018 same-corpus invariant). Granularity is the
**Section** (NOT the char-Chunk that ``vector_rag`` uses over ``docs/``) — this
keeps fusion trivial and avoids the wrong-retrieval-unit failure mode (FM3) by
construction.

Pipeline::

    wiki/ Sections (markdown_kb.parse_markdown + the same status-live qa filter)
      → one Document per Section (page_content = heading-path + body, so a
        short/heading-only Section still embeds a non-empty, meaningful text)
      → OpenAIEmbeddings(text-embedding-3-small) → in-memory FAISS index
      → search_with_distance(query, k) → ranked (Section, distance), filtered to
        the QUERY's language exactly like the BM25 path.

The index ships as a **committed seed** under ``.kb/hybrid_dense/`` (separate
from ``vector_rag``'s ``.kb/faiss_index/`` docs seed and ``markdown_kb``'s
``.kb/index.json`` BM25 seed), persisted atomically (tmp dir + ``os.replace``)
and force-added to git so a fresh clone answers Hybrid queries with no build
step. A guard test asserts the committed seed's metadata + 1:1 id alignment with
the committed BM25 wiki index so a stale seed cannot pass a green fresh-build
suite while breaking production (the #307 lesson).

What is reused from ``markdown_kb`` (ADR-0018 blessed cross-app coupling, the
same pattern ``vector_rag`` already uses):
  * ``Section`` — the retrieval unit stays ``Section`` (no new Document / Chunk /
    Article class, CODING_STANDARD §2.x). ``Section`` satisfies the
    ``CitableContent`` protocol so every downstream concern is reused unchanged.
  * ``parse_markdown`` + ``SOURCE_DIRS`` + ``_passes_index_filter`` — to derive
    the SAME filtered Section list (and therefore the same ids) BM25 builds from.
  * ``detect_lang`` / ``LANG_METADATA_KEY`` / ``_section_lang`` — so query-time
    language routing and index-time language tagging share one classifier and
    can never drift from the BM25 path.

LangChain's ``Document`` is an indexing-pipeline implementation detail and never
leaves this module — ``search`` returns ``Section``, not ``Document``
(CODING_STANDARD §2.4 no-LangChain-leak; hybrid_kb is an LLM-facing module so it
may import LangChain internally).

Out of scope for S1 (delivered by S2, issue #312): the per-arm OR-gate, the
dense distance-gate calibration, and RRF fusion. S1 ships only the dense index
build, the committed seed + atomic persist + fail-fast load, and the
language-filtered dense retrieval call.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ADR-0018 blessed cross-app reuse (the same recorded coupling vector_rag has on
# parse_markdown / detect_lang). Importing leaf functions only — no markdown_kb
# state is mutated from here. ``_passes_index_filter`` is THE filter that defines
# "already-filtered Section list", so reusing it (rather than re-deriving the
# status-live qa gate) is what keeps dense ids aligned 1:1 with BM25 ids. Any
# drift is caught by the committed-seed guard test, which checks the dense seed
# ids against the committed BM25 .kb/index.json.
from markdown_kb.app.indexer import (
    LANG_METADATA_KEY,
    SOURCE_DIRS,
    Section,
    _passes_index_filter,
    _section_lang,
    detect_lang,
    parse_markdown,
)

from .logger import log_event

__all__ = [
    "Section",
    "build_index",
    "search",
    "search_with_distance",
    "save_dense_index",
    "load_dense_index",
    "warm_embeddings_client",
    "DENSE_INDEX_DIR",
    "EMBEDDING_MODEL",
]

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
# Repo root: hybrid_kb/app/dense_index.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The dense-over-wiki seed lives in ITS OWN directory under .kb/ — separate from
# vector_rag's .kb/faiss_index/ (docs Chunks) and markdown_kb's .kb/index.json
# (BM25). The directory holds FAISS's own index.faiss + index.pkl plus our
# metadata.json. .kb/ is gitignored; this seed is force-committed (git add -f),
# like the other two seeds, so a fresh clone works without a build step.
DENSE_INDEX_DIR = _REPO_ROOT / ".kb" / "hybrid_dense"
METADATA_FILENAME = "metadata.json"

# Same embedding model the Vector RAG stack uses (ADR-0018) — keeps cost
# identical and the three-arm comparison embedding-consistent.
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# In-memory index state
# ---------------------------------------------------------------------------
# Single-process prototype model (CODING_STANDARD §2.7): module-level globals,
# swappable by tests via monkeypatch. The FAISS index is held in memory and
# persisted to DENSE_INDEX_DIR on build (see save_dense_index).
vectorstore: FAISS | None = None
_embeddings: OpenAIEmbeddings | None = None
sections_indexed = 0


# ---------------------------------------------------------------------------
# Embeddings (lazy singleton — CODING_STANDARD §2.7 / §10 lazy-singleton)
# ---------------------------------------------------------------------------
def get_embeddings() -> OpenAIEmbeddings:
    """Return the lazily-constructed OpenAI embeddings client.

    Raises RuntimeError when OPENAI_API_KEY is absent so a missing key fails
    fast with a clear message rather than deep inside the FAISS call. Hermetic
    tests swap this leaf (``get_embeddings``) for a deterministic offline fake,
    so the real FAISS build / save / load / search path runs without network.
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


def warm_embeddings_client() -> None:
    """Fire one tiny embedding call to prime the embeddings client's connection (issue #439).

    Mirrors ``vector_rag.app.indexer.warm_embeddings_client`` for Stack C's own
    embeddings singleton. Opt-in — called only from Gateway startup behind
    ``KB_WARMUP_PING`` (see ``gateway/app/warmup.py``).

    Best-effort: any failure (auth, quota, network) is caught and logged, never
    raised — a failed ping degrades to the pre-issue-#439 behaviour (the client
    still lazily constructs + connects on the next real call) and never blocks
    Gateway startup.
    """
    try:
        get_embeddings().embed_query("hi")
        log_event("startup_warmup", "client=hybrid_embeddings status=ok")
    except Exception as exc:
        log_event(
            "startup_warmup",
            f"client=hybrid_embeddings status=failed exc={type(exc).__name__}",
        )


# ---------------------------------------------------------------------------
# Wiki Section list (the BM25-aligned corpus)
# ---------------------------------------------------------------------------
def filtered_wiki_sections() -> list[Section]:
    """Return the SAME filtered wiki Section list ``markdown_kb`` builds BM25 over.

    Mirrors ``markdown_kb.app.indexer.build_index``'s production scan exactly
    (``SOURCE_DIRS`` = entities + concepts + qa; ``source_id=md_file.stem`` so
    ids are slug-based; the ``status: live`` qa gate via ``_passes_index_filter``)
    so the resulting Section ids are byte-identical to the BM25 Section ids — the
    ADR-0018 same-corpus invariant. Pure read over ``wiki/``: no index is
    persisted and the BM25 ``.kb/index.json`` is NOT written, so building the
    dense index never mutates the BM25 seed.
    """
    result: list[Section] = []
    for source_dir in SOURCE_DIRS:
        for md_file in sorted(source_dir.glob("**/*.md")):
            page_sections = parse_markdown(md_file, source_id=md_file.stem)
            # Same two-stage curation gate BM25 applies: qa pages need
            # frontmatter.status == "live"; entity/concept pages pass through.
            if not _passes_index_filter(md_file, page_sections):
                continue
            result.extend(page_sections)
    return result


def _embed_text(section: Section) -> str:
    """Return the text embedded for one Section (its dense-vector signal).

    Composes the heading-path breadcrumb with the body so a short or heading-only
    leaf Section (Rule 8 empty-body Section) still embeds a non-empty, meaningful
    text rather than the degenerate empty string. The breadcrumb gives a terse
    Section extra topical context the dense model can use. Never empty: every
    wiki Section carries at least a heading, so full 1:1 coverage of the BM25 id
    set is guaranteed.
    """
    breadcrumb = (
        " > ".join(section.heading_path) if section.heading_path else section.heading
    )
    body = section.content.strip()
    if body and breadcrumb:
        return f"{breadcrumb}\n{body}"
    return body or breadcrumb or section.id


def _section_to_document(section: Section) -> Document:
    """Map a domain ``Section`` to the LangChain ``Document`` embedded by FAISS.

    The metadata carries everything needed to reconstruct the ``Section`` at
    retrieval time WITHOUT touching ``markdown_kb`` state, plus the index-time
    language tag (``_section_lang``) so the FAISS search can filter by language
    exactly like the BM25 path. ``id`` IS the BM25 Section id (same-corpus
    invariant). BM25 ``tokens`` are deliberately NOT stored — the dense arm never
    needs them, and downstream citation/grounding reads id / heading_path /
    content only.
    """
    return Document(
        page_content=_embed_text(section),
        metadata={
            "id": section.id,
            "file": section.file,
            "heading": section.heading,
            "heading_path": list(section.heading_path),
            "content": section.content,
            LANG_METADATA_KEY: _section_lang(section),
        },
    )


def _section_from_document(doc: Document) -> Section:
    """Reconstruct a domain ``Section`` from a retrieved FAISS ``Document``.

    Confines the LangChain ``Document`` type to this module (CODING_STANDARD
    §2.4). ``tokens`` is reconstructed empty: the dense arm does not score with
    BM25 tokens, and every downstream consumer of a dense hit reads id /
    heading_path / content (the ``CitableContent`` surface), never ``tokens``.
    The language tag is carried through ``metadata`` so a reconstructed Section
    reports the same language it was indexed under.
    """
    meta = doc.metadata
    section_id = meta["id"]
    return Section(
        id=section_id,
        file=meta.get("file", ""),
        heading=meta.get("heading", ""),
        heading_path=list(meta.get("heading_path", [])),
        content=meta.get("content", ""),
        tokens=[],
        metadata={LANG_METADATA_KEY: meta.get(LANG_METADATA_KEY, detect_lang(""))},
    )


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------
def build_index(
    sections: list[Section] | None = None,
    index_dir: Path | None = None,
) -> int:
    """Build a dense FAISS index over the wiki Section corpus and persist it.

    ``sections`` defaults to :func:`filtered_wiki_sections` (the BM25-aligned
    list). A caller MAY pass an explicit Section list (e.g. the bake driver
    reusing ``markdown_kb``'s in-memory list, or a test fixture) — in which case
    ids align with whatever that list carries. One embedding per Section
    (Section granularity, NOT char-Chunk).

    On a successful build the index is persisted via :func:`save_dense_index`
    so a restart reloads it without re-embedding. An empty corpus clears any
    in-memory index but leaves a previously persisted one untouched.

    Returns ``sections_indexed``.
    """
    global vectorstore, sections_indexed

    if sections is None:
        sections = filtered_wiki_sections()

    if not sections:
        vectorstore = None
        sections_indexed = 0
        log_event("dense_index_built", "sections=0")
        return 0

    documents = [_section_to_document(s) for s in sections]
    vectorstore = _build_faiss(documents)
    sections_indexed = len(documents)
    save_dense_index(index_dir)
    log_event("dense_index_built", f"sections={sections_indexed}")
    return sections_indexed


def _build_faiss(documents: list[Document]) -> FAISS:
    """Construct the FAISS vectorstore from Section Documents (real-embedding path).

    The single network seam: hermetic tests monkeypatch ``get_embeddings`` to a
    deterministic offline fake, so this exercises the REAL FAISS build path
    without embedding cost.
    """
    return FAISS.from_documents(documents, get_embeddings())


# ---------------------------------------------------------------------------
# Search (dense arm — language-filtered, consistent with the BM25 path)
# ---------------------------------------------------------------------------
def search_with_distance(query: str, k: int = 3) -> list[tuple[Section, float]]:
    """Return the top-``k`` ``(Section, distance)`` pairs in the QUERY's language.

    Language-filtered retrieval, consistent with the BM25 path
    (``markdown_kb.app.indexer.search``): the query language is classified by the
    SAME ``detect_lang`` helper that tagged each Section at index time, and the
    FAISS search is restricted to Sections whose index-time ``lang`` tag matches.
    A Chinese query is scored only against ``zh`` Sections and an English query
    only against ``en`` Sections, closing the cross-language leak before any
    downstream gate — exactly the discipline the BM25 arm enforces.

    The FAISS distance (lower = closer; default L2) is the native dense score the
    S2 per-arm OR-gate will threshold; S1 surfaces it but does not gate on it.
    Returns domain ``Section`` objects only (no LangChain ``Document`` leak,
    §2.4). Empty when the index is not built OR when no Section matches the query
    language.
    """
    if vectorstore is None:
        return []
    query_lang = detect_lang(query)
    # fetch_k must exceed FAISS's default (20) so the post-retrieval metadata
    # filter has enough same-language candidates to fill k even when the query
    # language is the minority of the index — FAISS filters AFTER fetching
    # fetch_k neighbours. The wiki corpus is small (~10^2 Sections), so a
    # generous fetch_k effectively scans the whole index.
    scored = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter={LANG_METADATA_KEY: query_lang},
        fetch_k=max(k * 20, 200),
    )
    return [(_section_from_document(doc), float(distance)) for doc, distance in scored]


def search(query: str, k: int = 3) -> list[Section]:
    """Return the top-``k`` dense wiki Sections for ``query`` (distance dropped).

    Thin wrapper over :func:`search_with_distance`. Returns domain ``Section``
    objects (never LangChain ``Document``) so no LangChain type leaks past this
    module (CODING_STANDARD §2.4). Empty when the index has not been built.
    """
    return [section for section, _distance in search_with_distance(query, k=k)]


# ---------------------------------------------------------------------------
# Persistence (committed seed under .kb/hybrid_dense/ + metadata.json)
# ---------------------------------------------------------------------------
def save_dense_index(index_dir: Path | None = None) -> None:
    """Persist the in-memory FAISS index + ``metadata.json`` atomically.

    ``index_dir`` defaults to the module-level ``DENSE_INDEX_DIR``, resolved at
    call time so tests monkeypatching it (and a startup lifespan) see the current
    value rather than a def-time-bound default.

    Writes the whole index directory (FAISS's ``index.faiss`` / ``index.pkl``
    plus our ``metadata.json``) to a sibling tmp directory, then swaps it into
    place with :func:`os.replace` (CODING_STANDARD §2.6 atomic write). A crash
    mid-write therefore never leaves a half-written index for the next load.

    ``metadata.json`` carries ``sections_indexed`` and the ``embedding_model``
    so a reload can report the same count and detect an embedding-model mismatch
    without re-embedding the corpus. ``granularity`` records the Section-level
    contract so a reader cannot mistake this for the char-Chunk docs seed.

    No-op when there is no index to persist (``vectorstore is None``).
    """
    if vectorstore is None:
        return

    if index_dir is None:
        index_dir = DENSE_INDEX_DIR

    index_dir.parent.mkdir(parents=True, exist_ok=True)

    # Build the complete directory in a sibling tmp dir, then os.replace it over
    # the target as one atomic step. FAISS.save_local writes two files; writing
    # them straight into index_dir would expose a torn state to a concurrent load.
    tmp_dir = Path(tempfile.mkdtemp(dir=index_dir.parent, prefix="hybrid_dense_"))
    try:
        vectorstore.save_local(str(tmp_dir))
        metadata = {
            "sections_indexed": sections_indexed,
            "embedding_model": EMBEDDING_MODEL,
            "granularity": "section",
        }
        (tmp_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        # os.replace onto an existing directory fails on Windows, so clear the
        # old index dir first. The window between rmtree and replace is the only
        # non-atomic moment; acceptable for the single-process prototype model
        # (CODING_STANDARD §2.6 — multi-worker is post-prototype).
        if index_dir.exists():
            shutil.rmtree(index_dir)
        os.replace(tmp_dir, index_dir)
    except Exception:
        with contextlib.suppress(OSError):
            shutil.rmtree(tmp_dir)
        raise


def load_dense_index(index_dir: Path | None = None) -> int:
    """Load a persisted dense index into the module-level state.

    ``index_dir`` defaults to the module-level ``DENSE_INDEX_DIR``, resolved at
    call time (mirrors :func:`save_dense_index`).

    Returns ``sections_indexed``. Returns ``0`` and leaves the in-memory index
    unset when no persisted index exists.

    Fail-fast on corruption (CODING_STANDARD §4.1): a present-but-unreadable
    index (missing ``metadata.json``, unparseable JSON, or a FAISS payload that
    will not deserialize) raises rather than silently serving an empty index —
    so a corrupt or stale committed seed cannot degrade into a silent empty
    fallback (the #307 lesson). A successful load emits ``dense_index_loaded``.
    """
    global vectorstore, sections_indexed

    if index_dir is None:
        index_dir = DENSE_INDEX_DIR

    if not index_dir.exists():
        return 0

    metadata_path = index_dir / METADATA_FILENAME
    if not metadata_path.exists():
        # Present-but-incomplete index — fail fast rather than serve empty.
        raise RuntimeError(
            f"persisted dense index at {index_dir} is missing {METADATA_FILENAME}"
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
    sections_indexed = int(metadata.get("sections_indexed", 0))

    log_event("dense_index_loaded", f"sections={sections_indexed}")
    return sections_indexed
