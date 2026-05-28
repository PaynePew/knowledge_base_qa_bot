"""Deep module per Ousterhout. Public surface: ``query``, ``build_prompt``, ``get_llm``.

Vector RAG (Stack B) query path — retrieve Chunks, build a grounded prompt,
call the LLM.

This module owns Stack B's LLM call site, so it is an LLM-facing module and may
import LangChain (CODING_STANDARD §2.4). It consumes domain ``Chunk`` objects
from :mod:`vector_rag.app.indexer` (never LangChain ``Document``) so the
retrieval boundary stays framework-free.

The Phase 8 Slice 1 tracer exercises only the retrieval core
(``indexer.search``); the full grounded ``/chat`` answer path is unchanged
scaffold and is thickened in a later slice.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer
from .indexer import Chunk

# ADR-0001 strict-grounded contract: never hand weak context to the LLM. The
# literal Cannot Confirm sentinel mirrors markdown_kb's contract surface.
CANNOT_CONFIRM = "I cannot confirm from the knowledge base."

SYSTEM_PROMPT = (
    "You answer strictly from the provided CONTEXT. Cite every claim with a "
    "[Source: slug#heading] marker drawn from the context. If the CONTEXT does "
    f"not contain the answer, reply exactly: {CANNOT_CONFIRM}"
)

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return the lazily-constructed chat LLM (CODING_STANDARD §10 lazy-singleton)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query_text: str, chunks: list[Chunk]) -> str:
    """Render the CONTEXT (one block per Chunk) followed by the QUESTION."""
    blocks = [f"[Source: {chunk.source}]\n{chunk.content}" for chunk in chunks]
    context = "\n\n".join(blocks) if blocks else "(no context)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query_text}"


def query(question: str) -> dict:
    """Answer ``question`` from the FAISS index (scaffold answer path).

    The retrieval core (``indexer.search`` → ``Chunk`` list) is the
    Phase 8 Slice 1 deliverable; the grounded synthesis below is unchanged
    scaffold behaviour that later slices thicken.
    """
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    chunks = indexer.search(question, k=3)
    if not chunks:
        return {"answer": CANNOT_CONFIRM, "sources": []}

    response = get_llm().invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(question, chunks)),
        ]
    )

    sources = [
        {
            "source": chunk.source,
            "heading": chunk.heading_path[-1] if chunk.heading_path else "",
            "content": chunk.content[:240],
        }
        for chunk in chunks
    ]
    return {"answer": response.content, "sources": sources}
