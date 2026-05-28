"""Shallow module per Ousterhout. Public surface: ``app``.

FastAPI application entry point for Stack B (Vector RAG). Loads .env, wires the
router, and rehydrates the persisted FAISS index on startup via lifespan so a
restart serves /chat without re-embedding the corpus (PROMPT.md contract).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

# Env must be loaded BEFORE importing app modules so the embeddings/LLM
# singletons pick up OPENAI_API_KEY the same way uvicorn does.
load_dotenv(find_dotenv(usecwd=True))

from .indexer import load_vector_index  # noqa: E402
from .routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Rehydrate the persisted FAISS index from .kb/faiss_index/ before serving."""
    load_vector_index()
    yield


app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot", lifespan=lifespan)
app.include_router(router)
