"""Shallow module per Ousterhout. Public surface: ``app``.

FastAPI application entry point for Stack B (Vector RAG). The in-memory FAISS
index is built on demand via POST /index; persistence is a later slice, so
there is no startup index load.
"""

from __future__ import annotations

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot")
app.include_router(router)
