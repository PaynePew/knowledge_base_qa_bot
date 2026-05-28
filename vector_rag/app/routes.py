"""Shallow module per Ousterhout. Public surface: ``router``.

FastAPI routes for Stack B. Wires the deep retrieval/indexer modules to the
HTTP surface; carries no business logic (CODING_STANDARD §2.3).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .indexer import build_index
from .retrieval import query
from .schemas import ChatRequest, ChatResponse, IndexResponse

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs() -> IndexResponse:
    try:
        files_count, chunks_count = build_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(files_indexed=files_count, chunks_indexed=chunks_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    return query(req.query)
