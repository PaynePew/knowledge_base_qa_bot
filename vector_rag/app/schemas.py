"""Shallow module per Ousterhout. Public surface: the Pydantic request/response models.

Pydantic boundary schemas for Stack B's FastAPI surface. Primitives only — no
LangChain types cross this boundary (CODING_STANDARD §2.4).
"""

from __future__ import annotations

from pydantic import BaseModel


class IndexResponse(BaseModel):
    files_indexed: int
    chunks_indexed: int


class ChatRequest(BaseModel):
    query: str


class SourceInfo(BaseModel):
    source: str
    heading: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
