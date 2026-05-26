"""HTTP wiring for /health, /index, /chat. No domain logic."""

from fastapi import APIRouter

from .indexer import build_index
from .retrieval import query
from .schemas import ChatRequest, ChatResponse, IndexResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs() -> IndexResponse:
    files_count, sections_count = build_index()
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    # query() returns a raw dict; FastAPI serializes via response_model.
    return query(req.query)
