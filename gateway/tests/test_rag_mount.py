"""Gateway mount tests: GET /rag/health + POST /rag/index reachable (issue #138).

AC: gateway mounts /rag; env loaded before import.
  - GET /rag/health returns 200 {"status": "ok"}
  - POST /rag/index builds+persists index; returns IndexResponse shape
  - /wiki/* routes are unaffected

All tests are hermetic: fake embeddings (no OPENAI_API_KEY).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# Fake embeddings (mirrors test_chat_stream_rag.py to stay offline)
# ---------------------------------------------------------------------------


class _FakeEmbeddings(Embeddings):
    _DIM = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_rag_paths(tmp_path, monkeypatch):
    """Redirect FAISS_INDEX_DIR + LOG_PATH to tmp for isolation."""
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    yield
    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0


@pytest.fixture()
def gateway_client(monkeypatch):
    """TestClient for the Gateway app with fake embeddings injected."""
    fake = _FakeEmbeddings()
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: fake)

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


# ---------------------------------------------------------------------------
# AC: GET /rag/health reachable through gateway
# ---------------------------------------------------------------------------


def test_rag_health_reachable_through_gateway(gateway_client):
    """GET /rag/health returns 200 through the gateway mount (issue #138 AC)."""
    resp = gateway_client.get("/rag/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# AC: POST /rag/index reachable through gateway and builds index
# ---------------------------------------------------------------------------


def test_rag_index_reachable_through_gateway(gateway_client):
    """POST /rag/index returns 200 and IndexResponse shape (issue #138 AC)."""
    resp = gateway_client.post("/rag/index")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "files_indexed" in data, f"IndexResponse must have files_indexed: {data}"
    assert "chunks_indexed" in data, f"IndexResponse must have chunks_indexed: {data}"
    assert data["files_indexed"] > 0, "POST /rag/index must index at least one file"
    assert data["chunks_indexed"] > 0, "POST /rag/index must produce at least one chunk"


def test_rag_index_persists_faiss_index(gateway_client, tmp_path, monkeypatch):
    """POST /rag/index persists the FAISS index to the configured FAISS_INDEX_DIR."""
    # tmp_path redirect is already active (autouse fixture)
    gateway_client.post("/rag/index")
    # After indexing, FAISS_INDEX_DIR must exist and contain the persisted index
    assert vr_indexer.FAISS_INDEX_DIR.exists(), (
        "FAISS_INDEX_DIR must be created after POST /rag/index"
    )


# ---------------------------------------------------------------------------
# Existing /wiki/* routes unaffected
# ---------------------------------------------------------------------------


def test_wiki_health_still_works_after_rag_mount(gateway_client):
    """/wiki/health returns 200 — wiki mount is unaffected by the /rag mount."""
    resp = gateway_client.get("/wiki/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
