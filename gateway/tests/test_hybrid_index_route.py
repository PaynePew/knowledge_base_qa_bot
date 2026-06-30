"""Gateway route tests: POST /hybrid/index — operator-triggered dense re-embed (issue #348, ADR-0022).

Acceptance criteria tested:
  AC1 — POST /hybrid/index returns 200 with {"sections_indexed": N};
         calls hybrid_kb.app.dense_index.build_index() in-process;
         hybrid_kb gains NO FastAPI app.
  AC2 — After POST /hybrid/index, a wiki Section is findable via the dense arm
         (dense↔BM25 ids re-align by construction after a full re-embed).
  AC3 — /hybrid/index is in ADMIN_PATHS (admin semaphore + kill-switch + budget cap)
         and in _COST_ESTIMATES at ~$0.50; a request with KB_ADMIN_TOKEN set but no
         Bearer token is 401.

All tests are hermetic:
  * Fake embeddings (offline _FakeEmbeddings — real FAISS build/search path, no OpenAI).
  * DENSE_INDEX_DIR + hybrid_kb LOG_PATH redirected to tmp_path (§6.3).
  * The gateway conftest autouse redirects markdown_kb WIKI_DIR / LOG_PATH / INDEX_PATH;
    hybrid_kb paths are additional, added here.
  * No @pytest.mark.live test (AC5 constraint).
"""

from __future__ import annotations

import hashlib

import hybrid_kb.app.dense_index as hk_dense
import hybrid_kb.app.logger as hk_logger
import pytest
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Fake embeddings — mirrors hybrid_kb/tests/conftest.py's _FakeEmbeddings
# ---------------------------------------------------------------------------


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline stand-in for OpenAIEmbeddings (same as hybrid_kb conftest)."""

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
def _redirect_hybrid_paths(tmp_path, monkeypatch):
    """Redirect DENSE_INDEX_DIR + hybrid_kb LOG_PATH to tmp for isolation (§6.3).

    The gateway conftest autouse already redirects markdown_kb wiki paths;
    this adds the hybrid_kb-specific paths that POST /hybrid/index writes.

    Also resets the gateway budget singleton before and after each test.
    POST /hybrid/index is priced at $0.50; without resetting, 6 tests exhaust
    the $3.00 default cap and contaminate subsequent gateway tests with 503s.
    """
    import gateway.app.budget as _budget_mod

    monkeypatch.setattr(hk_dense, "DENSE_INDEX_DIR", tmp_path / ".kb" / "hybrid_dense")
    monkeypatch.setattr(hk_logger, "LOG_PATH", tmp_path / "hybrid_kb" / "log.md")
    # Reset the budget singleton so each test starts with $0.00 accumulated.
    _budget_mod.budget._totals.clear()
    yield
    # Teardown: reset in-memory index state and budget so tests don't bleed.
    hk_dense.vectorstore = None
    hk_dense.sections_indexed = 0
    _budget_mod.budget._totals.clear()


@pytest.fixture()
def gateway_client(monkeypatch):
    """TestClient for the Gateway app with offline fake embeddings injected."""
    fake = _FakeEmbeddings()
    monkeypatch.setattr(hk_dense, "get_embeddings", lambda: fake)

    from fastapi.testclient import TestClient

    from gateway.app.main import app as _gateway_app

    return TestClient(_gateway_app)


# ---------------------------------------------------------------------------
# AC1: POST /hybrid/index returns 200 with sections_indexed shape
# ---------------------------------------------------------------------------


def test_hybrid_index_returns_200_with_sections_count(gateway_client):
    """POST /hybrid/index returns 200 and {"sections_indexed": N} shape (AC1)."""
    resp = gateway_client.post("/hybrid/index")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "sections_indexed" in data, f"Response must have sections_indexed: {data}"
    assert isinstance(data["sections_indexed"], int), "sections_indexed must be int"
    assert data["sections_indexed"] >= 0, "sections_indexed must be non-negative"


def test_hybrid_index_persists_dense_index_dir(gateway_client):
    """POST /hybrid/index persists the FAISS index to DENSE_INDEX_DIR (AC1)."""
    gateway_client.post("/hybrid/index")
    assert hk_dense.DENSE_INDEX_DIR.exists(), (
        "DENSE_INDEX_DIR must exist after POST /hybrid/index (ADR-0022 AC1)"
    )


# ---------------------------------------------------------------------------
# AC2: after rebuild, sections are retrievable via the dense arm
# ---------------------------------------------------------------------------


def test_hybrid_index_makes_sections_dense_searchable(gateway_client, monkeypatch):
    """After POST /hybrid/index, synthetic sections are searchable via dense arm (AC2).

    Monkeypatches filtered_wiki_sections to a known 1-section corpus so we can
    assert the section is returned by dense_index.search() without touching the
    production wiki directory.
    """
    from markdown_kb.app.indexer import Section

    test_section = Section(
        id="dense-test#dense-test",
        file="dense-test",
        heading="Dense Test",
        heading_path=["Dense Test"],
        content="This is the dense arm test section content about shipping.",
        tokens=[],
        metadata={},
    )
    monkeypatch.setattr(hk_dense, "filtered_wiki_sections", lambda: [test_section])

    resp = gateway_client.post("/hybrid/index")
    assert resp.status_code == 200
    assert resp.json()["sections_indexed"] == 1

    # The section is now retrievable via the dense arm
    results = hk_dense.search("shipping", k=1)
    assert len(results) == 1, "Rebuilt dense index must return the synthetic section"
    assert results[0].id == "dense-test#dense-test", (
        "Dense search must return the section indexed by POST /hybrid/index"
    )


# ---------------------------------------------------------------------------
# AC3: /hybrid/index is wired into budget + middleware
# ---------------------------------------------------------------------------


def test_hybrid_index_in_admin_paths():
    """/hybrid/index is in ADMIN_PATHS — gated by admin semaphore + kill-switch (AC3)."""
    from gateway.app.middleware import ADMIN_PATHS

    assert "/hybrid/index" in ADMIN_PATHS, (
        "/hybrid/index must be in ADMIN_PATHS (ADR-0022 / issue #348 AC3)"
    )


def test_hybrid_index_cost_estimate_approx_fifty_cents():
    """/hybrid/index has a ~$0.50 cost estimate in _COST_ESTIMATES (AC3)."""
    from gateway.app.budget import estimate_cost

    cost = estimate_cost("/hybrid/index")
    assert cost == pytest.approx(0.50), (
        f"/hybrid/index estimate must be ~$0.50; got {cost} (ADR-0022 / issue #348 AC3)"
    )


def test_hybrid_index_blocked_by_admin_token_gate(monkeypatch):
    """When KB_ADMIN_TOKEN is set, POST /hybrid/index without Bearer → 401 (AC3).

    The token is read by ProdMiddleware at request time (os.getenv), so
    monkeypatch.setenv works without reloading the app.
    """
    from fastapi.testclient import TestClient

    fake = _FakeEmbeddings()
    monkeypatch.setattr(hk_dense, "get_embeddings", lambda: fake)
    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post("/hybrid/index")
    assert resp.status_code == 401, (
        f"Expected 401 when KB_ADMIN_TOKEN is set and no Bearer; got {resp.status_code}"
    )


def test_hybrid_index_admitted_with_correct_bearer(monkeypatch):
    """When KB_ADMIN_TOKEN is set, correct Bearer token admits the request (AC3)."""
    from fastapi.testclient import TestClient

    fake = _FakeEmbeddings()
    monkeypatch.setattr(hk_dense, "get_embeddings", lambda: fake)
    monkeypatch.setenv("KB_ADMIN_TOKEN", "test-secret-token")

    from gateway.app.main import app as _gateway_app

    client = TestClient(_gateway_app)
    resp = client.post(
        "/hybrid/index",
        headers={"Authorization": "Bearer test-secret-token"},
    )
    assert resp.status_code == 200, (
        f"Expected 200 with correct Bearer; got {resp.status_code}: {resp.text}"
    )
