"""Cold-Gateway lifespan regression test (issue #398).

Starlette's ``app.mount()`` does NOT run a mounted sub-app's lifespan — only
the top-level ASGI app receives ``lifespan.startup`` / ``lifespan.shutdown``.
Both ``markdown_kb`` and ``vector_rag`` rehydrate their index from disk in
their own lifespan, so a cold Gateway process (mount-only, no fix) served
every route — most dangerously ``POST /wiki/lint?include_c5=true`` — against
an empty in-memory BM25 index with no error, only ``llm_calls=0`` /
``c5_pairs_capped=0`` in the summary (a silent zero-audit).

AC coverage (issue #398):
  - AC1: a cold gateway TestClient (``with TestClient(app) as client``, no
    prior request) has a populated BM25 index (``indexer.sections`` non-empty)
    as soon as the app is up — no ``POST /wiki/index`` or ``/chat`` call needed.
  - AC2: cold-gateway ``POST /wiki/lint?include_c5=true`` (C5 judge stubbed via
    the existing test seam — ``monkeypatch get_lint_llm``, see
    ``markdown_kb/tests/lint/test_c5_unit.py``) produces F3 BM25 candidate
    pairs > 0 on a fixture corpus, proven via a regression pin: the fixture
    pages deliberately do NOT share ``frontmatter.sources`` (no F1 pair is
    possible), so any candidate pair judged can only have come from the F3
    BM25 self-query leg — which needs a populated ``indexer.sections``.
  - AC3: the same fix rehydrates vector_rag's persisted FAISS index on a cold
    Gateway too (audited: ``/rag/chat`` already had a lazy-load fallback from
    issue #133, so the Gateway lifespan is belt-and-suspenders there, not a
    new behaviour) — pinned by
    ``test_cold_gateway_populates_rag_vectorstore_on_startup``.
  - AC5: this file + its docstring is the regression pin for "``app.mount``
    does NOT run sub-app lifespans".

All tests are hermetic: fake wiki pages under ``tmp_path`` / fake embeddings,
C5 LLM stubbed (no OPENAI_API_KEY / network calls).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import markdown_kb.app.indexer as mk_indexer
import markdown_kb.app.lint as mk_lint
import markdown_kb.app.logger as mk_logger
import pytest
import vector_rag.app.indexer as vr_indexer
import vector_rag.app.logger as vr_logger
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

# vector_rag's build_index() indexes real docs/ (unlike markdown_kb's wiki-page
# BM25 index) — mirrors REAL_DOCS in test_chat_stream_rag_lazy.py.
REAL_DOCS = Path(__file__).resolve().parents[2] / "docs"

# ---------------------------------------------------------------------------
# Fixture wiki corpus: two concept pages with DIFFERENT frontmatter.sources
# (so F1 — the shared-source leg — cannot produce a candidate pair) but
# near-identical bodies (so the F3 BM25 self-query leg scores them above the
# KB_LINT_BM25_THRESHOLD default). Any C5 candidate pair observed here can
# only have come from F3, which is exactly the leg the issue's docstring
# names as going silently dark on a cold gateway (lint.py ~1501).
# ---------------------------------------------------------------------------

_SHARED_BODY = (
    "Approved refund requests are processed within five business days after "
    "the returned item arrives at our fulfillment center in Springfield."
)

_PAGE_TEMPLATE = """---
id: {slug}
type: concept
created: "2026-07-01T00:00:00Z"
updated: "2026-07-01T00:00:00Z"
sources:
  - {source}
status: live
open_questions: []
---

# {title}

{body}
"""


def _write_fixture_wiki(wiki_dir: Path) -> tuple[Path, Path]:
    """Write two F1-disjoint / F3-overlapping concept pages under wiki_dir/concepts/."""
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    entities_dir = wiki_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)

    alpha = concepts_dir / "alpha-refund-window.md"
    alpha.write_text(
        _PAGE_TEMPLATE.format(
            slug="alpha-refund-window",
            source="policy_alpha.md#window",
            title="Alpha Refund Window",
            body=_SHARED_BODY,
        ),
        encoding="utf-8",
    )
    beta = concepts_dir / "beta-refund-window.md"
    beta.write_text(
        _PAGE_TEMPLATE.format(
            slug="beta-refund-window",
            source="policy_beta.md#window",
            title="Beta Refund Window",
            body=_SHARED_BODY,
        ),
        encoding="utf-8",
    )
    return entities_dir, concepts_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def persisted_wiki_index_fresh_process(tmp_path, monkeypatch):
    """Build + persist a wiki BM25 index to disk, then simulate a fresh process.

    Mirrors the established lazy-load fixture pattern (issue #133/#148): build
    the index once (as a prior ``/wiki/index`` call would), assert it landed on
    disk, then clear the in-memory ``sections`` list so the only way the
    Gateway can serve a populated index is via its own startup logic.
    """
    wiki_dir = tmp_path / "wiki"
    entities_dir, concepts_dir = _write_fixture_wiki(wiki_dir)

    monkeypatch.setattr(mk_logger, "LOG_PATH", wiki_dir / "log.md")
    monkeypatch.setattr(mk_indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(mk_indexer, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(mk_indexer, "SOURCE_DIRS", [entities_dir, concepts_dir])

    # The Gateway lifespan under test enters BOTH sub-apps' lifespans, so the
    # vector_rag leg must be neutralized too or `load_vector_index()` reaches
    # the real committed .kb/faiss_index and the live embeddings seam
    # (CODING_STANDARD §6.3, incident #332): point FAISS at an empty tmp dir
    # (load no-ops) and fake the embeddings getter (belt and suspenders).
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")

    # lint.py rebinds its own WIKI_DIR/DOCS_DIR/LOG_PATH names (see
    # markdown_kb/app/_paths.py) — patch those too so POST /wiki/lint's
    # defaults resolve to the same tmp fixture, not the real repo wiki/.
    monkeypatch.setattr(mk_lint, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(mk_lint, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(mk_lint, "LOG_PATH", wiki_dir / "log.md")
    (tmp_path / "docs").mkdir(exist_ok=True)

    mk_indexer.build_index()  # default docs_dir -> production SOURCE_DIRS path (slug ids)
    assert mk_indexer.INDEX_PATH.exists(), "index must be persisted after build_index()"

    # Simulate a fresh Gateway process: no /wiki/index or /chat has run yet.
    mk_indexer.sections.clear()
    assert not mk_indexer.sections

    yield

    mk_indexer.sections.clear()


@pytest.fixture()
def stub_c5_llm(monkeypatch):
    """Stub the C5 judge via the existing test seam (mirrors test_c5_unit.py).

    Returns a 'tension' finding for every judged pair — non-'none' so it
    survives the severity filter in ``_check_c5_page_pair`` and is countable.
    """
    from markdown_kb.app.schemas import PagePairFinding

    def mock_invoke(messages):
        return PagePairFinding(
            severity="tension",
            page_a="alpha-refund-window",
            page_b="beta-refund-window",
            page_a_claim="five business days",
            page_b_claim="five business days",
            summary="Stubbed finding for cold-start F3 regression test",
            suggested_action="n/a",
        )

    mock_chain = MagicMock()
    mock_chain.invoke = mock_invoke
    monkeypatch.setattr(
        mk_lint,
        "get_lint_llm",
        lambda: MagicMock(with_structured_output=lambda schema: mock_chain),
    )


# ---------------------------------------------------------------------------
# AC1 — cold gateway TestClient has a populated index as soon as the app is up
# ---------------------------------------------------------------------------


def test_cold_gateway_populates_wiki_sections_on_startup(persisted_wiki_index_fresh_process):
    """Entering the Gateway's TestClient context populates indexer.sections.

    No POST /wiki/index and no POST /chat is issued — only ``with
    TestClient(app) as client`` (which drives the ASGI lifespan protocol).
    Before issue #398's fix, ``app.mount()`` swallowed the sub-apps' lifespans
    and this would stay empty.
    """
    from gateway.app.main import app as gateway_app

    assert not mk_indexer.sections, "precondition: fresh process has an empty index"

    with TestClient(gateway_app):
        assert mk_indexer.sections, (
            "indexer.sections must be populated by the Gateway lifespan alone — "
            "no /wiki/index or /chat call was made"
        )


# ---------------------------------------------------------------------------
# AC2 — cold-gateway POST /wiki/lint?include_c5=true produces F3 pairs > 0
# ---------------------------------------------------------------------------


def test_cold_gateway_lint_c5_produces_f3_candidate_pairs(
    persisted_wiki_index_fresh_process, stub_c5_llm
):
    """Cold-gateway POST /wiki/lint?include_c5=true finds F3 candidate pairs.

    The fixture pages deliberately share NO frontmatter.sources, so F1 cannot
    produce a candidate pair here — any pair judged came from the F3 BM25
    self-query leg, which reads ``indexer.sections`` (issue #398's F3
    docstring: "if the index is empty ... F3 produces no pairs — safe
    degradation"). A judged/capped count of 0 here would mean the regression
    is back.
    """
    from gateway.app.main import app as gateway_app

    with TestClient(gateway_app) as client:
        resp = client.post("/wiki/lint", params={"include_c5": "true"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    summary = resp.json()["summary"]
    total_c5_candidates = summary["llm_calls"] + summary["c5_pairs_capped"]
    assert total_c5_candidates > 0, (
        f"Expected >0 F3 candidate pairs on a cold gateway, got summary={summary}"
    )


# ---------------------------------------------------------------------------
# AC3 — vector_rag cold-start under the Gateway (audit evidence)
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


@pytest.fixture()
def persisted_rag_index_fresh_process(tmp_path, monkeypatch):
    """Persist a FAISS index to disk, then simulate a fresh process (vectorstore=None).

    Mirrors ``test_chat_stream_rag_lazy.py``'s proven fixture — the FAISS side
    of the same fresh-process simulation used for the wiki index above.
    """
    monkeypatch.setattr(vr_indexer, "FAISS_INDEX_DIR", tmp_path / ".kb" / "faiss_index")
    monkeypatch.setattr(vr_logger, "LOG_PATH", tmp_path / "vector_rag" / "log.md")
    monkeypatch.setattr(vr_indexer, "get_embeddings", lambda: _FakeEmbeddings())

    vr_indexer.build_index(REAL_DOCS)
    assert vr_indexer.FAISS_INDEX_DIR.exists(), "FAISS index must be persisted after build"

    # Simulate a fresh Gateway process: no /rag/index or /chat has run yet.
    vr_indexer.vectorstore = None

    yield

    vr_indexer.vectorstore = None
    vr_indexer.files_indexed = 0
    vr_indexer.chunks_indexed = 0


def test_cold_gateway_populates_rag_vectorstore_on_startup(persisted_rag_index_fresh_process):
    """The same Gateway lifespan also rehydrates vector_rag's FAISS index (AC3).

    Audit finding: ``/rag/chat`` already lazy-loads on a cache miss (issue
    #133), so this is belt-and-suspenders, not a new behaviour change — but it
    means C5-style "cold process silently serves an empty index" risk on the
    RAG side is closed the same way, not just patched around per-route.
    """
    from gateway.app.main import app as gateway_app

    assert vr_indexer.vectorstore is None, "precondition: fresh process has no vectorstore"

    with TestClient(gateway_app):
        assert vr_indexer.vectorstore is not None, (
            "vector_rag's FAISS index must be populated by the Gateway lifespan alone — "
            "no /rag/index or /chat call was made"
        )
