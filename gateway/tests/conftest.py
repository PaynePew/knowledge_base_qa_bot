"""Shared fixtures for the gateway test suite.

Provides:
  - pytest_collection_modifyitems: skip @pytest.mark.live unless -m live

Also loads .env so live tests pick up OPENAI_API_KEY, mirroring
how gateway.app.main does it via load_dotenv.
"""

from __future__ import annotations

import pytest
from dotenv import find_dotenv, load_dotenv

# Load .env BEFORE imports — mirrors gateway.app.main.
load_dotenv(find_dotenv(usecwd=True))


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.live tests unless explicitly selected with -m live."""
    marker_expr = config.option.markexpr if hasattr(config.option, "markexpr") else ""
    if "live" in marker_expr:
        return
    skip_live = pytest.mark.skip(reason="live test — run with: pytest -m live")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _neutralize_rag_distance_gate(monkeypatch):
    """Disable the calibrated RAG distance gate for the gateway suite.

    The RAG distance gate ships ON by default (calibrated ceiling 1.1 against REAL
    text-embedding-3-small distances — eval/rag_distance). Gateway RAG-stream tests
    build the index with offline fake embeddings whose hash-derived distances are not
    comparable to that ceiling, so an enabled gate would refuse every fake query and
    mask the grounded-answer / SSE-ordering assertions. A large permissive ceiling
    neutralises it (the wiki BM25 path is unaffected). The gate's own behaviour is
    covered in vector_rag/tests/test_rag_distance_gate.py.
    """
    monkeypatch.setenv("KB_RAG_DISTANCE_THRESHOLD", "1000.0")
