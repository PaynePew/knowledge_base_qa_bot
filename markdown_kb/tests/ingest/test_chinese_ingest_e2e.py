"""Hermetic end-to-end integration test for the Chinese /ingest → /index → /chat pipeline.

AC (issue #166 AC4):
  - Chinese Source → POST /ingest (mocked LLM) writes a Chinese Wiki Page.
  - POST /index builds BM25 index over that page.
  - POST /chat with a Chinese query returns sources that include a CJK-bearing Citation.
  - English Source still produces an English Wiki Page (no regression).

All LLM calls are mocked via the lazy-singleton getter pattern (ADR-0005 /
CODING_STANDARD §6.3). No OPENAI_API_KEY required. No live LLM calls.

Mocking strategy:
  - ingest LLM: mocked via monkeypatch on templates_module._ingest_llm + get_ingest_llm
    (same pattern as test_ingest_integration.py). The fake is schema-aware:
    classifier always returns "concept"; synthesis returns a CJK body for the
    Chinese source and an English body for the English source.
  - chat LLM: mocked via monkeypatch on retrieval_module.get_llm / get_retry_llm.
  - grounding verifier: autouse conftest fixture already mocks app.ingest.verify to
    claim_supported, covering the ingest path. The chat path's verify is patched
    inline here to claim_supported.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.templates as templates_module

# ---------------------------------------------------------------------------
# Fake LLM helpers
# ---------------------------------------------------------------------------

CHINESE_BODY = (
    "退款申請須在收到商品後14天內提出，且商品須保持原狀。[Source: 退款政策.md#退款申請窗口]"
)
ENGLISH_BODY = "Customers may return items within 30 days of purchase."


class _FakeSynthesisOutput:
    """Mirrors _PageSynthesisOutput without importing the private class."""

    def __init__(self, body: str, open_questions: list | None = None):
        self.body = body
        self.open_questions = open_questions or []


class _FakeClassifierOutput:
    """Mirrors _ClassifierOutput without importing the private class."""

    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_language_aware_fake_llm(
    chinese_basename: str,
    chinese_body: str = CHINESE_BODY,
    english_body: str = ENGLISH_BODY,
) -> MagicMock:
    """Return a fake ingest LLM that:
      - classifies everything as "concept"
      - returns a Chinese body when the human message mentions the Chinese filename
      - returns an English body otherwise

    The schema-dispatch follows test_ingest_integration.py's _make_schema_aware_fake_llm
    pattern: inspect the schema class to distinguish classifier from synthesis calls.
    """
    from app.templates import _ClassifierOutput  # accessible in tests

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            chain.invoke.return_value = _FakeClassifierOutput("concept")
        else:
            # Synthesis path: choose body based on content of the human message
            def _invoke(messages):
                human_text = ""
                for m in messages:
                    # messages are HumanMessage objects; access .content
                    if hasattr(m, "content") and m.content:
                        human_text += str(m.content)
                body = chinese_body if chinese_basename in human_text else english_body
                return _FakeSynthesisOutput(body)

            chain.invoke.side_effect = _invoke
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

# The committed Chinese demo Source lives at docs/demo-zh/退款政策.md.
# We resolve it relative to this test file to stay machine-agnostic.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEMO_ZH_DIR = _REPO_ROOT / "docs" / "demo-zh"
_CHINESE_SOURCE_NAME = "退款政策.md"

# Minimal English source (inline, no dependency on frozen fixture files)
_ENGLISH_SOURCE_CONTENT = """\
## Return Policy

Customers may return items within 30 days of purchase provided they are unused.
"""
_ENGLISH_SOURCE_NAME = "return_guide.md"


@pytest.fixture()
def chinese_e2e_client(tmp_path, monkeypatch):
    """Set up the full stack for the Chinese pipeline e2e test.

    1. Creates a docs dir with the committed Chinese demo Source + an English source.
    2. Patches the ingest LLM (language-aware fake), ingest DOCS_DIR, and WIKI_DIR.
    3. Returns (client, tmp_wiki_path).
    """
    # Create a docs dir with both sources
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Copy the committed Chinese demo Source into the test's docs dir
    chinese_src = _DEMO_ZH_DIR / _CHINESE_SOURCE_NAME
    assert chinese_src.exists(), (
        f"Chinese demo Source not found at {chinese_src}. "
        "Did the docs/demo-zh/ commit land on this branch?"
    )
    dest_zh = docs_dir / _CHINESE_SOURCE_NAME
    dest_zh.write_bytes(chinese_src.read_bytes())

    # Write the inline English source
    (docs_dir / _ENGLISH_SOURCE_NAME).write_text(_ENGLISH_SOURCE_CONTENT, encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    # Patch ingest LLM
    fake_llm = _make_language_aware_fake_llm(
        chinese_basename=_CHINESE_SOURCE_NAME,
        chinese_body=CHINESE_BODY,
        english_body=ENGLISH_BODY,
    )
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)

    import app.indexer as indexer_module
    import app.ingest as ingest_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)
    # SOURCE_DIRS is pre-baked at module load time as [WIKI_DIR/entities, WIKI_DIR/concepts,
    # WIKI_DIR/qa]. Patching WIKI_DIR alone does not update SOURCE_DIRS. Patch SOURCE_DIRS
    # explicitly so build_index() (called by /index with no args) scans the tmp wiki dir.
    monkeypatch.setattr(
        indexer_module,
        "SOURCE_DIRS",
        [wiki_dir / "entities", wiki_dir / "concepts", wiki_dir / "qa"],
    )
    monkeypatch.setattr(ingest_module, "DOCS_DIR", docs_dir)

    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app), tmp_path, docs_dir, wiki_dir


# ---------------------------------------------------------------------------
# AC4: Chinese Source → /ingest → /index → /chat carries CJK to Citation
# ---------------------------------------------------------------------------


def test_chinese_ingest_index_chat_cjk_citation(chinese_e2e_client, monkeypatch):
    """Full Chinese pipeline: ingest → index → chat returns CJK-bearing Citation.

    Steps:
      1. POST /ingest the Chinese source — mocked LLM writes a Chinese wiki page.
      2. POST /index — builds BM25 index over the wiki page.
      3. Mock the chat LLM to return a Chinese answer.
      4. POST /chat with a Chinese question.
      5. Assert: at least one source in the response contains CJK code-points
         (the Citation id includes the CJK heading slug).
    """
    from app.grounding import GroundingOutcome

    client, tmp_path, docs_dir, wiki_dir = chinese_e2e_client

    # Step 1: ingest the Chinese source
    resp_ingest = client.post("/ingest", json={"source": _CHINESE_SOURCE_NAME})
    assert resp_ingest.status_code == 200, (
        f"Expected 200 from /ingest, got {resp_ingest.status_code}: {resp_ingest.text}"
    )
    ingest_body = resp_ingest.json()
    assert ingest_body["failed_sources"] == [], (
        f"Unexpected ingest failures: {ingest_body['failed_sources']}"
    )
    assert len(ingest_body["results"]) >= 1, "Expected at least one ingest result"

    # Verify at least one written page path contains CJK (slug is Unicode)
    all_pages = [p for r in ingest_body["results"] for p in r["pages_written"]]
    assert all_pages, "Expected at least one page written"
    has_cjk_page = any(any("一" <= ch <= "鿿" for ch in page) for page in all_pages)
    assert has_cjk_page, f"Expected at least one CJK-slug wiki page written, got: {all_pages}"

    # Verify a Chinese wiki page file actually exists on disk
    concepts_dir = wiki_dir / "concepts"
    assert concepts_dir.exists(), "wiki/concepts/ dir should exist after ingest"
    written_files = list(concepts_dir.glob("*.md"))
    assert written_files, "Expected at least one .md file in wiki/concepts/"
    has_cjk_file = any(any("一" <= ch <= "鿿" for ch in f.name) for f in written_files)
    assert has_cjk_file, (
        f"Expected a CJK-named wiki page on disk, found: {[f.name for f in written_files]}"
    )

    # Step 2: build the BM25 index over the wiki pages
    resp_index = client.post("/index")
    assert resp_index.status_code == 200, (
        f"Expected 200 from /index, got {resp_index.status_code}: {resp_index.text}"
    )
    index_body = resp_index.json()
    assert index_body["sections_indexed"] >= 1, (
        f"Expected at least 1 section indexed, got: {index_body}"
    )

    # Step 3+4: mock chat LLM and POST /chat with a Chinese question
    import app.retrieval as ret_module

    CHINESE_ANSWER = "退款須在14天內申請，且商品須保持原狀。"

    class _FakeChatLLM:
        def invoke(self, messages):
            from tests.conftest import FakeLLMResponse

            return FakeLLMResponse(content=CHINESE_ANSWER)

    fake_chat_llm = _FakeChatLLM()
    monkeypatch.setattr(ret_module, "get_llm", lambda: fake_chat_llm)
    monkeypatch.setattr(ret_module, "get_retry_llm", lambda: fake_chat_llm)

    supported_outcome = GroundingOutcome(passed=True, reason="claim_supported", result=None)
    monkeypatch.setattr(
        ret_module.grounding_module,
        "verify",
        lambda *_a, **_kw: supported_outcome,
    )

    resp_chat = client.post("/chat", json={"query": "退款需要多少時間？"})
    assert resp_chat.status_code == 200, (
        f"Expected 200 from /chat, got {resp_chat.status_code}: {resp_chat.text}"
    )
    chat_body = resp_chat.json()

    # Step 5: assert sources contain CJK-bearing Citation
    sources = chat_body["sources"]
    assert sources, "Expected at least one source in /chat response"
    has_cjk_source = any(any("一" <= ch <= "鿿" for ch in s["source"]) for s in sources)
    assert has_cjk_source, (
        f"Expected CJK-bearing Citation in sources, got: {[s['source'] for s in sources]}"
    )


# ---------------------------------------------------------------------------
# AC4 regression: English Source still produces an English wiki page
# ---------------------------------------------------------------------------


def test_english_source_ingest_produces_english_page(chinese_e2e_client):
    """Regression: English source ingest still writes an English-slug wiki page."""
    client, tmp_path, docs_dir, wiki_dir = chinese_e2e_client

    resp = client.post("/ingest", json={"source": _ENGLISH_SOURCE_NAME})
    assert resp.status_code == 200, (
        f"Expected 200 from /ingest for English source, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["failed_sources"] == [], (
        f"Unexpected ingest failures for English source: {body['failed_sources']}"
    )
    all_pages = [p for r in body["results"] for p in r["pages_written"]]
    assert all_pages, "Expected at least one page written for English source"

    # English pages use ASCII slugs only
    for page in all_pages:
        assert all(ord(ch) < 128 for ch in page), (
            f"English source page should have ASCII-only path, got: {page!r}"
        )
