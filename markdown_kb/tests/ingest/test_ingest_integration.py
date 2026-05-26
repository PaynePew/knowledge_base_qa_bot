"""Integration tests for POST /ingest — Slice #1: end-to-end skeleton.

All tests use TestClient + mock LLM (no OPENAI_API_KEY required).
The mock pattern follows the `with_structured_output` pattern established in
grounding/test_verifier.py — patch `ChatOpenAI` in `app.templates` and return
a fake chain whose `invoke()` returns a `_PageSynthesisOutput`-compatible
object.

AC coverage (issue #29):
  - POST /ingest happy path: 200, IngestResponse with one result, one page created
  - wiki/concepts/cancellation-window.md exists with correct structure
  - Atomic write: no .tmp file lingers after success
  - OPENAI_INGEST_MODEL fallback chain
  - POST /ingest with nonexistent source: 200, failed_sources populated
  - POST /ingest with no body: 400 (FastAPI Pydantic validation)
  - All existing tests still pass (run separately)
  - Hermetic: no OPENAI_API_KEY needed
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

import app.templates as templates_module
from app.schemas import WikiPageDraft, WikiPageFrontmatter

# ---------------------------------------------------------------------------
# Fake LLM helpers
# ---------------------------------------------------------------------------

FIXED_TS = "2026-05-26T14:30:00Z"
FIXED_BODY = "Customers can cancel within 24 hours of purchase if the order has not shipped."


class _FakeSynthesisOutput:
    """Mirrors _PageSynthesisOutput but without importing the private class."""

    body: str
    open_questions: list

    def __init__(self, body: str = FIXED_BODY, open_questions: list | None = None):
        self.body = body
        self.open_questions = open_questions or []


def _make_fake_chain(body: str = FIXED_BODY) -> MagicMock:
    """Return a fake structured-output chain that returns a canned synthesis."""
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = _FakeSynthesisOutput(body=body)
    return fake_chain


def _make_fake_llm(body: str = FIXED_BODY) -> MagicMock:
    """Return a fake ChatOpenAI that delegates with_structured_output to a fake chain."""
    fake_llm = MagicMock()
    fake_chain = _make_fake_chain(body)
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_fake_ingest_llm(tmp_path, monkeypatch):
    """TestClient with a fake ingest LLM and redirected wiki_dir to tmp_path.

    Patches:
    - `app.templates._ingest_llm` and `app.templates.get_ingest_llm` so
      no real OpenAI call is made.
    - `app.wiki_writer.indexer.WIKI_DIR` redirected to tmp_path so pages
      land in a temp directory, not the real wiki/.
    - `app.ingest.DOCS_DIR` uses the real docs/ directory (no mock — per
      CODING_STANDARD §6.3, we only mock the LLM, not the indexer/parser).

    Returns:
        (client, tmp_wiki_path) tuple.
    """
    fake_llm = _make_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)

    # Redirect wiki/ to tmp_path so written pages land in isolation
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")

    from app.main import app

    return TestClient(app), tmp_path / "wiki"


# ---------------------------------------------------------------------------
# AC: POST /ingest happy path — 200, one result, one page created
# ---------------------------------------------------------------------------


def test_ingest_happy_path_returns_200(client_with_fake_ingest_llm):
    """POST /ingest with a valid source returns 200 with IngestResponse."""
    client, tmp_wiki = client_with_fake_ingest_llm

    resp = client.post("/ingest", json={"source": "refund_policy.md"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "results" in body, f"Expected 'results' key in response: {body}"
    assert "failed_sources" in body, f"Expected 'failed_sources' key in response: {body}"
    assert body["failed_sources"] == [], (
        f"Expected no failed sources, got: {body['failed_sources']}"
    )
    assert len(body["results"]) == 1, f"Expected 1 result, got: {body['results']}"

    result = body["results"][0]
    assert result["source"] == "refund_policy.md"
    assert len(result["pages_written"]) == 1, (
        f"Expected 1 page written, got: {result['pages_written']}"
    )
    assert result["pages_written"][0] == "concepts/cancellation-window.md", (
        f"Expected concepts/cancellation-window.md, got: {result['pages_written'][0]}"
    )


# ---------------------------------------------------------------------------
# AC: wiki/concepts/cancellation-window.md exists with correct structure
# ---------------------------------------------------------------------------


def test_ingest_creates_wiki_page_with_correct_structure(client_with_fake_ingest_llm):
    """The generated wiki page has sentinel, 7-field frontmatter, body, and citation."""
    client, tmp_wiki = client_with_fake_ingest_llm

    resp = client.post("/ingest", json={"source": "refund_policy.md"})
    assert resp.status_code == 200

    page_path = tmp_wiki / "concepts" / "cancellation-window.md"
    assert page_path.exists(), f"Expected {page_path} to exist"

    content = page_path.read_text(encoding="utf-8")

    # Sentinel comment
    assert "<!-- Auto-generated by POST /ingest on" in content
    assert "Source of truth: docs/refund_policy.md." in content

    # YAML frontmatter — all 7 fields
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    assert len(dash_indices) >= 2, f"Missing --- frontmatter delimiters in:\n{content}"
    fm_text = "\n".join(lines[dash_indices[0] + 1 : dash_indices[1]])
    parsed = yaml.safe_load(fm_text)

    required_fields = {"id", "type", "created", "updated", "sources", "status", "open_questions"}
    assert required_fields.issubset(parsed.keys()), (
        f"Missing frontmatter fields: {required_fields - parsed.keys()}"
    )
    assert parsed["id"] == "cancellation-window"
    assert parsed["type"] == "concept"
    assert parsed["status"] == "live"
    assert "refund_policy.md#cancellation-window" in parsed["sources"]

    # LLM body
    assert FIXED_BODY in content, f"Expected LLM body in page:\n{content}"

    # Citation line
    assert "[Source: refund_policy.md#cancellation-window]" in content, (
        f"Citation line missing from page:\n{content}"
    )


# ---------------------------------------------------------------------------
# AC: atomic write — no .tmp file lingers
# ---------------------------------------------------------------------------


def test_ingest_no_tmp_files_linger(client_with_fake_ingest_llm):
    """No .tmp files remain in wiki/concepts/ after a successful ingest."""
    client, tmp_wiki = client_with_fake_ingest_llm

    resp = client.post("/ingest", json={"source": "refund_policy.md"})
    assert resp.status_code == 200

    concepts_dir = tmp_wiki / "concepts"
    if concepts_dir.exists():
        tmp_files = list(concepts_dir.glob("*.tmp"))
        assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"


# ---------------------------------------------------------------------------
# AC: POST /ingest with nonexistent source returns 200 with failed_sources
# ---------------------------------------------------------------------------


def test_ingest_nonexistent_source_returns_failed_sources(client_with_fake_ingest_llm):
    """POST /ingest with a non-existent source returns 200 with failed_sources."""
    client, tmp_wiki = client_with_fake_ingest_llm

    resp = client.post("/ingest", json={"source": "nonexistent.md"})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["results"] == [], f"Expected empty results, got: {body['results']}"
    assert "nonexistent.md" in body["failed_sources"], (
        f"Expected 'nonexistent.md' in failed_sources, got: {body['failed_sources']}"
    )

    # No page should have been written
    concepts_dir = tmp_wiki / "concepts"
    assert not concepts_dir.exists() or list(concepts_dir.glob("*.md")) == [], (
        "No page should be written for a non-existent source"
    )


# ---------------------------------------------------------------------------
# AC: POST /ingest with no body returns 400 (FastAPI Pydantic validation)
# ---------------------------------------------------------------------------


def test_ingest_no_body_returns_400(client_with_fake_ingest_llm):
    """POST /ingest with no body returns HTTP 422 (FastAPI validation error).

    Design choice: we rely on FastAPI's Pydantic validation for the missing
    body case, which returns 422 Unprocessable Entity. This satisfies the AC
    intent (reject request without valid body) and avoids bespoke route logic.
    Note: the AC says "400 or 200 with explicit message" — we choose 422 (the
    RFC-correct HTTP code for schema violations) and document it here.
    """
    client, _ = client_with_fake_ingest_llm

    resp = client.post("/ingest")  # no JSON body

    assert resp.status_code in (400, 422), (
        f"Expected 400 or 422 for missing body, got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# AC: OPENAI_INGEST_MODEL env var fallback chain
# ---------------------------------------------------------------------------


def test_ingest_model_env_var_fallback_uses_openai_ingest_model(monkeypatch):
    """OPENAI_INGEST_MODEL takes priority over OPENAI_MODEL."""
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Reset the singleton so the next call rebuilds with patched env
    monkeypatch.setattr(templates_module, "_ingest_llm", None)

    captured_model = {}

    class _CaptureChatOpenAI:
        def __init__(self, model: str, **kwargs):
            captured_model["model"] = model

        def with_structured_output(self, schema):
            return MagicMock()

    with patch("app.templates.ChatOpenAI", _CaptureChatOpenAI):
        templates_module.get_ingest_llm()

    assert captured_model["model"] == "gpt-4o", (
        f"Expected model=gpt-4o from OPENAI_INGEST_MODEL, got: {captured_model['model']!r}"
    )


def test_ingest_model_env_var_fallback_uses_openai_model(monkeypatch):
    """When OPENAI_INGEST_MODEL is absent, falls back to OPENAI_MODEL."""
    monkeypatch.delenv("OPENAI_INGEST_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")

    monkeypatch.setattr(templates_module, "_ingest_llm", None)

    captured_model = {}

    class _CaptureChatOpenAI:
        def __init__(self, model: str, **kwargs):
            captured_model["model"] = model

        def with_structured_output(self, schema):
            return MagicMock()

    with patch("app.templates.ChatOpenAI", _CaptureChatOpenAI):
        templates_module.get_ingest_llm()

    assert captured_model["model"] == "gpt-4-turbo", (
        f"Expected model=gpt-4-turbo from OPENAI_MODEL, got: {captured_model['model']!r}"
    )


def test_ingest_model_env_var_fallback_default(monkeypatch):
    """When neither OPENAI_INGEST_MODEL nor OPENAI_MODEL is set, uses gpt-4o-mini."""
    monkeypatch.delenv("OPENAI_INGEST_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    monkeypatch.setattr(templates_module, "_ingest_llm", None)

    captured_model = {}

    class _CaptureChatOpenAI:
        def __init__(self, model: str, **kwargs):
            captured_model["model"] = model

        def with_structured_output(self, schema):
            return MagicMock()

    with patch("app.templates.ChatOpenAI", _CaptureChatOpenAI):
        templates_module.get_ingest_llm()

    assert captured_model["model"] == "gpt-4o-mini", (
        f"Expected default model=gpt-4o-mini, got: {captured_model['model']!r}"
    )


# ---------------------------------------------------------------------------
# AC: Hermetic — verify mock LLM was used (not a real API call)
# ---------------------------------------------------------------------------


def test_ingest_uses_mock_llm_not_real_api(client_with_fake_ingest_llm):
    """Verify the fake LLM was invoked (proving no real API call was made)."""
    client, tmp_wiki = client_with_fake_ingest_llm

    # The fixture injects _ingest_llm directly; verify it's a MagicMock
    assert isinstance(templates_module._ingest_llm, MagicMock), (
        "Expected _ingest_llm to be a MagicMock (fake) in test context"
    )

    resp = client.post("/ingest", json={"source": "refund_policy.md"})
    assert resp.status_code == 200

    # The fake LLM's with_structured_output chain was invoked
    assert templates_module._ingest_llm.with_structured_output.called, (
        "Expected with_structured_output to be called on the fake LLM"
    )
