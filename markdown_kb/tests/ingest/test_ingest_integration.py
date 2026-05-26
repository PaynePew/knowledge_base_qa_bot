"""Integration tests for POST /ingest — Slice #1 + Slice #2.

All tests use TestClient + mock LLM (no OPENAI_API_KEY required).
The mock pattern follows the `with_structured_output` pattern established in
grounding/test_verifier.py — patch `ChatOpenAI` in `app.templates` and return
a fake chain whose `invoke()` returns the appropriate canned output type
depending on the schema passed to `with_structured_output`.

AC coverage (issue #29 — Slice #1):
  - POST /ingest happy path: 200, IngestResponse with one result, one page created
  - wiki/concepts/cancellation-window.md exists with correct structure
  - Atomic write: no .tmp file lingers after success
  - OPENAI_INGEST_MODEL fallback chain
  - POST /ingest with nonexistent source: 200, failed_sources populated
  - All existing tests still pass (run separately)
  - Hermetic: no OPENAI_API_KEY needed

AC coverage (issue #30 — Slice #2):
  - POST /ingest with no body triggers batch mode (200, 3 results)
  - POST /ingest with body {source: X} still works (single-source filter)
  - Entity source produces one page in wiki/entities/<stem>.md
  - Slug collision: two sources producing same concept slug → overview + overview-2
  - Continue-on-error: APITimeoutError on one source → others still succeed
  - glob("**/*.md") picks up nested fixture
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


class _FakeClassifierOutput:
    """Mirrors _ClassifierOutput but without importing the private class."""

    type: str

    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_schema_aware_fake_llm(
    synthesis_body: str = FIXED_BODY,
    classifier_type: str = "concept",
) -> MagicMock:
    """Return a fake ChatOpenAI whose with_structured_output is schema-aware.

    When the classifier schema is passed, returns _FakeClassifierOutput.
    For any other schema (synthesis), returns _FakeSynthesisOutput.
    The distinction is made by inspecting the schema's field names.
    """
    from app.templates import _ClassifierOutput  # private but accessible in tests

    fake_llm = MagicMock()

    def _side_effect(schema):
        chain = MagicMock()
        if schema is _ClassifierOutput:
            chain.invoke.return_value = _FakeClassifierOutput(classifier_type)
        else:
            chain.invoke.return_value = _FakeSynthesisOutput(synthesis_body)
        return chain

    fake_llm.with_structured_output.side_effect = _side_effect
    return fake_llm


def _make_fake_chain(body: str = FIXED_BODY) -> MagicMock:
    """Return a fake structured-output chain that returns a canned synthesis."""
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = _FakeSynthesisOutput(body=body)
    return fake_chain


def _make_fake_llm(body: str = FIXED_BODY) -> MagicMock:
    """Return a simple fake ChatOpenAI (synthesis only, no classifier distinction).

    For tests that only need the synthesis path and never hit classify_source.
    """
    fake_llm = MagicMock()
    fake_chain = _make_fake_chain(body)
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_fake_ingest_llm(tmp_path, monkeypatch):
    """TestClient with a schema-aware fake ingest LLM and redirected wiki_dir.

    Patches:
    - `app.templates._ingest_llm` and `app.templates.get_ingest_llm` so
      no real OpenAI call is made.  The fake LLM is schema-aware:
      classifier calls get _FakeClassifierOutput("concept");
      synthesis calls get _FakeSynthesisOutput.
    - `app.indexer.WIKI_DIR` redirected to tmp_path so pages land in
      a temp directory, not the real wiki/.
    - `app.ingest.DOCS_DIR` uses the real docs/ directory (no mock — per
      CODING_STANDARD §6.3, we only mock the LLM, not the indexer/parser).

    Returns:
        (client, tmp_wiki_path) tuple.
    """
    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)

    # Redirect wiki/ to tmp_path so written pages land in isolation
    import app.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "WIKI_DIR", tmp_path / "wiki")

    from app.main import app

    return TestClient(app), tmp_path / "wiki"


# ---------------------------------------------------------------------------
# AC (Slice #1): POST /ingest happy path — 200, one result, one page created
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
    # Slice #2: concept sources produce N pages (one per section).
    # refund_policy.md has 3 sections → 3 pages; we check at least one.
    assert len(result["pages_written"]) >= 1, (
        f"Expected at least 1 page written, got: {result['pages_written']}"
    )
    assert "concepts/cancellation-window.md" in result["pages_written"], (
        f"Expected concepts/cancellation-window.md, got: {result['pages_written']}"
    )


# ---------------------------------------------------------------------------
# AC (Slice #1): wiki/concepts/cancellation-window.md correct structure
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
# AC (Slice #1): atomic write — no .tmp file lingers
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
# AC (Slice #1): POST /ingest with nonexistent source returns 200 with failed_sources
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
# AC (Slice #2): POST /ingest with no body returns 200 (batch mode)
# ---------------------------------------------------------------------------


def test_ingest_no_body_triggers_batch_mode(client_with_fake_ingest_llm):
    """POST /ingest with no body triggers batch mode: 200 with 3 IngestSourceResult entries.

    The real docs/ has 3 Sources (account_help.md, refund_policy.md,
    shipping_faq.md).  Batch mode should process all of them and return 3
    results entries (all classified as concept by the mock LLM).
    """
    client, tmp_wiki = client_with_fake_ingest_llm

    resp = client.post("/ingest")  # no body → batch mode

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["failed_sources"] == [], (
        f"Expected no failed sources in batch mode, got: {body['failed_sources']}"
    )
    assert len(body["results"]) == 3, (
        f"Expected 3 results (one per real doc), got: {len(body['results'])}"
    )


# ---------------------------------------------------------------------------
# AC (Slice #2): batch mode produces 9 concept pages
# ---------------------------------------------------------------------------


def test_ingest_batch_produces_9_concept_pages(client_with_fake_ingest_llm):
    """Batch ingest of the 3-source corpus produces 9 concept pages."""
    client, tmp_wiki = client_with_fake_ingest_llm

    resp = client.post("/ingest")
    assert resp.status_code == 200
    body = resp.json()

    all_pages = [p for r in body["results"] for p in r["pages_written"]]
    assert len(all_pages) == 9, f"Expected 9 concept pages, got {len(all_pages)}: {all_pages}"

    # All pages in concepts/ subdir
    for page in all_pages:
        assert page.startswith("concepts/"), f"Expected concepts/ prefix: {page}"

    # Spot-check a few expected slugs
    expected = {
        "concepts/cancellation-window.md",
        "concepts/change-email-address.md",
        "concepts/standard-shipping.md",
    }
    assert expected.issubset(set(all_pages)), (
        f"Expected {expected} to be subset of {set(all_pages)}"
    )


# ---------------------------------------------------------------------------
# AC (Slice #2): POST /ingest with body {source: X} still works (filter)
# ---------------------------------------------------------------------------


def test_ingest_single_source_filter_still_works(client_with_fake_ingest_llm):
    """POST /ingest with body {source: X} ingests only that one source."""
    client, tmp_wiki = client_with_fake_ingest_llm

    resp = client.post("/ingest", json={"source": "refund_policy.md"})
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["results"]) == 1
    assert body["results"][0]["source"] == "refund_policy.md"


# ---------------------------------------------------------------------------
# AC (Slice #2): entity source → one page in wiki/entities/<stem>.md
# ---------------------------------------------------------------------------


def test_ingest_entity_source_produces_one_entity_page(tmp_path, monkeypatch):
    """A Source classified as 'entity' produces one page in wiki/entities/."""
    import app.indexer as indexer_module

    # Create a minimal entity docs dir
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    entity_md = docs_dir / "my_product.md"
    entity_md.write_text(
        "# My Product\n\n## Overview\n\nA great product.\n\n## Features\n\nFast, reliable.\n",
        encoding="utf-8",
    )

    wiki_dir = tmp_path / "wiki"

    # Fake LLM that classifies as entity
    fake_llm = _make_schema_aware_fake_llm(classifier_type="entity")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.ingest import ingest_sources

    result = ingest_sources(["my_product.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert len(result.results) == 1
    source_result = result.results[0]
    assert len(source_result.pages_written) == 1
    # Entity page lands in entities/ subdir
    assert source_result.pages_written[0] == "entities/my-product.md", (
        f"Expected entities/my-product.md, got: {source_result.pages_written[0]}"
    )
    assert (wiki_dir / "entities" / "my-product.md").exists()


# ---------------------------------------------------------------------------
# AC (Slice #2): slug collision → overview + overview-2
# ---------------------------------------------------------------------------


def test_slug_collision_across_sources(tmp_path, monkeypatch):
    """Two sources both producing concept slug 'overview' → overview + overview-2."""
    import app.indexer as indexer_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Two sources each with a single "## Overview" section
    (docs_dir / "alpha.md").write_text("## Overview\n\nAlpha overview content.\n", encoding="utf-8")
    (docs_dir / "beta.md").write_text("## Overview\n\nBeta overview content.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.ingest import ingest_sources

    result = ingest_sources(["alpha.md", "beta.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"

    all_pages = [p for r in result.results for p in r.pages_written]
    assert "concepts/overview.md" in all_pages, f"Expected overview.md in {all_pages}"
    assert "concepts/overview-2.md" in all_pages, f"Expected overview-2.md in {all_pages}"

    assert (wiki_dir / "concepts" / "overview.md").exists()
    assert (wiki_dir / "concepts" / "overview-2.md").exists()


# ---------------------------------------------------------------------------
# AC (Slice #2): continue-on-error — APITimeoutError on one source
# ---------------------------------------------------------------------------


def test_continue_on_error_one_source_fails(tmp_path, monkeypatch):
    """APITimeoutError on one source's LLM call → that source in failed_sources,
    other sources still succeed."""
    import app.indexer as indexer_module

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "good.md").write_text("## Good Section\n\nThis one is fine.\n", encoding="utf-8")
    (docs_dir / "bad.md").write_text("## Bad Section\n\nThis one will fail.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    from openai import APITimeoutError

    # Track calls to classify: first call (good.md) → concept; second (bad.md) → raise
    call_counts: dict[str, int] = {"classify": 0, "generate": 0}

    def _error_on_bad_classify(content: str):
        call_counts["classify"] += 1
        if "bad" in content.lower() or "fail" in content.lower():
            raise APITimeoutError(
                request=None,  # type: ignore[arg-type]
                message="timeout",
            )
        return "concept"

    def _fake_generate(section, source_type, **kwargs):

        slug = "good-section"
        fm = WikiPageFrontmatter(
            id=slug,
            type="concept",
            created=FIXED_TS,
            updated=FIXED_TS,
            sources=[f"good.md#{slug}"],
            status="live",
            open_questions=[],
        )
        return WikiPageDraft(
            frontmatter=fm,
            body="Good content.",
            citation_line=f"[Source: good.md#{slug}]",
            slug=slug,
            heading="Good Section",
        )

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    # Patch the names as bound in ingest.py (not in templates.py), because
    # ingest.py does `from .templates import classify_source, generate_page`.
    import app.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "classify_source", _error_on_bad_classify)
    monkeypatch.setattr(ingest_module, "generate_page", _fake_generate)

    result = ingest_module.ingest_sources(
        ["good.md", "bad.md"],
        docs_dir=docs_dir,
        wiki_dir=wiki_dir,
    )

    assert "bad.md" in result.failed_sources, (
        f"Expected bad.md in failed_sources, got: {result.failed_sources}"
    )
    assert len(result.results) == 1, f"Expected 1 success result (good.md), got: {result.results}"
    assert result.results[0].source == "good.md"


# ---------------------------------------------------------------------------
# AC (Slice #2): glob("**/*.md") picks up nested fixture
# ---------------------------------------------------------------------------


def test_glob_nested_md_files_are_picked_up(tmp_path, monkeypatch):
    """ingest_sources(None) with a nested subfolder discovers nested .md files."""
    import app.indexer as indexer_module

    docs_dir = tmp_path / "docs"
    sub = docs_dir / "sub"
    sub.mkdir(parents=True)
    (sub / "nested.md").write_text("## Nested Section\n\nNested content.\n", encoding="utf-8")

    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm(classifier_type="concept")
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from app.ingest import ingest_sources

    result = ingest_sources(None, docs_dir=docs_dir, wiki_dir=wiki_dir)

    # nested.md should be discovered and processed
    all_sources = [r.source for r in result.results] + result.failed_sources
    assert "nested.md" in all_sources, (
        f"Expected nested.md to be processed (found or failed), got: {all_sources}"
    )
    assert result.results[0].source == "nested.md"


# ---------------------------------------------------------------------------
# AC (Slice #1): OPENAI_INGEST_MODEL env var fallback chain
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
# AC (Slice #1): Hermetic — verify mock LLM was used (not a real API call)
# ---------------------------------------------------------------------------


def test_ingest_uses_mock_llm_not_real_api(client_with_fake_ingest_llm):
    """Verify the fake LLM was invoked (proving no real API call was made).

    Also asserts the grounding verifier was mocked by the autouse conftest
    fixture — without that, /ingest secretly hits the real OpenAI verifier
    every test (issue #42). If this assertion ever fails, the autouse
    fixture in markdown_kb/tests/conftest.py has been removed or weakened.
    """
    client, tmp_wiki = client_with_fake_ingest_llm

    # The fixture injects _ingest_llm directly; verify it's a MagicMock
    assert isinstance(templates_module._ingest_llm, MagicMock), (
        "Expected _ingest_llm to be a MagicMock (fake) in test context"
    )

    # Regression guard (#42): conftest autouse must mock the grounding verifier
    # too. The real grounding.verify is the free function defined in
    # app/grounding.py; the autouse fixture replaces ingest_module.verify with
    # a lambda that returns a claim_supported outcome.
    import app.grounding as grounding_module
    import app.ingest as ingest_module

    assert ingest_module.verify is not grounding_module.verify, (
        "Expected ingest_module.verify to be mocked by the conftest autouse "
        "fixture; otherwise /ingest will make real OpenAI calls during the "
        "supposedly hermetic suite (issue #42)."
    )

    resp = client.post("/ingest", json={"source": "refund_policy.md"})
    assert resp.status_code == 200

    # The fake LLM's with_structured_output chain was invoked
    assert templates_module._ingest_llm.with_structured_output.called, (
        "Expected with_structured_output to be called on the fake LLM"
    )
