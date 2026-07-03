"""Tests for Slice #4 — Grounding Check integration, red links, ingest log kinds.

AC coverage (issue #32):
  - Mock verifier returns claim_supported → page status=live, pages_with_failed_grounding empty
  - Mock verifier returns claim_unsupported → page status=failed_grounding,
    grounding_failure block in frontmatter, page id in pages_with_failed_grounding
  - Mock verifier returns verifier_unavailable → same as claim_unsupported (fail-soft)
  - Red link rule block appears in prompt (golden assertion: "wikilinks" + "maximum 5" +
    "do NOT verify" substrings)
  - Mock LLM produces [[some-concept]] wikilinks → verbatim in page content on disk
  - All 5 log entry kinds emit at correct trigger points in correct format
  - ingest_source does NOT emit for failed Sources (only ingest_error does)
  - ingest_batch_completed.failed_grounding=F matches len(pages_with_failed_grounding)
  - Hermetic: no OPENAI_API_KEY needed (verifier mocked)
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import app.templates as templates_module
from app.grounding import GroundingClaim, GroundingOutcome, GroundingResult
from app.schemas import WikiPageDraft, WikiPageFrontmatter

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FIXED_TS = "2026-05-26T14:30:00Z"
ISO_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z")


# ---------------------------------------------------------------------------
# Fake LLM helpers (mirrors test_ingest_integration.py pattern)
# ---------------------------------------------------------------------------


class _FakeSynthesisOutput:
    def __init__(self, body: str = "Synthesised content.", open_questions: list | None = None):
        self.body = body
        self.open_questions = open_questions or []


class _FakeClassifierOutput:
    def __init__(self, source_type: str = "concept"):
        self.type = source_type


def _make_schema_aware_fake_llm(
    synthesis_body: str = "Synthesised content.",
    classifier_type: str = "concept",
) -> MagicMock:
    """Schema-aware fake LLM following the established test pattern."""
    from app.templates import _ClassifierOutput

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


# ---------------------------------------------------------------------------
# Grounding outcome builders
# ---------------------------------------------------------------------------


def _claim_supported_outcome() -> GroundingOutcome:
    result = GroundingResult(
        reasoning="All claims are supported by the cited sections.",
        claims=[
            GroundingClaim(
                text="Customers can cancel within 24 hours.",
                supported=True,
                citing_section_ids=["refund_policy.md#cancellation-window"],
            )
        ],
        unsupported_claims=[],
        passed=True,
    )
    return GroundingOutcome(passed=True, reason="claim_supported", result=result)


def _claim_unsupported_outcome() -> GroundingOutcome:
    result = GroundingResult(
        reasoning="One claim is not in the source.",
        claims=[
            GroundingClaim(
                text="next-day delivery guaranteed",
                supported=False,
                citing_section_ids=[],
            )
        ],
        unsupported_claims=["next-day delivery guaranteed"],
        passed=False,
    )
    return GroundingOutcome(passed=False, reason="claim_unsupported", result=result)


def _verifier_unavailable_outcome() -> GroundingOutcome:
    return GroundingOutcome(passed=False, reason="verifier_unavailable", result=None)


def _claim_unsupported_outcome_empty_flat_list() -> GroundingOutcome:
    """claims[] has a supported=False entry but the flat list was left empty.

    Mirrors the live corpus bug (#404): the LLM marks a claim unsupported in
    claims[] without mirroring it into the flat unsupported_claims field.
    """
    result = GroundingResult(
        reasoning="One claim is not in the source.",
        claims=[
            GroundingClaim(
                text="ships within 2 business days",
                supported=False,
                citing_section_ids=[],
            )
        ],
        unsupported_claims=[],
        passed=False,
    )
    return GroundingOutcome(passed=False, reason="claim_unsupported", result=result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def docs_dir(tmp_path) -> Path:
    """A minimal docs dir with one concept source."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "refund_policy.md").write_text(
        "## Cancellation Window\n\nCustomers can cancel within 24 hours of purchase.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def wiki_dir(tmp_path) -> Path:
    return tmp_path / "wiki"


def _patch_env(monkeypatch, docs_dir: Path, wiki_dir: Path, fake_llm: MagicMock) -> None:
    """Apply standard monkeypatches: fake LLM + redirected dirs."""
    import app.indexer as indexer_module

    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)


# ---------------------------------------------------------------------------
# AC: claim_supported → status=live, pages_with_failed_grounding empty
# ---------------------------------------------------------------------------


def test_grounding_supported_page_has_status_live(tmp_path, monkeypatch, docs_dir, wiki_dir):
    """When verifier returns claim_supported, page is written with status=live."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    fake_llm = _make_schema_aware_fake_llm()
    _patch_env(monkeypatch, docs_dir, wiki_dir, fake_llm)

    outcome = _claim_supported_outcome()

    with patch("app.ingest.verify", return_value=outcome):
        result = ingest_sources(["refund_policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], f"Unexpected failures: {result.failed_sources}"
    assert len(result.results) == 1

    # Page on disk: status must be live
    page_path = wiki_dir / "concepts" / "cancellation-window.md"
    assert page_path.exists(), f"Expected page at {page_path}"
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    assert len(dash_indices) >= 2
    fm = yaml.safe_load("\n".join(lines[dash_indices[0] + 1 : dash_indices[1]]))
    assert fm["status"] == "live", f"Expected status=live, got: {fm['status']}"
    assert "grounding_failure" not in fm, "grounding_failure block must be absent for live page"

    # Response: pages_with_failed_grounding is empty
    assert result.pages_with_failed_grounding == [], (
        f"Expected empty pages_with_failed_grounding, got: {result.pages_with_failed_grounding}"
    )


# ---------------------------------------------------------------------------
# AC: claim_unsupported → status=failed_grounding, frontmatter block, response lists page
# ---------------------------------------------------------------------------


def test_grounding_unsupported_page_has_failed_grounding_status(
    tmp_path, monkeypatch, docs_dir, wiki_dir
):
    """When verifier returns claim_unsupported, page written with status=failed_grounding."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    fake_llm = _make_schema_aware_fake_llm()
    _patch_env(monkeypatch, docs_dir, wiki_dir, fake_llm)

    outcome = _claim_unsupported_outcome()

    with patch("app.ingest.verify", return_value=outcome):
        result = ingest_sources(["refund_policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], (
        f"Source should still succeed (fail-soft): {result.failed_sources}"
    )
    assert len(result.results) == 1

    # Page on disk: status=failed_grounding + grounding_failure block
    page_path = wiki_dir / "concepts" / "cancellation-window.md"
    assert page_path.exists(), f"Expected page at {page_path}"
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    assert len(dash_indices) >= 2
    fm = yaml.safe_load("\n".join(lines[dash_indices[0] + 1 : dash_indices[1]]))

    assert fm["status"] == "failed_grounding", (
        f"Expected status=failed_grounding, got: {fm['status']}"
    )
    assert "grounding_failure" in fm, f"Expected grounding_failure block in frontmatter: {fm}"
    gf = fm["grounding_failure"]
    assert gf["reason"] == "claim_unsupported", f"Expected reason=claim_unsupported, got: {gf}"
    assert "next-day delivery guaranteed" in gf["unsupported_claims"], (
        f"Expected unsupported claim in list, got: {gf['unsupported_claims']}"
    )

    # Response: page id in pages_with_failed_grounding
    assert "cancellation-window" in result.pages_with_failed_grounding, (
        f"Expected 'cancellation-window' in pages_with_failed_grounding, "
        f"got: {result.pages_with_failed_grounding}"
    )


# ---------------------------------------------------------------------------
# AC: verifier_unavailable → same shape as claim_unsupported
# ---------------------------------------------------------------------------


def test_grounding_unavailable_treated_same_as_unsupported(
    tmp_path, monkeypatch, docs_dir, wiki_dir
):
    """verifier_unavailable (transient error after retry) → status=failed_grounding."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    fake_llm = _make_schema_aware_fake_llm()
    _patch_env(monkeypatch, docs_dir, wiki_dir, fake_llm)

    outcome = _verifier_unavailable_outcome()

    with patch("app.ingest.verify", return_value=outcome):
        result = ingest_sources(["refund_policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == [], "verifier_unavailable must not fail the source"

    page_path = wiki_dir / "concepts" / "cancellation-window.md"
    assert page_path.exists()
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm = yaml.safe_load("\n".join(lines[dash_indices[0] + 1 : dash_indices[1]]))

    assert fm["status"] == "failed_grounding", f"Expected failed_grounding, got: {fm['status']}"
    assert "grounding_failure" in fm
    gf = fm["grounding_failure"]
    assert gf["reason"] == "verifier_unavailable", (
        f"Expected reason=verifier_unavailable, got: {gf}"
    )
    # No unsupported_claims list when verifier was unavailable (result is None)
    assert gf.get("unsupported_claims", []) == [], (
        f"Expected empty unsupported_claims for unavailable outcome, got: {gf}"
    )

    assert "cancellation-window" in result.pages_with_failed_grounding


# ---------------------------------------------------------------------------
# AC (#404): unsupported_claims derived from authoritative claims[]
# ---------------------------------------------------------------------------


def test_grounding_failure_derives_claims_when_flat_list_empty(
    tmp_path, monkeypatch, docs_dir, wiki_dir
):
    """Empty flat unsupported_claims still persists the claims[]-derived text."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    fake_llm = _make_schema_aware_fake_llm()
    _patch_env(monkeypatch, docs_dir, wiki_dir, fake_llm)

    outcome = _claim_unsupported_outcome_empty_flat_list()

    with patch("app.ingest.verify", return_value=outcome):
        ingest_sources(["refund_policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    page_path = wiki_dir / "concepts" / "cancellation-window.md"
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm = yaml.safe_load("\n".join(lines[dash_indices[0] + 1 : dash_indices[1]]))

    gf = fm["grounding_failure"]
    assert gf["unsupported_claims"] == ["ships within 2 business days"], (
        f"Expected claims[]-derived text despite empty flat list, got: {gf}"
    )


def test_grounding_failure_unions_flat_list_entries_not_in_claims(
    tmp_path, monkeypatch, docs_dir, wiki_dir
):
    """A flat-list-only entry (not present in claims[]) survives the union (no regression)."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    fake_llm = _make_schema_aware_fake_llm()
    _patch_env(monkeypatch, docs_dir, wiki_dir, fake_llm)

    result = GroundingResult(
        reasoning="Two claims are not in the source.",
        claims=[
            GroundingClaim(
                text="ships within 2 business days",
                supported=False,
                citing_section_ids=[],
            )
        ],
        # "next-day delivery guaranteed" only appears in the flat list, not
        # mirrored into claims[] — must still survive the union.
        unsupported_claims=["ships within 2 business days", "next-day delivery guaranteed"],
        passed=False,
    )
    outcome = GroundingOutcome(passed=False, reason="claim_unsupported", result=result)

    with patch("app.ingest.verify", return_value=outcome):
        ingest_sources(["refund_policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    page_path = wiki_dir / "concepts" / "cancellation-window.md"
    content = page_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]
    fm = yaml.safe_load("\n".join(lines[dash_indices[0] + 1 : dash_indices[1]]))

    gf = fm["grounding_failure"]
    assert set(gf["unsupported_claims"]) == {
        "ships within 2 business days",
        "next-day delivery guaranteed",
    }, f"Expected union of claims[] and flat list, got: {gf}"


# ---------------------------------------------------------------------------
# AC: Red link rule block in prompt
# ---------------------------------------------------------------------------


def test_red_link_rule_block_appears_in_concept_prompt():
    """Concept system prompt contains the red link rule block substrings."""
    from app.templates import _CONCEPT_SYSTEM_PROMPT

    assert "wikilinks" in _CONCEPT_SYSTEM_PROMPT, "Expected 'wikilinks' in concept system prompt"
    assert "maximum 5" in _CONCEPT_SYSTEM_PROMPT, "Expected 'maximum 5' in concept system prompt"
    assert "do NOT verify" in _CONCEPT_SYSTEM_PROMPT, (
        "Expected 'do NOT verify' in concept system prompt"
    )


def test_red_link_rule_block_appears_in_entity_prompt():
    """Entity system prompt contains the red link rule block substrings."""
    from app.templates import _ENTITY_SYSTEM_PROMPT

    assert "wikilinks" in _ENTITY_SYSTEM_PROMPT, "Expected 'wikilinks' in entity system prompt"
    assert "maximum 5" in _ENTITY_SYSTEM_PROMPT, "Expected 'maximum 5' in entity system prompt"
    assert "do NOT verify" in _ENTITY_SYSTEM_PROMPT, (
        "Expected 'do NOT verify' in entity system prompt"
    )


# ---------------------------------------------------------------------------
# AC: [[wikilinks]] in LLM body are preserved verbatim on disk
# ---------------------------------------------------------------------------


def test_wikilinks_in_llm_body_preserved_on_disk(tmp_path, monkeypatch, docs_dir, wiki_dir):
    """If LLM body contains [[some-concept]], it must appear verbatim on disk."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    body_with_links = (
        "Customers can cancel within 24 hours. See also [[return-policy]] and [[shipping-sla]]."
    )
    fake_llm = _make_schema_aware_fake_llm(synthesis_body=body_with_links)
    _patch_env(monkeypatch, docs_dir, wiki_dir, fake_llm)

    outcome = _claim_supported_outcome()
    with patch("app.ingest.verify", return_value=outcome):
        result = ingest_sources(["refund_policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert result.failed_sources == []
    page_path = wiki_dir / "concepts" / "cancellation-window.md"
    assert page_path.exists()
    content = page_path.read_text(encoding="utf-8")
    assert "[[return-policy]]" in content, f"Expected [[return-policy]] in page:\n{content}"
    assert "[[shipping-sla]]" in content, f"Expected [[shipping-sla]] in page:\n{content}"


# ---------------------------------------------------------------------------
# AC: All 5 log entry kinds emit at correct trigger points
# ---------------------------------------------------------------------------


def test_log_ingest_batch_started_and_completed(tmp_path, monkeypatch):
    """ingest_batch_started and ingest_batch_completed log entries are emitted."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "simple.md").write_text("## Overview\n\nSimple content.\n", encoding="utf-8")
    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    outcome = _claim_supported_outcome()
    with (
        patch("app.ingest.verify", return_value=outcome),
        patch("app.ingest.log_event") as mock_log,
    ):
        ingest_sources(["simple.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    calls = [(call.args[0], call.args[1]) for call in mock_log.call_args_list]
    kinds = [c[0] for c in calls]

    # ingest_batch_started must be first
    assert "ingest_batch_started" in kinds, f"Expected ingest_batch_started in: {kinds}"
    assert kinds[0] == "ingest_batch_started", (
        f"ingest_batch_started must be first log entry, got: {kinds[0]}"
    )

    # ingest_batch_completed must be last
    assert "ingest_batch_completed" in kinds, f"Expected ingest_batch_completed in: {kinds}"
    assert kinds[-1] == "ingest_batch_completed", (
        f"ingest_batch_completed must be last log entry, got: {kinds[-1]}"
    )

    # ingest_source must appear (success path)
    assert "ingest_source" in kinds, f"Expected ingest_source in: {kinds}"

    # ingest_batch_started summary contains sources=N
    started_call = next(c for c in calls if c[0] == "ingest_batch_started")
    assert "sources=" in started_call[1], (
        f"Expected sources= in ingest_batch_started summary: {started_call[1]}"
    )

    # ingest_batch_completed summary contains required fields
    completed_call = next(c for c in calls if c[0] == "ingest_batch_completed")
    summary = completed_call[1]
    for field in ("sources=", "total_pages=", "llm_calls=", "cost_usd=", "failed_grounding="):
        assert field in summary, f"Expected '{field}' in ingest_batch_completed summary: {summary}"


def test_log_ingest_source_has_required_fields(tmp_path, monkeypatch):
    """ingest_source log entry has source=X type=Y pages_created=A pages_updated=B pages_deleted=C."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "simple.md").write_text("## Overview\n\nSimple content.\n", encoding="utf-8")
    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    outcome = _claim_supported_outcome()
    with (
        patch("app.ingest.verify", return_value=outcome),
        patch("app.ingest.log_event") as mock_log,
    ):
        ingest_sources(["simple.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    calls = [(call.args[0], call.args[1]) for call in mock_log.call_args_list]
    source_calls = [c for c in calls if c[0] == "ingest_source"]
    assert len(source_calls) == 1, f"Expected 1 ingest_source call, got: {source_calls}"

    summary = source_calls[0][1]
    for field in ("source=", "type=", "pages_created=", "pages_updated=", "pages_deleted="):
        assert field in summary, f"Expected '{field}' in ingest_source summary: {summary}"


def test_log_ingest_grounding_failed_emitted_on_verifier_fail(
    tmp_path, monkeypatch, docs_dir, wiki_dir
):
    """ingest_grounding_failed log entry emitted when verifier returns claim_unsupported."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    fake_llm = _make_schema_aware_fake_llm()
    _patch_env(monkeypatch, docs_dir, wiki_dir, fake_llm)

    outcome = _claim_unsupported_outcome()
    with (
        patch("app.ingest.verify", return_value=outcome),
        patch("app.ingest.log_event") as mock_log,
    ):
        ingest_sources(["refund_policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    calls = [(call.args[0], call.args[1]) for call in mock_log.call_args_list]
    grounding_failed_calls = [c for c in calls if c[0] == "ingest_grounding_failed"]
    assert len(grounding_failed_calls) >= 1, (
        f"Expected at least 1 ingest_grounding_failed entry, got: {[c[0] for c in calls]}"
    )
    summary = grounding_failed_calls[0][1]
    for field in ("page=", "reason=", "claims="):
        assert field in summary, f"Expected '{field}' in ingest_grounding_failed summary: {summary}"


def test_log_ingest_error_emitted_on_source_failure(tmp_path, monkeypatch):
    """ingest_error log entry emitted on source-level failure (not ingest_source)."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "good.md").write_text("## Overview\n\nGood content.\n", encoding="utf-8")
    (docs_dir / "bad.md").write_text("## Bad\n\nBad content.\n", encoding="utf-8")
    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from openai import APITimeoutError

    def _error_on_bad(content: str):
        if "bad" in content.lower():
            raise APITimeoutError(request=None, message="timeout")  # type: ignore[arg-type]
        return "concept"

    import app.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "classify_source", _error_on_bad)

    outcome = _claim_supported_outcome()
    with (
        patch("app.ingest.verify", return_value=outcome),
        patch("app.ingest.log_event") as mock_log,
    ):
        ingest_sources(["good.md", "bad.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    calls = [(call.args[0], call.args[1]) for call in mock_log.call_args_list]
    kinds = [c[0] for c in calls]

    assert "ingest_error" in kinds, f"Expected ingest_error for bad.md, got: {kinds}"
    error_calls = [c for c in calls if c[0] == "ingest_error"]
    assert any("bad.md" in c[1] for c in error_calls), (
        f"Expected 'bad.md' in ingest_error summary, got: {error_calls}"
    )

    # ingest_source must NOT appear for bad.md
    source_calls = [c for c in calls if c[0] == "ingest_source"]
    for sc in source_calls:
        assert "bad.md" not in sc[1], (
            f"ingest_source must NOT emit for failed source bad.md: {sc[1]}"
        )


# ---------------------------------------------------------------------------
# AC: ingest_source NOT emitted for failed Sources
# ---------------------------------------------------------------------------


def test_ingest_source_not_emitted_for_failed_source(tmp_path, monkeypatch):
    """ingest_source is NEVER emitted for a failed source — only ingest_error."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "failing.md").write_text("## Section\n\nContent.\n", encoding="utf-8")
    wiki_dir = tmp_path / "wiki"

    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    from openai import APITimeoutError

    import app.ingest as ingest_module

    monkeypatch.setattr(
        ingest_module,
        "classify_source",
        lambda _: (_ for _ in ()).throw(
            APITimeoutError(request=None, message="timeout")  # type: ignore[arg-type]
        ),
    )

    with patch("app.ingest.log_event") as mock_log:
        result = ingest_sources(["failing.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    calls = [(call.args[0], call.args[1]) for call in mock_log.call_args_list]
    kinds = [c[0] for c in calls]

    assert "ingest_source" not in kinds, (
        f"ingest_source must NOT appear for a failed source, got: {kinds}"
    )
    assert "ingest_error" in kinds, f"Expected ingest_error for failed source, got: {kinds}"
    assert "failing.md" in result.failed_sources


# ---------------------------------------------------------------------------
# AC: ingest_batch_completed.failed_grounding=F matches len(pages_with_failed_grounding)
# ---------------------------------------------------------------------------


def test_batch_completed_failed_grounding_count_matches_response(tmp_path, monkeypatch):
    """ingest_batch_completed failed_grounding=F matches len(pages_with_failed_grounding)."""
    import app.indexer as indexer_module
    from app.ingest import ingest_sources

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    # Two sections → two pages; both will fail grounding
    (docs_dir / "policy.md").write_text(
        "## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n",
        encoding="utf-8",
    )
    wiki_dir = tmp_path / "wiki"

    fake_llm = _make_schema_aware_fake_llm()
    monkeypatch.setattr(templates_module, "_ingest_llm", fake_llm)
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    outcome = _claim_unsupported_outcome()

    logged_completed: list[str] = []

    def _capture_log(kind: str, summary: str) -> None:
        if kind == "ingest_batch_completed":
            logged_completed.append(summary)

    with (
        patch("app.ingest.verify", return_value=outcome),
        patch("app.ingest.log_event", side_effect=_capture_log),
    ):
        result = ingest_sources(["policy.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert len(logged_completed) == 1, f"Expected 1 ingest_batch_completed, got: {logged_completed}"
    summary = logged_completed[0]

    # Extract failed_grounding=F value from summary
    match = re.search(r"failed_grounding=(\d+)", summary)
    assert match, f"Expected failed_grounding=N in summary: {summary}"
    logged_count = int(match.group(1))

    response_count = len(result.pages_with_failed_grounding)
    assert logged_count == response_count, (
        f"ingest_batch_completed failed_grounding={logged_count} "
        f"!= len(pages_with_failed_grounding)={response_count}"
    )
    # Both should be 2 (both pages failed grounding)
    assert response_count == 2, (
        f"Expected 2 failed grounding pages, got: {result.pages_with_failed_grounding}"
    )
