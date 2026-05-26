"""Integration tests for ChatResponse.grounding field — Slice 4 (issue #12).

Tests mirror the PROMPT.md verification curl cases and assert the grounding
field shape for all six GroundingInfo.reason values.

Mock-mode tests (default, no OPENAI_API_KEY needed):
    Use fake LLM stubs and monkeypatched grounding.verify() so the full
    route → retrieval → schema mapping is exercised without real API calls.

Live-mode tests (opt-in with pytest -m live, requires OPENAI_API_KEY):
    Same four PROMPT.md scenarios hit the real endpoint with real OpenAI calls.

Run with:
    uv run pytest                 # mock-mode only (skips @pytest.mark.live)
    uv run pytest -m live         # live-mode only
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import app.indexer as indexer
import app.logger as logger_module
import app.retrieval as retrieval_module
from app.grounding import GroundingClaim, GroundingOutcome, GroundingResult
from app.retrieval import CANNOT_CONFIRM_PHRASE

from .conftest import REAL_DOCS, FakeLLMResponse

REFUND_SECTION_ID = "refund_policy.md#refund-timeline"
EMAIL_SECTION_ID = "account_help.md#change-email-address"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLLM:
    """Returns a canned grounded answer containing the given section id."""

    def __init__(self, source_id: str, content: str | None = None):
        self.source_id = source_id
        self._content = content

    def invoke(self, messages: list) -> FakeLLMResponse:
        body = self._content or f"Answer text. [Source: {self.source_id}]"
        return FakeLLMResponse(content=body)


def _approved_outcome(source_id: str) -> GroundingOutcome:
    """Build a GroundingOutcome(passed=True) with one supported claim."""
    return GroundingOutcome(
        passed=True,
        reason="claim_supported",
        result=GroundingResult(
            reasoning="All claims trace to the cited section.",
            claims=[
                GroundingClaim(
                    text="The claim is supported.",
                    supported=True,
                    citing_section_ids=[source_id],
                )
            ],
            unsupported_claims=[],
            passed=True,
        ),
        retries_attempted=0,
    )


def _rejected_outcome() -> GroundingOutcome:
    """Build a GroundingOutcome(passed=False, reason=claim_unsupported)."""
    return GroundingOutcome(
        passed=False,
        reason="claim_unsupported",
        result=GroundingResult(
            reasoning="The draft contained an unsupported claim.",
            claims=[
                GroundingClaim(
                    text="Unsupported extra claim.",
                    supported=False,
                    citing_section_ids=[],
                )
            ],
            unsupported_claims=["Unsupported extra claim."],
            passed=False,
        ),
        retries_attempted=0,
    )


def _unavailable_outcome() -> GroundingOutcome:
    """Build a GroundingOutcome(passed=False, reason=verifier_unavailable)."""
    return GroundingOutcome(
        passed=False,
        reason="verifier_unavailable",
        result=None,
        retries_attempted=2,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Chat before indexing → grounding.reason == "index_missing"
# ---------------------------------------------------------------------------


def test_grounding_index_missing(monkeypatch):
    """POST /chat before POST /index → grounding.passed=False, reason=index_missing.

    Mirrors PROMPT.md verification case: chat before indexing.
    """
    indexer.sections.clear()

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()

    assert "grounding" in body, "ChatResponse must have 'grounding' field"
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] == "index_missing", (
        f"Expected reason=index_missing, got: {body['grounding']['reason']!r}"
    )
    assert body["sources"] == [], "sources must be [] when index is missing"


# ---------------------------------------------------------------------------
# Scenario 2: Refund query → claim_supported, claims non-empty, cites section
# ---------------------------------------------------------------------------


def test_grounding_claim_supported_refund(indexed_corpus, monkeypatch):
    """POST /chat 'How long do refunds take?' → grounding.passed=True, claim_supported.

    Mirrors PROMPT.md verification case 1.
    """
    fake_llm = FakeLLM(source_id=REFUND_SECTION_ID)
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(REFUND_SECTION_ID),
    )

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["grounding"]["passed"] is True
    assert body["grounding"]["reason"] == "claim_supported"
    claims = body["grounding"]["claims"]
    assert claims is not None and len(claims) > 0, f"Expected non-empty claims, got: {claims}"
    # At least one claim cites the refund section
    citing_ids = [cid for c in claims for cid in c["citing_section_ids"]]
    assert REFUND_SECTION_ID in citing_ids, (
        f"Expected {REFUND_SECTION_ID!r} in citing_section_ids, got: {citing_ids}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: Email query → claim_supported, claims non-empty, cites section
# ---------------------------------------------------------------------------


def test_grounding_claim_supported_email(indexed_corpus, monkeypatch):
    """POST /chat 'Can I change my email address?' → grounding.passed=True, claim_supported.

    Mirrors PROMPT.md verification case 2.
    """
    fake_llm = FakeLLM(source_id=EMAIL_SECTION_ID)
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: _approved_outcome(EMAIL_SECTION_ID),
    )

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Can I change my email address?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["grounding"]["passed"] is True
    assert body["grounding"]["reason"] == "claim_supported"
    claims = body["grounding"]["claims"]
    assert claims is not None and len(claims) > 0
    citing_ids = [cid for c in claims for cid in c["citing_section_ids"]]
    assert EMAIL_SECTION_ID in citing_ids, (
        f"Expected {EMAIL_SECTION_ID!r} in citing_section_ids, got: {citing_ids}"
    )


# ---------------------------------------------------------------------------
# Scenario 4: Out-of-scope query → pre-LLM gate (retrieval_empty | below_threshold)
# ---------------------------------------------------------------------------


def test_grounding_pre_llm_gate_restaurants(indexed_corpus, monkeypatch):
    """POST /chat 'Which restaurants are nearby?' → below_threshold or retrieval_empty.

    Mirrors PROMPT.md verification case 3. LLM must NOT be invoked.
    """

    class _SentinelLLM:
        def invoke(self, messages):
            raise AssertionError("LLM must not be called for out-of-scope query")

    monkeypatch.setattr(retrieval_module, "_llm", _SentinelLLM())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: _SentinelLLM())

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == CANNOT_CONFIRM_PHRASE
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] in ("retrieval_empty", "below_threshold"), (
        f"Expected pre-LLM gate reason, got: {body['grounding']['reason']!r}"
    )


# ---------------------------------------------------------------------------
# Scenario 5: grounding.verify() returns claim_unsupported → Cannot Confirm
# ---------------------------------------------------------------------------


def test_grounding_claim_unsupported(indexed_corpus, monkeypatch):
    """When grounding.verify() rejects the draft → Cannot Confirm + claim_unsupported.

    Exercises ADR-0004 Block & Replace failure contract.
    """
    fake_llm = FakeLLM(
        source_id=REFUND_SECTION_ID,
        content="Refunds take 3 days. Also, we offer free worldwide shipping.",
    )
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: _rejected_outcome(),
    )

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == CANNOT_CONFIRM_PHRASE
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] == "claim_unsupported"
    assert (
        body["grounding"]["unsupported_claims"] is not None
        and len(body["grounding"]["unsupported_claims"]) > 0
    ), f"Expected unsupported_claims to be non-empty, got: {body['grounding']}"


# ---------------------------------------------------------------------------
# Scenario 6: grounding.verify() unavailable → Cannot Confirm + verifier_unavailable
# ---------------------------------------------------------------------------


def test_grounding_verifier_unavailable(indexed_corpus, monkeypatch):
    """When grounding.verify() fails (all retries exhausted) → verifier_unavailable.

    Exercises ADR-0004 fail-closed invariant: even on verifier failure, the
    answer is Cannot Confirm (not the draft).
    """
    fake_llm = FakeLLM(source_id=REFUND_SECTION_ID)
    monkeypatch.setattr(retrieval_module, "_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        retrieval_module.grounding_module,
        "verify",
        lambda draft, sections: _unavailable_outcome(),
    )

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == CANNOT_CONFIRM_PHRASE
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] == "verifier_unavailable"
    # claims must be None when verifier did not complete (no GroundingResult)
    assert body["grounding"]["claims"] is None, (
        f"Expected claims=None for verifier_unavailable, got: {body['grounding']['claims']}"
    )


# ---------------------------------------------------------------------------
# sources populated on below_threshold (ADR-0004 / PRD User Story 22)
# ---------------------------------------------------------------------------


def test_sources_populated_on_below_threshold(indexed_corpus, monkeypatch):
    """Even on below_threshold Cannot Confirm, sources is non-empty if retrieval ran.

    PRD User Story 22: client can display what the bot looked at even on fallback.
    """
    # Set threshold very high so ANY BM25 result is still below threshold
    monkeypatch.setattr(retrieval_module, "_SCORE_THRESHOLD", 9999.0)

    class _SentinelLLM:
        def invoke(self, messages):
            raise AssertionError("LLM must not be called below threshold")

    monkeypatch.setattr(retrieval_module, "_llm", _SentinelLLM())
    monkeypatch.setattr(retrieval_module, "get_llm", lambda: _SentinelLLM())

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == CANNOT_CONFIRM_PHRASE
    assert body["grounding"]["reason"] == "below_threshold"
    # sources must be non-empty — retrieval ran before the gate fired
    assert len(body["sources"]) > 0, (
        f"Expected non-empty sources on below_threshold, got: {body['sources']}"
    )


# ---------------------------------------------------------------------------
# Live integration tests (opt-in, require OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_grounding_live_refund_claim_supported(tmp_path, monkeypatch):
    """PROMPT.md case 1 live: 'How long do refunds take?' → claim_supported.

    Requires OPENAI_API_KEY. Tests the full chain: real LLM + real verifier.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

    monkeypatch.setattr(indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(retrieval_module, "_llm", None)

    indexer.build_index(REAL_DOCS)

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "How long do refunds take?"})

    assert resp.status_code == 200
    body = resp.json()

    assert "grounding" in body
    assert body["grounding"]["passed"] is True, (
        f"Expected grounding.passed=True for supported query, got: {body['grounding']}"
    )
    assert body["grounding"]["reason"] == "claim_supported"
    assert body["grounding"]["claims"] is not None and len(body["grounding"]["claims"]) > 0
    refund_in_sources = any("refund_policy.md#" in s.get("source", "") for s in body["sources"])
    assert refund_in_sources, (
        f"Expected refund_policy.md source, got: {[s['source'] for s in body['sources']]}"
    )

    indexer.sections.clear()


@pytest.mark.live
def test_grounding_live_email_claim_supported(tmp_path, monkeypatch):
    """PROMPT.md case 2 live: 'Can I change my email address?' → claim_supported."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

    monkeypatch.setattr(indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(retrieval_module, "_llm", None)

    indexer.build_index(REAL_DOCS)

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Can I change my email address?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["grounding"]["passed"] is True
    assert body["grounding"]["reason"] == "claim_supported"
    assert body["grounding"]["claims"] is not None and len(body["grounding"]["claims"]) > 0
    email_in_sources = any("account_help.md#" in s.get("source", "") for s in body["sources"])
    assert email_in_sources, (
        f"Expected account_help.md source, got: {[s['source'] for s in body['sources']]}"
    )

    indexer.sections.clear()


@pytest.mark.live
def test_grounding_live_restaurants_pre_llm_gate(tmp_path, monkeypatch):
    """PROMPT.md case 3 live: 'Which restaurants are nearby?' → pre-LLM gate."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.fail("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

    monkeypatch.setattr(indexer, "INDEX_PATH", tmp_path / ".kb" / "index.json")
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setattr(retrieval_module, "_llm", None)

    indexer.build_index(REAL_DOCS)

    from app.main import app

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "Which restaurants are nearby?"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == CANNOT_CONFIRM_PHRASE
    assert body["grounding"]["passed"] is False
    assert body["grounding"]["reason"] in ("retrieval_empty", "below_threshold"), (
        f"Expected pre-LLM gate reason, got: {body['grounding']['reason']!r}"
    )

    indexer.sections.clear()
