"""``answer_over_sections`` — hermetic tests (issue #673).

External behaviour only (CODING_STANDARD §0.2 / §6.2 / §6.3): only the two
LLM-facing lazy-singleton getters are mocked (the synthesis ``get_llm`` and
the shared grounding verifier ``get_verifier_llm``) — no deep module
(``grounding.verify`` itself, page expansion, prompt building) is mocked.
"""

from __future__ import annotations

import markdown_kb.app.indexer as bm25_indexer
import pytest

import hybrid_kb.app.query as query_module
from markdown_kb.app import grounding as grounding_module
from markdown_kb.app.grounding import GroundingClaim, GroundingResult
from markdown_kb.app.indexer import Section

REFUND_ID = "refund-policy#refund-policy"


def _wiki_section(section_id: str, content: str, heading_path: list[str]) -> Section:
    return Section(
        id=section_id,
        file=section_id.split("#")[0],
        heading=heading_path[-1],
        heading_path=heading_path,
        content=content,
        tokens=bm25_indexer.tokenize(content),
        metadata={"lang": "en"},
    )


_REFUND_SECTION = _wiki_section(
    REFUND_ID,
    "Refund policy: refunds are processed within seven business days after approval.",
    ["Refund Policy"],
)


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.call_count = 0

    def invoke(self, messages: list):
        self.call_count += 1
        return _FakeLLMResponse(content=self._content)


class _FakeVerifierChain:
    def __init__(self, result: GroundingResult):
        self._result = result

    def invoke(self, message):
        return self._result


class _FakeVerifierLLM:
    def __init__(self, result: GroundingResult):
        self._result = result

    def with_structured_output(self, schema):
        return _FakeVerifierChain(self._result)


def _passing_result(cited_id: str) -> GroundingResult:
    return GroundingResult(
        reasoning="ok",
        claims=[
            GroundingClaim(text="ok", supported=True, citing_section_ids=[cited_id])
        ],
        unsupported_claims=[],
        passed=True,
    )


@pytest.fixture()
def wired_bm25_sections(monkeypatch):
    """Populate ``indexer.sections`` (the module-level page list) with the
    same Sections passed to ``answer_over_sections`` — ``expand_to_pages``
    (reused unchanged from ``_draft_and_verify``) resolves a hit's parent
    page from that global list, not from its own input."""
    monkeypatch.setattr(bm25_indexer, "sections", [_REFUND_SECTION])


@pytest.fixture()
def fake_verifier(monkeypatch):
    def _install(cited_id: str = REFUND_ID) -> None:
        monkeypatch.setattr(
            grounding_module,
            "get_verifier_llm",
            lambda: _FakeVerifierLLM(_passing_result(cited_id)),
        )

    return _install


def _patch_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(query_module, "get_llm", lambda: llm)


# ---------------------------------------------------------------------------
# Happy path — reuses _draft_and_verify, same shape as query()
# ---------------------------------------------------------------------------
def test_answers_over_a_caller_supplied_section_pool(
    monkeypatch, fake_verifier, wired_bm25_sections
):
    fake_verifier(REFUND_ID)
    fake_llm = FakeLLM(f"Refunds take about a week. [Source: {REFUND_ID}]")
    _patch_llm(monkeypatch, fake_llm)

    result = query_module.answer_over_sections(
        "how long do refunds take", [_REFUND_SECTION]
    )

    assert result["answer"] == f"Refunds take about a week. [Source: {REFUND_ID}]"
    assert result["grounding_outcome"].passed is True
    assert fake_llm.call_count == 1


def test_sources_mirror_the_shared_citation_shape(
    monkeypatch, fake_verifier, wired_bm25_sections
):
    fake_verifier(REFUND_ID)
    _patch_llm(monkeypatch, FakeLLM(f"Refunds take a week. [Source: {REFUND_ID}]"))

    result = query_module.answer_over_sections(
        "how long do refunds take", [_REFUND_SECTION]
    )

    sources = result["sources"]
    assert sources
    assert all({"source", "heading", "content"} <= set(s) for s in sources)
    assert REFUND_ID in {s["source"] for s in sources}


# ---------------------------------------------------------------------------
# No retrieval gate of its own — only the empty-pool refusal
# ---------------------------------------------------------------------------
def test_empty_section_pool_refuses_without_an_llm_call(monkeypatch):
    sentinel = FakeLLM("should never be called")
    _patch_llm(monkeypatch, sentinel)

    result = query_module.answer_over_sections("anything", [])

    assert result["answer"] == query_module.CANNOT_CONFIRM_PHRASE
    assert result["grounding_outcome"].passed is False
    assert result["grounding_outcome"].reason == "retrieval_empty"
    assert sentinel.call_count == 0


def test_grounding_verifier_failure_still_returns_cannot_confirm(
    monkeypatch, wired_bm25_sections
):
    monkeypatch.setattr(
        grounding_module,
        "get_verifier_llm",
        lambda: _FakeVerifierLLM(
            GroundingResult(
                reasoning="unsupported",
                claims=[
                    GroundingClaim(text="x", supported=False, citing_section_ids=[])
                ],
                unsupported_claims=["x"],
                passed=False,
            )
        ),
    )
    _patch_llm(monkeypatch, FakeLLM(f"Refunds take a week. [Source: {REFUND_ID}]"))

    result = query_module.answer_over_sections(
        "how long do refunds take", [_REFUND_SECTION]
    )

    assert result["answer"] == query_module.CANNOT_CONFIRM_PHRASE
    assert result["grounding_outcome"].passed is False
