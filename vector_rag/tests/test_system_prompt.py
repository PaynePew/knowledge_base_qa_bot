"""SYSTEM_PROMPT drift guard + build_prompt structure (issue #103).

vector_rag's SYSTEM_PROMPT is its OWN literal of the ADR-0001 strict-grounded
contract — deliberately NOT imported from markdown_kb (the apps stay
decoupled). This smoke test guards against the two literals drifting apart: if
markdown_kb tightens its contract, the duplicate must be updated in lockstep or
this test fails, surfacing the divergence.

build_prompt is asserted to mirror markdown_kb's structure (CONTEXT before
QUESTION, one [Source: ...] header + Heading: breadcrumb per Chunk), filled
exclusively with vector_rag's own Chunks.
"""

from __future__ import annotations

from markdown_kb.app.prompt_builder import SYSTEM_PROMPT as MK_SYSTEM_PROMPT

from vector_rag.app.indexer import Chunk
from vector_rag.app.retrieval import CANNOT_CONFIRM_PHRASE, SYSTEM_PROMPT, build_prompt

# The Cannot Confirm sentinel is duplicated across apps by design; pin it so a
# typo in either app's literal is caught (CODING_STANDARD §3.3).
EXPECTED_CANNOT_CONFIRM = "I cannot confirm from the knowledge base."


def test_system_prompt_matches_markdown_kb_contract():
    """vector_rag's SYSTEM_PROMPT is byte-identical to markdown_kb's contract.

    The literal is intentionally duplicated (decoupled apps); this drift guard
    is the trip-wire that keeps the duplicate honest.
    """
    assert SYSTEM_PROMPT == MK_SYSTEM_PROMPT, (
        "vector_rag SYSTEM_PROMPT has drifted from markdown_kb's strict-grounded "
        "contract. Update the duplicate in vector_rag/app/retrieval.py to match, "
        "or, if the divergence is intentional, supersede this guard with a new ADR."
    )


def test_system_prompt_encodes_adr_0001_contract():
    """SYSTEM_PROMPT carries the key ADR-0001 obligations regardless of import source."""
    prompt = SYSTEM_PROMPT
    assert "[Source: filename#heading]" in prompt, "must specify the citation format"
    assert EXPECTED_CANNOT_CONFIRM in prompt, "must frame the exact Cannot Confirm phrase"
    assert "ONLY" in prompt, "must constrain answers to CONTEXT only"


def test_cannot_confirm_phrase_is_the_shared_sentinel():
    assert CANNOT_CONFIRM_PHRASE == EXPECTED_CANNOT_CONFIRM


def test_build_prompt_structure():
    """build_prompt: CONTEXT before QUESTION, [Source: ...] header, Heading: breadcrumb."""
    chunks = [
        Chunk(
            id="refund_policy.md#refund-timeline",
            source="refund_policy.md#refund-timeline",
            heading_path=["Refund Policy", "Refund Timeline"],
            content="Approved refunds are processed within 5-7 business days.",
        ),
        Chunk(
            id="account_help.md#change-email-address",
            source="account_help.md#change-email-address",
            heading_path=["Account Help", "Change Email Address"],
            content="Go to Settings to update your email.",
        ),
    ]
    prompt = build_prompt("How long do refunds take?", chunks)

    ctx_pos = prompt.find("CONTEXT:")
    q_pos = prompt.find("QUESTION:")
    assert ctx_pos != -1 and q_pos != -1
    assert ctx_pos < q_pos, "CONTEXT: must appear before QUESTION:"

    for chunk in chunks:
        assert f"[Source: {chunk.source}]" in prompt
        assert f"Heading: {' > '.join(chunk.heading_path)}" in prompt
        assert chunk.content in prompt

    assert prompt.rstrip().endswith("How long do refunds take?")
