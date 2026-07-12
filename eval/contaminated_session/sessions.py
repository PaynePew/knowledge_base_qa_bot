"""Shallow module. Hand-written fixture sessions for the contaminated-session
rewrite-drift eval (#608, split from #579's "(b)" report finding).

Each ``ContaminatedSessionCase`` is a turn-1-WRONG-answer / turn-2-on-topic-
follow-up pair, hand-authored (never LLM-generated at test time — CODING_STANDARD
§6.5) to reproduce the two drift shapes #579 reported:

  - ``wrong_topic_contamination``: the wrong turn-1 answer drags in an
    unrelated topic's vocabulary (Shipping) into what should stay a Refund
    Policy conversation — a probe for retrieval-flip risk.
  - ``wrong_fact_same_topic``: the wrong turn-1 answer states an incorrect
    number for an otherwise on-topic fact — a probe for whether the wrong
    figure gets baked into the rewritten query text even without a retrieval
    flip.

History turn dicts mirror ``gateway.app.conversation_store``'s documented
shape verbatim (fixture fidelity, CODING_STANDARD §6.5) — every key that
module's docstring lists as required is present, even though ``rewrite_query``
itself only reads ``question`` / ``answer`` (``query_rewriting._build_user_message``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContaminatedSessionCase:
    """One characterization case.

    ``contaminated_history`` carries the wrong turn-1 answer;
    ``clean_history`` is the control (empty — no prior turn at all, so
    ``rewrite_query`` takes its turn-1-passthrough branch and never calls the
    LLM for the control arm). ``followup_question`` is the literal, raw turn-2
    ask — never pre-resolved by the fixture itself.
    """

    name: str
    contaminated_history: list[dict]
    clean_history: list[dict]
    followup_question: str
    note: str


def _turn(question: str, answer: str) -> dict:
    """Build one conversation_store turn dict (module docstring's required keys).

    Only ``question`` / ``answer`` vary across fixtures — ``stack`` /
    ``grounding_reason`` / ``ts`` are fixed, deterministic filler since
    ``rewrite_query`` never reads them (query_rewriting._build_user_message
    surfaces only question + answer to the model).
    """
    return {
        "question": question,
        "answer": answer,
        "stack": "wiki",
        "grounding_reason": "claim_supported",
        "ts": "2026-01-01T00:00:00Z",
    }


CASES: tuple[ContaminatedSessionCase, ...] = (
    ContaminatedSessionCase(
        name="wrong_topic_contamination",
        contaminated_history=[
            _turn(
                "How long do refunds take?",
                "Refunds are handled by our shipping carrier and typically "
                "take 5-7 business days from when your order ships.",
            )
        ],
        clean_history=[],
        followup_question="And what if I don't have the receipt?",
        note=(
            "Turn 1's WRONG answer conflates the refund timeline with the "
            "Shipping Policy's carrier delivery timeline (the real refund "
            "window is 14 days and is unrelated to the carrier). The literal "
            "follow-up is on-topic for Refund Policy > Store Credit Refunds — "
            "a topic-drift probe: does the shipping-flavoured wrong turn "
            "steer the rewrite (and therefore retrieval) toward Shipping "
            "Policy instead?"
        ),
    ),
    ContaminatedSessionCase(
        name="wrong_fact_same_topic",
        contaminated_history=[
            _turn(
                "What's your refund window?",
                "You have 60 days from the delivery date to request a refund.",
            )
        ],
        clean_history=[],
        followup_question="Does that apply if I return it as store credit too?",
        note=(
            "Turn 1's WRONG answer states 60 days (the real policy is 14 "
            "days). The literal follow-up stays on-topic (still Refund "
            "Policy) — a fact-drift probe: does the wrong '60 days' figure "
            "get baked into the rewritten query text even when the retrieved "
            "Section does not flip?"
        ),
    ),
)
