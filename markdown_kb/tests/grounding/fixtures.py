"""Type-I verifier judgment fixture skeleton for grounding.verify().

Each fixture is a VerifierCase record:
    draft               — the draft answer text the verifier judges
    sections            — list of SampleSection objects satisfying CitableContent
    expected_passed     — expected GroundingOutcome.passed value
    expected_unsupported_claims — expected unsupported_claims list (None = "don't care")

All sample Section data is defined here, decoupled from docs/ so fixtures
remain valid when the knowledge base evolves to wiki/ content.

The 7 categories follow the PRD / Testing Decisions:
    1. clean_support             — draft near-identical to a section          → passed=True
    2. paraphrase_support        — draft paraphrases section content          → passed=True
    3. addition_unsupported      — draft adds a fact not in sections          → passed=False
    4. inference_unsupported     — draft logically infers beyond sections     → passed=False
    5. partial_support           — half supported, half not                   → passed=False
    6. empty_sections            — sections list empty                        → passed=False
    7. synthesis_across_sections — combines facts from two sections, each ok  → passed=True
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Minimal stand-in that satisfies CitableContent (structural Protocol)
# ---------------------------------------------------------------------------


@dataclass
class SampleSection:
    """Lightweight stand-in for markdown_kb's Section type.

    Owns its content directly so fixtures are not coupled to docs/ files.
    Three required fields match the CitableContent Protocol:
        id, heading_path, content.
    """

    id: str
    heading_path: list[str]
    content: str


# ---------------------------------------------------------------------------
# Fixture record
# ---------------------------------------------------------------------------


@dataclass
class VerifierCase:
    """One judgment fixture consumed by test_verifier.py."""

    name: str
    draft: str
    sections: list[SampleSection]
    expected_passed: bool
    expected_unsupported_claims: list[str] | None  # None = "don't assert list contents"


# ---------------------------------------------------------------------------
# Sample sections shared across fixtures
# ---------------------------------------------------------------------------

_SECTION_REFUNDS = SampleSection(
    id="refunds-policy",
    heading_path=["Policies", "Refund Policy"],
    content=(
        "Refunds are processed within 5 to 7 business days of receiving the returned item. "
        "Eligible items must be unused and in original packaging."
    ),
)

_SECTION_SHIPPING = SampleSection(
    id="shipping-info",
    heading_path=["Shipping", "Domestic Shipping"],
    content=(
        "Standard domestic shipping takes 3 to 5 business days. "
        "Express shipping takes 1 to 2 business days and costs an additional $15."
    ),
)

_SECTION_RETURNS = SampleSection(
    id="returns-window",
    heading_path=["Policies", "Returns"],
    content=(
        "Customers may return items within 30 days of purchase. "
        "Items must be accompanied by the original receipt."
    ),
)

# ---------------------------------------------------------------------------
# 7 anchor judgment cases
# ---------------------------------------------------------------------------

CLEAN_SUPPORT = VerifierCase(
    name="clean_support",
    draft=(
        "Refunds are processed within 5 to 7 business days of receiving the returned item. "
        "Items must be unused and in original packaging to be eligible."
    ),
    sections=[_SECTION_REFUNDS],
    expected_passed=True,
    expected_unsupported_claims=None,
)

PARAPHRASE_SUPPORT = VerifierCase(
    name="paraphrase_support",
    draft=(
        "Once we receive your return, you can expect your refund to arrive "
        "in roughly a week — typically 5 to 7 business days."
    ),
    sections=[_SECTION_REFUNDS],
    expected_passed=True,
    expected_unsupported_claims=None,
)

ADDITION_UNSUPPORTED = VerifierCase(
    name="addition_unsupported",
    draft=(
        "Refunds are processed within 5 to 7 business days. "
        "Additionally, customers receive a 10% loyalty credit on their next order."
    ),
    sections=[_SECTION_REFUNDS],
    expected_passed=False,
    expected_unsupported_claims=["10% loyalty credit on their next order"],
)

INFERENCE_UNSUPPORTED = VerifierCase(
    name="inference_unsupported",
    draft=(
        "Because refunds take 5 to 7 business days, the company clearly prioritises "
        "manual review over automated processing for all returns."
    ),
    sections=[_SECTION_REFUNDS],
    expected_passed=False,
    expected_unsupported_claims=None,  # verifier decides exact claim text
)

PARTIAL_SUPPORT = VerifierCase(
    name="partial_support",
    draft=(
        "Standard domestic shipping takes 3 to 5 business days. "
        "International shipping is available to over 50 countries."
    ),
    sections=[_SECTION_SHIPPING],
    expected_passed=False,
    expected_unsupported_claims=None,  # the international shipping claim is unsupported
)

EMPTY_SECTIONS = VerifierCase(
    name="empty_sections",
    draft="Refunds are processed within 5 to 7 business days.",
    sections=[],
    expected_passed=False,
    expected_unsupported_claims=None,
)

SYNTHESIS_ACROSS_SECTIONS = VerifierCase(
    name="synthesis_across_sections",
    draft=(
        "You can return items within 30 days of purchase with your original receipt, "
        "and once we receive the return, your refund will be processed in 5 to 7 business days."
    ),
    sections=[_SECTION_RETURNS, _SECTION_REFUNDS],
    expected_passed=True,
    expected_unsupported_claims=None,
)

# ---------------------------------------------------------------------------
# Ordered list for parametrize consumption
# ---------------------------------------------------------------------------

ALL_CASES: list[VerifierCase] = [
    CLEAN_SUPPORT,
    PARAPHRASE_SUPPORT,
    ADDITION_UNSUPPORTED,
    INFERENCE_UNSUPPORTED,
    PARTIAL_SUPPORT,
    EMPTY_SECTIONS,
    SYNTHESIS_ACROSS_SECTIONS,
]
