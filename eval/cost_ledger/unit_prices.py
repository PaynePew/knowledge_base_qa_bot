"""Shallow module per Ousterhout. Pinned USD unit-price appendix for the cost
ledger.

PRD #654: "primary units are calls and tokens, USD is secondary with unit
prices pinned in an appendix." Prices below are USD per 1,000,000 tokens,
hand-pinned to OpenAI's published API pricing as of 2026-07-24 (issue #657)
— this module NEVER performs a live pricing lookup. Re-pin by editing
``PINNED_PRICES`` directly (and updating the date in this docstring); a model
absent from the table returns ``None`` from ``estimate_usd`` rather than
guessing.

Covers the models named across ADR-0005's LLM-facing surface enumeration:
``OPENAI_MODEL`` / ``OPENAI_VERIFIER_MODEL`` / ``OPENAI_INGEST_MODEL`` /
``OPENAI_TRANSCRIBE_MODEL`` all default to ``gpt-4o-mini``;
``vector_rag``/``hybrid_kb``'s embedding getters default to
``text-embedding-3-small``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import UsageMetadata


@dataclass(frozen=True)
class UnitPrice:
    """USD per 1,000,000 tokens, input and output priced separately."""

    input_per_million: float
    output_per_million: float


# Appendix — pinned 2026-07-24 from https://openai.com/api/pricing (issue #657)
# and https://platform.claude.com/docs/en/pricing (issue #672's Family B
# generator; Anthropic first-party API rates).
PINNED_PRICES: dict[str, UnitPrice] = {
    "gpt-4o-mini": UnitPrice(input_per_million=0.15, output_per_million=0.60),
    "gpt-4o": UnitPrice(input_per_million=2.50, output_per_million=10.00),
    "text-embedding-3-small": UnitPrice(input_per_million=0.02, output_per_million=0.0),
    "text-embedding-3-large": UnitPrice(input_per_million=0.13, output_per_million=0.0),
    "claude-haiku-4-5": UnitPrice(input_per_million=1.00, output_per_million=5.00),
}


def estimate_usd(model: str, usage: UsageMetadata) -> float | None:
    """Secondary USD estimate for one call's usage, or ``None`` when `model`
    has no pinned price in ``PINNED_PRICES``."""
    price = PINNED_PRICES.get(model)
    if price is None:
        return None
    return (usage.input_tokens / 1_000_000) * price.input_per_million + (
        usage.output_tokens / 1_000_000
    ) * price.output_per_million
