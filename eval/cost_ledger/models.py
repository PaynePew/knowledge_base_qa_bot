"""Shallow module per Ousterhout. Data models for the LLM cost ledger (issue
#657, PRD #654 user stories 18-20 / 22)."""

from __future__ import annotations

from dataclasses import dataclass

# The four-part cost model's three MEASURED phases (PRD #654: build / query /
# update; the fourth part, cost-per-grounded-correct-answer, is a derived
# metric over these, not a phase of its own).
PHASES = frozenset({"build", "query", "update"})


@dataclass(frozen=True)
class UsageMetadata:
    """Token counts for one LLM call, in the shape LangChain's
    ``AIMessage.usage_metadata`` already exposes (``input_tokens`` /
    ``output_tokens`` / ``total_tokens``).

    A call whose usage is not observable at its SDK surface (e.g. a
    ``with_structured_output()`` chain without ``include_raw=True`` — see
    ``hooks.py``) still gets recorded, with all-zero token counts, so the
    calls axis (the PRD's primary unit) stays accurate even when the tokens
    axis is invisible for that call shape.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_raw(cls, raw: dict | None) -> UsageMetadata:
        """Build from a raw ``usage_metadata``-shaped mapping (or ``None``/empty)."""
        if not raw:
            return cls()
        return cls(
            input_tokens=int(raw.get("input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
        )


@dataclass(frozen=True)
class LedgerCall:
    """One recorded LLM call: which stack, which experiment phase, which
    model, and its usage."""

    stack: str
    phase: str
    model: str
    usage: UsageMetadata


@dataclass(frozen=True)
class StackPhaseTotals:
    """Aggregated calls/tokens (and secondary USD) for one (stack, phase) cell.

    ``usd`` is ``None`` when no aggregated call's model has a pinned unit
    price (``unit_prices.py``) — never a silent 0.0, so a caller can tell
    "zero cost" apart from "cost unknown".
    """

    stack: str
    phase: str
    calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usd: float | None
