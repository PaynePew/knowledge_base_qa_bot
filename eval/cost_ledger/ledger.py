"""Deep module per Ousterhout. Public surface: ``CostLedger``.

Per-stack, per-phase LLM call/token ledger (PRD #654 user stories 18-20 / 22;
ADR-0045's cost axis). Calls and tokens are the PRIMARY units; USD is
secondary and derived from ``unit_prices.py``'s pinned appendix, never a live
lookup (see that module's docstring). A ``CostLedger`` only accumulates and
aggregates what ``record()`` is given — it does not itself call any LLM or
know how to extract usage from a LangChain response (that seam lives in
``hooks.py``), so this module never sees a LangChain type (CODING_STANDARD
§2.4).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import PHASES, LedgerCall, StackPhaseTotals, UsageMetadata
from .unit_prices import estimate_usd


class CostLedger:
    """Accumulates ``LedgerCall`` entries and aggregates them per (stack, phase)."""

    def __init__(self) -> None:
        self._calls: list[LedgerCall] = []

    def record(
        self, *, stack: str, phase: str, model: str, usage: UsageMetadata
    ) -> None:
        """Record one LLM call.

        Raises ``ValueError`` when `phase` is not one of the PRD's three
        measured phases (``build`` / ``query`` / ``update``) — a typo'd phase
        would otherwise silently vanish from every per-phase aggregate.
        """
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {sorted(PHASES)}, got {phase!r}")
        self._calls.append(
            LedgerCall(stack=stack, phase=phase, model=model, usage=usage)
        )

    @property
    def calls(self) -> list[LedgerCall]:
        """All recorded calls, in recording order. A copy — mutating the
        returned list does not affect the ledger."""
        return list(self._calls)

    def totals(
        self, *, stack: str | None = None, phase: str | None = None
    ) -> StackPhaseTotals:
        """Aggregate every recorded call matching the given filter.

        Either or both of `stack` / `phase` may be omitted to aggregate
        across all stacks / phases (the returned ``StackPhaseTotals.stack`` /
        ``.phase`` reads ``"*"`` for an omitted filter). ``usd`` sums only
        the matching calls whose model has a pinned price — partial cost
        visibility beats none — and is ``None`` only when NO matching call's
        model is priced.
        """
        matching = [
            c
            for c in self._calls
            if (stack is None or c.stack == stack)
            and (phase is None or c.phase == phase)
        ]
        return _aggregate(
            stack if stack is not None else "*",
            phase if phase is not None else "*",
            matching,
        )

    def totals_by_stack_phase(self) -> dict[tuple[str, str], StackPhaseTotals]:
        """One ``StackPhaseTotals`` per distinct (stack, phase) pair actually recorded."""
        grouped: dict[tuple[str, str], list[LedgerCall]] = defaultdict(list)
        for c in self._calls:
            grouped[(c.stack, c.phase)].append(c)
        return {
            key: _aggregate(key[0], key[1], group) for key, group in grouped.items()
        }


def _aggregate(stack: str, phase: str, calls: Iterable[LedgerCall]) -> StackPhaseTotals:
    calls = list(calls)
    input_tokens = sum(c.usage.input_tokens for c in calls)
    output_tokens = sum(c.usage.output_tokens for c in calls)
    total_tokens = sum(c.usage.total_tokens for c in calls)
    priced = [p for c in calls if (p := estimate_usd(c.model, c.usage)) is not None]
    return StackPhaseTotals(
        stack=stack,
        phase=phase,
        calls=len(calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        usd=sum(priced) if priced else None,
    )
