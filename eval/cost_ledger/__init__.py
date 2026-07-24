"""LLM cost ledger — per-stack, per-phase call/token accounting for the corpus
v3 fair experiment (PRD #654 user stories 18-20 / 22; ADR-0045's cost axis).

Calls and tokens are the primary units; USD is a secondary estimate derived
from ``unit_prices.py``'s pinned appendix (never a live lookup). See
``ledger.py`` for the ``CostLedger`` accumulator, ``hooks.py`` for wiring an
LLM-facing module's lazy-singleton getter into a ledger, and ``models.py`` /
``unit_prices.py`` for the data shapes.
"""
