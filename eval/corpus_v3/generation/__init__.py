"""Corpus v3 query generation spec (issue #660, ADR-0045 Prerequisite 4).

The multi-family, stratum-labelled query-generation pipeline for the corpus
v3 fair experiment (PRD #654 user stories 5, 7, 8). This package holds the
deterministic, offline-testable building blocks a later corpus-build issue's
generator script composes once the adversarial corpus (redundancy +
contradictions + version evolution) exists:

  - ``overlap``    — deterministic query-document lexical-overlap stratum
                      computation (ADR-0045 Prerequisite 3: overlap stratified
                      at generation time, not left implicit).
  - ``gen_schema`` — the ``with_structured_output`` Pydantic draft schema an
                      LLM fills, plus ``to_query`` stitching the
                      generator-owned bookkeeping fields onto it.
  - ``qc``         — the generation-time QC gate: every generated
                      :class:`~eval.corpus_v3.query_schema.Query` must record
                      its ``generating_family``, its ``key_tokens`` must
                      survive tokenisation, and its ``language`` label must
                      match its text's detected script (the zh slice's "own
                      gate").

See ``eval/corpus_v3/generation/SPEC.md`` for the full query-generation spec
this package implements (model-family requirement, overlap timing, zh
gating, and the LLM-call convention — seeded, temperature-0, excluded from
the default test suite — for the actual generator script this scaffolds).
LLM output content is never asserted in tests (CODING_STANDARD §6.2); only
these deterministic seams are.

The actual generator script this spec anticipated (issue #672) adds:

  - ``targets``           — deterministic derivation + sha256-ordered
                             sampling of which corpus Section(s) each
                             generated query targets, per scenario stratum.
  - ``prompts``            — per-(stratum, language) prompt construction
                             (structural contract only; wording is untested).
  - ``artifact``           — committed query-set YAML assembly, including the
                             target-vs-actual deviation header (AC 1).
  - ``generate_queries``   — the CLI orchestrator (``__main__``, not
                             collected by pytest), gated by the #662-pattern
                             pre-spend cost guard (``eval.corpus_v3.cost_guard``).
"""

from __future__ import annotations
