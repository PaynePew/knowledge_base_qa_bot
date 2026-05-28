"""Phase 8 Paraphrase generation pipeline (PRD #100, issue #102).

The generator (``generate_paraphrases.py``, NOT run under pytest) drives
gpt-4o-mini over the concept-page corpus to synthesise the five Core Paraphrase
Types, then merges the two hand-written probe types into the committed
``queries.yaml``. This package holds the deterministic, offline-testable
building blocks the generator composes:

  - ``sampling``  — sha256-based deterministic Gold Section ordering/sampling.
  - ``qc``        — the Key-Token QC gate (reject all-stopword sets; flag
                    low-distinctiveness tokens via an IDF-like check).
  - ``gen_schema``— the ``with_structured_output`` Pydantic schema the LLM fills.
  - ``templates`` — the five per-type prompt templates (one file per Core type).

LLM output content is non-deterministic and therefore never asserted; the tests
cover only these deterministic seams (CODING_STANDARD §6.2).
"""
