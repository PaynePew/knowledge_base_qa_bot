"""corpus v3 fair-experiment eval package (PRD #654, ADR-0045).

Sibling to ``eval.paraphrase_comparison``, with its own committed fixtures and
production isolation — the production Sources and ``wiki/`` are never touched.

This slice (#655) is the tracer bullet: the package scaffold plus one complete
arm end-to-end — dense-over-wiki standalone, the missing 2x2 cell named in
ADR-0045 Prerequisite 1 (the hybrid stack's dense arm evaluated without RRF
fusion, so corpus effect and algorithm effect separate cleanly from the v2
eval's confound). Retrieval-arm adapters live in ``.stacks``; every arm returns
``list[.models.RetrievedItem]`` — the common normalised shape later slices'
arms (stack A / B / C over the v3 corpus) also return, registered by name in
``stacks.ARM_REGISTRY`` so later slices add arms without conflicting here.
"""

from __future__ import annotations
