"""Shallow module per Ousterhout. Public surface: ``RetrievedItem``.

Domain data shape shared by every corpus v3 retrieval-arm adapter
(``stacks.ARM_REGISTRY``). Mirrors ``eval.paraphrase_comparison.models
.RetrievedItem`` in shape (ADR-0045 Prerequisite 1: every arm normalises to a
common retrieved-item shape so corpus effect and algorithm effect separate
cleanly) but is defined independently here — this package owns its own
committed fixtures and stays isolated from the v2 eval package (PRD #654:
"a new eval package, sibling to the existing paraphrase-comparison eval ...
with its own committed fixtures ... and production isolation").

Unlike the v2 shape, ``source_section_id`` is NOT resolved to a docs-native
gold id here — symmetric gold-label mapping across corpora is ADR-0045
Prerequisite 3 (de-biased harness), out of scope for this slice. It carries
whichever corpus-neutral Section id the retrieving stack natively returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievedItem:
    """One retrieval-arm hit, normalised to a common corpus-neutral shape.

    ``source_section_id`` is the retrieving stack's native Section id (e.g. a
    wiki page's slug-based id for a wiki-backed arm). ``content`` is the
    retrieved text. ``heading_path`` is the Section's heading breadcrumb.
    """

    source_section_id: str
    content: str
    heading_path: list[str] = field(default_factory=list)
