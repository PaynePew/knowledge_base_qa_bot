"""Deep module per Ousterhout. Public surface: ``SynthesizerConfig``,
``build_styling_config``, ``build_filtration_config``, ``build_synthesizer``,
``build_section_contexts``, ``post_filter_by_score``, ``goldens_to_queries``.

DeepEval Synthesizer pipeline for Phase 8.5 query generation (issue #144).

This module wires DeepEval's Synthesizer to generate **query text only** from
per-Gold-Section contexts.  The LLM's role is confined to authoring the query
string; the Gold Section id, paraphrase type, and all bookkeeping fields are
assigned deterministically by this module.

Architecture
------------
The pipeline has three deterministic seams (offline-testable without any API
key) and one live seam (real OpenAI calls, opt-in via ``--live`` / issue #145):

Deterministic seams:
  1. ``build_section_contexts`` — maps GoldSections + Section bodies to the
     ``(contexts, source_files)`` pair that the Synthesizer consumes.  The
     ``source_files`` list carries one Gold Section id per context so DeepEval
     propagates it to ``Golden.source_file`` on each returned Golden.
  2. ``post_filter_by_score`` — quality-score threshold gate.  DeepEval stores
     the critic's score in ``Golden.additional_metadata["synthetic_input_quality"]``.
     The threshold parameter alone does not hard-drop inside DeepEval (it controls
     retry attempts); this function applies a hard post-filter after generation so
     the threshold is the *output* gate.
  3. ``goldens_to_queries`` — converts Goldens to ``Paraphrase`` objects by
     reading ``golden.source_file`` as ``gold_docs_section_id``.  Raises on a
     ``None`` source_file (requires the caller to always pass ``source_files``).

Live seam:
  ``build_synthesizer`` — creates the Synthesizer with StylingConfig + FiltrationConfig.
  Called only when OPENAI_API_KEY is set.  The live call itself is ``generate_goldens_from_contexts``,
  which the caller performs (see generate_paraphrases_v2.py for the orchestration CLI).

Generator design decisions (PRD #137)
--------------------------------------
- Generator model: ``gpt-4o`` (strong model, better style diversity than mini).
- Critic model: ``gpt-4o-mini`` (same OpenAI family — same-family justification:
  the answer key is deterministic so the critic only judges query quality, not
  correctness; cross-family adds cost without benefit).
- ``StylingConfig.task`` steers toward natural customer questions that avoid
  copying the passage vocabulary, fulfilling AC2.
- ``StylingConfig.scenario`` gives Acme Shop domain context so generated queries
  are on-topic for the e-commerce help-desk setting.
- ``include_expected_output=False`` — we do not want DeepEval's Synthesizer to
  invent an expected answer; the answer key is deterministic and owned by the
  corpus (issue #139).
- Demo tier default: 50 Paraphrases per Core Type (PRD #137 §Dataset size).

Coverage guarantee (AC3)
------------------------
``build_section_contexts`` accepts every Gold Section in the pool without
filtering.  The Synthesizer then receives one context per section, ensuring
every Gold Section is *offered*.  The actual generated count per section is
controlled by ``max_goldens_per_context`` (= 1 for 1 query/section coverage).

Stdout via ``print`` is NOT used here — this is committed library code
(CODING_STANDARD §5.1).  Log channels must use ``log_event`` if added.

LangChain types (StylingConfig, FiltrationConfig, Synthesizer) are kept
inside this module and the CLI layer; they never leak past the LLM-call
wrapper boundary (CODING_STANDARD §2.4 / project trap #3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden

from ..models import Paraphrase, ParaphraseType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default generator model (strong; gpt-4o class for query diversity — PRD #137).
_DEFAULT_GENERATOR_MODEL = "gpt-4o"

# Default critic model (same OpenAI family as generator — PRD #137 justification).
_DEFAULT_CRITIC_MODEL = "gpt-4o-mini"

# Demo-tier quantity: ≈50 Paraphrases per Core Type (PRD #137 §Dataset size).
_DEMO_TIER_PER_TYPE = 50

# Default quality threshold for post-filtering.  Goldens whose critic score is
# strictly below this value are dropped.  FiltrationConfig's internal threshold
# controls retry attempts during generation; this controls the *output* gate.
_DEFAULT_QUALITY_THRESHOLD = 0.5


@dataclass
class SynthesizerConfig:
    """Configuration for the DeepEval Synthesizer pipeline (issue #144).

    ``per_type_count`` is the number of generated Paraphrases per Core Type
    (Demo tier default ≈ 50, PRD #137).  ``generator_model`` and ``critic_model``
    are OpenAI model names; the critic must be same-family as the generator
    (PRD #137 — with a deterministic answer key, the critic only judges query
    quality, not correctness).  ``quality_threshold`` is the post-filter gate:
    Goldens whose ``synthetic_input_quality`` score is below this value are
    dropped from the final paraphrase set.
    """

    per_type_count: int = _DEMO_TIER_PER_TYPE
    generator_model: str = _DEFAULT_GENERATOR_MODEL
    critic_model: str = _DEFAULT_CRITIC_MODEL
    quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD


# ---------------------------------------------------------------------------
# Config builders (deterministic — offline testable)
# ---------------------------------------------------------------------------


def build_styling_config():
    """Return a StylingConfig that steers Synthesizer output away from passage vocabulary.

    The config sets ``task`` and ``scenario`` so the Synthesizer generates
    natural customer questions — phrased in the customer's own words — rather
    than paraphrases that copy the help-centre document wording (AC2).

    Lazy import keeps DeepEval types inside this module (project trap #3).
    """
    from deepeval.synthesizer.config import StylingConfig  # noqa: PLC0415

    return StylingConfig(
        scenario=(
            "An Acme Shop customer browsing the help centre or contacting support "
            "with a question about their order, account, returns, shipping, or "
            "any other store policy."
        ),
        task=(
            "Generate a natural, realistic customer question that a real user would "
            "type into a search bar or chat window. The question must be phrased in "
            "plain, conversational English — avoiding the exact wording and technical "
            "vocabulary used in the help-centre passage — so that only a system with "
            "strong semantic retrieval can match it to the right section."
        ),
        input_format=(
            "A short question (one sentence, typically 10–25 words) written in the "
            "first or second person, as a customer would phrase it."
        ),
    )


def build_filtration_config(
    critic_model: str,
    quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD,
):
    """Return a FiltrationConfig with the given same-family critic model.

    ``critic_model`` must be from the same provider family as the generator
    (PRD #137).  ``quality_threshold`` is propagated to
    ``FiltrationConfig.synthetic_input_quality_threshold`` and controls both
    the internal retry logic AND the downstream ``post_filter_by_score`` gate
    (callers should use the same value for both).

    Lazy import keeps DeepEval types inside this module (project trap #3).
    """
    from deepeval.synthesizer.config import FiltrationConfig  # noqa: PLC0415

    return FiltrationConfig(
        critic_model=critic_model,
        synthetic_input_quality_threshold=quality_threshold,
        max_quality_retries=3,
    )


# ---------------------------------------------------------------------------
# Synthesizer factory (live — requires OPENAI_API_KEY at call time)
# ---------------------------------------------------------------------------


def build_synthesizer(config: SynthesizerConfig):
    """Return a configured DeepEval Synthesizer (live — requires OPENAI_API_KEY).

    Wires the generator model, StylingConfig (avoids passage vocabulary), and
    FiltrationConfig (same-family critic, quality threshold).  The Synthesizer
    is constructed here but the actual generation call (``generate_goldens_from_contexts``)
    is the caller's responsibility so it can be gated behind OPENAI_API_KEY,
    ``--live``, and ``@pytest.mark.live``.

    Lazy import keeps DeepEval types inside this module (project trap #3).
    """
    from deepeval.synthesizer import Synthesizer  # noqa: PLC0415

    return Synthesizer(
        model=config.generator_model,
        async_mode=False,  # sync mode: simpler error propagation for CLI use
        filtration_config=build_filtration_config(
            critic_model=config.critic_model,
            quality_threshold=config.quality_threshold,
        ),
        styling_config=build_styling_config(),
    )


# ---------------------------------------------------------------------------
# Deterministic seam 1: build_section_contexts
# ---------------------------------------------------------------------------


def build_section_contexts(
    gold_sections: list,
    section_bodies: dict[str, tuple[str, str]],
) -> tuple[list[list[str]], list[str]]:
    """Map Gold Sections to Synthesizer inputs with deterministic source tracking.

    For each ``GoldSection`` in ``gold_sections``, look up the ``(heading, body)``
    pair from ``section_bodies`` and build a single-element context list.  The
    ``source_files`` list carries the Gold Section id at the same index so
    DeepEval propagates it to ``Golden.source_file`` on generation.

    Returns ``(contexts, source_files)`` where:
    - ``contexts`` is ``List[List[str]]`` — one entry per Gold Section, each
      entry is a list of strings (the Section body, headed by its heading).
    - ``source_files`` is ``List[str]`` — parallel list of Gold Section ids,
      passed as ``source_files`` to the Synthesizer.

    Coverage guarantee: every Gold Section in ``gold_sections`` receives exactly
    one context entry.  The caller passes the full Gold Section pool (not a
    pre-sampled subset) so every section is offered to the Synthesizer (AC3).

    ``section_bodies`` must contain an entry for every id in ``gold_sections``;
    missing entries raise ``KeyError`` (fail-fast, CODING_STANDARD §4.1).
    """
    contexts: list[list[str]] = []
    source_files: list[str] = []

    for sec in gold_sections:
        heading, body = section_bodies[sec.section_id]
        # A single context element: heading + body as one string.  DeepEval
        # treats each context element as a retrieval unit; a single combined
        # string keeps the section coherent and avoids cross-section leakage.
        context_text = f"{heading}\n\n{body}" if heading else body
        contexts.append([context_text])
        source_files.append(sec.section_id)

    return contexts, source_files


# ---------------------------------------------------------------------------
# Deterministic seam 2: post_filter_by_score
# ---------------------------------------------------------------------------


def post_filter_by_score(
    goldens: list[Golden],
    threshold: float,
) -> list[Golden]:
    """Return only Goldens whose critic quality score meets or exceeds ``threshold``.

    DeepEval stores the critic's quality score in
    ``golden.additional_metadata["synthetic_input_quality"]``.  A missing key
    or ``None`` additional_metadata is treated as score=0 (i.e. dropped) to
    avoid silently admitting unscored Goldens.

    The ``FiltrationConfig.synthetic_input_quality_threshold`` parameter controls
    DeepEval's *internal* retry loop; this function is the *output* hard-drop gate.
    Both should use the same threshold value so the retry policy and the output
    gate are consistent (``SynthesizerConfig.quality_threshold`` holds the
    single source of truth).
    """
    kept: list[Golden] = []
    for g in goldens:
        meta = g.additional_metadata
        if not meta:
            continue
        score = meta.get("synthetic_input_quality")
        if score is None:
            continue
        if score >= threshold:
            kept.append(g)
    return kept


# ---------------------------------------------------------------------------
# Deterministic seam 3: goldens_to_queries
# ---------------------------------------------------------------------------


def goldens_to_queries(
    goldens: list[Golden],
    *,
    paraphrase_type: ParaphraseType,
    id_prefix: str,
) -> list[Paraphrase]:
    """Convert DeepEval Goldens to ``Paraphrase`` objects with deterministic ids.

    Reads ``golden.source_file`` as ``gold_docs_section_id`` — the deterministic
    assignment guaranteed because ``build_section_contexts`` passed the Gold
    Section id as ``source_files[i]`` to the Synthesizer.

    ``paraphrase_type`` and ``id_prefix`` are caller-supplied; the id is
    ``"{id_prefix}-{N:03d}"`` for N in 1-indexed order.

    Raises ``ValueError`` if any Golden has ``source_file=None`` — that indicates
    the caller omitted ``source_files`` from the Synthesizer call, which would
    break the deterministic id assignment.

    Only the query text (``golden.input``) comes from the LLM; all other fields
    are deterministic (AC1 / PRD #137 §Deterministic answer key).
    """
    queries: list[Paraphrase] = []
    for idx, golden in enumerate(goldens, start=1):
        if golden.source_file is None:
            raise ValueError(
                f"golden.source_file is None for golden #{idx} — "
                "pass source_files= to the Synthesizer call so Gold Section ids "
                "ride through deterministically (issue #144, AC4)."
            )
        queries.append(
            Paraphrase(
                paraphrase_id=f"{id_prefix}-{idx:03d}",
                paraphrase_type=paraphrase_type,  # type: ignore[arg-type]
                text=golden.input,
                gold_docs_section_id=golden.source_file,
                # Key tokens are left empty here; they are derived deterministically
                # from the Gold Section body by rekey.py (issue #139) after generation.
                key_tokens_docs=[],
                key_tokens_wiki=[],
                generation_notes="",
            )
        )
    return queries
