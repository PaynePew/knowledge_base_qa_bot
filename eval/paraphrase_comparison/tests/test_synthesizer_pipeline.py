"""Offline unit tests for the DeepEval Synthesizer pipeline seams (issue #144).

Deterministic seams under test:
  - build_section_contexts: maps GoldSections + Section bodies to Synthesizer inputs
  - post_filter_by_score: quality-score threshold gate on Golden objects
  - goldens_to_paraphrases: deterministic Gold Section id from golden.source_file
  - quantity parameterisation: per_type_count flows through to context list size
  - coverage guarantee: every Gold Section is offered (no silent exclusions)
  - SynthesizerConfig defaults (Demo tier ≈ 50/Core Type)

All tests run offline — no OPENAI_API_KEY, no real LLM calls.
The live Synthesizer call (real API) is opt-in via @pytest.mark.live + OPENAI_API_KEY
(one smoke-test at the bottom). LLM output content is never asserted (CODING_STANDARD §6.2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.paraphrase_comparison.generation.sampling import GoldSection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

_GOLD_SECTIONS = [
    GoldSection("returns_policy.md#return-window", "return-window"),
    GoldSection("returns_policy.md#refund-processing-time", "refund-processing-time"),
    GoldSection("shipping_options.md#standard-delivery", "standard-delivery"),
    GoldSection("shipping_options.md#order-tracking", "order-tracking"),
    GoldSection("payment_methods.md#accepted-cards", "accepted-cards"),
]

_SECTION_BODIES: dict[str, tuple[str, str]] = {
    "returns_policy.md#return-window": (
        "Return Window",
        "You have 30 days to return any item.",
    ),
    "returns_policy.md#refund-processing-time": (
        "Refund Processing Time",
        "Refunds take 5-7 business days.",
    ),
    "shipping_options.md#standard-delivery": (
        "Standard Delivery",
        "Standard delivery takes 3-5 days.",
    ),
    "shipping_options.md#order-tracking": (
        "Order Tracking",
        "Track your order in the account portal.",
    ),
    "payment_methods.md#accepted-cards": (
        "Accepted Cards",
        "We accept Visa, Mastercard, and Amex.",
    ),
}


# ---------------------------------------------------------------------------
# build_section_contexts
# ---------------------------------------------------------------------------


class TestBuildSectionContexts:
    """build_section_contexts maps GoldSections to Synthesizer context lists."""

    def test_returns_one_context_per_section(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_section_contexts,
        )

        contexts, source_files = build_section_contexts(_GOLD_SECTIONS, _SECTION_BODIES)

        assert len(contexts) == len(_GOLD_SECTIONS)
        assert len(source_files) == len(_GOLD_SECTIONS)

    def test_source_files_are_section_ids(self):
        """source_files must be the Gold Section id strings, not filenames."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_section_contexts,
        )

        contexts, source_files = build_section_contexts(_GOLD_SECTIONS, _SECTION_BODIES)

        expected_ids = [s.section_id for s in _GOLD_SECTIONS]
        assert source_files == expected_ids

    def test_each_context_contains_section_body(self):
        """Each context (list[str]) must include the Section body text."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_section_contexts,
        )

        contexts, _ = build_section_contexts(_GOLD_SECTIONS, _SECTION_BODIES)

        for ctx, sec in zip(contexts, _GOLD_SECTIONS):
            heading, body = _SECTION_BODIES[sec.section_id]
            # The combined context strings should include the body somewhere.
            full_text = " ".join(ctx)
            assert body in full_text, (
                f"Section body not found in context for {sec.section_id!r}"
            )

    def test_contexts_are_list_of_string_lists(self):
        """Synthesizer expects List[List[str]] for contexts."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_section_contexts,
        )

        contexts, _ = build_section_contexts(_GOLD_SECTIONS, _SECTION_BODIES)

        for ctx in contexts:
            assert isinstance(ctx, list)
            assert all(isinstance(s, str) for s in ctx)


# ---------------------------------------------------------------------------
# post_filter_by_score
# ---------------------------------------------------------------------------


class TestPostFilterByScore:
    """post_filter_by_score keeps Goldens above the quality threshold."""

    def _make_golden(self, quality_score: float, source_file: str = "a.md#section"):
        """Construct a minimal Golden-like object with a quality score."""
        # We use a real Golden to avoid mocking deep internal types.
        from deepeval.dataset.golden import Golden

        return Golden(
            input="a question",
            source_file=source_file,
            additional_metadata={"synthetic_input_quality": quality_score},
        )

    def test_keeps_goldens_above_threshold(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            post_filter_by_score,
        )

        goldens = [
            self._make_golden(0.9),
            self._make_golden(0.7),
            self._make_golden(0.3),
        ]
        kept = post_filter_by_score(goldens, threshold=0.5)
        assert len(kept) == 2
        for g in kept:
            assert g.additional_metadata["synthetic_input_quality"] >= 0.5

    def test_drops_goldens_below_threshold(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            post_filter_by_score,
        )

        goldens = [self._make_golden(0.1), self._make_golden(0.2)]
        kept = post_filter_by_score(goldens, threshold=0.5)
        assert kept == []

    def test_keeps_all_when_threshold_zero(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            post_filter_by_score,
        )

        goldens = [
            self._make_golden(0.0),
            self._make_golden(0.5),
            self._make_golden(1.0),
        ]
        kept = post_filter_by_score(goldens, threshold=0.0)
        assert len(kept) == 3

    def test_missing_quality_score_drops_golden(self):
        """A Golden without additional_metadata quality key must be dropped (safe default)."""
        from deepeval.dataset.golden import Golden
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            post_filter_by_score,
        )

        g = Golden(input="q", source_file="a.md#s", additional_metadata={})
        kept = post_filter_by_score([g], threshold=0.5)
        assert kept == [], "Golden without quality score should be dropped"

    def test_none_additional_metadata_drops_golden(self):
        """A Golden with additional_metadata=None must be dropped."""
        from deepeval.dataset.golden import Golden
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            post_filter_by_score,
        )

        g = Golden(input="q", source_file="a.md#s")
        kept = post_filter_by_score([g], threshold=0.5)
        assert kept == [], "Golden with None additional_metadata should be dropped"


# ---------------------------------------------------------------------------
# goldens_to_queries
# ---------------------------------------------------------------------------


class TestGoldensToQueries:
    """goldens_to_queries derives Gold Section id deterministically from source_file."""

    def _make_golden(self, source_file: str, input_text: str = "a question") -> object:
        from deepeval.dataset.golden import Golden

        return Golden(
            input=input_text,
            source_file=source_file,
            additional_metadata={"synthetic_input_quality": 0.8},
        )

    def test_source_file_becomes_gold_section_id(self):
        """The golden.source_file is used verbatim as gold_docs_section_id."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            goldens_to_queries,
        )

        section_id = "returns_policy.md#return-window"
        goldens = [self._make_golden(source_file=section_id)]
        queries = goldens_to_queries(
            goldens, paraphrase_type="synonym_swap", id_prefix="syn"
        )
        assert len(queries) == 1
        assert queries[0].gold_docs_section_id == section_id

    def test_query_text_from_golden_input(self):
        """The query text is the golden.input, never the context body."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            goldens_to_queries,
        )

        text = "How long do I have to return an item?"
        goldens = [
            self._make_golden("returns_policy.md#return-window", input_text=text)
        ]
        queries = goldens_to_queries(
            goldens, paraphrase_type="synonym_swap", id_prefix="syn"
        )
        assert queries[0].text == text

    def test_ids_are_sequential_with_prefix(self):
        """Paraphrase ids are '{id_prefix}-{N:03d}' in order."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            goldens_to_queries,
        )

        goldens = [
            self._make_golden("a.md#one"),
            self._make_golden("b.md#two"),
            self._make_golden("c.md#three"),
        ]
        queries = goldens_to_queries(
            goldens, paraphrase_type="word_reorder", id_prefix="wr"
        )
        assert queries[0].paraphrase_id == "wr-001"
        assert queries[1].paraphrase_id == "wr-002"
        assert queries[2].paraphrase_id == "wr-003"

    def test_paraphrase_type_propagated(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            goldens_to_queries,
        )

        goldens = [self._make_golden("a.md#s")]
        queries = goldens_to_queries(
            goldens, paraphrase_type="verbosity_expansion", id_prefix="ve"
        )
        assert queries[0].paraphrase_type == "verbosity_expansion"

    def test_empty_goldens_returns_empty_list(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            goldens_to_queries,
        )

        queries = goldens_to_queries(
            [], paraphrase_type="synonym_swap", id_prefix="syn"
        )
        assert queries == []

    def test_none_source_file_raises(self):
        """A Golden with source_file=None cannot yield a deterministic Gold Section id."""
        from deepeval.dataset.golden import Golden
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            goldens_to_queries,
        )

        g = Golden(input="q", source_file=None)
        with pytest.raises(ValueError, match="source_file"):
            goldens_to_queries([g], paraphrase_type="synonym_swap", id_prefix="syn")


# ---------------------------------------------------------------------------
# SynthesizerConfig defaults
# ---------------------------------------------------------------------------


class TestSynthesizerConfig:
    """SynthesizerConfig encodes the Demo-tier defaults and is parameterisable."""

    def test_default_per_type_count_is_demo_tier(self):
        """Demo tier ≈ 50/Core Type (PRD #137)."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            SynthesizerConfig,
        )

        cfg = SynthesizerConfig()
        assert cfg.per_type_count == 50

    def test_per_type_count_is_parameterisable(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            SynthesizerConfig,
        )

        cfg = SynthesizerConfig(per_type_count=10)
        assert cfg.per_type_count == 10

    def test_generator_model_is_gpt4o_class(self):
        """Generator uses a strong model (gpt-4o class) per PRD #137."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            SynthesizerConfig,
        )

        cfg = SynthesizerConfig()
        assert "gpt-4o" in cfg.generator_model

    def test_critic_model_same_family_as_generator(self):
        """Same-family critic: critic_model must come from the same provider family."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            SynthesizerConfig,
        )

        cfg = SynthesizerConfig()
        # Both must be OpenAI models (same family — PRD #137 justification)
        assert (
            "gpt" in cfg.generator_model.lower() or "o1" in cfg.generator_model.lower()
        )
        assert "gpt" in cfg.critic_model.lower() or "o1" in cfg.critic_model.lower()

    def test_quality_threshold_between_zero_and_one(self):
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            SynthesizerConfig,
        )

        cfg = SynthesizerConfig()
        assert 0.0 <= cfg.quality_threshold <= 1.0


# ---------------------------------------------------------------------------
# Coverage: every Gold Section offered
# ---------------------------------------------------------------------------


class TestCoverageGuarantee:
    """build_section_contexts must offer every Gold Section to the Synthesizer."""

    def test_all_gold_sections_have_context_entry(self):
        """len(contexts) == len(gold_sections) — no silent exclusions."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_section_contexts,
        )

        contexts, source_files = build_section_contexts(_GOLD_SECTIONS, _SECTION_BODIES)

        assert len(contexts) == len(_GOLD_SECTIONS)
        assert set(source_files) == {s.section_id for s in _GOLD_SECTIONS}

    def test_corpus_gold_sections_all_have_bodies(self):
        """derive_gold_sections + _docs_section_bodies cover the same 42 IDs."""
        from eval.paraphrase_comparison.generation.sampling import derive_gold_sections
        from eval.paraphrase_comparison.generate_paraphrases import _docs_section_bodies

        from pathlib import Path

        # We'll use the committed corpus snapshot for this test
        corpus_dir = Path(__file__).resolve().parent.parent / "corpus"
        sections = derive_gold_sections(corpus_dir)
        bodies = _docs_section_bodies()

        # Every Gold Section must have a matching body entry
        missing = [s.section_id for s in sections if s.section_id not in bodies]
        assert not missing, f"Gold Sections with no body in corpus: {missing[:5]}"

    def test_coverage_count_at_least_42(self):
        """The full corpus offers ≥42 Gold Sections (was 28 of 42 in the old bespoke generator)."""
        from eval.paraphrase_comparison.generation.sampling import derive_gold_sections

        corpus_dir = Path(__file__).resolve().parent.parent / "corpus"
        sections = derive_gold_sections(corpus_dir)
        assert len(sections) >= 42, f"Expected ≥42 Gold Sections, got {len(sections)}"


# ---------------------------------------------------------------------------
# build_styling_config / build_filtration_config
# ---------------------------------------------------------------------------


class TestConfigBuilders:
    """Config builders return correct DeepEval config objects."""

    def test_build_styling_config_returns_styling_config_instance(self):
        from deepeval.synthesizer.config import StylingConfig
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_styling_config,
        )

        cfg = build_styling_config()
        assert isinstance(cfg, StylingConfig)

    def test_styling_config_steers_away_from_passage_vocabulary(self):
        """StylingConfig must have a task or scenario that references natural language."""
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_styling_config,
        )

        cfg = build_styling_config()
        combined = " ".join(
            filter(None, [cfg.task, cfg.scenario, cfg.input_format])
        ).lower()
        # Must include at least one of these steering terms
        steering_terms = [
            "customer",
            "question",
            "natural",
            "avoid",
            "vocabulary",
            "rephrase",
        ]
        assert any(term in combined for term in steering_terms), (
            f"StylingConfig must steer toward natural customer questions; got: {combined!r}"
        )

    def test_build_filtration_config_returns_filtration_config_instance(
        self, monkeypatch
    ):
        """FiltrationConfig.__post_init__ calls initialize_model which needs OPENAI_API_KEY.

        We monkeypatch the env var to a dummy value so the model is instantiated
        without a real API call — this tests config structure, not API connectivity.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-offline-test")

        from deepeval.synthesizer.config import FiltrationConfig
        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_filtration_config,
        )

        cfg = build_filtration_config("gpt-4o-mini")
        assert isinstance(cfg, FiltrationConfig)

    def test_filtration_config_quality_threshold_propagated(self, monkeypatch):
        """Threshold is propagated to FiltrationConfig.synthetic_input_quality_threshold."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-offline-test")

        from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
            build_filtration_config,
        )

        cfg = build_filtration_config("gpt-4o-mini", quality_threshold=0.7)
        assert cfg.synthetic_input_quality_threshold == 0.7


# ---------------------------------------------------------------------------
# Live smoke (opt-in — requires OPENAI_API_KEY; gated by @pytest.mark.live)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_synthesizer_generates_query_from_section_context(tmp_path):
    """LIVE: Synthesizer produces ≥1 Golden with a non-empty input and matching source_file.

    This is the ONE live test for this surface (ADR-0005: one live test per LLM surface).
    It exercises the full pipeline against a single Section context and asserts only
    structural properties (input non-empty, source_file propagated) — never content.
    Requires OPENAI_API_KEY. Run with: pytest -m live
    """
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — live test skipped")

    from eval.paraphrase_comparison.generation.synthesizer_pipeline import (
        SynthesizerConfig,
        build_section_contexts,
        build_synthesizer,
        post_filter_by_score,
    )

    config = SynthesizerConfig(per_type_count=1)
    sections = [GoldSection("returns_policy.md#return-window", "return-window")]
    bodies = {
        "returns_policy.md#return-window": (
            "Return Window",
            "You have 30 days to return any item in original condition.",
        )
    }

    contexts, source_files = build_section_contexts(sections, bodies)
    synth = build_synthesizer(config)

    goldens = synth.generate_goldens_from_contexts(
        contexts=contexts,
        source_files=source_files,
        max_goldens_per_context=1,
        include_expected_output=False,
    )
    filtered = post_filter_by_score(goldens, threshold=config.quality_threshold)

    # Structural assertions only — never content
    assert len(goldens) >= 1, "Synthesizer must return ≥1 Golden"
    for g in goldens:
        assert g.input, "Golden.input must be non-empty"
        assert g.source_file == "returns_policy.md#return-window", (
            "Golden.source_file must carry back the Gold Section id"
        )
    # Post-filtered set is a subset
    assert len(filtered) <= len(goldens)
