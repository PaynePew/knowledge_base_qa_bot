"""Paraphrase schema parsing tests (external behaviour only, CODING_STANDARD §0.2).

Covers the on-disk ``queries.yaml`` round-trip through ``load_paraphrases`` and
the generator's structured-output stitching (``gen_schema.to_paraphrase``). The
schema field names are load-bearing — ``loader.py`` reads them by name and the
LLM's structured output must match (CONTEXT.md § Phase 8 > Paraphrase).
"""

from __future__ import annotations

import textwrap

from eval.paraphrase_comparison.generation.gen_schema import (
    ParaphraseDraft,
    to_paraphrase,
)
from eval.paraphrase_comparison.loader import load_paraphrases
from eval.paraphrase_comparison.models import Paraphrase


def test_loader_parses_paraphrase_fields(tmp_path):
    yaml_text = textwrap.dedent(
        """
        metadata:
          generator_model: gpt-4o-mini
          total: 1
        paraphrases:
          - paraphrase_id: synonym_swap-001
            paraphrase_type: synonym_swap
            text: "How long for a refund?"
            gold_docs_section_id: returns_policy.md#return-window
            key_tokens_docs: [return, refund, days]
            key_tokens_wiki: [thirty, refund, packaging]
            generation_notes: "note"
        """
    )
    path = tmp_path / "queries.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    paraphrases = load_paraphrases(path)
    assert len(paraphrases) == 1
    p = paraphrases[0]
    assert isinstance(p, Paraphrase)
    assert p.paraphrase_id == "synonym_swap-001"
    assert p.paraphrase_type == "synonym_swap"
    assert p.gold_docs_section_id == "returns_policy.md#return-window"
    assert p.key_tokens_docs == ["return", "refund", "days"]
    assert p.key_tokens_wiki == ["thirty", "refund", "packaging"]
    assert p.generation_notes == "note"


def test_generation_notes_defaults_to_empty_when_absent(tmp_path):
    yaml_text = textwrap.dedent(
        """
        paraphrases:
          - paraphrase_id: word_reorder-001
            paraphrase_type: word_reorder
            text: "Tracking number — where?"
            gold_docs_section_id: shipping_options.md#order-tracking
            key_tokens_docs: [tracking, number]
            key_tokens_wiki: [tracking, parcel]
        """
    )
    path = tmp_path / "queries.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    p = load_paraphrases(path)[0]
    assert p.generation_notes == ""


def test_key_tokens_property_is_lowercased_union():
    p = Paraphrase(
        paraphrase_id="t1",
        paraphrase_type="synonym_swap",
        text="x",
        gold_docs_section_id="a.md#b",
        key_tokens_docs=["Refund", "Days"],
        key_tokens_wiki=["refund", "Thirty"],
    )
    # The dual-side union the C5c metric uses — lowercased, de-duplicated.
    assert p.key_tokens == {"refund", "days", "thirty"}


def test_to_paraphrase_stitches_generator_owned_fields():
    draft = ParaphraseDraft(
        text="When does my reimbursement land?",
        key_tokens_docs=["refund", "business", "days"],
        key_tokens_wiki=["refund", "inspection"],
        generation_notes="targeted sub-fact = processing time",
    )
    p = to_paraphrase(
        draft,
        paraphrase_id="synonym_swap-002",
        paraphrase_type="synonym_swap",
        gold_docs_section_id="returns_policy.md#refund-processing-time",
    )
    # The LLM never supplies id/type/gold — the generator owns them.
    assert p.paraphrase_id == "synonym_swap-002"
    assert p.paraphrase_type == "synonym_swap"
    assert p.gold_docs_section_id == "returns_policy.md#refund-processing-time"
    assert p.text == "When does my reimbursement land?"
    assert p.key_tokens_docs == ["refund", "business", "days"]


def test_committed_queries_yaml_round_trips():
    # The real committed set must parse without raising and carry the expected
    # field shapes on every entry.
    paraphrases = load_paraphrases()
    assert paraphrases
    for p in paraphrases:
        assert p.paraphrase_id
        assert p.paraphrase_type
        assert p.text
        assert p.gold_docs_section_id
        assert p.key_tokens_docs and p.key_tokens_wiki
