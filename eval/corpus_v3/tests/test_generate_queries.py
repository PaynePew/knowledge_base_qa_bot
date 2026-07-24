"""Corpus v3 query generator orchestration tests — external behaviour only
(CODING_STANDARD §0.2). Every LLM call is faked (CODING_STANDARD §6.3-style
seam swap); no network call, no OPENAI_API_KEY required, no LLM output
content is asserted (§6.2) -- only counts, cost-guard wiring, and the
artifact's structural shape.

Covers issue #672.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import yaml

from eval.corpus_v3.generation import generate_queries as gq
from eval.corpus_v3.generation.gen_schema import QueryDraft
from eval.corpus_v3.generation.targets import GenerationTarget
from eval.corpus_v3.query_schema import Query, dump_queries


@dataclass
class _FakeRaw:
    usage_metadata: dict


class _FakeLLM:
    """Deterministic stand-in for ``_get_family_a_llm()``'s
    ``with_structured_output(..., include_raw=True)`` chain. Every call
    returns a well-formed, QC-passing draft unless ``reject_every`` is set,
    in which case every draft carries all-stopword key tokens (the QC gate's
    cheapest reliable rejection trigger)."""

    def __init__(
        self,
        *,
        reject_every: bool = False,
        input_tokens: int = 100,
        output_tokens: int = 50,
    ):
        self.reject_every = reject_every
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        key_tokens = ["the", "a"] if self.reject_every else ["reference", "passage"]
        draft = QueryDraft(
            text="What does the reference passage say?",
            key_tokens=key_tokens,
            generation_notes="",
        )
        raw = _FakeRaw(
            usage_metadata={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            }
        )
        return {"raw": raw, "parsed": draft}


def _target(stratum="factoid") -> GenerationTarget:
    return GenerationTarget(
        scenario_stratum=stratum,
        group_id="g1",
        heading="H",
        gold_section_ids=["a.md#h"] if stratum != "unanswerable" else [],
        reference_ids=["a.md#h"],
        reference_text="The reference passage body about returns.",
    )


# ---------------------------------------------------------------------------
# Offline refusal
# ---------------------------------------------------------------------------
def test_main_refuses_without_openai_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(gq, "load_dotenv", lambda *a, **k: None)
    out_path = tmp_path / "queries.yaml"
    code = gq.main(["--output", str(out_path)])
    assert code == 1
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# Human slice loading
# ---------------------------------------------------------------------------
def test_load_human_slice_returns_empty_list_when_file_absent(tmp_path):
    assert gq.load_human_slice(tmp_path / "missing.yaml") == []


def test_load_human_slice_loads_and_validates_present_file(tmp_path):
    path = tmp_path / "human_slice.yaml"
    query = Query(
        query_id="human-001",
        text="What is the return window?",
        scenario_stratum="factoid",
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=["a.md#h"],
        key_tokens=["return", "window"],
        generating_family="human",
    )
    dump_queries([query], path)
    loaded = gq.load_human_slice(path)
    assert loaded == [query]


def test_load_human_slice_raises_on_malformed_file(tmp_path):
    path = tmp_path / "human_slice.yaml"
    path.write_text("queries:\n  - text: no other fields\n", encoding="utf-8")
    with pytest.raises(ValueError):
        gq.load_human_slice(path)


# ---------------------------------------------------------------------------
# Generation loop (deterministic seams, fake LLM)
# ---------------------------------------------------------------------------
def test_run_generation_accepts_well_formed_drafts():
    llm = _FakeLLM()
    ledger = gq.CostLedger()
    cells = [("factoid", "en", 3)]
    monkeypatch_targets = {
        "factoid": [_target("factoid")],
        "cross_doc": [],
        "version_conflict": [],
        "unanswerable": [],
    }

    import eval.corpus_v3.generation.generate_queries as mod

    original = mod.derive_generation_targets
    mod.derive_generation_targets = lambda groups: monkeypatch_targets
    try:
        queries, counts = gq.run_generation(llm, ledger, cells=cells)
    finally:
        mod.derive_generation_targets = original

    assert len(queries) == 3
    assert counts[0].target == 3
    assert counts[0].actual == 3
    assert counts[0].qc_rejected == 0
    assert llm.calls == 3
    assert ledger.totals(phase="query").calls == 3


def test_run_generation_drops_qc_rejected_drafts_and_counts_them():
    llm = _FakeLLM(reject_every=True)
    ledger = gq.CostLedger()
    cells = [("factoid", "en", 2)]

    import eval.corpus_v3.generation.generate_queries as mod

    original = mod.derive_generation_targets
    mod.derive_generation_targets = lambda groups: {
        "factoid": [_target("factoid")],
        "cross_doc": [],
        "version_conflict": [],
        "unanswerable": [],
    }
    try:
        queries, counts = gq.run_generation(llm, ledger, cells=cells)
    finally:
        mod.derive_generation_targets = original

    assert queries == []
    assert counts[0].actual == 0
    assert counts[0].qc_rejected == 2


# ---------------------------------------------------------------------------
# Cost guard wiring (end-to-end main(), fake LLM, real cost math)
# ---------------------------------------------------------------------------
def test_main_halts_and_writes_nothing_when_projected_spend_exceeds_cap(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(gq, "load_dotenv", lambda *a, **k: None)
    # Expensive fake calls (gpt-4o-mini pricing) so a small pilot already
    # projects well past the $10 cap over the real 4,236-query plan.
    monkeypatch.setattr(
        gq,
        "_get_family_a_llm",
        lambda: _FakeLLM(input_tokens=50_000, output_tokens=50_000),
    )
    out_path = tmp_path / "queries.yaml"
    human_slice_path = _human_slice_file(tmp_path)
    code = gq.main(
        [
            "--output",
            str(out_path),
            "--human-slice",
            str(human_slice_path),
            "--pilot-calls",
            "2",
        ]
    )
    assert code == 1
    assert not out_path.exists()


def test_main_refuses_when_no_family_b_source_is_available(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(gq, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(gq, "_get_family_a_llm", lambda: _FakeLLM())
    out_path = tmp_path / "queries.yaml"
    code = gq.main(
        [
            "--output",
            str(out_path),
            "--human-slice",
            str(tmp_path / "no-such-human-slice.yaml"),
        ]
    )
    assert code == 1
    assert not out_path.exists()


def _human_slice_file(tmp_path) -> object:
    path = tmp_path / "human_slice.yaml"
    query = Query(
        query_id="human-001",
        text="What is the store's return window?",
        scenario_stratum="factoid",
        overlap_stratum="high_overlap",
        language="en",
        gold_section_ids=["a.md#h"],
        key_tokens=["return", "window"],
        generating_family="human",
    )
    dump_queries([query], path)
    return path


def test_main_writes_artifact_with_deviations_when_qc_rejects_some(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(gq, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(gq, "_get_family_a_llm", lambda: _FakeLLM(reject_every=True))

    small_targets = {
        "factoid": [_target("factoid")],
        "cross_doc": [_target("cross_doc")],
        "version_conflict": [_target("version_conflict")],
        "unanswerable": [_target("unanswerable")],
    }
    monkeypatch.setattr(gq, "derive_generation_targets", lambda groups: small_targets)
    monkeypatch.setattr(gq, "EN_TARGET_PER_STRATUM", 2)
    monkeypatch.setattr(gq, "ZH_TARGET_PER_STRATUM", 1)

    out_path = tmp_path / "queries.yaml"
    human_slice_path = _human_slice_file(tmp_path)
    code = gq.main(
        [
            "--output",
            str(out_path),
            "--human-slice",
            str(human_slice_path),
            "--pilot-calls",
            "1",
        ]
    )
    assert code == 0
    assert out_path.exists()
    data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    # 1 human-authored query survives; every LLM-generated draft was QC-rejected.
    assert len(data["queries"]) == 1
    assert data["queries"][0]["generating_family"] == "human"
    assert data["metadata"]["deviations"]  # every LLM cell fell short -> non-empty
    assert data["metadata"]["generator_family_b"] == "human"
