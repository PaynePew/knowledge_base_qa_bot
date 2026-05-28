"""L2 cross-family Spot-check tests (external behaviour only, CODING_STANDARD §0.2).

The default suite runs OFFLINE: zone selection is a pure deterministic function
over fixture verdicts (no model call), and the judge call is MOCKED via a stub
client (no live Anthropic call anywhere here — §6.3). Two behaviours are asserted:

  1. **Zone selection** — the right (Paraphrase, Stack) items land in the
     Marginal / Disagreement / Control zones, deterministically.
  2. **Judge wiring + aggregation** — ``run_spotcheck`` assembles the subset,
     calls the (stubbed) judge once per item, and aggregates by-zone agreement
     against the deterministic L1 verdict correctly.
  3. **Flag behaviour** — ``--judge`` set without ``ANTHROPIC_API_KEY`` fail-fasts.
"""

from __future__ import annotations

import pytest

from eval.paraphrase_comparison import spotcheck
from eval.paraphrase_comparison.models import Paraphrase, RetrievedItem
from eval.paraphrase_comparison.spotcheck import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_MODELS,
    ZONE_CONTROL,
    ZONE_DISAGREEMENT,
    ZONE_MARGINAL,
    JudgeUnavailableError,
    JudgeVerdict,
    build_spotcheck_subset,
    run_spotcheck,
)

# ---------------------------------------------------------------------------
# Fixture verdicts: hand-built so each zone has a known, deterministic member.
# ---------------------------------------------------------------------------
GOLD = "returns_policy.md#return-window"
OTHER = "shipping_options.md#standard-delivery"
KEY = ["refund", "packaging", "receipt", "thirty", "days"]


def _para(pid: str, gold: str = GOLD) -> Paraphrase:
    return Paraphrase(
        paraphrase_id=pid,
        paraphrase_type="synonym_swap",
        text=f"query for {pid}",
        gold_docs_section_id=gold,
        key_tokens_docs=KEY,
        key_tokens_wiki=[],
    )


def _item(source: str, content: str) -> RetrievedItem:
    return RetrievedItem(source_section_id=source, content=content)


def _retriever(table: dict[str, list[RetrievedItem]]):
    """Build a retrieval callable returning the fixture items for a paraphrase text."""

    def retrieve(query: str, k: int = 3) -> list[RetrievedItem]:
        return list(table.get(query, []))[:k]

    return retrieve


# ---------------------------------------------------------------------------
# Zone selection — deterministic, no judge call
# ---------------------------------------------------------------------------
def test_marginal_zone_picks_correct_id_with_overlap_at_or_below_threshold():
    # top-1 matches the gold AND shares exactly ONE key token -> marginal hit.
    para = _para("syn-1")
    a = _retriever({para.text: [_item(GOLD, "issued a refund yesterday")]})  # 1 token
    b = _retriever({para.text: [_item(GOLD, "refund packaging receipt thirty")]})  # >1

    subset = build_spotcheck_subset(
        [para], a, b, zones=(ZONE_MARGINAL,), marginal_threshold=1
    )
    marginal_a = [it for it in subset if it.stack == "Stack A"]
    assert marginal_a, "Stack A top-1 with single-token overlap is a marginal hit"
    assert ZONE_MARGINAL in marginal_a[0].zones
    # Stack B's top-1 overlaps 4 key tokens (> threshold) -> NOT marginal.
    assert not [it for it in subset if it.stack == "Stack B"]


def test_marginal_zone_excludes_zero_overlap_clear_miss():
    # top-1 matches gold id but shares NO key token -> a clear miss, not marginal.
    para = _para("syn-2")
    a = _retriever({para.text: [_item(GOLD, "entirely unrelated body text")]})
    b = _retriever({para.text: []})
    subset = build_spotcheck_subset([para], a, b, zones=(ZONE_MARGINAL,))
    assert subset == []


def test_disagreement_zone_when_stack_top1_verdicts_differ():
    para = _para("syn-3")
    # Stack A hits (gold + token), Stack B misses (wrong id) -> disagreement.
    a = _retriever(
        {para.text: [_item(GOLD, "refund and packaging within thirty days")]}
    )
    b = _retriever({para.text: [_item(OTHER, "fast delivery options")]})
    subset = build_spotcheck_subset([para], a, b, zones=(ZONE_DISAGREEMENT,))
    stacks = {it.stack for it in subset}
    assert stacks == {"Stack A", "Stack B"}
    for it in subset:
        assert ZONE_DISAGREEMENT in it.zones
    a_item = next(it for it in subset if it.stack == "Stack A")
    b_item = next(it for it in subset if it.stack == "Stack B")
    assert a_item.l1_hit is True and b_item.l1_hit is False


def test_no_disagreement_when_both_stacks_agree():
    para = _para("syn-4")
    # both hit -> no disagreement member.
    a = _retriever({para.text: [_item(GOLD, "refund packaging thirty days receipt")]})
    b = _retriever({para.text: [_item(GOLD, "refund packaging thirty days")]})
    subset = build_spotcheck_subset([para], a, b, zones=(ZONE_DISAGREEMENT,))
    assert subset == []


def test_control_zone_is_seeded_and_reproducible():
    # 6 clear hits + 6 clear misses -> control samples 5 of each deterministically.
    paras = []
    a = {}
    b = {}
    for i in range(6):
        p = _para(f"hit-{i}")
        paras.append(p)
        a[p.text] = [_item(OTHER, "noise")]
        b[p.text] = [_item(GOLD, "refund packaging receipt thirty days")]  # clear hit
    for i in range(6):
        p = _para(f"miss-{i}")
        paras.append(p)
        a[p.text] = [_item(OTHER, "noise")]
        b[p.text] = [_item(OTHER, "totally unrelated content")]  # clear miss

    kwargs = dict(zones=(ZONE_CONTROL,), control_sample_size=5)
    sub1 = build_spotcheck_subset(paras, _retriever(a), _retriever(b), **kwargs)
    sub2 = build_spotcheck_subset(paras, _retriever(a), _retriever(b), **kwargs)

    # 5 clear-hit + 5 clear-miss = 10 control members, all Stack B.
    assert len(sub1) == 10
    assert all(ZONE_CONTROL in it.zones and it.stack == "Stack B" for it in sub1)
    hits = [it for it in sub1 if it.l1_hit]
    misses = [it for it in sub1 if not it.l1_hit]
    assert len(hits) == 5 and len(misses) == 5
    # Seeded -> two builds pick the IDENTICAL members.
    assert [it.paraphrase_id for it in sub1] == [it.paraphrase_id for it in sub2]


def test_item_in_multiple_zones_carries_all_zones():
    # A marginal hit on Stack A that also disagrees with Stack B.
    para = _para("syn-5")
    a = _retriever(
        {para.text: [_item(GOLD, "a single refund mention")]}
    )  # marginal hit
    b = _retriever(
        {para.text: [_item(OTHER, "shipping speeds")]}
    )  # miss -> disagreement
    subset = build_spotcheck_subset(
        [para], a, b, zones=(ZONE_MARGINAL, ZONE_DISAGREEMENT), marginal_threshold=1
    )
    a_item = next(it for it in subset if it.stack == "Stack A")
    assert set(a_item.zones) == {ZONE_MARGINAL, ZONE_DISAGREEMENT}
    # The item is still a single judged unit (de-duplicated by paraphrase+stack).
    assert len([it for it in subset if it.stack == "Stack A"]) == 1


# ---------------------------------------------------------------------------
# Judge wiring — MOCKED (no live Anthropic call), aggregation correctness
# ---------------------------------------------------------------------------
class _StubJudgeClient:
    """Records each judge call and replies per a substring->answers map (no network).

    Keys are matched as substrings of the assembled user prompt (which embeds both
    the QUESTION and the CONTENT), so a test can target a specific item by its
    distinctive content. First matching key wins; default is ``False``.
    """

    def __init__(self, answers_by_substring: dict[str, bool]):
        self._answers = answers_by_substring
        self.calls: list[dict] = []
        self.messages = self  # client.messages.create(...) entry point

    def create(self, *, model, max_tokens, system, messages):
        self.calls.append({"model": model, "messages": messages})
        # Echo a verdict the module's _parse_verdict can read.
        user = messages[0]["content"]
        answers = next(
            (v for substring, v in self._answers.items() if substring in user), False
        )
        return _StubResponse(answers)


class _StubResponse:
    def __init__(self, answers: bool):
        import json

        payload = json.dumps({"answers": answers, "reasoning": "stub"})
        self.content = [_StubBlock(payload)]


class _StubBlock:
    def __init__(self, text: str):
        self.text = text


def test_run_spotcheck_calls_judge_per_item_and_aggregates_agreement(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Two paraphrases: one disagreement pair (A hit / B miss), judged so the
    # judge AGREES with both L1 verdicts; control adds clear hits/misses.
    p_dis = _para("syn-dis")
    a = {p_dis.text: [_item(GOLD, "refund packaging thirty days receipt")]}  # A hit
    b = {p_dis.text: [_item(OTHER, "shipping content")]}  # B miss

    # Judge answers keyed by distinctive content: agrees with L1 (A-item content
    # answers -> True == L1 hit; B-item content does not -> False == L1 miss).
    answers = {"refund packaging thirty days receipt": True}
    stub = _StubJudgeClient(answers_by_substring=answers)
    monkeypatch.setattr(spotcheck, "_judge_client", lambda: stub)

    result = run_spotcheck(
        [p_dis],
        _retriever(a),
        _retriever(b),
        zones=(ZONE_DISAGREEMENT,),
    )

    # Judge was called once per subset item (the disagreement pair = 2 items).
    assert len(stub.calls) == result.total_size == 2
    # Stack A item: L1 hit=True, judge answers True -> agree.
    # Stack B item: L1 hit=False, judge answers False (query not in map) -> agree.
    assert result.agreement_by_zone[ZONE_DISAGREEMENT] == 1.0
    assert result.subset_size_by_zone[ZONE_DISAGREEMENT] == 2
    assert result.judge_model == DEFAULT_JUDGE_MODEL
    # No Anthropic type leaks: verdicts are primitive (bool + str).
    for _item_, verdict in result.verdicts:
        assert isinstance(verdict, JudgeVerdict)
        assert isinstance(verdict.answers, bool)
        assert isinstance(verdict.reasoning, str)


def test_run_spotcheck_reports_disagreement_when_judge_differs_from_l1(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # L1 says Stack B is a clear hit; judge says it does NOT answer -> 0 agreement.
    p = _para("syn-control")
    a = {p.text: [_item(OTHER, "noise")]}
    b = {p.text: [_item(GOLD, "refund packaging receipt thirty days")]}  # clear hit
    stub = _StubJudgeClient(answers_by_substring={})  # judge always answers False
    monkeypatch.setattr(spotcheck, "_judge_client", lambda: stub)

    result = run_spotcheck(
        [p], _retriever(a), _retriever(b), zones=(ZONE_CONTROL,), control_sample_size=5
    )
    # The single clear-hit control item: L1=True, judge=False -> 0% agreement,
    # which is exactly the mis-calibration signal the control zone exists to flag.
    assert result.subset_size_by_zone[ZONE_CONTROL] == 1
    assert result.agreement_by_zone[ZONE_CONTROL] == 0.0


def test_judge_models_are_the_three_documented_choices():
    assert JUDGE_MODELS == ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7")
    assert DEFAULT_JUDGE_MODEL == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Flag behaviour — fail-fast without the key, no live call
# ---------------------------------------------------------------------------
def test_run_spotcheck_fails_fast_without_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    para = _para("syn-x")
    with pytest.raises(JudgeUnavailableError) as exc:
        run_spotcheck([para], _retriever({}), _retriever({}))
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_run_spotcheck_fails_fast_when_key_is_empty(monkeypatch):
    # An empty string must be treated as absent (the real environment here).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(JudgeUnavailableError):
        run_spotcheck([_para("syn-y")], _retriever({}), _retriever({}))
