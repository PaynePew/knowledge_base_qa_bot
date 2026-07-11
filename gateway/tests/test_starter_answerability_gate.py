"""Offline answerability probe for the reader UI's starter presets (issue #571).

Companion to ``test_ui_bilingual_starters.py`` (issue #289 / PRD #284). That
file's ``test_starter_presets_clear_pre_llm_gate`` pins only the TOP-1 BM25
score against the pre-LLM Cannot-Confirm threshold (ADR-0001). Issue #571
names the gap that check cannot see: ``markdown_kb.app.retrieval
._retrieve_and_gate`` score-gates on ``ranked[0]`` alone, but the answer
synthesis stage receives the FULL top-``k=3`` window (every entry in
``ranked`` becomes a ``sources`` entry the LLM/verifier can cite). A preset
can clear the top-1 score gate while the Section that actually carries the
answer fact sits at rank 2 or 3 -- or the top-1 Section can be on a different
topic entirely, with the real fact only reachable further down the window.
Neither failure mode is visible to a threshold-only check.

This module implements item 1 of #571 only: per starter preset, assert that
the CONTENT of the top-``k`` BM25 hits (the same ``k=3`` the production
pre-LLM gate retrieves) contains a pinned fact substring lifted verbatim from
the corpus -- no LLM call, no network, a pure read of the committed
``.kb/index.json`` seed. Item 2 (an optional live smoke in the deploy gate)
is explicitly deferred by the issue's scope decision and is NOT implemented
here.

Per CODING_STANDARD §6.2 ("Don't assert: BM25 score absolute values"), this
probe does not re-assert raw score magnitudes -- the sibling file already
owns that check. It asserts CONTENT PRESENCE within the retrieved window,
which §6.2 allows explicitly ("Section IDs, ranking order" / "exact literal
sentinel strings").

The preset question strings below are pinned copies of the ``PRESET_QUESTIONS``
entries in ``gateway/static/index.html``, matching the sibling file's own
pinned copies. This file does not re-verify their presence in the UI text --
that is AC1 of #289, already covered by ``test_ui_bilingual_starters.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The committed baked BM25 seed. See test_ui_bilingual_starters.py's module
# docstring for why this probe reads it directly rather than through the
# gateway suite's autouse INDEX_PATH-to-tmp redirect (#303 write-isolation):
# it is a READ-ONLY check of the committed seed, and load_index_json's
# index_loaded log entry still goes to the redirected tmp LOG_PATH, so the
# probe writes nothing.
_COMMITTED_INDEX = Path(__file__).resolve().parents[2] / ".kb" / "index.json"

# The same k the production pre-LLM gate retrieves at
# (markdown_kb.app.retrieval._retrieve_and_gate calls indexer.search(question,
# k=3)) -- every one of these k Sections reaches the answer-synthesis prompt,
# not just rank 0, so answerability must be checked across the whole window.
_TOP_K = 3

# preset question -> a fact substring lifted verbatim from the corpus Section
# that actually answers it. The fact may rank anywhere in the top-k window,
# not necessarily rank 0 -- see module docstring.
ZH_PRESET_FACTS = {
    "退款要幾天才會退到帳戶？": "5-7 個工作天",
    "你們接受哪些付款方式？": "VISA、MasterCard 和 JCB",
    "紅利點數怎麼累積？": "每消費新台幣 100 元累積 1 點",
    "你們配送到哪些國家？": "日本、韓國、香港、澳門、新加坡、馬來西亞、美國、加拿大",
}
EN_PRESET_FACTS = {
    "How long do refunds take?": "reviewed within 3 business days",
    "What payment methods do you accept?": "VISA, MasterCard, and JCB",
    "How do I earn reward points?": "Earn 1 point per NT$100 spent",
    "Which countries do you ship to?": (
        "Japan, South Korea, Hong Kong, Macau, Singapore, Malaysia, the United States, and Canada"
    ),
}


def _assert_top_k_contains_fact(
    preset_facts: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert every preset's pinned fact substring appears somewhere in its top-k window."""
    import markdown_kb.app.indexer as ix

    # Snapshot/restore the module-level sections list (CODING_STANDARD §6.5):
    # this probe loads the COMMITTED index by explicit path rather than
    # through the per-test tmp INDEX_PATH redirect, so the load below must
    # not leak into other tests that assume a fresh/empty index.
    monkeypatch.setattr(ix, "sections", ix.sections)

    _files, sections = ix.load_index_json(_COMMITTED_INDEX)
    assert sections > 0, "baked .kb/index.json must be present and non-empty for the probe"

    for question, fact in preset_facts.items():
        hits = ix.search(question, _TOP_K)
        assert hits, f"preset returned no BM25 hits (would Cannot-Confirm): {question!r}"
        window = "\n".join(sec.content for sec, _score in hits)
        assert fact in window, (
            f"expected answer fact {fact!r} not found in the top-{_TOP_K} retrieved "
            f"content for preset {question!r} -- the answer would not be groundable "
            f"even if some Section in the window clears the score gate"
        )


def test_en_starter_presets_top_k_contains_expected_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every EN starter preset's answer fact is present in its top-k retrieval window (#571 item 1)."""
    _assert_top_k_contains_fact(EN_PRESET_FACTS, monkeypatch)


def test_zh_starter_presets_top_k_contains_expected_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ZH starter preset's answer fact is present in its top-k retrieval window (#571 item 1)."""
    _assert_top_k_contains_fact(ZH_PRESET_FACTS, monkeypatch)
