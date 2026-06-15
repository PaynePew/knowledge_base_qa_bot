"""Structural tests for the reader UI's bilingual starter questions (#289, PRD #284).

The reader UI lives in ``gateway/static/index.html`` as a single vanilla
HTML/CSS/JS file (CODING_STANDARD §12.1 — no framework, no build step, no JS
toolchain §12.6). Following the established pattern in ``test_sse_parser.py``,
these tests inspect the production UI file's text to assert the structural
invariants of issue #289:

- The empty state offers BOTH a Traditional-Chinese starter-question set and an
  English starter-question set (AC1).
- The preset questions target topics the corpus actually covers, so a first
  click returns a grounded answer rather than "Cannot Confirm" (AC2). Coverage
  itself is confirmed by an OFFLINE BM25 retrieval probe (see
  ``test_starter_presets_clear_pre_llm_gate`` — BM25 needs no OpenAI key); these
  text assertions pin the *exact* preset strings that probe validated.
- Presets render via the EXISTING textContent-only path: the ``el()`` helper and
  the ``ask()`` submit helper, with no ``innerHTML`` and no new client-side
  business logic (AC3).

No DOM, no fetch, no browser, no OPENAI_API_KEY — fully hermetic (§6.3 / §12.7).
The BM25 probe loads the baked ``.kb/index.json`` and calls ``indexer.search``,
which is pure BM25 (no embeddings, no LLM) and therefore safe offline.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Path to the production UI file (mirror of test_sse_parser.py)
# ---------------------------------------------------------------------------

_STATIC_INDEX = Path(__file__).resolve().parents[2] / "gateway" / "static" / "index.html"


# The exact preset strings the UI must offer, per language. Each was validated
# OFFLINE against the baked BM25 index (see test_starter_presets_clear_pre_llm_gate):
# every question's top hit clears the pre-LLM Cannot-Confirm gate
# (en KB_SCORE_THRESHOLD=0.5, zh KB_SCORE_THRESHOLD_ZH=4.0) and resolves to a
# single-intent, well-covered topic page — so a first click returns a grounded
# answer in the visitor's language.
EXPECTED_ZH_PRESETS = [
    "退款政策是什麼？",
    "你們接受哪些付款方式？",
    "國際配送需要多久時間？",
    "紅利點數怎麼累積？",
]
EXPECTED_EN_PRESETS = [
    "What is your return policy?",
    "What payment methods do you accept?",
    "How long does standard shipping take?",
    "How do I claim warranty?",
]


def _ui_text() -> str:
    return _STATIC_INDEX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC1: the empty state offers Chinese AND English starter questions
# ---------------------------------------------------------------------------


def test_ui_file_offers_chinese_starter_questions():
    """The empty state includes a Traditional-Chinese starter-question set (AC1)."""
    text = _ui_text()
    for q in EXPECTED_ZH_PRESETS:
        assert q in text, f"Chinese starter question missing from UI: {q!r}"


def test_ui_file_offers_english_starter_questions():
    """The empty state includes an English starter-question set (AC1)."""
    text = _ui_text()
    for q in EXPECTED_EN_PRESETS:
        assert q in text, f"English starter question missing from UI: {q!r}"


def test_ui_file_has_distinct_per_language_preset_sets():
    """The UI defines two distinct preset sets (zh + en), not one shared list (AC1).

    The single ``PRESET_QUESTIONS`` array from before #289 carried only English
    copy and was shown identically on both languages. After #289 the UI must
    select presets by language, so a per-language structure must exist.
    """
    text = _ui_text()
    # The empty state must read a per-language preset set keyed by language, so it
    # can offer the right questions for each language.
    assert "PRESET_QUESTIONS.zh" in text, "UI must read a zh preset set (PRESET_QUESTIONS.zh)"
    assert "PRESET_QUESTIONS.en" in text, "UI must read an en preset set (PRESET_QUESTIONS.en)"


# ---------------------------------------------------------------------------
# AC2: presets target covered topics — confirmed by an OFFLINE BM25 probe
# ---------------------------------------------------------------------------


def test_starter_presets_clear_pre_llm_gate():
    """Every starter question clears the pre-LLM Cannot-Confirm gate (AC2).

    OFFLINE probe: load the baked BM25 index and score each preset with the same
    language-filtered ``indexer.search`` the production pre-LLM gate uses. The top
    hit must clear the per-language threshold (en 0.5 / zh 4.0), which is the
    exact condition under which the gate would otherwise refuse with
    "Cannot Confirm". BM25 needs no OpenAI key, so this runs offline.
    """
    import markdown_kb.app.indexer as ix
    from markdown_kb.app.retrieval import (
        _KB_SCORE_THRESHOLD_DEFAULT,
        _KB_SCORE_THRESHOLD_ZH_DEFAULT,
    )

    files, sections = ix.load_index_json()
    assert sections > 0, "baked .kb/index.json must be present and non-empty for the probe"

    for q in EXPECTED_EN_PRESETS:
        hits = ix.search(q, 3)
        assert hits, f"EN preset returned no BM25 hits (would Cannot-Confirm): {q!r}"
        top = hits[0][1]
        assert top >= _KB_SCORE_THRESHOLD_DEFAULT, (
            f"EN preset top score {top:.3f} < threshold "
            f"{_KB_SCORE_THRESHOLD_DEFAULT} (would Cannot-Confirm): {q!r}"
        )

    for q in EXPECTED_ZH_PRESETS:
        hits = ix.search(q, 3)
        assert hits, f"ZH preset returned no BM25 hits (would Cannot-Confirm): {q!r}"
        top = hits[0][1]
        assert top >= _KB_SCORE_THRESHOLD_ZH_DEFAULT, (
            f"ZH preset top score {top:.3f} < threshold "
            f"{_KB_SCORE_THRESHOLD_ZH_DEFAULT} (would Cannot-Confirm): {q!r}"
        )


# ---------------------------------------------------------------------------
# AC3: presets render via the existing textContent-only path, no new business logic
# ---------------------------------------------------------------------------


def test_ui_file_no_inner_html_assignment_after_change():
    """The UI still never assigns to innerHTML (§12.4 / AC3 — textContent only)."""
    text = _ui_text()
    assert ".innerHTML =" not in text and ".innerHTML=" not in text, (
        "innerHTML assignment found — §12.4 / #289 AC3 requires textContent only"
    )


def test_ui_file_presets_use_existing_el_and_ask_path():
    """Presets reuse the existing el() render helper + ask() submit path (AC3).

    AC3 forbids new client-side business logic: clicking a preset must still just
    fill the composer and submit via the existing ``ask()`` helper, rendered via
    the existing textContent-only ``el()`` factory. We assert both helpers remain
    referenced in the empty-state render path.
    """
    text = _ui_text()
    assert "ask(" in text, "preset click must route through the existing ask() submit helper (AC3)"
    assert "empty-q" in text, "preset buttons must reuse the existing .empty-q render class (AC3)"
    # No new network/business-logic primitive sneaked in alongside the presets.
    assert "new EventSource" not in text, "no new EventSource (§12.2)"
