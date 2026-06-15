"""Unit tests for ``detect_lang`` — the shared CJK-ratio language classifier.

Issue #285 (parent PRD #284): ``detect_lang(text) -> "zh" | "en"`` consolidates
the scattered "is this CJK?" logic (``indexer._is_cjk`` for ADR-0014 bigram
tokenisation, ``retrieval._is_cjk_query`` for #261 threshold routing) into one
tested unit used by both index-time tagging and query-time routing.

Contract (PRD #284 "Implementation Decisions"):
  - Chinese-dominant content  -> "zh"
  - English-dominant content  -> "en"
  - Mixed                     -> dominant language by CJK character ratio
  - empty / whitespace / symbol-only -> a defined default ("en")

Pure function, no I/O — fast and deterministic.
"""

from __future__ import annotations

import pytest

from app.indexer import detect_lang

# The defined default for content with no language signal (empty / whitespace /
# symbols / digits only). English is the default per PRD #284: it is the larger
# corpus and the fail-closed language for the bilingual demo.
DEFAULT_LANG = "en"


# ---------------------------------------------------------------------------
# zh: Chinese-dominant content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "退款需要多久",
        "可以用禮品卡嗎",
        "大量訂購有折扣嗎",
        "我們的退款政策是七天內全額退款。",
    ],
)
def test_chinese_text_is_zh(text):
    assert detect_lang(text) == "zh"


# ---------------------------------------------------------------------------
# en: English-dominant content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "How long does a refund take?",
        "Can I use a gift card?",
        "Our refund policy is a full refund within seven days.",
        "bulk orders qualify for a discount",
    ],
)
def test_english_text_is_en(text):
    assert detect_lang(text) == "en"


# ---------------------------------------------------------------------------
# mixed: dominant-by-ratio
# ---------------------------------------------------------------------------


def test_mixed_chinese_dominant_is_zh():
    # Mostly Chinese with a stray English token — CJK ratio dominates.
    assert detect_lang("退款政策 refund 七天內全額退款，謝謝") == "zh"


def test_mixed_english_dominant_is_en():
    # Mostly English with a stray Chinese token — CJK ratio is below the gate.
    assert detect_lang("Our refund policy gives a full refund within seven days 退款") == "en"


# ---------------------------------------------------------------------------
# default: empty / whitespace / symbol-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\t  \n",
        "!!! ??? ...",
        "1234567890",
        "--- *** ###",
    ],
)
def test_empty_or_symbol_only_is_default(text):
    assert detect_lang(text) == DEFAULT_LANG


# ---------------------------------------------------------------------------
# determinism / purity
# ---------------------------------------------------------------------------


def test_detect_lang_is_pure_and_deterministic():
    text = "退款需要多久 refund timeline"
    assert detect_lang(text) == detect_lang(text)
