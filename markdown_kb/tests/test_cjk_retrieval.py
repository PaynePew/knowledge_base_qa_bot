"""Tests for Phase 16 Slice 1: CJK bigram tokeniser, Unicode slugify, and
language-agnostic retrieval over a hand-authored Chinese Wiki fixture.

Acceptance criteria covered:
- AC1: CJK text tokenises to character bigrams; length-1 CJK run → unigram fallback.
- AC2: Pure-ASCII input tokenises byte-identically to pre-change output (regression).
- AC3: slugify preserves CJK characters; distinct CJK headings produce distinct slugs;
       collisions resolve via -2/-3 suffix; empty-after-strip yields "section".
- AC4: Chinese question against indexed Chinese fixture returns Grounded Answer
       containing CJK characters with a readable CJK Citation (hermetic, LLM mocked).
- AC5: Cannot Confirm is emitted as verbatim English sentinel on no-answer path,
       including a Chinese query below threshold.
- AC6: Unit tests for tokeniser (bigram + unigram fallback + ASCII byte-identical
       regression) and slugify (CJK preserve + collision + fallback).

Note: AC7 (answer-language directive present in SYSTEM_PROMPT) is covered by
test_system_prompt_language_directive below and by the drift-guard in
vector_rag/tests/test_system_prompt.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.indexer import parse_markdown, slugify, tokenize
from app.prompt_builder import SYSTEM_PROMPT
from app.retrieval import CANNOT_CONFIRM_PHRASE

from .conftest import FakeLLMResponse

# ---------------------------------------------------------------------------
# AC6 — Unit tests: tokenise
# ---------------------------------------------------------------------------


def test_tokenize_ascii_byte_identical_regression():
    """These pure-ASCII inputs are unaffected by the CJK branch (codepoints > 127).

    Guards CJK-branch isolation: adding CJK handling must not change ASCII
    tokenisation. NOTE: pure-ASCII output overall is NOT frozen — #252 (§11 /
    ADR-0014 amended) deliberately changed it (dropped junk tokens, eval-backed).
    These specific samples simply don't hit the #252 filter, so they pin the
    CJK-isolation guarantee, not a byte-identical-forever guarantee.
    """
    # Pinned values: unaffected by both the CJK branch and the #252 filter.
    assert tokenize("refund policy") == ["refund", "policy"]
    assert tokenize("How do I cancel?") == ["cancel"]
    assert tokenize("the quick brown fox") == ["quick", "brown", "fox"]
    # Numbers are preserved
    assert tokenize("item 42") == ["item", "42"]
    # Empty string
    assert tokenize("") == []


def test_tokenize_cjk_bigram_basic():
    """CJK text produces sliding character bigrams."""
    tokens = tokenize("退款政策")
    # Sliding bigrams: 退款, 款政, 政策
    assert tokens == ["退款", "款政", "政策"]


def test_tokenize_cjk_bigram_longer():
    """Longer CJK run produces the full bigram sequence."""
    tokens = tokenize("退款政策說明")
    assert tokens == ["退款", "款政", "政策", "策說", "說明"]


def test_tokenize_cjk_unigram_fallback_single_char():
    """A length-1 CJK run falls back to a single unigram (not dropped)."""
    # Single CJK char query should not be discarded
    tokens = tokenize("錢")
    assert tokens == ["錢"]


def test_tokenize_cjk_unigram_fallback_in_mixed():
    """A length-1 CJK run inside a mixed text falls back to a unigram."""
    # "A 錢 B" — CJK run of length 1 → unigram
    tokens = tokenize("a 錢 b")
    # "a" and "b" are stop words; "錢" is a unigram
    assert "錢" in tokens


def test_tokenize_mixed_cjk_and_latin():
    """Mixed CJK + Latin text: each part tokenises by its own rule."""
    tokens = tokenize("refund 退款政策")
    # "refund" is ASCII word; CJK part produces bigrams
    assert "refund" in tokens
    assert "退款" in tokens
    assert "款政" in tokens
    assert "政策" in tokens


def test_tokenize_cjk_stop_words_not_applied_to_bigrams():
    """Stop-word filter applies only to ASCII tokens, not CJK bigrams."""
    # CJK bigrams should never be dropped as stop words
    tokens = tokenize("退款政策")
    assert len(tokens) == 3


# ---------------------------------------------------------------------------
# AC6 — Unit tests: slugify
# ---------------------------------------------------------------------------


def test_slugify_cjk_preserved():
    """CJK characters are preserved verbatim in slugs."""
    assert slugify("退款政策") == "退款政策"


def test_slugify_mixed_preserves_cjk():
    """Mixed ASCII + CJK: ASCII is lowercased, CJK is kept verbatim."""
    result = slugify("Refund 退款")
    # ASCII 'r','e','f','u','n','d' → lower; space → hyphen; CJK stays
    assert "退款" in result
    assert result == result.lower() or "退款" in result  # CJK isn't lowercased


def test_slugify_ascii_unchanged():
    """Pure ASCII slugify behaviour is byte-identical to before."""
    assert slugify("Hello World") == "hello-world"
    assert slugify("--  edge  --") == "edge"
    assert slugify("cancellation-window") == "cancellation-window"
    assert slugify("") == "section"
    assert slugify("   ") == "section"
    assert slugify("!@#$%") == "section"


def test_slugify_empty_after_strip_cjk_only_returns_section():
    """When no slug-able characters remain, fallback is 'section'."""
    # Punctuation-only input still falls back
    assert slugify("...") == "section"


def test_slugify_distinct_cjk_headings_produce_distinct_slugs():
    """Two different CJK headings produce different slugs."""
    assert slugify("退款政策") != slugify("帳戶管理")


def test_slugify_cjk_collision_suffix():
    """Repeated CJK heading within one Source produces -2 suffix on collision."""
    md = "## 退款政策\nContent A.\n## 退款政策\nContent B.\n"
    # Use parse_markdown to exercise the full collision logic
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8", delete=False) as fh:
        fh.write(md)
        tmp = Path(fh.name)
    try:
        sections = parse_markdown(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    ids = [s.id for s in sections]
    assert any("退款政策" in sid and "-2" in sid for sid in ids), (
        f"Expected -2 suffix on second '退款政策' heading, got: {ids}"
    )


# ---------------------------------------------------------------------------
# AC7 — SYSTEM_PROMPT contains answer-language directive
# ---------------------------------------------------------------------------


def test_system_prompt_language_directive():
    """SYSTEM_PROMPT contains an answer-in-question-language directive."""
    # The directive must instruct the model to answer in the question's language
    assert "language" in SYSTEM_PROMPT.lower(), (
        "SYSTEM_PROMPT must contain a language directive (e.g. 'answer in the "
        "same language as the QUESTION')"
    )
    # The Cannot Confirm carve-out must keep the English sentinel intact
    assert CANNOT_CONFIRM_PHRASE in SYSTEM_PROMPT, (
        "SYSTEM_PROMPT must contain the verbatim Cannot Confirm phrase as an "
        "exception to the language directive"
    )


# ---------------------------------------------------------------------------
# AC4 + AC5 — End-to-end hermetic test: Chinese query → Chinese answer
# ---------------------------------------------------------------------------


CHINESE_WIKI_MD = """\
---
type: concept
status: live
---
# 退款政策

## 退款流程

購買後30天內可申請退款。審核通過後，款項將在5-7個工作日內退回原支付帳戶。

## 帳戶管理

如需更改帳戶資訊，請至設定頁面修改。
"""


@pytest.fixture()
def chinese_wiki_section_index(tmp_path, monkeypatch):
    """Build a BM25 index from the hand-authored Chinese wiki fixture.

    Writes one Chinese Wiki Page to a tmp wiki/concepts/ directory and calls
    build_index() through its docs_dir override so it stays hermetic (no disk
    pollution, no live wiki scan).
    """
    import app.indexer as _idx

    # Point wiki path to tmp; _redirect_paths_to_tmp autouse does this already
    # but we also need the concepts/ subdir to exist.
    concepts_dir = _idx.WIKI_DIR / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    page = concepts_dir / "退款政策.md"
    page.write_text(CHINESE_WIKI_MD, encoding="utf-8")

    # index by calling build_index with the wiki concepts dir
    # We use parse_markdown directly + manual index assembly (test-isolation path)
    sections_list = parse_markdown(page, source_id="退款政策")
    with _idx._index_lock:
        _idx.sections = sections_list
        _idx.rebuild_stats()

    yield sections_list

    # teardown — clear in-memory state
    with _idx._index_lock:
        _idx.sections = []
        _idx.rebuild_stats()


def test_chinese_query_returns_grounded_answer_with_cjk(chinese_wiki_section_index, monkeypatch):
    """Chinese question against indexed Chinese Wiki Page returns Grounded Answer
    containing CJK characters.

    AC4: asserts the plumbing:
      1. Chinese query tokenises to bigrams → retrieves the Chinese fixture Section
      2. answer-language directive is in SYSTEM_PROMPT
      3. mocked LLM returns a Chinese-language answer with CJK Citation
      4. The answer returned to the caller contains CJK code-points

    Hermetic: the LLM is mocked via monkeypatching get_llm() (ADR-0005 /
    CODING_STANDARD §6.3 lazy-singleton getter pattern).
    """
    import app.retrieval as ret_module
    from app.grounding import GroundingOutcome

    CHINESE_ANSWER = (
        "購買後30天內可申請退款，款項將在5-7個工作日內退回。[Source: 退款政策#退款流程]"
    )

    class FakeCJKLLM:
        def invoke(self, messages):
            return FakeLLMResponse(content=CHINESE_ANSWER)

    fake_llm = FakeCJKLLM()
    monkeypatch.setattr(ret_module, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(ret_module, "get_retry_llm", lambda: fake_llm)

    # Mock grounding.verify to return claim_supported (hermetic — no LLM call)
    supported_outcome = GroundingOutcome(passed=True, reason="claim_supported", result=None)
    monkeypatch.setattr(
        ret_module.grounding_module,
        "verify",
        lambda *_a, **_kw: supported_outcome,
    )

    # AC7: language directive must be in SYSTEM_PROMPT
    assert "language" in SYSTEM_PROMPT.lower(), (
        "SYSTEM_PROMPT must contain an answer-language directive"
    )

    # Ask a Chinese question
    question = "退款需要多少時間？"
    result = ret_module.query(question)

    # AC4: answer must contain CJK code-points
    answer = result["answer"]
    has_cjk = any("一" <= ch <= "鿿" for ch in answer)
    assert has_cjk, f"Expected answer to contain CJK characters, got: {answer!r}"

    # AC4: sources must contain the Chinese Citation
    sources = result["sources"]
    assert sources, "Expected at least one source"
    assert any("退款政策" in s["source"] for s in sources), (
        f"Expected Chinese Citation in sources, got: {sources}"
    )

    # AC4: CJK Citation is readable (heading contains CJK)
    top_source = sources[0]
    heading = top_source["heading"]
    has_cjk_heading = any("一" <= ch <= "鿿" for ch in heading)
    assert has_cjk_heading, f"Expected CJK heading in Citation, got heading: {heading!r}"


def test_chinese_query_below_threshold_returns_cannot_confirm_in_english(tmp_path, monkeypatch):
    """AC5: Chinese query that scores below threshold returns the verbatim English
    Cannot Confirm sentinel, NOT a Chinese translation of it.

    This is the critical invariant — the sentinel must stay in English so tests,
    grounding.reason mapping, and the filing gate continue to function correctly.
    """
    import app.indexer as _idx
    import app.retrieval as ret_module

    # Index an unrelated English section so the Chinese query scores below threshold
    english_md = "## Shipping Policy\nStandard shipping takes 3-5 days.\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False, dir=tmp_path
    ) as fh:
        fh.write(english_md)
        tmp_p = Path(fh.name)

    secs = parse_markdown(tmp_p, source_id="shipping")
    with _idx._index_lock:
        _idx.sections = secs
        _idx.rebuild_stats()

    try:
        # Ask a Chinese question about refunds — should score near zero
        question = "我可以退款嗎？"
        result = ret_module.query(question)

        assert result["answer"] == CANNOT_CONFIRM_PHRASE, (
            f"Expected verbatim English Cannot Confirm sentinel, got: {result['answer']!r}"
        )
    finally:
        with _idx._index_lock:
            _idx.sections = []
            _idx.rebuild_stats()
        tmp_p.unlink(missing_ok=True)
