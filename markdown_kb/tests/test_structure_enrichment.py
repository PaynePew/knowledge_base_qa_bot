"""Unit tests for ``app.structure_enrichment`` (ADR-0033 decision 2, issue #512).

AC coverage:
  - Longform predicate: table-driven over heading count / preamble share /
    oversized-section cases; filename-, size-, and page-count-blind.
  - Well-headed Sources bypass enrichment byte-identically (no LLM call, no
    frontmatter change).
  - Mocked-LLM: headings materialized at proposed boundaries; every
    resulting Section <= the per-section cap (mechanical re-split fallback
    covered).
  - Page furniture mirroring the REAL scanned-book transcript's byte-shape
    (per-page \\xa0 padding of varying width, page counters whose numbers
    increment every page — never byte-identical lines) is stripped;
    standalone short chapter-numeral lines and non-repeating content are
    untouched.
  - Boundary anchors are located in normalized space (whitespace incl.
    \\xa0 removed, CJK punctuation width folded); a single unfindable anchor
    is skipped when >= 2 usable boundaries remain, otherwise fails soft.
  - Enrichment LLM failure fails soft to the un-enriched transcript.

The LLM is mocked at the lazy-singleton getter (``get_enrichment_llm``), per
CODING_STANDARD §6.3 — never the deep-module entry points themselves.
"""

from __future__ import annotations

import string
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app.indexer as indexer_module
import app.structure_enrichment as se

FILLER = "Lorem ipsum filler text about nothing in particular. "

NBSP = "\xa0"

# CJK filler for the anchor-normalization fixtures (scanned-book shape).
CJK_FILLER = "這一段填充文字僅為湊足長度而寫，內容並無特別意義，只是讓文件超過長篇門檻而已。"


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    """Redirect the Wiki Log so unit runs never append to the repo's wiki/log.md."""
    import app.logger as logger_module

    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")


def _padded(prefix: str, repeats: int = 40) -> str:
    """A paragraph starting with ``prefix`` verbatim, padded past the min-chars floor share."""
    return prefix + " " + (FILLER * repeats)


def _make_fake_llm(chapters: list[SimpleNamespace]) -> MagicMock:
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = SimpleNamespace(chapters=chapters)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


def _scanned_page(page_no: int, content: str, total: int = 63) -> str:
    """One 'page' mirroring the real transcript's byte-shape (蘇格拉底的申辯):

    - running header: timestamp + \\xa0 padding whose WIDTH VARIES per page +
      literal ``Document``;
    - running footer: share URL + \\xa0 padding + a page counter whose
      numbers INCREMENT every page (``2/63``, ``3/63``, …).

    No two furniture lines are byte-identical — only their normalized keys
    coincide.
    """
    header = f"2023/9/19 晚上10:21{NBSP * (10 + page_no)}Document"
    footer = (
        f"https://pan.baidu.com/s/12qkC2ivlBArkl2l-W-jerw{NBSP * (80 - page_no)}{page_no}/{total}"
    )
    return f"{header}\n\n{content}\n\n{footer}"


# ---------------------------------------------------------------------------
# Longform predicate — table-driven
# ---------------------------------------------------------------------------


def test_is_longform_false_below_min_chars_floor(monkeypatch):
    """A tiny zero-heading doc never counts as longform (nothing to segment)."""
    monkeypatch.delenv("KB_LONGFORM_MIN_CHARS", raising=False)
    assert se.is_longform("hello world, no headings here.") is False


def test_is_longform_true_zero_headings_long_body():
    body = "\n\n".join(_padded(f"Paragraph {i} starts here.", 20) for i in range(5))
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is True


def test_is_longform_true_one_heading_long_body():
    body = "# Stray Page Header\n\n" + "\n\n".join(
        _padded(f"Paragraph {i} starts here.", 20) for i in range(5)
    )
    assert se.is_longform(body) is True


def test_is_longform_true_dominant_preamble():
    preamble = _padded("The preamble dominates this document.", 60)
    body = preamble + "\n\n## Heading A\n\nShort body A.\n\n## Heading B\n\nShort body B.\n"
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is True


def test_is_longform_false_well_headed_handbook():
    """Multiple headings, small preamble, no oversized Section => NOT longform."""
    sections = "\n\n".join(
        f"## Chapter {i}\n\n{_padded(f'Chapter {i} content.', 15)}" for i in range(4)
    )
    body = sections
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is False


def test_is_longform_true_oversized_section(monkeypatch):
    """Well-headed on the surface, but one Section blows the per-section cap."""
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "40")  # cap ~= 120 chars
    body = (
        "## Short Chapter\n\nJust a little text.\n\n"
        f"## Huge Chapter\n\n{_padded('This chapter is way too big.', 40)}\n"
    )
    assert len(body.strip()) >= 2000
    assert se.is_longform(body) is True


# ---------------------------------------------------------------------------
# Page-furniture stripping
# ---------------------------------------------------------------------------


def _unique_word(i: int) -> str:
    return string.ascii_lowercase[i // 26] + string.ascii_lowercase[i % 26]


def test_strip_page_furniture_removes_34_pages_of_real_shape_furniture():
    """Mirrors the real transcript byte-shape: 34 pages whose header/footer
    lines are never byte-identical (\\xa0 padding width and page-counter
    numbers vary per page) are still all stripped."""
    contents = [f"Unique prose about topic {_unique_word(i)} on this page." for i in range(34)]
    body = "\n\n".join(_scanned_page(i + 1, content) for i, content in enumerate(contents))

    stripped, removed = se._strip_page_furniture(body)

    assert removed == 68  # 34 headers + 34 footers
    assert "pan.baidu" not in stripped
    assert "2023/9/19" not in stripped
    assert "Document" not in stripped
    for content in contents:
        assert content in stripped


def test_strip_page_furniture_strips_standalone_page_counters():
    """Bare incrementing page-counter lines (``1/63`` … ``5/63``) are furniture
    despite being short — the pure-counter shape is exempt from the
    min-chars floor."""
    blocks = []
    for i in range(5):
        blocks.append(f"Unique prose about topic {_unique_word(i)} on this page.")
        blocks.append(f"{i + 1}/63")
    body = "\n\n".join(blocks)

    stripped, removed = se._strip_page_furniture(body)

    assert removed == 5
    assert "/63" not in stripped
    for i in range(5):
        assert f"topic {_unique_word(i)}" in stripped


def test_strip_page_furniture_folds_ocr_confusables():
    """The same footer OCR'd with l/I/1 jitter across pages (observed live on
    the real transcript: …ivlBArkl2l…, …ivIBArkl2…, …ivIBArkl2I…) still
    counts as ONE repeating furniture line."""
    url_variants = [
        "https://pan.baidu.com/s/12qkC2ivlBArkl2l-W-jerw",
        "https://pan.baidu.com/s/12qkC2ivIBArkl2-W-jerw",
        "https://pan.baidu.com/s/12qkC2ivIBArkl2I-W-jerw",
        "https://pan.baidu.com/s/12qkC2ivlBArkl2-W-jerw",
    ]
    blocks = []
    for i, url in enumerate(url_variants):
        blocks.append(f"Unique prose about topic {_unique_word(i)} on this page.")
        blocks.append(f"{url}{NBSP * (9 - i)}{i + 1}/63")
    body = "\n\n".join(blocks)

    stripped, removed = se._strip_page_furniture(body)

    assert removed == 4
    assert "pan.baidu" not in stripped
    for i in range(4):
        assert f"topic {_unique_word(i)}" in stripped


def test_strip_page_furniture_strips_truncated_tail_variants():
    """A furniture line whose trailing token OCR occasionally drops (the real
    transcript has 2x ``2023/9/19 晚上10:21`` without the usual ``Document``)
    is corroborated by its full form and stripped — but a UNIQUE line that
    merely prefixes furniture (e.g. the book title inside a running header)
    survives."""
    blocks = ["苏格拉底的申辩篇"]  # unique title line — prefix of the running header below
    for i in range(4):
        blocks.append(f"Unique prose about topic {_unique_word(i)} on this page.")
        blocks.append(f"2023/9/19 晚上10:21{NBSP * (12 + i)}Document")
        blocks.append(f"苏格拉底的申辩篇{NBSP * (7 + i)}{20 + i}")
    for i in range(2):
        blocks.append(f"2023/9/19 晚上10:2{i}")  # truncated: no trailing Document
    body = "\n\n".join(blocks)

    stripped, removed = se._strip_page_furniture(body)

    assert removed == 10  # 4 full timestamps + 4 running headers + 2 truncated timestamps
    assert "2023/9/19" not in stripped
    assert stripped.splitlines()[0] == "苏格拉底的申辩篇", "unique title line must survive"


def test_strip_page_furniture_keeps_short_chapter_numeral_lines():
    """Standalone CJK chapter-numeral lines (「一」…) are body STRUCTURE: even
    repeated past the threshold, they stay below the min-chars floor and are
    never stripped."""
    blocks = []
    for i in range(4):
        blocks.append("一")
        blocks.append(f"第{'一二三四'[i]}部分的正文內容各自不同，不會被視為家具。")
    body = "\n\n".join(blocks)

    stripped, removed = se._strip_page_furniture(body)

    assert removed == 0
    assert stripped == body


def test_strip_page_furniture_leaves_headings_and_rules_alone():
    """A repeated heading or horizontal rule is never treated as furniture."""
    body = "\n\n".join(["## Same Title", "body one", "---", "## Same Title", "body two", "---"] * 2)
    stripped, removed = se._strip_page_furniture(body)
    assert removed == 0
    assert stripped == body


def test_strip_page_furniture_untouched_when_no_repeats():
    body = "\n\n".join(f"Unique paragraph about {_unique_word(i)} here." for i in range(10))
    stripped, removed = se._strip_page_furniture(body)
    assert removed == 0
    assert stripped == body


# ---------------------------------------------------------------------------
# enrich_structure — bypass (well-headed), mocked-LLM success, fail-soft
# ---------------------------------------------------------------------------


def test_enrich_structure_bypasses_well_headed_source_byte_identical(monkeypatch):
    """A well-headed Source is untouched: no LLM call, enriched=False, body unchanged."""

    def _boom() -> None:
        raise AssertionError("get_enrichment_llm must not be called for a well-headed Source")

    monkeypatch.setattr(se, "get_enrichment_llm", _boom)

    body = "\n\n".join(
        f"## Chapter {i}\n\n{_padded(f'Chapter {i} content.', 15)}" for i in range(4)
    )
    result = se.enrich_structure(body, filename="handbook.md")

    assert result.enriched is False
    assert result.reason is None
    assert result.body == body
    assert result.enriched_chars == 0


def test_enrich_structure_materializes_headings_at_proposed_boundaries(monkeypatch):
    para1 = _padded("Chapter One begins here.", 30)
    para2 = _padded("Chapter Two starts now.", 30)
    para3 = _padded("Chapter Three opens here.", 30)
    body = "\n\n".join([para1, para2, para3])
    assert se.is_longform(body) is True  # zero headings, well past the min-chars floor

    chapters = [
        SimpleNamespace(title="Chapter One", boundary_anchor="Chapter One begins here."),
        SimpleNamespace(title="Chapter Two", boundary_anchor="Chapter Two starts now."),
        SimpleNamespace(title="Chapter Three", boundary_anchor="Chapter Three opens here."),
    ]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    result = se.enrich_structure(body, filename="book.md")

    assert result.enriched is True
    assert result.reason is None

    # Headings appear in the proposed order.
    idx1 = result.body.index("## Chapter One")
    idx2 = result.body.index("## Chapter Two")
    idx3 = result.body.index("## Chapter Three")
    assert idx1 < idx2 < idx3

    sections = indexer_module.parse_markdown_body(result.body, source_prefix="book.md")
    headings = [s.heading for s in sections]
    assert headings == ["Chapter One", "Chapter Two", "Chapter Three"]
    assert "Chapter One begins here." in sections[0].content
    assert "Chapter Two starts now." in sections[1].content
    assert "Chapter Three opens here." in sections[2].content

    # Issue #513: enriched_chars is the summed length of the inserted
    # `## title` heading lines — never the furniture-removal count.
    expected_enriched_chars = sum(
        len(f"## {title}") for title in ("Chapter One", "Chapter Two", "Chapter Three")
    )
    assert result.enriched_chars == expected_enriched_chars


def _fixed_len_paragraph(prefix: str, target_len: int) -> str:
    """A paragraph of EXACTLY ``target_len`` chars, starting with ``prefix`` verbatim."""
    body = prefix + " "
    pad = "filler word "
    while len(body) < target_len:
        body += pad
    return body[:target_len]


def test_enrich_structure_oversized_chapter_mechanically_resplit(monkeypatch):
    """A proposed chapter that blows the cap gets paragraph-boundary re-split.

    Each individual paragraph (150 chars => 50 estimated tokens) stays safely
    UNDER the cap on its own; only the combined big chapter (12 paragraphs)
    blows it, so the mechanical re-split fallback must pack multiple whole
    paragraphs per emitted Section, never exceeding the cap.
    """
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "100")  # cap ~= 300 chars/section

    # Word-unique prefixes: real prose paragraphs never repeat modulo digits,
    # so the fixture must not either (the furniture detector masks digit runs).
    paragraphs = [
        _fixed_len_paragraph(f"Paragraph {_unique_word(i)} of the big chapter.", 150)
        for i in range(15)
    ]
    body = "\n\n".join(paragraphs)
    assert len(body.strip()) >= 2000

    chapters = [
        SimpleNamespace(title="Intro", boundary_anchor="Paragraph aa of"),
        SimpleNamespace(title="Big Chapter", boundary_anchor="Paragraph ad of"),
    ]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    result = se.enrich_structure(body, filename="big.md")

    assert result.enriched is True
    sections = indexer_module.parse_markdown_body(result.body, source_prefix="big.md")
    big_sections = [sec for sec in sections if sec.heading.startswith("Big Chapter")]
    assert len(big_sections) > 1, "Expected the oversized chapter to be re-split into >1 Section"
    cap = 100
    for sec in sections:
        assert sec.heading.startswith(("Intro", "Big Chapter"))
        assert se.ingest_module.estimate_tokens(sec.content) <= cap

    # Issue #513: the mechanical re-split's `(cont. N)` heading lines count
    # toward enriched_chars too — every inserted heading line does.
    assert result.enriched_chars == sum(len(f"## {sec.heading}") for sec in sections)


def test_enrich_structure_fails_soft_on_llm_error(monkeypatch):
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("simulated model failure")
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    body = "\n\n".join(_padded(f"Paragraph {i} starts here.", 20) for i in range(5))
    assert se.is_longform(body) is True

    result = se.enrich_structure(body, filename="broken.md")

    assert result.enriched is False
    assert result.body == body, "Un-enriched transcript must be returned byte-identical on failure"
    assert result.reason is not None
    assert "simulated model failure" in result.reason
    assert result.enriched_chars == 0, "fail-soft must never report added structure"


def test_enrich_structure_fails_soft_on_unfindable_boundary_anchor(monkeypatch):
    """A proposal whose anchor text is not in the document degrades gracefully."""
    chapters = [SimpleNamespace(title="Ghost Chapter", boundary_anchor="text that never appears")]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    body = "\n\n".join(_padded(f"Paragraph {i} starts here.", 20) for i in range(5))

    result = se.enrich_structure(body, filename="ghost.md")

    assert result.enriched is False
    assert result.body == body
    assert result.reason is not None


# ---------------------------------------------------------------------------
# Boundary anchors — normalized search (real scanned-CJK byte-shape)
# ---------------------------------------------------------------------------


def test_materialize_headings_tolerate_nbsp_wrap_and_width_variance(monkeypatch):
    """Anchors are found despite \\xa0 runs, line wrapping, and 全形/半形
    punctuation divergence between the document and the model's quote.

    Mirrors the real failure: the transcript wraps mid-sentence and pads with
    \\xa0; the model quotes the opening line with full-width commas and plain
    spaces — an exact ``str.find`` never matches.
    """
    # Document side: half-width commas, an \xa0 run, and a hard wrap
    # mid-anchor (real scanned-book shape).
    ch1_open = f"听了原告的控诉,雅典公民们,各位心里怎么想,我不知道;{NBSP * 12}17A"
    ch2_open = "年轻的人，公民们，当着你们像孩子似的\n说瞎话是不恰当的。我恳切地请求你们"
    body = ch1_open + "\n" + (CJK_FILLER * 40) + "\n\n" + ch2_open + "\n" + (CJK_FILLER * 40) + "\n"
    assert se.is_longform(body) is True

    chapters = [
        # Model side: full-width commas where the document has half-width.
        SimpleNamespace(
            title="第一部分",
            boundary_anchor="听了原告的控诉，雅典公民们，各位心里怎么想，我不知道",
        ),
        # Model side: single line with no wrap where the document wraps.
        SimpleNamespace(
            title="第二部分",
            boundary_anchor="年轻的人，公民们，当着你们像孩子似的说瞎话是不恰当的。我恳",
        ),
    ]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    result = se.enrich_structure(body, filename="apology.md")

    assert result.enriched is True
    assert result.reason is None
    assert result.body.index("## 第一部分") < result.body.index("## 第二部分")

    sections = indexer_module.parse_markdown_body(result.body, source_prefix="apology.md")
    assert [s.heading for s in sections] == ["第一部分", "第二部分"]
    assert "听了原告的控诉" in sections[0].content
    assert "年轻的人" in sections[1].content


def test_enrich_structure_skips_unfindable_anchor_when_two_usable_remain(monkeypatch):
    """One bad anchor out of three: the chapter is skipped, the other two
    still materialize (no all-or-nothing abort)."""
    para1 = _padded("Chapter One begins here.", 30)
    para2 = _padded("Chapter Two starts now.", 30)
    body = "\n\n".join([para1, para2])

    chapters = [
        SimpleNamespace(title="Chapter One", boundary_anchor="Chapter One begins here."),
        SimpleNamespace(title="Ghost Chapter", boundary_anchor="text that never appears"),
        SimpleNamespace(title="Chapter Two", boundary_anchor="Chapter Two starts now."),
    ]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    result = se.enrich_structure(body, filename="partial.md")

    assert result.enriched is True
    assert result.reason is None
    assert "## Chapter One" in result.body
    assert "## Chapter Two" in result.body
    assert "Ghost Chapter" not in result.body


def test_enrich_structure_fails_soft_below_two_usable_boundaries(monkeypatch):
    """With only one findable anchor left, materialization is not attempted:
    fail-soft to the un-enriched body, reason carries the failure counts."""
    para1 = _padded("Chapter One begins here.", 30)
    para2 = _padded("Chapter Two starts now.", 30)
    body = "\n\n".join([para1, para2])

    chapters = [
        SimpleNamespace(title="Chapter One", boundary_anchor="Chapter One begins here."),
        SimpleNamespace(title="Ghost Chapter", boundary_anchor="text that never appears"),
    ]
    fake_llm = _make_fake_llm(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_llm)

    result = se.enrich_structure(body, filename="thin.md")

    assert result.enriched is False
    assert result.body == body
    assert result.reason is not None
    assert "1/2" in result.reason
    assert "usable boundaries" in result.reason
