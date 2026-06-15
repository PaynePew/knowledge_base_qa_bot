"""Index-time ``lang`` tagging correctness for the BM25 (Wiki/Stack A) build.

Issue #285: building the BM25 index must tag every Section with a ``lang`` value
in its metadata, derived from the Section's CONTENT (never filename or folder)
via ``detect_lang``. This slice only ADDS the tag — retrieval results and answers
are unchanged (nothing filters on ``lang`` yet).

Behavioural test (asserts the external index state, not implementation details):
build over a tiny mixed-language fixture corpus, then check each Section carries
the correct ``lang``. The filename is deliberately language-misleading (an
English filename holding Chinese content, and vice versa) to prove the tag is
content-derived, not name-derived.
"""

from __future__ import annotations

import app.indexer as indexer_module
from app.indexer import build_index

# Chinese body under an *English* filename — proves content (not filename) drives
# the tag.
_ZH_DOC = """# 退款政策

## 退款時間

退款會在七個工作天內處理完成，款項退回原付款方式。

## 取消訂單

訂單出貨前都可以免費取消，出貨後請依退貨流程辦理。
"""

# English body under a *Chinese* filename — the mirror case.
_EN_DOC = """# Refund Policy

## Refund Timeline

Refunds are processed within seven business days to the original payment method.

## Cancellation

Orders can be cancelled free of charge any time before they ship.
"""


def _write_corpus(docs_dir):
    docs_dir.mkdir(parents=True, exist_ok=True)
    # English filename, Chinese content.
    (docs_dir / "refund_policy.md").write_text(_ZH_DOC, encoding="utf-8")
    # Chinese filename, English content.
    (docs_dir / "退款.md").write_text(_EN_DOC, encoding="utf-8")


def test_bm25_sections_tagged_with_content_language(tmp_path, monkeypatch):
    """Every BM25 Section carries metadata['lang'] derived from its content."""
    docs_dir = tmp_path / "docs"
    _write_corpus(docs_dir)

    monkeypatch.setattr(indexer_module, "INDEX_PATH", tmp_path / ".kb" / "index.json")

    build_index(docs_dir)

    secs = indexer_module.sections
    assert secs, "Expected at least one Section indexed"

    # Every Section must carry a lang tag, and it must be one of the two values.
    for sec in secs:
        assert "lang" in sec.metadata, f"Section {sec.id} missing 'lang' metadata"
        assert sec.metadata["lang"] in ("zh", "en")

    # The Chinese-content file (English filename) → all zh.
    zh_secs = [s for s in secs if s.file == "refund_policy.md"]
    assert zh_secs, "Expected Sections from the Chinese-content file"
    assert all(s.metadata["lang"] == "zh" for s in zh_secs), (
        f"Chinese content must tag zh regardless of English filename, got "
        f"{[(s.id, s.metadata['lang']) for s in zh_secs]}"
    )

    # The English-content file (Chinese filename) → all en.
    en_secs = [s for s in secs if s.file == "退款.md"]
    assert en_secs, "Expected Sections from the English-content file"
    assert all(s.metadata["lang"] == "en" for s in en_secs), (
        f"English content must tag en regardless of Chinese filename, got "
        f"{[(s.id, s.metadata['lang']) for s in en_secs]}"
    )


def test_bm25_lang_tag_survives_index_json_roundtrip(tmp_path, monkeypatch):
    """The lang tag persists through write_index_json + load_index_json."""
    docs_dir = tmp_path / "docs"
    _write_corpus(docs_dir)

    index_path = tmp_path / ".kb" / "index.json"
    monkeypatch.setattr(indexer_module, "INDEX_PATH", index_path)

    build_index(docs_dir)
    before = {s.id: s.metadata.get("lang") for s in indexer_module.sections}

    # Drop in-memory state and reload from the persisted JSON.
    indexer_module.sections.clear()
    indexer_module.load_index_json(index_path)
    after = {s.id: s.metadata.get("lang") for s in indexer_module.sections}

    assert before == after, "lang tag must round-trip through the persisted index"
    assert all(v in ("zh", "en") for v in after.values())
