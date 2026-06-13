"""Tests for the ingest size guard — migrated from byte guard to token guard.

KB_INGEST_MAX_BYTES has been retired (Fix 2).  The per-section hard token cap
(KB_INGEST_MAX_SECTION_TOKENS, _max_section_tokens()) replaced it.  These tests
are kept as a brief smoke layer to document the migration; the full token-guard
test suite lives in test_ingest_token_guard.py.

All tests hermetic — no OPENAI_API_KEY required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import app.indexer as indexer_module
import app.ingest as ingest_module
import app.templates as templates_module


def test_oversized_section_fails_without_llm_call(tmp_path, monkeypatch):
    """A Source with a section over KB_INGEST_MAX_SECTION_TOKENS is rejected before any LLM call.

    Migrated from byte-guard test: the per-section token cap now gates ingest
    instead of the retired KB_INGEST_MAX_BYTES byte ceiling.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Set a tiny token cap so a small fixture trips it.
    # _estimate_tokens = len(content) // 3; cap = 10 → need > 30 chars.
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "10")
    oversized = "## Heading\n\n" + ("x" * 100) + "\n"
    (docs_dir / "big.md").write_text(oversized, encoding="utf-8")

    # If the guard regresses, classify would reach the LLM — assert it never does.
    fake_llm = MagicMock()
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["big.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert "big.md" in result.failed_sources
    assert not fake_llm.with_structured_output.called, (
        "section token guard must fire before any LLM call"
    )
    reason = result.failed_reasons.get("big.md", "")
    assert "too large" in reason.lower()


def test_under_limit_source_reaches_classifier(tmp_path, monkeypatch):
    """A Source under the section token cap passes the guard and reaches the classifier."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Generous section cap — short fixture passes easily
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "500")
    (docs_dir / "small.md").write_text("## Topic\n\nShort body.\n", encoding="utf-8")

    class _ReachedClassifier(Exception):
        pass

    fake_llm = MagicMock()

    def _raise_on_use(_schema):
        raise _ReachedClassifier

    fake_llm.with_structured_output.side_effect = _raise_on_use
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["small.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert fake_llm.with_structured_output.called, (
        "guard must let an under-section-cap Source through to classify"
    )
    assert "too large" not in (result.failed_reasons.get("small.md") or "").lower()


def test_default_section_cap_is_6k(monkeypatch):
    """Unset env → default per-section token ceiling is 6000.

    Replaces the retired test_default_limit_is_256_kib (KB_INGEST_MAX_BYTES).
    """
    monkeypatch.delenv("KB_INGEST_MAX_SECTION_TOKENS", raising=False)
    assert ingest_module._max_section_tokens() == 6000
