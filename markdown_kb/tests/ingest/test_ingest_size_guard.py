"""Tests for the ingest size guard — reject oversized Sources before any LLM call.

``classify_source`` (and the entity-synthesis path) send the WHOLE Source in a
single LLM call, so the hard ceiling is the ingest model's context window.  An
oversized Source must fail fast — *before* parse/classify — as a per-source
failure (the batch continues) carrying a reason, and must make NO LLM call.

The limit is ``KB_INGEST_MAX_BYTES`` (bytes, read at call time so a restart-free
env override takes effect), default 256 KiB.

All tests hermetic — no OPENAI_API_KEY required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import app.indexer as indexer_module
import app.ingest as ingest_module
import app.templates as templates_module


def test_oversized_source_fails_without_llm_call(tmp_path, monkeypatch):
    """A Source over KB_INGEST_MAX_BYTES is rejected before any LLM call."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    # Tiny configured limit so a small fixture trips it.
    monkeypatch.setenv("KB_INGEST_MAX_BYTES", "1024")
    oversized = "## Heading\n\n" + ("x" * 2000) + "\n"
    (docs_dir / "big.md").write_text(oversized, encoding="utf-8")

    # If the guard regresses, classify would reach the LLM — assert it never does.
    fake_llm = MagicMock()
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["big.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert "big.md" in result.failed_sources
    assert not fake_llm.with_structured_output.called, (
        "size guard must fire before any LLM call"
    )
    reason = result.failed_reasons.get("big.md", "")
    assert "too large" in reason.lower()
    assert "KB_INGEST_MAX_BYTES" in reason


def test_under_limit_source_reaches_classifier(tmp_path, monkeypatch):
    """A Source under the limit passes the guard and reaches the classifier."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    wiki_dir = tmp_path / "wiki"

    monkeypatch.setenv("KB_INGEST_MAX_BYTES", str(1024 * 1024))  # 1 MiB — generous
    (docs_dir / "small.md").write_text("## Topic\n\nShort body.\n", encoding="utf-8")

    class _ReachedClassifier(Exception):
        pass

    fake_llm = MagicMock()

    def _raise_on_use(_schema):
        # Proves we got past the guard into classify_source; stop the pipeline
        # cleanly so the test stays hermetic (no synth/grounding/write).
        raise _ReachedClassifier

    fake_llm.with_structured_output.side_effect = _raise_on_use
    monkeypatch.setattr(templates_module, "get_ingest_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer_module, "WIKI_DIR", wiki_dir)

    result = ingest_module.ingest_sources(["small.md"], docs_dir=docs_dir, wiki_dir=wiki_dir)

    assert fake_llm.with_structured_output.called, "guard must let an under-limit Source through"
    # Failed at classify (our sentinel), NOT blocked by the size guard.
    assert "too large" not in (result.failed_reasons.get("small.md") or "").lower()


def test_default_limit_is_256_kib(monkeypatch):
    """Unset env → default ceiling is 256 KiB."""
    monkeypatch.delenv("KB_INGEST_MAX_BYTES", raising=False)
    assert ingest_module._max_ingest_bytes() == 256 * 1024
