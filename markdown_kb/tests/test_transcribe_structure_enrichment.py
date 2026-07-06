"""Transcriber-side coverage for Structure Enrichment (ADR-0033 decision 2, issue #512).

AC coverage:
  - The transcription system prompt carries the "omit repeated page
    furniture" rule (ADR-0032 amendment).
  - The forced Transcribe entry (``transcribe_source`` / ``POST /transcribe``)
    wires Structure Enrichment the same way as the auto-route in importer.py:
    a longform-triggering transcript gains `structure: enriched` frontmatter.

The Transcribe LLM is mocked at ``get_transcribe_llm`` (per-page canned
body) and the enrichment LLM at ``structure_enrichment.get_enrichment_llm`` —
never any deep-module entry point (CODING_STANDARD §6.3).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from .conftest import FakeLLMResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw_import"

FILLER = "Lorem ipsum filler text about nothing in particular. "


def test_transcribe_system_prompt_has_page_furniture_rule():
    import app.transcriber as transcriber_module

    prompt = transcriber_module._TRANSCRIBE_SYSTEM_PROMPT
    assert "furniture" in prompt.lower()
    assert "page number" in prompt.lower() or "page numbers" in prompt.lower()


@pytest.fixture()
def transcribe_env(tmp_path, monkeypatch):
    import app.logger as logger_module
    import app.transcriber as transcriber_module

    raw_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()

    monkeypatch.setattr(transcriber_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(transcriber_module, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("KB_TRANSCRIBE_ENABLED", "true")

    return {"raw_dir": raw_dir, "docs_dir": docs_dir}


def _fake_llm_with_chapters(chapters: list[SimpleNamespace]) -> MagicMock:
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = SimpleNamespace(chapters=chapters)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain
    return fake_llm


def test_force_transcribe_wires_structure_enrichment(transcribe_env, monkeypatch):
    """A single-page transcript whose canned body is long + zero-heading gets enriched."""
    import app.structure_enrichment as se
    import app.transcriber as transcriber_module

    long_body = "\n\n".join(f"Paragraph {i} opens here. " + (FILLER * 6) for i in range(8))
    assert len(long_body.strip()) >= 2000

    fake_transcribe_llm = SimpleNamespace(
        invoke=lambda messages: FakeLLMResponse(content=long_body)
    )
    monkeypatch.setattr(transcriber_module, "get_transcribe_llm", lambda: fake_transcribe_llm)

    chapters = [SimpleNamespace(title="Part One", boundary_anchor="Paragraph 0 opens here.")]
    fake_enrichment_llm = _fake_llm_with_chapters(chapters)
    monkeypatch.setattr(se, "get_enrichment_llm", lambda: fake_enrichment_llm)

    (transcribe_env["raw_dir"] / "sample_english.pdf").write_bytes(
        (FIXTURES / "sample_english.pdf").read_bytes()
    )
    result = transcriber_module.transcribe_source("sample_english.pdf")

    assert result.status == "created"
    content = (transcribe_env["docs_dir"] / "sample_english.md").read_text(encoding="utf-8")
    assert "structure: enriched" in content
    assert "## Part One" in content
    assert result.origin == "transcribed"
