"""Determinism guard (ingest stack): the curator-time ingest synthesis LLM
(`get_ingest_llm`) must be constructed at ``temperature=0``.

Why: with the langchain default temperature the ingest synthesis samples
non-deterministically, so re-running ingest over the same Sources produces a
different wiki layer each time and the baked seed is not reproducible. The
Source-type classification step (`classify_source`) in particular must be
deterministic. Pinning ``temperature=0`` keeps the synthesized wiki layer
faithful to its Sources and makes re-ingest a clean, reviewable diff.

Mirrors the answer-path determinism guards in
``markdown_kb/tests/test_llm_determinism.py`` (PR #283). No live OpenAI call:
``get_ingest_llm`` constructs ``ChatOpenAI`` inline, so we monkeypatch that
symbol with a spy that records the construction kwargs (like the verifier
guard), and we also read the ``temperature`` field off a real client built
without an API key (construction only — no network).

The lazy singleton ``templates._ingest_llm`` is reset before each call so the
client is rebuilt under the test's conditions (same pattern as
``test_ingest_llm_retries.py``).
"""

from __future__ import annotations

import app.templates as templates_module


def test_get_ingest_llm_pinned_to_temperature_zero(monkeypatch):
    """get_ingest_llm() must construct the ingest LLM with temperature=0.

    Spy on the ``ChatOpenAI`` symbol ``templates`` uses so the construction
    kwargs are captured directly (no real client, no API key, no network).
    """
    captured: dict = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(templates_module, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(templates_module, "_ingest_llm", None)  # force fresh build

    templates_module.get_ingest_llm()

    assert captured.get("temperature") == 0, (
        "ingest synthesis LLM must be deterministic (temperature=0)"
    )


def test_get_ingest_llm_temperature_field_is_zero(monkeypatch):
    """The constructed client exposes temperature=0 (guards against drift)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-ingest-determinism")
    monkeypatch.setattr(templates_module, "_ingest_llm", None)  # force fresh build

    llm = templates_module.get_ingest_llm()

    assert llm.temperature == 0, "ingest synthesis LLM must be temperature=0"
