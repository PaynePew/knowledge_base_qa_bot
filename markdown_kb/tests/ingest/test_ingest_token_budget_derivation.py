"""Tests for model-derived ingest token budget (Level 1 — de-frost the literal).

Replaces the frozen ``KB_INGEST_MAX_TOKENS=64000`` literal with a budget derived
from the *configured* ingest model's context window:

    budget = ingest_model_context_window(model) * KB_INGEST_TOKEN_FRACTION

Precedence (two-tier knob):
    1. KB_INGEST_MAX_TOKENS  — absolute override (wins outright)
    2. window(model) * KB_INGEST_TOKEN_FRACTION (default 0.5)

Design pins exercised here:
- fraction default 0.5 → gpt-4o-mini (128K) reproduces the old 64000 exactly
  (zero behaviour regression for the default model).
- unknown model → pessimistic fallback window (never over-estimate capacity).
- the per-section cap stays a *named constant*, NOT model-derived (it is a
  quality cap, not a capacity cap).

All tests hermetic — no OPENAI_API_KEY required, model resolved from env only.
"""

from __future__ import annotations

import app.ingest as ingest_module
import app.templates as templates_module


def _clear_budget_env(monkeypatch):
    """Strip every env var that influences the derived budget → clean baseline."""
    for var in (
        "KB_INGEST_MAX_TOKENS",
        "KB_INGEST_TOKEN_FRACTION",
        "OPENAI_INGEST_MODEL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# templates.ingest_model_context_window — the model→window lookup
# ---------------------------------------------------------------------------


def test_context_window_known_model(monkeypatch):
    """A known model resolves to its documented context window."""
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4o-mini")
    assert templates_module.ingest_model_context_window() == 128_000


def test_context_window_large_model(monkeypatch):
    """gpt-4.1 family carries a 1M-token window."""
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4.1")
    assert templates_module.ingest_model_context_window() == 1_000_000


def test_context_window_unknown_model_pessimistic(monkeypatch):
    """Unknown model → pessimistic fallback (small window, never over-estimate)."""
    _clear_budget_env(monkeypatch)
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "totally-made-up-model-9000")
    window = templates_module.ingest_model_context_window()
    assert window == 32_000
    # Fallback must be conservative: smaller than the smallest mainstream model
    # we'd actually configure, so an unrecognised model under-fills not overflows.
    assert window < 128_000


def test_model_name_resolution_chain(monkeypatch):
    """OPENAI_INGEST_MODEL > OPENAI_MODEL > gpt-4o-mini (single source of truth)."""
    _clear_budget_env(monkeypatch)
    assert templates_module._ingest_model_name() == "gpt-4o-mini"

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert templates_module._ingest_model_name() == "gpt-4o"

    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4.1")
    assert templates_module._ingest_model_name() == "gpt-4.1"


# ---------------------------------------------------------------------------
# _max_ingest_tokens — derivation, override precedence, fraction knob
# ---------------------------------------------------------------------------


def test_default_model_budget_unchanged(monkeypatch):
    """Backward-compat pin: default model + default fraction → still 64000."""
    _clear_budget_env(monkeypatch)
    # gpt-4o-mini 128_000 * 0.5 == 64_000 (the old frozen literal)
    assert ingest_module._max_ingest_tokens() == 64_000


def test_budget_scales_with_model_window(monkeypatch):
    """Swapping to a 1M-window model re-scales the budget — no literal touched."""
    _clear_budget_env(monkeypatch)
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4.1")
    # 1_000_000 * 0.5
    assert ingest_module._max_ingest_tokens() == 500_000


def test_absolute_override_beats_derivation(monkeypatch):
    """KB_INGEST_MAX_TOKENS is an absolute escape hatch — wins over the model."""
    _clear_budget_env(monkeypatch)
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4.1")  # would derive 500_000
    monkeypatch.setenv("KB_INGEST_MAX_TOKENS", "12345")
    assert ingest_module._max_ingest_tokens() == 12_345


def test_fraction_knob_tunes_derivation(monkeypatch):
    """KB_INGEST_TOKEN_FRACTION proportionally tunes the derived budget."""
    _clear_budget_env(monkeypatch)
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4o-mini")  # 128_000 window
    monkeypatch.setenv("KB_INGEST_TOKEN_FRACTION", "0.25")
    assert ingest_module._max_ingest_tokens() == 32_000


def test_unknown_model_budget_uses_fallback(monkeypatch):
    """Unknown model → fallback window * fraction (32_000 * 0.5)."""
    _clear_budget_env(monkeypatch)
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "totally-made-up-model-9000")
    assert ingest_module._max_ingest_tokens() == 16_000


# ---------------------------------------------------------------------------
# _max_section_tokens — stays a NAMED CONSTANT, deliberately NOT derived
# ---------------------------------------------------------------------------


def test_section_cap_default_is_named_constant(monkeypatch):
    """Per-section cap default is 6000 — a quality cap, independent of the model."""
    monkeypatch.delenv("KB_INGEST_MAX_SECTION_TOKENS", raising=False)
    assert ingest_module._max_section_tokens() == 6_000


def test_section_cap_does_not_scale_with_model(monkeypatch):
    """The section cap must NOT grow on a big-window model (it is not a capacity cap)."""
    monkeypatch.delenv("KB_INGEST_MAX_SECTION_TOKENS", raising=False)
    monkeypatch.setenv("OPENAI_INGEST_MODEL", "gpt-4.1")  # 1M window
    # If someone wrongly derived it from the window it would balloon; it must not.
    assert ingest_module._max_section_tokens() == 6_000


def test_section_cap_env_override(monkeypatch):
    """KB_INGEST_MAX_SECTION_TOKENS still overrides the named-constant default."""
    monkeypatch.setenv("KB_INGEST_MAX_SECTION_TOKENS", "999")
    assert ingest_module._max_section_tokens() == 999
