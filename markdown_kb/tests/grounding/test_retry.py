"""Retry behaviour tests for grounding.verify() — mock OpenAI client only.

Tests cover each error class per ADR-0004 Q5:
  - Transient (A): timeout, 5xx, 429 → up to 2 retries with backoff
  - Malformed (B): ValidationError, invalid JSON → 1 retry
  - Refusal (C): empty completion → 0 retries
  - Hard (D): AuthenticationError → 0 retries, prominent log

All tests mock the ChatOpenAI invocation so no real API calls are made.
time.sleep is also patched so tests run fast (no actual backoff waits).

The 5-second latency budget assertion uses time.monotonic; because sleep is
patched, elapsed wall time is always well under 5s on any machine.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import httpx
import openai
import pydantic
import pytest

import app.grounding as grounding
from app.grounding import (
    GroundingClaim,
    GroundingOutcome,
    GroundingResult,
    VerifierErrorType,
)

from .fixtures import SampleSection

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# Non-empty placeholder sections for tests that exercise the verifier's LLM
# machinery (model selection / retry / logging / error mapping) — NOT grounding
# semantics.  verify() short-circuits on EMPTY sections (issue #191, fail-closed
# before any LLM call), so these tests must pass a non-empty cited set to reach
# the LLM path they are asserting on.  The content is irrelevant: ChatOpenAI is
# mocked in every one of these tests.
_STUB_SECTIONS = [
    SampleSection(
        id="stub-section",
        heading_path=["Stub"],
        content="Placeholder content for verifier-machinery tests.",
    )
]


def _make_success_result() -> GroundingResult:
    """Return a valid GroundingResult representing a fully supported draft."""
    return GroundingResult(
        reasoning="All claims are supported by the cited sections.",
        claims=[
            GroundingClaim(
                text="The draft claim is supported.",
                supported=True,
                citing_section_ids=["sec-1"],
            )
        ],
        unsupported_claims=[],
        passed=True,
    )


def _make_unsupported_result() -> GroundingResult:
    """Return a valid GroundingResult representing a draft with unsupported claims."""
    return GroundingResult(
        reasoning="One claim is unsupported.",
        claims=[
            GroundingClaim(
                text="An unsupported claim.",
                supported=False,
                citing_section_ids=[],
            )
        ],
        unsupported_claims=["An unsupported claim."],
        passed=False,
    )


def _make_timeout_exc() -> openai.APITimeoutError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APITimeoutError(request=req)


def _make_server_error_exc() -> openai.APIStatusError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(503, request=req)
    return openai.APIStatusError("Internal Server Error", response=resp, body=None)


def _make_rate_limit_exc() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return openai.RateLimitError("Rate limit exceeded", response=resp, body=None)


def _make_auth_exc() -> openai.AuthenticationError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    return openai.AuthenticationError("Unauthorized", response=resp, body=None)


def _setup_chain(side_effects: list) -> tuple[MagicMock, MagicMock]:
    """Set up a fake LLM + chain pair with the given invoke() side effects.

    Returns (fake_llm, fake_chain).
    """
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = side_effects

    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_chain

    return fake_llm, fake_chain


# ---------------------------------------------------------------------------
# Transient error (A) — recover on 2nd attempt
# ---------------------------------------------------------------------------


def test_transient_timeout_first_call_then_success(tmp_path, monkeypatch) -> None:
    """First call raises APITimeoutError; second succeeds → passed=True, retries=1."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, fake_chain = _setup_chain([_make_timeout_exc(), _make_success_result()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        outcome = grounding.verify(
            draft="The draft claim is supported.",
            sections=_STUB_SECTIONS,
        )

    assert outcome.passed is True
    assert outcome.reason == "claim_supported"
    assert outcome.retries_attempted == 1
    assert outcome.error_type is None
    assert fake_chain.invoke.call_count == 2


def test_transient_5xx_first_call_then_success(tmp_path, monkeypatch) -> None:
    """First call raises 503 APIStatusError; second succeeds → passed=True, retries=1."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, fake_chain = _setup_chain([_make_server_error_exc(), _make_success_result()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        outcome = grounding.verify(draft="The draft claim is supported.", sections=_STUB_SECTIONS)

    assert outcome.passed is True
    assert outcome.retries_attempted == 1


def test_transient_rate_limit_first_call_then_success(tmp_path, monkeypatch) -> None:
    """First call raises RateLimitError (429); second succeeds → passed=True, retries=1."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, fake_chain = _setup_chain([_make_rate_limit_exc(), _make_success_result()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        outcome = grounding.verify(draft="The draft claim is supported.", sections=_STUB_SECTIONS)

    assert outcome.passed is True
    assert outcome.retries_attempted == 1


# ---------------------------------------------------------------------------
# Transient error (A) — all retries exhausted
# ---------------------------------------------------------------------------


def test_transient_exhausted_three_timeouts(tmp_path, monkeypatch) -> None:
    """Three consecutive timeouts → verifier_unavailable, error_type=TIMEOUT, retries=2."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, fake_chain = _setup_chain(
        [_make_timeout_exc(), _make_timeout_exc(), _make_timeout_exc()]
    )

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        outcome = grounding.verify(draft="The draft claim is supported.", sections=_STUB_SECTIONS)

    assert outcome.passed is False
    assert outcome.reason == "verifier_unavailable"
    assert outcome.error_type == VerifierErrorType.TIMEOUT
    assert outcome.retries_attempted == 2
    # 3 attempts total: initial + 2 retries
    assert fake_chain.invoke.call_count == 3


def test_transient_exhausted_applies_backoff(tmp_path, monkeypatch) -> None:
    """Transient retries apply backoff delays (200ms then 800ms) between attempts."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, fake_chain = _setup_chain(
        [_make_timeout_exc(), _make_timeout_exc(), _make_timeout_exc()]
    )

    sleep_mock = MagicMock()
    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep", sleep_mock),
    ):
        grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)

    # Should have slept twice: 0.2s after first failure, 0.8s after second
    assert sleep_mock.call_count == 2
    calls = sleep_mock.call_args_list
    assert calls[0] == call(0.2)
    assert calls[1] == call(0.8)


# ---------------------------------------------------------------------------
# Malformed response (B) — recover on 2nd attempt
# ---------------------------------------------------------------------------


def test_malformed_then_valid_result(tmp_path, monkeypatch) -> None:
    """First call raises ValidationError; second returns valid → passed=True, retries=1."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    # Simulate a pydantic ValidationError (malformed structured output)
    validation_error = pydantic.ValidationError.from_exception_data(
        title="GroundingResult",
        input_type="python",
        line_errors=[
            {
                "type": "missing",
                "loc": ("reasoning",),
                "msg": "Field required",
                "input": {},
                "ctx": {},
            }
        ],
    )

    fake_llm, fake_chain = _setup_chain([validation_error, _make_success_result()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        outcome = grounding.verify(draft="The draft claim is supported.", sections=_STUB_SECTIONS)

    assert outcome.passed is True
    assert outcome.reason == "claim_supported"
    assert outcome.retries_attempted == 1
    assert fake_chain.invoke.call_count == 2


def test_malformed_exhausted_after_one_retry(tmp_path, monkeypatch) -> None:
    """Two consecutive ValidationErrors → verifier_unavailable, error_type=MALFORMED_JSON, retries=1."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    validation_error = pydantic.ValidationError.from_exception_data(
        title="GroundingResult",
        input_type="python",
        line_errors=[
            {
                "type": "missing",
                "loc": ("reasoning",),
                "msg": "Field required",
                "input": {},
                "ctx": {},
            }
        ],
    )

    fake_llm, fake_chain = _setup_chain([validation_error, validation_error])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        outcome = grounding.verify(draft="The draft claim is supported.", sections=_STUB_SECTIONS)

    assert outcome.passed is False
    assert outcome.reason == "verifier_unavailable"
    assert outcome.error_type == VerifierErrorType.MALFORMED_JSON
    assert outcome.retries_attempted == 1
    assert fake_chain.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Refusal (C) — 0 retries
# ---------------------------------------------------------------------------


def test_refusal_empty_completion_no_retry(tmp_path, monkeypatch) -> None:
    """Model returns empty claims + empty reasoning → verifier_unavailable, error_type=REFUSAL, no retry."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    refusal_result = GroundingResult(
        reasoning="",  # empty = refusal signal
        claims=[],  # empty = refusal signal
        unsupported_claims=[],
        passed=False,
    )

    fake_llm, fake_chain = _setup_chain([refusal_result])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        outcome = grounding.verify(draft="The draft claim.", sections=_STUB_SECTIONS)

    assert outcome.passed is False
    assert outcome.reason == "verifier_unavailable"
    assert outcome.error_type == VerifierErrorType.REFUSAL
    assert outcome.retries_attempted == 0
    # Only one attempt — no retry for refusal
    assert fake_chain.invoke.call_count == 1


# ---------------------------------------------------------------------------
# Hard error (D) — 0 retries, prominent log
# ---------------------------------------------------------------------------


def test_hard_auth_error_no_retry(tmp_path, monkeypatch) -> None:
    """AuthenticationError → verifier_unavailable, error_type=AUTH, retries=0."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, fake_chain = _setup_chain([_make_auth_exc()])

    sleep_mock = MagicMock()
    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep", sleep_mock),
    ):
        outcome = grounding.verify(draft="The draft claim.", sections=_STUB_SECTIONS)

    assert outcome.passed is False
    assert outcome.reason == "verifier_unavailable"
    assert outcome.error_type == VerifierErrorType.AUTH
    assert outcome.retries_attempted == 0
    # No sleep — hard error bails immediately
    sleep_mock.assert_not_called()
    # Only one attempt
    assert fake_chain.invoke.call_count == 1


def test_hard_auth_error_logs_hard_error(tmp_path, monkeypatch) -> None:
    """AuthenticationError emits a prominent server log with HARD_ERROR marker."""
    log_path = tmp_path / "wiki" / "log.md"
    monkeypatch.setattr("app.logger.LOG_PATH", log_path)

    fake_llm, _ = _setup_chain([_make_auth_exc()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        grounding.verify(draft="The draft claim.", sections=_STUB_SECTIONS)

    log_text = log_path.read_text()
    assert "HARD_ERROR" in log_text
    assert "auth" in log_text


# ---------------------------------------------------------------------------
# 5-second budget assertion
# ---------------------------------------------------------------------------


def test_latency_budget_never_exceeded(tmp_path, monkeypatch) -> None:
    """Total elapsed time for three-timeout scenario is under 5s (sleep is real but 0)."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, _ = _setup_chain([_make_timeout_exc(), _make_timeout_exc(), _make_timeout_exc()])

    # Patch sleep to zero so the test runs fast but we measure the real elapsed time
    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        t0 = time.monotonic()
        outcome = grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)
        elapsed = time.monotonic() - t0

    assert elapsed < 5.0, f"Expected elapsed < 5s, got {elapsed:.3f}s"
    assert outcome.reason == "verifier_unavailable"


# ---------------------------------------------------------------------------
# verify() never raises on verifier-side failures (AC #9)
# ---------------------------------------------------------------------------


def test_verify_never_raises_on_auth_error(tmp_path, monkeypatch) -> None:
    """verify() returns GroundingOutcome; it never raises on verifier-side failure."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")

    fake_llm, _ = _setup_chain([_make_auth_exc()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        # Must not raise
        outcome = grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)

    assert isinstance(outcome, GroundingOutcome)


def test_verify_raises_on_programmer_error() -> None:
    """verify() raises TypeError when called with wrong input types (programmer error)."""
    with pytest.raises(TypeError, match="draft must be str"):
        grounding.verify(draft=123, sections=_STUB_SECTIONS)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="sections must be list"):
        grounding.verify(draft="The draft.", sections="not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OPENAI_VERIFIER_MODEL env var resolution (AC #3)
# ---------------------------------------------------------------------------


def test_verifier_model_env_override(tmp_path, monkeypatch) -> None:
    """OPENAI_VERIFIER_MODEL env var overrides the model passed to ChatOpenAI."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.setenv("OPENAI_VERIFIER_MODEL", "gpt-4o")

    captured_model: list[str] = []

    def fake_chat_openai(model: str, **_kwargs) -> MagicMock:
        # **_kwargs tolerates temperature=0 (and any future construction kwarg);
        # this spy asserts model selection only — temperature is covered by
        # test_llm_determinism.test_verifier_llm_pinned_to_temperature_zero.
        captured_model.append(model)
        fake_llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = _make_success_result()
        fake_llm.with_structured_output.return_value = fake_chain
        return fake_llm

    with patch("app.grounding.ChatOpenAI", side_effect=fake_chat_openai):
        grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)

    assert captured_model == ["gpt-4o"]


def test_verifier_model_fallback_to_openai_model(tmp_path, monkeypatch) -> None:
    """Without OPENAI_VERIFIER_MODEL, falls through to OPENAI_MODEL env var."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.delenv("OPENAI_VERIFIER_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini-custom")

    captured_model: list[str] = []

    def fake_chat_openai(model: str, **_kwargs) -> MagicMock:
        # **_kwargs tolerates temperature=0 (and any future construction kwarg);
        # this spy asserts model selection only — temperature is covered by
        # test_llm_determinism.test_verifier_llm_pinned_to_temperature_zero.
        captured_model.append(model)
        fake_llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = _make_success_result()
        fake_llm.with_structured_output.return_value = fake_chain
        return fake_llm

    with patch("app.grounding.ChatOpenAI", side_effect=fake_chat_openai):
        grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)

    assert captured_model == ["gpt-4o-mini-custom"]


def test_verifier_model_default_is_gpt4o_mini(tmp_path, monkeypatch) -> None:
    """Without either env var, defaults to gpt-4o-mini."""
    monkeypatch.setattr("app.logger.LOG_PATH", tmp_path / "wiki" / "log.md")
    monkeypatch.delenv("OPENAI_VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    captured_model: list[str] = []

    def fake_chat_openai(model: str, **_kwargs) -> MagicMock:
        # **_kwargs tolerates temperature=0 (and any future construction kwarg);
        # this spy asserts model selection only — temperature is covered by
        # test_llm_determinism.test_verifier_llm_pinned_to_temperature_zero.
        captured_model.append(model)
        fake_llm = MagicMock()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = _make_success_result()
        fake_llm.with_structured_output.return_value = fake_chain
        return fake_llm

    with patch("app.grounding.ChatOpenAI", side_effect=fake_chat_openai):
        grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)

    assert captured_model == ["gpt-4o-mini"]


# ---------------------------------------------------------------------------
# Logging assertions (AC #10)
# ---------------------------------------------------------------------------


def test_successful_verify_logs_outcome_and_latency(tmp_path, monkeypatch) -> None:
    """On success, log_event records reason, retries, and latency."""
    log_path = tmp_path / "wiki" / "log.md"
    monkeypatch.setattr("app.logger.LOG_PATH", log_path)

    fake_llm, _ = _setup_chain([_make_success_result()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)

    log_text = log_path.read_text()
    assert "grounding_verify" in log_text
    assert "reason=claim_supported" in log_text
    assert "retries=0" in log_text
    assert "latency=" in log_text


def test_verifier_unavailable_logs_granular_error_type(tmp_path, monkeypatch) -> None:
    """On verifier_unavailable, log records the granular error_type."""
    log_path = tmp_path / "wiki" / "log.md"
    monkeypatch.setattr("app.logger.LOG_PATH", log_path)

    fake_llm, _ = _setup_chain([_make_timeout_exc(), _make_timeout_exc(), _make_timeout_exc()])

    with (
        patch("app.grounding.ChatOpenAI", return_value=fake_llm),
        patch("app.grounding.time.sleep"),
    ):
        grounding.verify(draft="The draft.", sections=_STUB_SECTIONS)

    log_text = log_path.read_text()
    assert "grounding_verifier_error" in log_text
    assert "timeout" in log_text
    assert "latency=" in log_text
