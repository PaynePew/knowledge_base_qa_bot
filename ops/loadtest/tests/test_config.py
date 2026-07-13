"""Fast hermetic unit tests for ops/loadtest/config.py (issue #600).

No server, no subprocess — pure parsing/merging logic only. This is the
"fast hermetic unit tests (config parsing / summarize math)" CI keeps per
the issue's scope; the load run itself is manual-local only.
"""

from __future__ import annotations

import pytest

from ops.loadtest import config


def test_parse_env_overrides_single_pair():
    assert config.parse_env_overrides(["KB_MAX_INFLIGHT=2"]) == {"KB_MAX_INFLIGHT": "2"}


def test_parse_env_overrides_multiple_pairs():
    result = config.parse_env_overrides(["A=1", "B=2"])
    assert result == {"A": "1", "B": "2"}


def test_parse_env_overrides_empty_list():
    assert config.parse_env_overrides([]) == {}


def test_parse_env_overrides_value_may_contain_equals():
    assert config.parse_env_overrides(["OPENAI_API_BASE=http://x/v1?a=b"]) == {
        "OPENAI_API_BASE": "http://x/v1?a=b"
    }


def test_parse_env_overrides_rejects_missing_equals():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        config.parse_env_overrides(["NOVALUE"])


def test_parse_env_overrides_rejects_empty_key():
    with pytest.raises(ValueError, match="empty key"):
        config.parse_env_overrides(["=value"])


def test_resolve_env_later_layer_wins():
    merged = config.resolve_env({"A": "1", "B": "1"}, {"B": "2"})
    assert merged == {"A": "1", "B": "2"}


def test_resolve_env_no_layers():
    assert config.resolve_env() == {}


def test_resolve_env_does_not_mutate_input_layers():
    base = {"A": "1"}
    override = {"A": "2"}
    config.resolve_env(base, override)
    assert base == {"A": "1"}
    assert override == {"A": "2"}


def test_harness_base_env_disables_rate_limit_and_raises_budget_cap():
    # These two defaults exist specifically so the concurrency knobs under
    # test are the only thing bounding a scenario (see config.py docstring).
    assert config.HARNESS_BASE_ENV["KB_RATE_LIMIT_PER_IP"] == "0"
    assert float(config.HARNESS_BASE_ENV["KB_DAILY_USD_CAP"]) > 3.0


def test_known_knobs_defaults_match_documented_table():
    # Mirrors the issue #600 technical brief's knob-default table verbatim —
    # a regression here means the report's methodology section has drifted
    # from what the harness actually documents.
    assert config.KNOWN_KNOBS["KB_MAX_INFLIGHT"].default == 6
    assert config.KNOWN_KNOBS["KB_MAX_ADMIN"].default == 2
    assert config.KNOWN_KNOBS["KB_SSE_MAX_CONCURRENT"].default == 6
    assert config.KNOWN_KNOBS["KB_TRANSCRIBE_CONCURRENCY"].default == 16
    assert config.KNOWN_KNOBS["KB_TRANSCRIBE_PAGE_COUNT_CONCURRENCY"].default == 4
    assert config.KNOWN_KNOBS["KB_TRANSCRIBE_MAX_CONCURRENT_JOBS"].default == 2
    assert config.KNOWN_KNOBS["KB_IMPORT_MAX_CONCURRENT_JOBS"].default == 2
