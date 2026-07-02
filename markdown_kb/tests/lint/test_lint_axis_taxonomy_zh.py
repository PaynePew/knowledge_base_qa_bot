"""Unit tests for the Lint Axis taxonomy's zh strings (issue #365, tier-A S5).

S1 (issue #361) shipped ``LINT_CHECK_TAXONOMY`` / ``LINT_AXIS_ORDER`` with
English-only labels. This slice adds the Traditional-Chinese counterparts —
``LintCheckMeta.label_zh`` per check and ``LINT_AXIS_LABEL_ZH`` per axis —
as the single source of truth the Operator Console's zh/en header toggle
reads from (issue #365 AC "single source, no per-interface duplication").
These tests pin the taxonomy side only; the Console's structural bilingual
test (``gateway/tests/test_ui_console_lint_bilingual.py``) pins that the
Console's own zh mirror renders the same strings.
"""

from __future__ import annotations

from app.lint import LINT_AXIS_LABEL_ZH, LINT_AXIS_ORDER, LINT_CHECK_TAXONOMY

# The zh short label every check must resolve to, mirrored 1:1 into the
# Console (gateway/static/console.html's LINT_CHECK_LABEL_ZH).
_EXPECTED_LABEL_ZH_BY_CODE = {
    "C6": "過時",
    "C3": "驗證失敗",
    "C11": "孤立頁面",
    "C5": "矛盾",
    "C4": "重複",
    "C1": "覆蓋缺口",
    "C2": "失效連結",
    "C8": "待升級",
    "C10": "格式錯誤",
    "C9": "資料過舊",
}


class TestLintCheckLabelZh:
    """LintCheckMeta.label_zh: every wired check resolves to its zh short label."""

    def test_every_check_has_a_non_empty_label_zh(self):
        for code, meta in LINT_CHECK_TAXONOMY.items():
            assert meta.label_zh, f"{code}: label_zh must not be empty"

    def test_each_check_resolves_to_its_expected_label_zh(self):
        for code, expected_zh in _EXPECTED_LABEL_ZH_BY_CODE.items():
            assert LINT_CHECK_TAXONOMY[code].label_zh == expected_zh

    def test_label_zh_is_distinct_from_the_english_label(self):
        """A translated string, not a copy-paste of the English label."""
        for code, meta in LINT_CHECK_TAXONOMY.items():
            assert meta.label_zh != meta.label, f"{code}: label_zh must differ from label"

    def test_english_label_is_unchanged_by_the_zh_addition(self):
        """Adding label_zh must not disturb the existing English label field
        (S1's test_lint_axis_taxonomy.py already pins these; re-asserted here
        as the specific regression this slice could introduce)."""
        expected_label = {
            "C6": "stale",
            "C3": "failed-grounding",
            "C11": "orphan",
            "C5": "contradiction",
            "C4": "collision",
            "C1": "coverage-gap",
            "C2": "red-link",
            "C8": "promotion",
            "C10": "invalid-schema",
            "C9": "stale-qa",
        }
        for code, expected in expected_label.items():
            assert LINT_CHECK_TAXONOMY[code].label == expected


class TestLintAxisLabelZh:
    """LINT_AXIS_LABEL_ZH: every axis identifier resolves to its zh display string."""

    def test_covers_every_axis_in_the_stable_order(self):
        assert set(LINT_AXIS_LABEL_ZH.keys()) == set(LINT_AXIS_ORDER)

    def test_each_axis_resolves_to_its_expected_zh_label(self):
        expected = {
            "Freshness": "新鮮度",
            "Coherence": "一致性",
            "Coverage": "覆蓋率",
            "Lifecycle": "生命週期",
        }
        assert expected == LINT_AXIS_LABEL_ZH

    def test_axis_identifiers_in_lint_axis_order_stay_english(self):
        """The stable axis identifiers (used as dict keys / report headers
        elsewhere) must NOT be translated — only LINT_AXIS_LABEL_ZH's values
        are bilingual display strings (issue #365 AC scope)."""
        assert LINT_AXIS_ORDER == ("Freshness", "Coherence", "Coverage", "Lifecycle")
