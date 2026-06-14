"""Shallow module per Ousterhout. Language selector for the negative-case eval (#256).

The harness is English by default; setting ``KB_EVAL_LANG=zh`` points the
``runner`` / ``calibrate`` CLIs at the committed Traditional-Chinese corpus
(``corpus_zh``) + case sets so the same gate-level, LLM-free methodology
(#249 / #253) runs for Chinese. ``report_suffix`` keeps a zh run from clobbering the
committed English ``report.md`` / ``calibration_report.md``.

Importing the English defaults here (not re-deriving them) means existing call sites
and tests are byte-for-byte unchanged; only the two CLI ``main`` functions consult
``resolve_lang``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .cases import NEGATIVE_CASES
from .cases_zh import NEGATIVE_CASES_ZH
from .driver import CORPUS_DIR
from .models import NegativeCase, PositiveCase
from .positive_cases import POSITIVE_CASES
from .positive_cases_zh import POSITIVE_CASES_ZH

_PKG_ROOT = Path(__file__).resolve().parent

CORPUS_DIR_EN = CORPUS_DIR  # the English ./corpus (driver default)
CORPUS_DIR_ZH = _PKG_ROOT / "corpus_zh"

DEFAULT_LANG = "en"
ENV_VAR = "KB_EVAL_LANG"


@dataclass(frozen=True)
class LangConfig:
    """Everything a CLI ``main`` needs to evaluate one language."""

    lang: str
    corpus_dir: Path
    positive_cases: list[PositiveCase]
    negative_cases: list[NegativeCase]
    report_suffix: str  # "" for en, "_zh" for zh


_CONFIGS: dict[str, LangConfig] = {
    "en": LangConfig("en", CORPUS_DIR_EN, POSITIVE_CASES, NEGATIVE_CASES, ""),
    "zh": LangConfig("zh", CORPUS_DIR_ZH, POSITIVE_CASES_ZH, NEGATIVE_CASES_ZH, "_zh"),
}


def resolve_lang(lang: str | None = None) -> LangConfig:
    """Resolve the eval language from the arg or ``KB_EVAL_LANG`` (default ``en``).

    An explicitly-requested but unsupported language raises ``ValueError`` rather
    than silently running English (fail loud).
    """
    chosen = (lang or os.getenv(ENV_VAR) or DEFAULT_LANG).lower()
    if chosen not in _CONFIGS:
        supported = ", ".join(sorted(_CONFIGS))
        raise ValueError(
            f"Unsupported {ENV_VAR}={chosen!r}; supported languages: {supported}."
        )
    return _CONFIGS[chosen]
