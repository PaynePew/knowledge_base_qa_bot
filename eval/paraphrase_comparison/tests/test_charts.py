"""Chart generation tests (external behaviour only, CODING_STANDARD §0.2).

Asserts the matplotlib chart files are produced without error and that Core and
Structural-probe types are drawn as SEPARATE figures (PRD #100 — no naive
cross-type aggregate). Per the issue #104 test note, we do NOT assert pixel
content: only that the expected PNG files exist and are non-empty.
"""

from __future__ import annotations

from eval.paraphrase_comparison.charts import render_charts
from eval.paraphrase_comparison.models import (
    CORE_PARAPHRASE_TYPES,
    PROBE_PARAPHRASE_TYPES,
)
from eval.paraphrase_comparison.runner import StackScores

_ALL_TYPES = (*CORE_PARAPHRASE_TYPES, *PROBE_PARAPHRASE_TYPES)


def _stack(name: str, hit: float, mrr: float) -> StackScores:
    return StackScores(
        stack=name,
        k=3,
        by_type={t: hit for t in _ALL_TYPES},
        mrr_by_type={t: mrr for t in _ALL_TYPES},
        n_by_type={t: 8 for t in _ALL_TYPES},
    )


def test_render_charts_produces_png_files(tmp_path):
    a = _stack("Stack A", hit=0.8, mrr=0.7)
    b = _stack("Stack B", hit=0.6, mrr=0.5)

    written = render_charts(a, b, charts_dir=tmp_path)

    assert written, "expected at least one chart to be written"
    for path in written:
        assert path.exists(), f"{path} was not written"
        assert path.stat().st_size > 0, f"{path} is empty"
        assert path.suffix == ".png"


def test_core_and_probes_are_separate_figures(tmp_path):
    a = _stack("Stack A", hit=0.8, mrr=0.7)
    b = _stack("Stack B", hit=0.6, mrr=0.5)

    render_charts(a, b, charts_dir=tmp_path)
    names = {p.name for p in tmp_path.glob("*.png")}

    # Each family gets its own hit_rate, delta, and MRR figure.
    assert any(n.startswith("core_") for n in names)
    assert any(n.startswith("probes_") for n in names)
    assert "core_hit_rate_at_3.png" in names
    assert "core_delta_hit_rate_at_3.png" in names
    assert "core_mrr_at_3.png" in names
    assert "probes_hit_rate_at_3.png" in names


def test_render_charts_skips_absent_types(tmp_path):
    # Only Core types present (e.g. an offline subset run): no probe figures.
    a = StackScores(
        stack="Stack A",
        k=3,
        by_type={t: 0.5 for t in CORE_PARAPHRASE_TYPES},
        mrr_by_type={t: 0.4 for t in CORE_PARAPHRASE_TYPES},
        n_by_type={t: 8 for t in CORE_PARAPHRASE_TYPES},
    )
    b = StackScores(
        stack="Stack B",
        k=3,
        by_type={t: 0.3 for t in CORE_PARAPHRASE_TYPES},
        mrr_by_type={t: 0.2 for t in CORE_PARAPHRASE_TYPES},
        n_by_type={t: 8 for t in CORE_PARAPHRASE_TYPES},
    )

    render_charts(a, b, charts_dir=tmp_path)
    names = {p.name for p in tmp_path.glob("*.png")}

    assert any(n.startswith("core_") for n in names)
    assert not any(n.startswith("probes_") for n in names)
