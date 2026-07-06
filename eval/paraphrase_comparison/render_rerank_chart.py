"""Render the Stack C -> Stack C + rerank comparison chart (ADR-0019, #310).

The 4th-arm reranker (``KB_HYBRID_RERANK``, default-off, eval-only) is measured
inside ``run_comparison`` with the real ``bge-reranker-v2-m3`` cross-encoder, and
its numbers land in ``report.md`` under "Reranker Evaluation". That report already
carries the per-type tables; this module draws the matching PNG for the README's
evaluation section so a reader can see the precision lift at a glance without the
2.3 GB model on their box.

The numbers below are the **committed real-embedding measurement** from
``report.md`` (the ``## Reranker Evaluation`` section) — the single source of
truth. To refresh them after a re-measure, re-run the comparison with the reranker
arm and copy the rendered per-type hit@3 values here:

    uv sync --group rerank                                      # optional extra
    uv run python -m eval.paraphrase_comparison.run_comparison  # rerank on by default

Style, colours, and the atomic tmp + ``os.replace`` save discipline mirror
``charts.py`` (CODING_STANDARD §2.6) so the two chart families read as one set.
Run it as a module::

    uv run python -m eval.paraphrase_comparison.render_rerank_chart
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display server in CI / AFK runs
import matplotlib.pyplot as plt  # noqa: E402

from .charts import _COLOR_STACK_C  # green, Stack C (Hybrid, RRF)
from .loader import replace_atomic

_PKG_ROOT = Path(__file__).resolve().parent
CHARTS_DIR = _PKG_ROOT / "charts"
OUTPUT = CHARTS_DIR / "rerank_hit_rate_at_3.png"

_COLOR_RERANK = "#8172b3"  # purple, Stack C + cross-encoder rerank

# --- Committed real-embedding measurement (report.md ## Reranker Evaluation) ---
# hit_rate@3 per Paraphrase Type: (Stack C, Stack C + rerank).
_CORE: dict[str, tuple[float, float]] = {
    "synonym_swap": (0.880, 0.960),
    "word_reorder": (0.940, 0.960),
    "verbosity_expansion": (0.960, 0.960),
    "specificity_narrowing": (0.940, 0.960),
    "implicit_reference": (0.900, 0.960),
}
_PROBES: dict[str, tuple[float, float]] = {
    "typo_fatfinger": (0.400, 1.000),
    "industry_jargon": (0.600, 0.800),
}
# Core macro-average hit@3 and the measured dev-box latency, for the caption.
_CORE_MACRO = (0.924, 0.960)  # Stack C -> Stack C + rerank
_ADDED_LATENCY_MS = 5754.3


def _panel(ax, data: dict[str, tuple[float, float]], title: str) -> None:
    """Grouped-bar panel: Stack C beside Stack C + rerank, one group per type."""
    types = list(data)
    positions = range(len(types))
    width = 0.38
    c_vals = [data[t][0] for t in types]
    d_vals = [data[t][1] for t in types]
    bars_c = ax.bar(
        [p - width / 2 for p in positions],
        c_vals,
        width,
        label="Stack C (Hybrid, RRF)",
        color=_COLOR_STACK_C,
    )
    bars_d = ax.bar(
        [p + width / 2 for p in positions],
        d_vals,
        width,
        label="Stack C + rerank (cross-encoder)",
        color=_COLOR_RERANK,
    )
    for bars in (bars_c, bars_d):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_title(title, fontsize=11)


def render_rerank_chart(output: Path = OUTPUT) -> Path:
    """Draw the two-panel reranker comparison and atomic-save it to ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_core, ax_probe) = plt.subplots(
        1, 2, figsize=(11.0, 5.4), gridspec_kw={"width_ratios": [5, 2]}
    )
    _panel(ax_core, _CORE, "Core paraphrases (natural rewrites)")
    _panel(ax_probe, _PROBES, "Structural probes (adversarial)")
    ax_core.set_ylabel("hit_rate@3")

    # Reserve the top ~22% for a two-line title, then the legend, then the panels
    # (panel titles are drawn inside their axes, below this band); a generous
    # bottom/left margin keeps the 30°-rotated type labels from clipping.
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.08, right=0.975, wspace=0.22)
    fig.text(
        0.5,
        0.965,
        "Reranker evaluation — Stack C vs Stack C + rerank (hit_rate@3)",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.915,
        f"eval-only, default-off (ADR-0019) · Core macro {_CORE_MACRO[0]:.3f} → "
        f"{_CORE_MACRO[1]:.3f} · +{_ADDED_LATENCY_MS/1000:.1f} s/query on the dev "
        "box, never loaded on the VPS tenant",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    handles, labels = ax_core.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fd, tmp_name = tempfile.mkstemp(
        dir=output.parent, suffix=".tmp", prefix=f"{output.stem}_"
    )
    os.close(fd)
    try:
        fig.savefig(tmp_name, format="png", dpi=120)
        replace_atomic(tmp_name, output)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    finally:
        plt.close(fig)
    return output


if __name__ == "__main__":
    path = render_rerank_chart()
    print(f"wrote {path}")
