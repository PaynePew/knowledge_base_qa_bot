"""Deep module per Ousterhout. Public surface: ``render_charts``, ``CHARTS_DIR``.

matplotlib chart generation for the Phase 8 retrieval comparison report
(PRD #100, issue #104). Turns two ``StackScores`` (Stack A = Wiki + BM25,
Stack B = Vector RAG) into the committed PNGs the report embeds.

PRD #100 forbids a naive cross-type aggregate, so every chart draws the **Core**
Paraphrase Types and the **Structural probe** types as SEPARATE figures — a
reader can never read a probe's deliberately-rigged result as part of the Core
story. Three chart kinds per family:

  1. grouped-bar of per-type hit_rate@k (Stack A vs Stack B);
  2. diverging-delta bar of the signed Δ (B − A) per type, coloured by winner
     (Stack B win = one colour, Stack A win = another, tie = grey);
  3. grouped-bar of per-type MRR (Stack A vs Stack B).

Files are written via the same atomic tmp + ``os.replace`` discipline as the
report (CODING_STANDARD §2.6): matplotlib saves to a sibling ``.tmp`` path which
is then atomically renamed, so a crash mid-render never leaves a half-written
PNG for the next reader.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Force the non-interactive Agg backend BEFORE importing pyplot: the report is
# generated headless (CI / AFK agent), where no display server exists.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

from .loader import replace_atomic
from .models import CORE_PARAPHRASE_TYPES, PROBE_PARAPHRASE_TYPES

if TYPE_CHECKING:
    # Import only for type-checking: runner imports charts at runtime, so a
    # runtime import here would be circular. `from __future__ import annotations`
    # keeps the StackScores annotations as strings, so no runtime import needed.
    from .runner import StackScores

_PKG_ROOT = Path(__file__).resolve().parent
CHARTS_DIR = _PKG_ROOT / "charts"

# Colour roles (winner-coded). Stack B is the "challenger" Vector RAG arm;
# Stack C is the Hybrid (RRF over the wiki corpus) arm.
_COLOR_STACK_A = "#4c72b0"  # Wiki + BM25
_COLOR_STACK_B = "#dd8452"  # Vector RAG
_COLOR_STACK_C = "#55a868"  # Hybrid (BM25 + dense, RRF)
_COLOR_TIE = "#999999"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_charts(
    stack_a: StackScores,
    stack_b: StackScores,
    charts_dir: Path = CHARTS_DIR,
    stack_c: StackScores | None = None,
) -> list[Path]:
    """Render all comparison charts (Core + probes, separate figures) to ``charts_dir``.

    Returns the list of PNG paths written, in a stable order. Each is produced
    via atomic tmp + ``os.replace`` so partial writes never surface. Types absent
    from every Stack's score map are skipped (a chart is only drawn for the types
    actually present, so an offline subset run still produces valid figures).

    ``stack_c`` (Hybrid) is the Phase 13 third arm: when present the grouped-bar
    hit_rate and MRR figures gain a third bar per type. The signed-delta figure
    stays the legacy Wiki-vs-RAG (B − A) view — the three-arm omnibus + post-hoc
    in the report is the authoritative three-way comparison.
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    k = stack_a.k
    written: list[Path] = []
    three_arms = stack_c is not None
    arms_label = "A vs B vs C" if three_arms else "Stack A vs Stack B"

    families = [
        ("core", CORE_PARAPHRASE_TYPES),
        ("probes", PROBE_PARAPHRASE_TYPES),
    ]
    for family, all_types in families:
        present = [t for t in all_types if t in stack_a.by_type or t in stack_b.by_type]
        if not present:
            continue
        types = present
        c_hit = {t: stack_c.by_type.get(t, 0.0) for t in types} if three_arms else None
        c_mrr = (
            {t: stack_c.mrr_by_type.get(t, 0.0) for t in types} if three_arms else None
        )
        written.append(
            _grouped_bar(
                charts_dir / f"{family}_hit_rate_at_{k}.png",
                types,
                {t: stack_a.by_type.get(t, 0.0) for t in types},
                {t: stack_b.by_type.get(t, 0.0) for t in types},
                title=f"{family.capitalize()} — hit_rate@{k} ({arms_label})",
                ylabel=f"hit_rate@{k}",
                c_by_type=c_hit,
            )
        )
        written.append(
            _diverging_delta(
                charts_dir / f"{family}_delta_hit_rate_at_{k}.png",
                types,
                {t: stack_a.by_type.get(t, 0.0) for t in types},
                {t: stack_b.by_type.get(t, 0.0) for t in types},
                title=f"{family.capitalize()} — Δ hit_rate@{k} (B − A), coloured by winner",
            )
        )
        written.append(
            _grouped_bar(
                charts_dir / f"{family}_mrr_at_{k}.png",
                types,
                {t: stack_a.mrr_by_type.get(t, 0.0) for t in types},
                {t: stack_b.mrr_by_type.get(t, 0.0) for t in types},
                title=f"{family.capitalize()} — MRR@{k} ({arms_label})",
                ylabel=f"MRR@{k}",
                c_by_type=c_mrr,
            )
        )
    return written


# ---------------------------------------------------------------------------
# Chart kinds
# ---------------------------------------------------------------------------
def _grouped_bar(
    path: Path,
    types: list[str],
    a_by_type: dict[str, float],
    b_by_type: dict[str, float],
    title: str,
    ylabel: str,
    c_by_type: dict[str, float] | None = None,
) -> Path:
    """Grouped-bar of a per-type metric (Stack A beside B, plus C when given)."""
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(types)), 4.5))
    positions = range(len(types))
    if c_by_type is not None:
        width = 0.27
        offsets = (-width, 0.0, width)
        series = [
            (a_by_type, "Stack A (Wiki + BM25)", _COLOR_STACK_A),
            (b_by_type, "Stack B (Vector RAG)", _COLOR_STACK_B),
            (c_by_type, "Stack C (Hybrid, RRF)", _COLOR_STACK_C),
        ]
    else:
        width = 0.38
        offsets = (-width / 2, width / 2)
        series = [
            (a_by_type, "Stack A (Wiki + BM25)", _COLOR_STACK_A),
            (b_by_type, "Stack B (Vector RAG)", _COLOR_STACK_B),
        ]
    for offset, (by_type, label, color) in zip(offsets, series):
        ax.bar(
            [p + offset for p in positions],
            [by_type[t] for t in types],
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _savefig_atomic(fig, path)


def _diverging_delta(
    path: Path,
    types: list[str],
    a_by_type: dict[str, float],
    b_by_type: dict[str, float],
    title: str,
) -> Path:
    """Horizontal diverging bar of Δ = B − A per type, coloured by winner; atomic-write."""
    deltas = [b_by_type[t] - a_by_type[t] for t in types]
    colors = [
        _COLOR_STACK_B if d > 0 else _COLOR_STACK_A if d < 0 else _COLOR_TIE
        for d in deltas
    ]
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.7 * len(types) + 1.5)))
    positions = range(len(types))
    ax.barh(list(positions), deltas, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(types)
    ax.set_xlabel("Δ (Stack B − Stack A); positive = Vector RAG wins")
    ax.set_title(title)
    fig.tight_layout()
    return _savefig_atomic(fig, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _savefig_atomic(fig, path: Path) -> Path:
    """Save ``fig`` to ``path`` via tmp + ``replace_atomic`` (CODING_STANDARD §2.6)."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=f"{path.stem}_"
    )
    os.close(fd)  # matplotlib opens the file itself; we only needed a unique name
    try:
        fig.savefig(tmp_name, format="png", dpi=120)
        replace_atomic(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    finally:
        plt.close(fig)
    return path
