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

# Colour roles (winner-coded). Stack B is the "challenger" Vector RAG arm.
_COLOR_STACK_A = "#4c72b0"  # Wiki + BM25
_COLOR_STACK_B = "#dd8452"  # Vector RAG
_COLOR_TIE = "#999999"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_charts(
    stack_a: StackScores,
    stack_b: StackScores,
    charts_dir: Path = CHARTS_DIR,
) -> list[Path]:
    """Render all comparison charts (Core + probes, separate figures) to ``charts_dir``.

    Returns the list of PNG paths written, in a stable order. Each is produced
    via atomic tmp + ``os.replace`` so partial writes never surface. Types absent
    from both Stacks' score maps are skipped (a chart is only drawn for the types
    actually present, so an offline subset run still produces valid figures).
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    k = stack_a.k
    written: list[Path] = []

    families = [
        ("core", CORE_PARAPHRASE_TYPES),
        ("probes", PROBE_PARAPHRASE_TYPES),
    ]
    for family, all_types in families:
        types = [t for t in all_types if t in stack_a.by_type or t in stack_b.by_type]
        if not types:
            continue
        written.append(
            _grouped_bar(
                charts_dir / f"{family}_hit_rate_at_{k}.png",
                types,
                {t: stack_a.by_type.get(t, 0.0) for t in types},
                {t: stack_b.by_type.get(t, 0.0) for t in types},
                title=f"{family.capitalize()} — hit_rate@{k} (Stack A vs Stack B)",
                ylabel=f"hit_rate@{k}",
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
                title=f"{family.capitalize()} — MRR@{k} (Stack A vs Stack B)",
                ylabel=f"MRR@{k}",
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
) -> Path:
    """Grouped-bar of a per-type metric, Stack A beside Stack B; atomic-write the PNG."""
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(types)), 4.5))
    positions = range(len(types))
    width = 0.38
    a_vals = [a_by_type[t] for t in types]
    b_vals = [b_by_type[t] for t in types]
    ax.bar(
        [p - width / 2 for p in positions],
        a_vals,
        width,
        label="Stack A (Wiki + BM25)",
        color=_COLOR_STACK_A,
    )
    ax.bar(
        [p + width / 2 for p in positions],
        b_vals,
        width,
        label="Stack B (Vector RAG)",
        color=_COLOR_STACK_B,
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
