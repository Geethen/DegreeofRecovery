"""Shared journal-quality matplotlib style for the v5 buffer-decision figures.

One import, one `apply_style()` call, and every figure in the buffer-decision
report inherits a consistent, publication-ready look:
  - Nimbus Sans (Helvetica-metric) / Liberation Sans fallback, sized for print
  - hairline spines (top/right removed), subtle ticks pointing out
  - colourblind-safe Okabe-Ito categorical palette + perceptually-uniform maps
  - vector-friendly defaults (TrueType fonts embedded in PDF/SVG, 600 dpi raster)

Use `savefig_dual(fig, stem)` to write both a 600-dpi PNG (for the HTML report)
and a vector PDF (for the manuscript) from one call.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe categorical palette (the project's class colours map
# onto it already: nature=bluish-green, crop=blue, built=orange).
OKABE_ITO = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "bluish_green": "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey":         "#7F7F7F",
}

CLASS_COLORS = {
    "stable_nature": "#009E73",
    "stable_crop":   "#0072B2",
    "stable_built":  "#E69F00",
    "built_loss":    "#D55E00",
    "crop_loss":     "#CC79A7",
}

CLASS_LABELS = {
    "stable_nature": "Stable natural",
    "stable_crop":   "Stable cropland",
    "stable_built":  "Stable built-up",
    "built_loss":    "Built-loss",
    "crop_loss":     "Crop-loss",
}

# Axis accent colours used consistently across panels and the report cards.
AXIS_COLORS = {
    "contamination":   "#9E2A2B",   # deep red
    "separability":    "#009E73",   # green
    "autocorrelation": "#0072B2",   # blue
    "confidence":      "#CC79A7",   # purple
    "retention":       "#7F7F7F",   # grey
    "overall":         "#111111",   # near-black
}

# Sequential map for desirability/axis panels (perceptually uniform, prints well
# in greyscale). Highlight (winner) drawn in a contrasting accent.
SEQ_CMAP = "viridis"
SEQ_CMAP_D = "cividis"          # for the overall-D panel (distinct from axes)
DIVERGE_CMAP = "RdBu_r"
BAD_COLOR = "#e9e9e9"           # disqualified / undefined cells
HILITE = "#D81B60"              # winner highlight (magenta-red, pops on viridis)


def apply_style() -> None:
    """Set rcParams for the whole process. Idempotent."""
    mpl.rcParams.update({
        # --- typography -----------------------------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Helvetica",
                            "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
        # --- spines / ticks -------------------------------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # --- lines / grid ---------------------------------------------------
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "axes.grid": False,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        # --- figure ---------------------------------------------------------
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 600,
        # --- vector friendliness -------------------------------------------
        "pdf.fonttype": 42,        # embed TrueType (editable text in Illustrator)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def style_heatmap_axis(ax, col_labels, row_labels, xlabel, ylabel,
                       title=None) -> None:
    """Common tick/label treatment for an (inner x outer) heatmap cell grid."""
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=6)
    # heatmaps look cleaner without spines
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def cmap_with_bad(name: str, bad=BAD_COLOR):
    cm = plt.get_cmap(name).copy()
    cm.set_bad(bad)
    return cm


def savefig_dual(fig, stem: Path, png: bool = True, pdf: bool = True) -> None:
    """Write `<stem>.png` (600 dpi raster for HTML) and `<stem>.pdf` (vector)."""
    stem = Path(stem)
    if png:
        fig.savefig(stem.with_suffix(".png"), dpi=600)
    if pdf:
        fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
