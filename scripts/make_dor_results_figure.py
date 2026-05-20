"""Publication-quality main-text figure for the DoR analysis.

Four panels, 2 x 2:
  a  Violin + box: candidate-abandonment DoR by class (with class icons)
  b  Violin + box: stable-site DoR by class (with class icons)
  c  Horizontal stacked bar: candidate-abandonment outcome proportions
  d  Horizontal stacked bar: stable-site outcome proportions

Style matches v4/scripts/reporting/make_supp_quality_figure.py so the main
and supplementary figures share a visual identity (Okabe-Ito-derived
palette, DejaVu Sans, identical panel labels and tick conventions).

Output: v4/plots/dor_results_main.{png,pdf}  (180 mm wide, 600 dpi PNG)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
V3_SCORES = REPO / "v3" / "data" / "test_site_dor_v3.csv"
V4_SCORES = REPO / "v4" / "data" / "test_site_dor_v4.csv"
OUT_PNG = REPO / "v4" / "plots" / "dor_results_main.png"
OUT_PDF = REPO / "v4" / "plots" / "dor_results_main.pdf"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette (shared with supplementary figure)
# ---------------------------------------------------------------------------
RECOVERING = "#009E73"      # Okabe-Ito green
DEGRADED = "#D55E00"        # Okabe-Ito vermillion
INDISTINGUISHABLE = "#999999"
NO_DATA = "#E5E5E5"

BLUE = "#0072B2"            # candidate-abandonment / stable cropland
ORANGE = "#E69F00"          # built-loss / stable built-up
GREEN = "#009E73"           # stable near-natural
GREY = "#999999"
DARK = "#222222"

CAT_COLORS = {
    "recovering": RECOVERING,
    "degraded": DEGRADED,
    "indistinguishable": INDISTINGUISHABLE,
    "no data": NO_DATA,
}
CAT_ORDER = ["recovering", "degraded", "indistinguishable", "no data"]

CAND_LABELS = {"crop_loss": "Cropland-loss", "built_loss": "Built-loss"}
STABLE_LABELS = {
    "stable_nature": "Stable\nnear-natural",
    "stable_crop": "Stable\ncropland",
    "stable_built": "Stable\nbuilt-up",
}
CAND_COLORS = {"crop_loss": BLUE, "built_loss": ORANGE}
STABLE_COLORS = {
    "stable_nature": GREEN,
    "stable_crop": BLUE,
    "stable_built": ORANGE,
}

# Pooled candidate-abandonment threshold from the methods
CANDIDATE_THRESHOLD = 0.4859
# Per-class stable thresholds
STABLE_THRESHOLDS = {
    "stable_nature": 0.4861,
    "stable_crop": 0.4823,
    "stable_built": 0.4948,
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def set_journal_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#EAEAEA", linewidth=0.45)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.18, 1.10, text, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# Simple vector icons (no external dependencies)
# ---------------------------------------------------------------------------
def _draw_icon(ax: plt.Axes, kind: str, x: float, y: float, size: float,
               color: str) -> None:
    """Draw a small class icon at axes-fraction coords (x, y).

    size is in axes-fraction units (height of icon bounding box).
    """
    # Translate axes-fraction to display, draw in axes coords.
    half = size / 2

    if kind == "leaf":
        # Stylised leaf: tear-drop oval tilted slightly, with central vein.
        leaf = mpatches.Ellipse((x, y), width=size * 0.65, height=size,
                                angle=20,
                                facecolor=color, edgecolor=color,
                                transform=ax.transAxes, clip_on=False,
                                zorder=10, linewidth=0.6)
        ax.add_patch(leaf)
        # Midvein: rotated line aligned with the ellipse major axis
        ang = np.deg2rad(20)
        dx = np.sin(ang) * half * 0.85
        dy = np.cos(ang) * half * 0.85
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
                color="white", linewidth=0.8,
                transform=ax.transAxes, clip_on=False, zorder=11)

    elif kind == "wheat":
        # Stylised wheat sheaf: vertical stem + three pairs of grain ovals.
        ax.plot([x, x], [y - half, y + half * 0.4],
                color=color, linewidth=1.2,
                transform=ax.transAxes, clip_on=False, zorder=10)
        for i, frac in enumerate([0.15, -0.15, -0.45]):
            for side in (-1, 1):
                ang_x = x + side * half * 0.4
                ang_y = y + half * frac
                e = mpatches.Ellipse((ang_x, ang_y),
                                     width=half * 0.45, height=half * 0.7,
                                     angle=30 * side,
                                     facecolor=color, edgecolor=color,
                                     transform=ax.transAxes,
                                     clip_on=False, zorder=11, linewidth=0.4)
                ax.add_patch(e)

    elif kind == "building":
        # Simple house: rectangle + triangular roof + door.
        body_w = half * 1.4
        body_h = half * 1.2
        body = Rectangle((x - body_w / 2, y - half * 0.95),
                         body_w, body_h,
                         facecolor=color, edgecolor=color,
                         transform=ax.transAxes, clip_on=False,
                         zorder=10, linewidth=0.4)
        ax.add_patch(body)
        # Roof
        roof = mpatches.Polygon(
            [
                (x - body_w / 2 - half * 0.1, y + half * 0.25),
                (x + body_w / 2 + half * 0.1, y + half * 0.25),
                (x, y + half * 0.95),
            ],
            facecolor=color, edgecolor=color,
            transform=ax.transAxes, clip_on=False, zorder=11,
            linewidth=0.4,
        )
        ax.add_patch(roof)
        # Door
        door = Rectangle((x - half * 0.16, y - half * 0.95),
                         half * 0.32, half * 0.55,
                         facecolor="white", edgecolor="white",
                         transform=ax.transAxes, clip_on=False,
                         zorder=12, linewidth=0)
        ax.add_patch(door)

    elif kind == "wheat-building":
        # Composite: a small wheat sheaf and a small building side by side
        _draw_icon(ax, "wheat", x - size * 0.32, y, size * 0.85, color)
        _draw_icon(ax, "building", x + size * 0.32, y, size * 0.85, color)


CAND_ICONS = {"crop_loss": "wheat", "built_loss": "building"}
STABLE_ICONS = {
    "stable_nature": "leaf",
    "stable_crop": "wheat",
    "stable_built": "building",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def proportions(df: pd.DataFrame, group_col: str, label_map: dict) -> pd.DataFrame:
    rows = []
    for grp, label in label_map.items():
        sub = df[df[group_col] == grp]
        n = len(sub)
        for cat in CAT_ORDER:
            count = (sub["category"] == cat).sum()
            rows.append({"group_key": grp, "group": label, "category": cat,
                         "prop": count / n if n else 0, "count": count, "n": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def violin_panel(ax: plt.Axes, df: pd.DataFrame, group_col: str,
                 label_map: dict, color_map: dict, icon_map: dict,
                 threshold: float | dict | None,
                 title: str) -> None:
    groups = list(label_map.keys())
    labels = list(label_map.values())
    rng = np.random.default_rng(42)

    box_data = [df[df[group_col] == g]["dor_knn"].dropna().values
                for g in groups]

    parts = ax.violinplot(box_data, positions=range(len(groups)),
                          widths=0.78, showmeans=False, showmedians=False,
                          showextrema=False)
    for body, grp in zip(parts["bodies"], groups):
        body.set_facecolor(color_map[grp])
        body.set_alpha(0.22)
        body.set_edgecolor(color_map[grp])
        body.set_linewidth(0.6)

    bp = ax.boxplot(
        box_data, positions=range(len(groups)), widths=0.22,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="white", linewidth=1.3),
        whiskerprops=dict(linewidth=0.7, color=DARK),
        capprops=dict(linewidth=0.7, color=DARK),
        boxprops=dict(linewidth=0.7),
    )
    for patch, grp in zip(bp["boxes"], groups):
        patch.set_facecolor(color_map[grp])
        patch.set_alpha(0.95)
        patch.set_edgecolor(color_map[grp])

    # Jittered scatter
    for i, (grp, vals) in enumerate(zip(groups, box_data)):
        xj = rng.uniform(-0.30, 0.30, len(vals)) + i
        ax.scatter(xj, vals, s=4.5, color=color_map[grp], alpha=0.35,
                   linewidths=0, zorder=2)

    # Threshold(s)
    if isinstance(threshold, dict):
        for i, grp in enumerate(groups):
            t = threshold[grp]
            ax.hlines(t, i - 0.42, i + 0.42, color=DARK,
                      linewidth=0.9, linestyle="--", zorder=4)
        ax.text(0.02, 0.97, "per-class threshold",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6, color=DARK,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85))
    elif threshold is not None:
        ax.axhline(threshold, color=DARK, linewidth=0.9, linestyle="--",
                   zorder=4)
        ax.text(0.02, 0.97, f"threshold = {threshold:.3f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6, color=DARK,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, ha="center")
    # colour the x-tick labels to match groups
    for tick, grp in zip(ax.get_xticklabels(), groups):
        tick.set_color(color_map[grp])
        tick.set_fontweight("bold")

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("DoR score")
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.set_title(title, pad=16)

    clean_axis(ax, grid_axis="y")

    # n= labels below x-tick labels
    for i, grp in enumerate(groups):
        n = (df[group_col] == grp).sum()
        ax.text(i, -0.20, f"n = {n}", ha="center", va="top",
                fontsize=6, color=GREY,
                transform=ax.get_xaxis_transform())

    # Icons above the plot area, between title and top spine
    n_groups = len(groups)
    for i, grp in enumerate(groups):
        frac = (i + 0.5) / n_groups
        _draw_icon(ax, icon_map[grp], frac, 1.05, 0.08, color_map[grp])


def stacked_bar_panel(ax: plt.Axes, props: pd.DataFrame, label_map: dict,
                      color_map: dict,
                      title: str, show_legend: bool = False) -> None:
    """Horizontal stacked bars with inline % labels and right-side n totals."""
    groups = list(label_map.keys())
    labels = [label_map[g].replace("\n", " ") for g in groups]
    y_pos = np.arange(len(groups))[::-1]

    left = np.zeros(len(groups))
    for cat in CAT_ORDER:
        sub = props[props["category"] == cat].set_index("group_key")
        vals = np.array([sub.loc[g, "prop"] if g in sub.index else 0
                         for g in groups])
        bars = ax.barh(y_pos, vals, left=left,
                       color=CAT_COLORS[cat], height=0.55,
                       edgecolor="white", linewidth=0.6,
                       label=cat.capitalize())
        for rect, v, l in zip(bars, vals, left):
            if v >= 0.08:
                ax.text(l + v / 2, rect.get_y() + rect.get_height() / 2,
                        f"{v*100:.0f}%",
                        ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
        left += vals

    # n totals on the far right
    for i, grp in enumerate(groups):
        n = int(props[props["group_key"] == grp]["n"].iloc[0])
        ax.text(1.02, y_pos[i], f"n = {n}", va="center", ha="left",
                fontsize=6, color=GREY,
                transform=ax.get_yaxis_transform())

    # Coloured y-tick labels (no icons here to avoid overlap; colour conveys class)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    for tick, grp in zip(ax.get_yticklabels(), groups):
        tick.set_color(color_map[grp])
        tick.set_fontweight("bold")

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Proportion of sites")
    ax.set_title(title, pad=8)
    clean_axis(ax, grid_axis="x")

    if show_legend:
        handles = [mpatches.Patch(facecolor=CAT_COLORS[c], edgecolor="white",
                                  label=c.capitalize())
                   for c in CAT_ORDER]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.32), ncol=4,
                  frameon=False, handlelength=1.0, handleheight=1.0,
                  columnspacing=1.4, borderpad=0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_journal_style()

    cand = pd.read_csv(V3_SCORES).rename(columns={"category_knn": "category"})
    stab = pd.read_csv(V4_SCORES).rename(columns={"category_knn": "category"})

    cand_props = proportions(cand, "parent_label", CAND_LABELS)
    stab_props = proportions(stab, "parent_label", STABLE_LABELS)

    # 180 mm x ~130 mm (two-column)
    fig = plt.figure(figsize=(7.09, 5.7))
    gs = fig.add_gridspec(
        2, 2,
        left=0.10, right=0.93,
        top=0.91, bottom=0.13,
        hspace=0.55, wspace=0.36,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    violin_panel(ax_a, cand, "parent_label", CAND_LABELS, CAND_COLORS,
                 CAND_ICONS, threshold=CANDIDATE_THRESHOLD,
                 title="Candidate abandonment sites")
    panel_label(ax_a, "a")

    violin_panel(ax_b, stab, "parent_label", STABLE_LABELS, STABLE_COLORS,
                 STABLE_ICONS, threshold=STABLE_THRESHOLDS,
                 title="Stable-site controls")
    panel_label(ax_b, "b")

    stacked_bar_panel(ax_c, cand_props, CAND_LABELS, CAND_COLORS,
                      title="Candidate abandonment — outcomes",
                      show_legend=False)
    panel_label(ax_c, "c")

    stacked_bar_panel(ax_d, stab_props, STABLE_LABELS, STABLE_COLORS,
                      title="Stable-site controls — outcomes",
                      show_legend=True)
    panel_label(ax_d, "d")

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}  ({OUT_PNG.stat().st_size/1024:.0f} KB)")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
