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

import re
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from svgpath2mpl import parse_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
V3_SCORES = REPO / "v3" / "data" / "test_site_dor_v3.csv"
V4_SCORES = REPO / "v4" / "data" / "test_site_dor_v4.csv"
OUT_PNG = REPO / "v4" / "plots" / "dor_results_main.png"
ICON_DIR = REPO / "v4" / "plots" / "icons"
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
    "stable_nature": "Stable\nnatural",
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


def panel_label(ax: plt.Axes, text: str, x: float = -0.18) -> None:
    ax.text(x, 1.10, text, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# Font Awesome SVG icons → matplotlib Path patches
# ---------------------------------------------------------------------------
_VIEWBOX_RE = re.compile(r'viewBox="([^"]+)"')
_PATH_D_RE = re.compile(r'<path[^>]*\bd="([^"]+)"')


@lru_cache(maxsize=None)
def _load_fa_icon(name: str) -> tuple:
    """Load a Font Awesome SVG icon as (mpl Path, (w, h)) in viewBox units.

    The path is y-flipped (SVG y is down, matplotlib y is up) and translated
    so the icon's bounding box has its lower-left corner at (0, 0).
    """
    svg = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    vb = list(map(float, _VIEWBOX_RE.search(svg).group(1).split()))
    _, _, vb_w, vb_h = vb
    d = _PATH_D_RE.search(svg).group(1)
    path = parse_path(d)
    # Flip vertically (SVG → mpl convention).
    verts = path.vertices.copy()
    verts[:, 1] = vb_h - verts[:, 1]
    path.vertices[:] = verts
    # Shift to origin using the actual rendered bbox so disparate icons share
    # a consistent baseline.
    bb = path.get_extents()
    path.vertices[:, 0] -= bb.x0
    path.vertices[:, 1] -= bb.y0
    w = bb.width
    h = bb.height
    return path, (w, h)


def _draw_fa_icon(ax: plt.Axes, name: str, cx: float, base_y: float,
                  height: float, color: str) -> None:
    """Draw a Font Awesome icon, scaled so its rendered height is `height`
    in axes fraction. The icon is horizontally centred at cx; its baseline
    (lower edge of bbox) is at base_y.

    Axes-fraction units are anisotropic (width != height in display pixels),
    so we correct width-scaling using the axes aspect to keep icons un-skewed.
    """
    path, (w, h) = _load_fa_icon(name)
    # Aspect = display-px-per-axes-fraction. Use it to choose the x-scale
    # that yields equal display-height and display-width per icon-unit.
    bbox = ax.get_window_extent()
    if bbox.width == 0 or bbox.height == 0 or h == 0:
        return
    aspect = bbox.height / bbox.width  # >1 if axes are taller than wide
    # Axes-frac per icon-unit (vertical) is height/h. Horizontal must satisfy
    # x_axes_frac * bbox.width == y_axes_frac * bbox.height (per icon unit)
    # so x = y * aspect.
    sy = height / h
    sx = sy * aspect
    icon_w_axes = w * sx
    # Build the transform: scale → translate → axes coords
    tr = (mtransforms.Affine2D()
          .scale(sx, sy)
          .translate(cx - icon_w_axes / 2, base_y)
          + ax.transAxes)
    patch = PathPatch(path, transform=tr,
                      facecolor=color, edgecolor="none",
                      clip_on=False, zorder=12)
    ax.add_patch(patch)


def _draw_pan_icon(ax: plt.Axes, kind: str, cx: float, base_y: float,
                   h: float, color: str) -> None:
    """Dispatch: draw a class icon sitting on a pan with base at base_y."""
    _draw_fa_icon(ax, kind, cx, base_y, h, color)


def _draw_scale(ax: plt.Axes, x: float, y: float, size: float,
                left_kind: str, right_kind: str,
                tilt: str,
                left_color: str, right_color: str,
                beam_color: str = DARK) -> None:
    """Draw a small balance scale with class icons sitting on each pan.

    (x, y) is the centre of the scale's fulcrum/base in axes-fraction coords.
    `size` is the total icon footprint height (axes-fraction).
    `tilt` is one of {"left", "right", "balanced"} — which pan is lower.
    """
    # Beam half-length and tilt angle
    arm = size * 0.55
    if tilt == "balanced":
        ang = 0.0
    elif tilt == "left":
        ang = np.deg2rad(13)   # beam tilts down on the left
    else:  # "right"
        ang = np.deg2rad(-13)

    # Use a tilt-shape transform that accounts for axes aspect, so the
    # angle reads visually even when the axes are not square.
    bbox = ax.get_window_extent()
    aspect = bbox.height / bbox.width if bbox.width else 1.0

    beam_cy = y + size * 0.18  # beam sits above the fulcrum tip
    dx = arm * np.cos(ang)
    dy = arm * np.sin(ang) * aspect

    lx, ly = x - dx, beam_cy - dy   # left pan attaches here
    rx, ry = x + dx, beam_cy + dy   # right pan attaches here

    # Fulcrum (triangle) and base
    base_w = size * 0.42
    base_h = size * 0.06
    base = Rectangle((x - base_w / 2, y - base_h),
                     base_w, base_h,
                     facecolor=beam_color, edgecolor=beam_color,
                     transform=ax.transAxes, clip_on=False,
                     zorder=8, linewidth=0)
    ax.add_patch(base)
    fulcrum = mpatches.Polygon(
        [
            (x - size * 0.18, y),
            (x + size * 0.18, y),
            (x, beam_cy),
        ],
        facecolor=beam_color, edgecolor=beam_color,
        transform=ax.transAxes, clip_on=False, zorder=9,
        linewidth=0,
    )
    ax.add_patch(fulcrum)

    # Beam
    ax.plot([lx, rx], [ly, ry], color=beam_color, linewidth=1.2,
            solid_capstyle="round",
            transform=ax.transAxes, clip_on=False, zorder=10)

    # Pan hangers
    pan_drop = size * 0.18
    ax.plot([lx, lx], [ly, ly - pan_drop], color=beam_color, linewidth=0.7,
            transform=ax.transAxes, clip_on=False, zorder=10)
    ax.plot([rx, rx], [ry, ry - pan_drop], color=beam_color, linewidth=0.7,
            transform=ax.transAxes, clip_on=False, zorder=10)

    # Pans (shallow arcs drawn as thin ellipses)
    pan_w = size * 0.36
    pan_h = size * 0.08
    left_pan_y = ly - pan_drop
    right_pan_y = ry - pan_drop
    ax.add_patch(mpatches.Ellipse((lx, left_pan_y), width=pan_w, height=pan_h,
                                  facecolor=beam_color, edgecolor=beam_color,
                                  transform=ax.transAxes, clip_on=False,
                                  zorder=11, linewidth=0))
    ax.add_patch(mpatches.Ellipse((rx, right_pan_y), width=pan_w, height=pan_h,
                                  facecolor=beam_color, edgecolor=beam_color,
                                  transform=ax.transAxes, clip_on=False,
                                  zorder=11, linewidth=0))

    # Icons sitting on each pan
    icon_h = size * 0.62
    _draw_pan_icon(ax, left_kind, lx, left_pan_y + pan_h * 0.3,
                   icon_h, left_color)
    _draw_pan_icon(ax, right_kind, rx, right_pan_y + pan_h * 0.3,
                   icon_h, right_color)


# Scale configuration per class:
# (left_icon, right_icon, tilt, left_color, right_color)
# Convention: non-natural reference on the left, near-natural (tree) on the right.
# Tilt direction indicates which reference pool the icon depicts as "heavier".
CAND_SCALE = {
    "crop_loss": ("tractor", "tree", "left", BLUE, GREEN),
    "built_loss": ("building", "tree", "left", ORANGE, GREEN),
}
STABLE_SCALE = {
    "stable_nature": ("tree", "tree", "balanced", GREEN, GREEN),
    "stable_crop": ("tractor", "tractor", "balanced", BLUE, BLUE),
    "stable_built": ("building", "building", "balanced", ORANGE, ORANGE),
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
                 label_map: dict, color_map: dict, scale_map: dict,
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
    ax.set_title(title, pad=56)

    clean_axis(ax, grid_axis="y")

    # n= labels below x-tick labels
    for i, grp in enumerate(groups):
        n = (df[group_col] == grp).sum()
        ax.text(i, -0.20, f"n = {n}", ha="center", va="top",
                fontsize=6, color=GREY,
                transform=ax.get_xaxis_transform())

    # Scale icons above the plot area, between title and top spine
    n_groups = len(groups)
    # Shrink scale slightly when more groups need to share the same width
    scale_size = 0.26 if n_groups <= 2 else 0.20
    for i, grp in enumerate(groups):
        frac = (i + 0.5) / n_groups
        left_kind, right_kind, tilt, lc, rc = scale_map[grp]
        _draw_scale(ax, frac, 1.08, scale_size,
                    left_kind, right_kind, tilt, lc, rc)


def stacked_bar_panel(ax: plt.Axes, props: pd.DataFrame, label_map: dict,
                      color_map: dict,
                      title: str, show_legend: bool = False,
                      slot_count: int | None = None) -> None:
    """Horizontal stacked bars with inline % labels and right-side n totals."""
    groups = list(label_map.keys())
    labels = [label_map[g].replace("\n", " ") for g in groups]
    slots = max(slot_count or len(groups), len(groups))
    y_pos = (np.arange(len(groups))[::-1]
             + (slots - len(groups)) / 2.0)

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
    ax.set_ylim(-0.5, slots - 0.5)
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

    # 180 mm x ~155 mm (two-column)
    fig = plt.figure(figsize=(7.09, 6.5))
    gs = fig.add_gridspec(
        2, 2,
        left=0.14, right=0.94,
        top=0.87, bottom=0.11,
        hspace=0.85, wspace=0.45,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    violin_panel(ax_a, cand, "parent_label", CAND_LABELS, CAND_COLORS,
                 CAND_SCALE, threshold=CANDIDATE_THRESHOLD,
                 title="Candidate abandonment sites")
    panel_label(ax_a, "a")

    violin_panel(ax_b, stab, "parent_label", STABLE_LABELS, STABLE_COLORS,
                 STABLE_SCALE, threshold=STABLE_THRESHOLDS,
                 title="Stable-site controls")
    panel_label(ax_b, "b")

    stacked_bar_panel(ax_c, cand_props, CAND_LABELS, CAND_COLORS,
                      title="Candidate abandonment — outcomes",
                      show_legend=False, slot_count=3)
    panel_label(ax_c, "c", x=-0.28)

    stacked_bar_panel(ax_d, stab_props, STABLE_LABELS, STABLE_COLORS,
                      title="Stable-site controls — outcomes",
                      show_legend=True, slot_count=3)
    panel_label(ax_d, "d", x=-0.28)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    print(f"Saved {OUT_PNG}  ({OUT_PNG.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
