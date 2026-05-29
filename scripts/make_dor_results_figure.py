"""Publication-quality main-text figure for the DoR analysis (greyscale).

Two panels, side by side:
  a  Violin + box: DoR by class — built-loss abandonment alongside the
     stable-natural and stable-built-up controls, with per-class thresholds.
  b  Horizontal stacked bars: outcome proportions for the same three classes.

The cropland-loss abandonment class and the stable-cropland control class are
excluded from this analysis, so the figure covers built-loss only plus the two
remaining stable controls.

The figure is greyscale: classes are distinguished by position, bold labels and
balance-scale icons; outcome categories are distinguished by greyscale fill
shades plus hatch patterns so they remain separable in black-and-white print.

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
PLOTS_DIR = REPO / "v4" / "plots"
ICON_DIR = PLOTS_DIR / "icons"
# Two versions are written: greyscale and colour. Pick between them later.
OUT_GREY = PLOTS_DIR / "dor_results_main_grey.png"
OUT_COLOUR = PLOTS_DIR / "dor_results_main_colour.png"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palettes — a greyscale and a colour version are rendered to separate files.
# ---------------------------------------------------------------------------
DARK = "#222222"
GREY = "#888888"

CAT_ORDER = ["recovering", "degraded", "indistinguishable", "no data"]

# Greyscale palette: distinct greys, with hatch overlays so outcome categories
# print-separate in black-and-white. The lightest shade is kept dark enough to
# read against white.
GREY_PALETTE = {
    # Class shades (violins / bars / tick labels). Darkest = built-loss, the
    # abandonment class of interest.
    "class_shade": {
        "stable_nature": "#9A9A9A",
        "stable_built":  "#6E6E6E",
        "built_loss":    "#2E2E2E",
    },
    # Outcome categories: fill + hatch.
    "cat_style": {
        "recovering":        dict(facecolor="#E2E2E2", hatch="///",  label="Recovering"),
        "degraded":          dict(facecolor="#4D4D4D", hatch="",     label="Degraded"),
        "indistinguishable": dict(facecolor="#B0B0B0", hatch="...",  label="Indistinguishable"),
        "no data":           dict(facecolor="#D2D2D2", hatch="xxx",  label="No data"),
    },
    # Text colour over each category fill.
    "cat_text": {
        "recovering": DARK, "degraded": "white",
        "indistinguishable": DARK, "no data": DARK,
    },
}

# Colour palette: Okabe-Ito-derived, colour-blind-safe. Hatch is dropped (no
# longer needed to separate categories) but kept available if wanted in print.
COLOUR_PALETTE = {
    "class_shade": {
        "stable_nature": "#009E73",   # green
        "stable_built":  "#E69F00",   # orange
        "built_loss":    "#0072B2",   # blue — abandonment class of interest
    },
    "cat_style": {
        "recovering":        dict(facecolor="#009E73", hatch="",  label="Recovering"),
        "degraded":          dict(facecolor="#D55E00", hatch="",  label="Degraded"),
        "indistinguishable": dict(facecolor="#999999", hatch="",  label="Indistinguishable"),
        "no data":           dict(facecolor="#E5E5E5", hatch="",  label="No data"),
    },
    "cat_text": {
        "recovering": "white", "degraded": "white",
        "indistinguishable": "white", "no data": DARK,
    },
}

# Active palette — rebound by apply_palette() before each render.
CLASS_SHADE: dict = GREY_PALETTE["class_shade"]
CAT_STYLE: dict = GREY_PALETTE["cat_style"]
CAT_TEXT: dict = GREY_PALETTE["cat_text"]


def apply_palette(name: str) -> None:
    """Rebind the active palette globals to 'grey' or 'colour'."""
    global CLASS_SHADE, CAT_STYLE, CAT_TEXT
    pal = {"grey": GREY_PALETTE, "colour": COLOUR_PALETTE}[name]
    CLASS_SHADE = pal["class_shade"]
    CAT_STYLE = pal["cat_style"]
    CAT_TEXT = pal["cat_text"]


# Stable controls first, built-loss (the class of interest) last.
CLASS_ORDER = ["stable_nature", "stable_built", "built_loss"]
CLASS_LABELS = {
    "built_loss":    "Built-loss",
    "stable_nature": "Stable\nnatural",
    "stable_built":  "Stable\nbuilt-up",
}
# Source CSV per class: built-loss from the candidate scores, stable from v4.
CLASS_SOURCE = {
    "built_loss":    "cand",
    "stable_nature": "stab",
    "stable_built":  "stab",
}

# Per-class thresholds. Built-loss uses the pooled candidate threshold; the
# stable controls use their per-class fitted thresholds.
CLASS_THRESHOLD = {
    "built_loss":    0.4859,
    "stable_nature": 0.4861,
    "stable_built":  0.4948,
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
        "hatch.linewidth": 0.6,
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
    bbox = ax.get_window_extent()
    if bbox.width == 0 or bbox.height == 0 or h == 0:
        return
    aspect = bbox.height / bbox.width  # >1 if axes are taller than wide
    sy = height / h
    sx = sy * aspect
    icon_w_axes = w * sx
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
    arm = size * 0.55
    if tilt == "balanced":
        ang = 0.0
    elif tilt == "left":
        ang = np.deg2rad(13)   # beam tilts down on the left
    else:  # "right"
        ang = np.deg2rad(-13)

    bbox = ax.get_window_extent()
    aspect = bbox.height / bbox.width if bbox.width else 1.0

    beam_cy = y + size * 0.18  # beam sits above the fulcrum tip
    dx = arm * np.cos(ang)
    dy = arm * np.sin(ang) * aspect

    lx, ly = x - dx, beam_cy - dy   # left pan attaches here
    rx, ry = x + dx, beam_cy + dy   # right pan attaches here

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

    ax.plot([lx, rx], [ly, ry], color=beam_color, linewidth=1.2,
            solid_capstyle="round",
            transform=ax.transAxes, clip_on=False, zorder=10)

    pan_drop = size * 0.18
    ax.plot([lx, lx], [ly, ly - pan_drop], color=beam_color, linewidth=0.7,
            transform=ax.transAxes, clip_on=False, zorder=10)
    ax.plot([rx, rx], [ry, ry - pan_drop], color=beam_color, linewidth=0.7,
            transform=ax.transAxes, clip_on=False, zorder=10)

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

    icon_h = size * 0.62
    _draw_pan_icon(ax, left_kind, lx, left_pan_y + pan_h * 0.3,
                   icon_h, left_color)
    _draw_pan_icon(ax, right_kind, rx, right_pan_y + pan_h * 0.3,
                   icon_h, right_color)


# Scale configuration per class:
# (left_icon, right_icon, tilt, left_color, right_color)
# Convention: non-natural reference on the left, near-natural (tree) on the
# right. Tilt indicates which reference pool the class sits closer to.
# Icons are drawn in the class shade so the scale matches its violin/bar.
CLASS_SCALE = {
    "built_loss":    ("building", "tree", "left"),
    "stable_nature": ("tree", "tree", "balanced"),
    "stable_built":  ("building", "building", "balanced"),
}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def violin_panel(ax: plt.Axes, data_by_class: dict[str, np.ndarray],
                 n_by_class: dict[str, int], title: str) -> None:
    groups = CLASS_ORDER
    labels = [CLASS_LABELS[g] for g in groups]
    rng = np.random.default_rng(42)

    box_data = [data_by_class[g] for g in groups]

    parts = ax.violinplot(box_data, positions=range(len(groups)),
                          widths=0.78, showmeans=False, showmedians=False,
                          showextrema=False)
    for body, grp in zip(parts["bodies"], groups):
        body.set_facecolor(CLASS_SHADE[grp])
        body.set_alpha(0.30)
        body.set_edgecolor(CLASS_SHADE[grp])
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
        patch.set_facecolor(CLASS_SHADE[grp])
        patch.set_alpha(0.95)
        patch.set_edgecolor(CLASS_SHADE[grp])

    # Jittered scatter
    for i, (grp, vals) in enumerate(zip(groups, box_data)):
        xj = rng.uniform(-0.30, 0.30, len(vals)) + i
        ax.scatter(xj, vals, s=4.5, color=CLASS_SHADE[grp], alpha=0.40,
                   linewidths=0, zorder=2)

    # Per-class thresholds
    for i, grp in enumerate(groups):
        t = CLASS_THRESHOLD[grp]
        ax.hlines(t, i - 0.42, i + 0.42, color=DARK,
                  linewidth=0.9, linestyle="--", zorder=4)
    ax.text(0.02, 0.97, "per-class threshold",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6, color=DARK,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85))

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, ha="center")
    for tick, grp in zip(ax.get_xticklabels(), groups):
        tick.set_color(CLASS_SHADE[grp])
        tick.set_fontweight("bold")

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("DoR score")
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.set_title(title, pad=56)

    clean_axis(ax, grid_axis="y")

    # n= labels below x-tick labels
    for i, grp in enumerate(groups):
        ax.text(i, -0.20, f"n = {n_by_class[grp]}", ha="center", va="top",
                fontsize=6, color=GREY,
                transform=ax.get_xaxis_transform())

    # Scale icons above the plot area, between title and top spine
    n_groups = len(groups)
    scale_size = 0.22
    for i, grp in enumerate(groups):
        frac = (i + 0.5) / n_groups
        left_kind, right_kind, tilt = CLASS_SCALE[grp]
        shade = CLASS_SHADE[grp]
        _draw_scale(ax, frac, 1.08, scale_size,
                    left_kind, right_kind, tilt, shade, shade)


def stacked_bar_panel(ax: plt.Axes, props: pd.DataFrame,
                      title: str, show_legend: bool = True) -> None:
    """Horizontal stacked bars with inline % labels and right-side n totals."""
    groups = CLASS_ORDER
    labels = [CLASS_LABELS[g].replace("\n", " ") for g in groups]
    y_pos = np.arange(len(groups))[::-1]

    left = np.zeros(len(groups))
    for cat in CAT_ORDER:
        sub = props[props["category"] == cat].set_index("group_key")
        vals = np.array([sub.loc[g, "prop"] if g in sub.index else 0
                         for g in groups])
        style = CAT_STYLE[cat]
        bars = ax.barh(y_pos, vals, left=left,
                       facecolor=style["facecolor"], hatch=style["hatch"],
                       height=0.55, edgecolor=DARK, linewidth=0.6,
                       label=style["label"])
        for rect, v, l in zip(bars, vals, left):
            if v >= 0.08:
                ax.text(l + v / 2, rect.get_y() + rect.get_height() / 2,
                        f"{v*100:.0f}%",
                        ha="center", va="center",
                        fontsize=6.5, color=CAT_TEXT[cat], fontweight="bold")
        left += vals

    # n totals on the far right
    for i, grp in enumerate(groups):
        n = int(props[props["group_key"] == grp]["n"].iloc[0])
        ax.text(1.02, y_pos[i], f"n = {n}", va="center", ha="left",
                fontsize=6, color=GREY,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    for tick, grp in zip(ax.get_yticklabels(), groups):
        tick.set_color(CLASS_SHADE[grp])
        tick.set_fontweight("bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Proportion of sites")
    ax.set_title(title, pad=8)
    clean_axis(ax, grid_axis="x")

    if show_legend:
        handles = [mpatches.Patch(facecolor=CAT_STYLE[c]["facecolor"],
                                  hatch=CAT_STYLE[c]["hatch"],
                                  edgecolor=DARK, linewidth=0.6,
                                  label=CAT_STYLE[c]["label"])
                   for c in CAT_ORDER]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.28), ncol=4,
                  frameon=False, handlelength=1.4, handleheight=1.2,
                  columnspacing=1.2, borderpad=0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def render(data_by_class: dict[str, np.ndarray], n_by_class: dict[str, int],
           props: pd.DataFrame, out_png: Path) -> None:
    """Build the two-panel figure with the currently active palette."""
    # 180 mm × ~85 mm (two-column, single row)
    fig = plt.figure(figsize=(7.09, 3.5))
    gs = fig.add_gridspec(
        1, 2,
        left=0.10, right=0.95,
        top=0.78, bottom=0.22,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    violin_panel(ax_a, data_by_class, n_by_class,
                 title="Degree of Recovery score")
    panel_label(ax_a, "a")

    stacked_bar_panel(ax_b, props, title="Outcome proportions",
                      show_legend=True)
    panel_label(ax_b, "b", x=-0.24)

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}  ({out_png.stat().st_size/1024:.0f} KB)")


def main() -> None:
    set_journal_style()

    cand = pd.read_csv(V3_SCORES).rename(columns={"category_knn": "category"})
    stab = pd.read_csv(V4_SCORES).rename(columns={"category_knn": "category"})
    src = {"cand": cand, "stab": stab}

    # DoR scores and outcome counts per class, drawn from the right source.
    data_by_class: dict[str, np.ndarray] = {}
    n_by_class: dict[str, int] = {}
    prop_rows = []
    for grp in CLASS_ORDER:
        df = src[CLASS_SOURCE[grp]]
        sub = df[df["parent_label"] == grp]
        data_by_class[grp] = sub["dor_knn"].dropna().values
        n_by_class[grp] = len(sub)
        n = len(sub)
        for cat in CAT_ORDER:
            count = int((sub["category"] == cat).sum())
            prop_rows.append({"group_key": grp, "category": cat,
                              "prop": count / n if n else 0,
                              "count": count, "n": n})
    props = pd.DataFrame(prop_rows)

    for palette, out_png in (("grey", OUT_GREY), ("colour", OUT_COLOUR)):
        apply_palette(palette)
        render(data_by_class, n_by_class, props, out_png)


if __name__ == "__main__":
    main()
