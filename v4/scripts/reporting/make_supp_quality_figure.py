"""Make a multi-panel supplementary figure for DoR design diagnostics.

The figure combines existing v2/v3/v4 diagnostics that justify the main
design choices described in paper_methods.md:

  A. reference sample-size calibration (CI width vs reference count)
  B. adaptive buffer radius used by v4 stable references
  C. k-nearest-neighbour choice for the cosine scorer
  D. v4 internal reference-label validation at the final operating point
  E. v4 per-class thresholds vs v3 pooled threshold transfer
  F. v4 stable-site categorical outcomes
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import duckdb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from svgpath2mpl import parse_path


ROOT = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT / "v4" / "plots"
ICON_DIR = PLOTS_DIR / "icons"
# Two versions are written: greyscale and colour. Pick between them later.
OUT_GREY = PLOTS_DIR / "supp_dor_quality_metrics_grey.png"
OUT_COLOUR = PLOTS_DIR / "supp_dor_quality_metrics_colour.png"

V2_NEFF = ROOT / "v2" / "data" / "neff_calibration_mask_on" / "neff_calibration_summary.csv"
V3_K_SWEEP = ROOT / "v3" / "data" / "k_sweep_summary.csv"
V4_REFS = ROOT / "v4" / "data" / "v4_stable_refs_alphaearth.parquet"
V4_SUMMARY = ROOT / "v4" / "data" / "within_parent_summary_v4.csv"
V4_SCORES = ROOT / "v4" / "data" / "test_site_dor_v4.csv"

GREY = "#999999"
BLACK = "#222222"
DARK = "#222222"

CLASS_ORDER = ["stable_nature", "stable_built"]
CLASS_LABELS = {
    "stable_nature": "Stable\nnatural",
    "stable_built": "Stable\nbuilt-up",
}
CATEGORY_ORDER = ["recovering", "indistinguishable", "degraded", "no_data"]

# ---------------------------------------------------------------------------
# Palettes — greyscale and colour, matching the main-text figure conventions.
# Two series share an axis in some panels; in greyscale they separate by
# marker/linestyle (lines) or hatch (bars), in colour by hue.
# ---------------------------------------------------------------------------
GREY_PALETTE = {
    "category_style": {
        "recovering":        dict(facecolor="#E2E2E2", hatch="///"),
        "degraded":          dict(facecolor="#4D4D4D", hatch=""),
        "indistinguishable": dict(facecolor="#B0B0B0", hatch="..."),
        "no_data":           dict(facecolor="#D2D2D2", hatch="xxx"),
    },
    "class_colors": {
        "stable_nature": "#9A9A9A",
        "stable_built":  "#6E6E6E",
    },
    "series_dark": "#333333",
    "series_mid":  "#777777",
    "abstain_fill": "#FFFFFF",
    "abstain_hatch": "...",
}
COLOUR_PALETTE = {
    "category_style": {
        "recovering":        dict(facecolor="#009E73", hatch=""),
        "degraded":          dict(facecolor="#D55E00", hatch=""),
        "indistinguishable": dict(facecolor="#999999", hatch=""),
        "no_data":           dict(facecolor="#E5E5E5", hatch=""),
    },
    "class_colors": {
        "stable_nature": "#009E73",   # green
        "stable_built":  "#E69F00",   # orange
    },
    "series_dark": "#0072B2",   # blue
    "series_mid":  "#E69F00",   # orange
    "abstain_fill": "#999999",
    "abstain_hatch": "",
}

# Active palette globals — rebound by apply_palette() before each render.
CATEGORY_STYLE: dict = GREY_PALETTE["category_style"]
CLASS_COLORS: dict = GREY_PALETTE["class_colors"]
SERIES_DARK: str = GREY_PALETTE["series_dark"]
SERIES_MID: str = GREY_PALETTE["series_mid"]
ABSTAIN_FILL: str = GREY_PALETTE["abstain_fill"]
ABSTAIN_HATCH: str = GREY_PALETTE["abstain_hatch"]


def apply_palette(name: str) -> None:
    """Rebind the active palette globals to 'grey' or 'colour'."""
    global CATEGORY_STYLE, CLASS_COLORS, SERIES_DARK, SERIES_MID
    global ABSTAIN_FILL, ABSTAIN_HATCH, STABLE_SCALE
    pal = {"grey": GREY_PALETTE, "colour": COLOUR_PALETTE}[name]
    CATEGORY_STYLE = pal["category_style"]
    CLASS_COLORS = pal["class_colors"]
    SERIES_DARK = pal["series_dark"]
    SERIES_MID = pal["series_mid"]
    ABSTAIN_FILL = pal["abstain_fill"]
    ABSTAIN_HATCH = pal["abstain_hatch"]
    # Balance-scale icons follow the class shades.
    STABLE_SCALE = {
        "stable_nature": ("tree", "tree", "balanced",
                          CLASS_COLORS["stable_nature"], CLASS_COLORS["stable_nature"]),
        "stable_built": ("building", "building", "balanced",
                         CLASS_COLORS["stable_built"], CLASS_COLORS["stable_built"]),
    }


# ---------------------------------------------------------------------------
# Scale/balance icons (shared visual identity with the main-text figure)
# ---------------------------------------------------------------------------
_VIEWBOX_RE = re.compile(r'viewBox="([^"]+)"')
_PATH_D_RE = re.compile(r'<path[^>]*\bd="([^"]+)"')


@lru_cache(maxsize=None)
def _load_fa_icon(name: str):
    svg = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    vb = list(map(float, _VIEWBOX_RE.search(svg).group(1).split()))
    _, _, vb_w, vb_h = vb
    d = _PATH_D_RE.search(svg).group(1)
    path = parse_path(d)
    verts = path.vertices.copy()
    verts[:, 1] = vb_h - verts[:, 1]
    path.vertices[:] = verts
    bb = path.get_extents()
    path.vertices[:, 0] -= bb.x0
    path.vertices[:, 1] -= bb.y0
    return path, (bb.width, bb.height)


def _draw_fa_icon(ax: plt.Axes, name: str, cx: float, base_y: float,
                  height: float, color: str) -> None:
    path, (w, h) = _load_fa_icon(name)
    bbox = ax.get_window_extent()
    if bbox.width == 0 or bbox.height == 0 or h == 0:
        return
    aspect = bbox.height / bbox.width
    sy = height / h
    sx = sy * aspect
    icon_w_axes = w * sx
    tr = (mtransforms.Affine2D()
          .scale(sx, sy)
          .translate(cx - icon_w_axes / 2, base_y)
          + ax.transAxes)
    ax.add_patch(PathPatch(path, transform=tr,
                           facecolor=color, edgecolor="none",
                           clip_on=False, zorder=12))


def _draw_pan_icon(ax, kind, cx, base_y, h, color):
    _draw_fa_icon(ax, kind, cx, base_y, h, color)


def _draw_scale(ax: plt.Axes, x: float, y: float, size: float,
                left_kind: str, right_kind: str, tilt: str,
                left_color: str, right_color: str,
                beam_color: str = DARK) -> None:
    arm = size * 0.55
    if tilt == "balanced":
        ang = 0.0
    elif tilt == "left":
        ang = np.deg2rad(13)
    else:
        ang = np.deg2rad(-13)

    bbox = ax.get_window_extent()
    aspect = bbox.height / bbox.width if bbox.width else 1.0
    beam_cy = y + size * 0.18
    dx = arm * np.cos(ang)
    dy = arm * np.sin(ang) * aspect
    lx, ly = x - dx, beam_cy - dy
    rx, ry = x + dx, beam_cy + dy

    base_w = size * 0.42
    base_h = size * 0.06
    ax.add_patch(Rectangle((x - base_w / 2, y - base_h), base_w, base_h,
                           facecolor=beam_color, edgecolor=beam_color,
                           transform=ax.transAxes, clip_on=False,
                           zorder=8, linewidth=0))
    ax.add_patch(mpatches.Polygon(
        [(x - size * 0.18, y), (x + size * 0.18, y), (x, beam_cy)],
        facecolor=beam_color, edgecolor=beam_color,
        transform=ax.transAxes, clip_on=False, zorder=9, linewidth=0))
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
    _draw_pan_icon(ax, left_kind, lx, left_pan_y + pan_h * 0.3, icon_h,
                   left_color)
    _draw_pan_icon(ax, right_kind, rx, right_pan_y + pan_h * 0.3, icon_h,
                   right_color)


# STABLE_SCALE is (re)built per palette by apply_palette(); initialise it here
# so add_class_scales() can resolve the name before the first apply_palette().
apply_palette("grey")


def add_class_scales(ax: plt.Axes, class_keys: list, y: float = 1.10,
                     size: float = 0.16) -> None:
    """Draw scale icons above each x-tick position for class-grouped panels."""
    n = len(class_keys)
    for i, key in enumerate(class_keys):
        if key not in STABLE_SCALE:
            continue
        frac = (i + 0.5) / n
        left_kind, right_kind, tilt, lc, rc = STABLE_SCALE[key]
        _draw_scale(ax, frac, y, size, left_kind, right_kind, tilt, lc, rc)


def set_journal_style() -> None:
    plt.rcParams.update(
        {
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
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#EAEAEA", linewidth=0.45)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.18,
        1.10,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def colour_class_ticks(ax: plt.Axes, class_keys: list, axis: str = "x") -> None:
    """Recolour and bold tick labels to match the class palette."""
    ticks = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for tick, key in zip(ticks, class_keys):
        if key in CLASS_COLORS:
            tick.set_color(CLASS_COLORS[key])
            tick.set_fontweight("bold")


def weighted_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in df.groupby("parent_label", sort=False):
        n = sub["n"].sum()
        rows.append(
            {
                "parent_label": label,
                "error_rate": float((sub["error_rate"] * sub["n"]).sum() / n),
                "abstain_rate": float((sub["abstain_rate"] * sub["n"]).sum() / n),
            }
        )
    return pd.DataFrame(rows)


def render(neff, k_sweep, v4_summary, scores, buffers, out_png: Path) -> None:
    """Build the six-panel diagnostic figure with the currently active palette."""
    # 180 mm × 130 mm — matches main-text Figure 1 proportions
    fig, axes = plt.subplots(2, 3, figsize=(7.09, 5.1), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # A. Sample-size diagnostics.
    ax_a.plot(neff["size"], neff["median_ci_width"], marker="o", linestyle="-",
              color=SERIES_DARK, label="Median")
    ax_a.plot(neff["size"], neff["p90_ci_width"], marker="s", linestyle="--",
              color=SERIES_MID, label="90th percentile")
    ax_a.axvline(30, color=GREY, linestyle=":", linewidth=0.9)
    ax_a.axvline(100, color=BLACK, linestyle="--", linewidth=0.9)
    ax_a.set_title("Reference sample-size calibration")
    ax_a.set_xlabel("Reference pixels per pool")
    ax_a.set_ylabel("Bootstrap CI width")
    ax_a.legend(frameon=False, loc="center right", handlelength=1.6)
    clean_axis(ax_a)
    panel_label(ax_a, "a")

    # B. Buffer radius use.
    buffer_order = [1, 1.5, 2, 3, 5, 8]
    counts = buffers["buffer_km"].round(1).value_counts().reindex(buffer_order, fill_value=0)
    ax_b.bar([str(x) for x in buffer_order], counts.values, color=SERIES_DARK, alpha=0.85)
    ax_b.set_title("Adaptive buffer radius used")
    ax_b.set_xlabel("Radius covering selected refs (km)")
    ax_b.set_ylabel("Parents")
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    # C. k choice.
    mean_cos = k_sweep[k_sweep["variant"] == "mean_cos"].sort_values("k")
    ax_c.plot(mean_cos["k"], mean_cos["bal_err"] * 100, marker="o", linestyle="-",
              color=SERIES_DARK, label="Balanced error (%)")
    ax_c.plot(mean_cos["k"], mean_cos["brier"] * 100, marker="s", linestyle="--",
              color=SERIES_MID, label="Brier × 100")
    ax_c.axvline(5, color=BLACK, linestyle="--", linewidth=0.9)
    ax_c.text(5, 0.95, "k = 5", transform=ax_c.get_xaxis_transform(), ha="center", va="top", fontsize=6)
    ax_c.set_title("kNN scorer selection")
    ax_c.set_xlabel("k nearest references")
    ax_c.set_ylabel("Validation metric")
    ax_c.legend(frameon=False, loc="lower right", handlelength=1.6)
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    # D. Internal validation.
    val = v4_summary[
        (v4_summary["step"] == "step4_knn") & (v4_summary["parent_label"].isin(CLASS_ORDER))
    ]
    val_rates = weighted_rates(val).set_index("parent_label").reindex(CLASS_ORDER)
    x = np.arange(len(CLASS_ORDER))
    width = 0.38
    ax_d.bar(x - width / 2, val_rates["error_rate"] * 100, width, label="Error",
             facecolor=SERIES_DARK, edgecolor=BLACK, linewidth=0.5)
    ax_d.bar(
        x + width / 2,
        val_rates["abstain_rate"] * 100,
        width,
        label="Indistinguishable",
        facecolor=ABSTAIN_FILL, edgecolor=BLACK, linewidth=0.5, hatch=ABSTAIN_HATCH,
    )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
    ax_d.set_title("Internal validation", pad=26)
    ax_d.set_ylabel("Held-out references (%)")
    ax_d.legend(frameon=False, loc="upper left", handlelength=1.2)
    clean_axis(ax_d)
    colour_class_ticks(ax_d, CLASS_ORDER, axis="x")
    add_class_scales(ax_d, CLASS_ORDER)
    panel_label(ax_d, "d")

    # E. Threshold transfer from v3 pooled threshold to v4 per-class thresholds.
    transfer = (
        scores.assign(changed=scores["category_knn"] != scores["category_knn_v3t"])
        .groupby("parent_label")["changed"]
        .agg(["sum", "count"])
        .reindex(CLASS_ORDER)
    )
    pct_changed = transfer["sum"] / transfer["count"] * 100
    ax_e.bar(x, pct_changed, color=SERIES_MID, edgecolor=BLACK, linewidth=0.5)
    for i, cls in enumerate(CLASS_ORDER):
        ax_e.text(i, pct_changed.loc[cls] + 0.12, f"{int(transfer.loc[cls, 'sum'])}", ha="center", fontsize=6)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
    ax_e.set_title("Threshold transfer check", pad=26)
    ax_e.set_ylabel("Sites changing class (%)")
    ax_e.set_ylim(0, max(4.0, float(pct_changed.max()) + 1.0))
    clean_axis(ax_e)
    colour_class_ticks(ax_e, CLASS_ORDER, axis="x")
    add_class_scales(ax_e, CLASS_ORDER)
    panel_label(ax_e, "e")

    # F. Final stable-site categories.
    cat = (
        scores.groupby(["parent_label", "category_knn"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=CLASS_ORDER, columns=CATEGORY_ORDER, fill_value=0)
    )
    cat_pct = cat.div(cat.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(CLASS_ORDER))
    for category in CATEGORY_ORDER:
        vals = cat_pct[category].to_numpy()
        style = CATEGORY_STYLE[category]
        ax_f.bar(
            x,
            vals,
            bottom=bottom,
            facecolor=style["facecolor"],
            hatch=style["hatch"],
            label=category.replace("_", " "),
            edgecolor=BLACK,
            linewidth=0.5,
        )
        bottom += vals
    ax_f.set_xticks(x)
    ax_f.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
    ax_f.set_title("Stable-site DoR outcomes", pad=26)
    ax_f.set_ylabel("Sites (%)")
    ax_f.set_ylim(0, 100)
    ax_f.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.2,
    )
    clean_axis(ax_f)
    colour_class_ticks(ax_f, CLASS_ORDER, axis="x")
    add_class_scales(ax_f, CLASS_ORDER)
    panel_label(ax_f, "f")

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    set_journal_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    neff = pd.read_csv(V2_NEFF)
    k_sweep = pd.read_csv(V3_K_SWEEP)
    v4_summary = pd.read_csv(V4_SUMMARY)
    scores = pd.read_csv(V4_SCORES)

    con = duckdb.connect()
    buffers = con.execute(
        """
        SELECT parent_id, max(buffer_m_used) AS buffer_m_used
        FROM read_parquet(?)
        WHERE strategy = 'random_100'
        GROUP BY parent_id
        """,
        [str(V4_REFS)],
    ).df()
    con.close()
    buffers["buffer_km"] = buffers["buffer_m_used"] / 1000.0

    for palette, out_png in (("grey", OUT_GREY), ("colour", OUT_COLOUR)):
        apply_palette(palette)
        render(neff, k_sweep, v4_summary, scores, buffers, out_png)


if __name__ == "__main__":
    main()
