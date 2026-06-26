"""Two-panel comparison: v5 buffer DoR vs ecoregion-percentile DoR.

Vertical grey box+strip plots (classes on x, regeneration score on y; white-
diamond mean, per-class threshold tick), non-crop classes only, with the
local-buffer DoR and the ecoregion-percentile DoR side by side on a shared
y-axis so the two regeneration views can be compared directly.

Left panel  : v5 buffer DoR (dor_knn, 0-1)  from v5/data/test_site_dor_v5.csv
Right panel : ecoregion DoR percentile (pct_dor/100, 0-1) from
              v5/data/test_site_ecoregion_percentile.csv

Outputs (greyscale + colour):
  v5/plots/dor_v5_vs_ecoregion_grey.{png,pdf}
  v5/plots/dor_v5_vs_ecoregion_colour.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

BASE = Path("/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery")
V5_DATA = BASE / "v5" / "data"
PLOTS = BASE / "v5" / "plots"
SCORES = V5_DATA / "test_site_dor_v5.csv"
ECO = V5_DATA / "test_site_ecoregion_percentile.csv"
THRESH = BASE / "v4" / "data" / "calibrated_thresholds_v4.json"

# Non-crop classes, plotted left-to-right.
ORDER = ["built_loss", "stable_built", "stable_nature"]
PRETTY = {
    "built_loss":    "Artificial land\nreversion",
    "stable_built":  "Stable\nartificial land",
    "stable_nature": "Stable\nnatural land",
}


def _strip(ax, x, vals, colour, mean_y, thr=None, rng=None):
    """Vertical box, jittered strip on top, then white-diamond mean."""
    rng = rng or np.random.default_rng(0)
    # 1) box underneath (semi-transparent so the overlaid points stay visible)
    bp = ax.boxplot(vals, positions=[x], vert=True, widths=0.5,
                    showfliers=False, patch_artist=True, zorder=2,
                    medianprops=dict(color="#111111", linewidth=1.3),
                    whiskerprops=dict(color="#444444", linewidth=1.0),
                    capprops=dict(color="#444444", linewidth=1.0),
                    boxprops=dict(facecolor="#d9d9d9", edgecolor="#444444",
                                  linewidth=1.0, alpha=0.6))
    # 2) jittered points ON TOP of the box
    jit = (rng.random(len(vals)) - 0.5) * 0.34
    ax.scatter(np.full(len(vals), x) + jit, vals, s=7, color=colour,
               alpha=0.55, edgecolors="none", zorder=4, rasterized=True)
    # 3) threshold tick (horizontal), then mean diamond, topmost
    if thr is not None:
        ax.plot([x - 0.27, x + 0.27], [thr, thr], color="#111111",
                linestyle="--", linewidth=1.0, zorder=5)
    ax.scatter([x], [mean_y], marker="D", s=46, facecolor="white",
               edgecolor="#111111", linewidth=1.1, zorder=6)


def _panel(ax, data, ylabel, title, thresholds, colour_by_class):
    rng = np.random.default_rng(42)
    for i, cls in enumerate(ORDER):
        vals = data[cls]["vals"]
        if len(vals) == 0:
            continue
        colour = fs.CLASS_COLORS[cls] if colour_by_class else "#7a7a7a"
        thr = thresholds.get(cls) if thresholds else None
        _strip(ax, i, vals, colour, float(np.mean(vals)), thr=thr, rng=rng)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{PRETTY[c]}\nn = {data[c]['n']}" for c in ORDER])
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(0, 1)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="x", length=0)


def build(colour_by_class: bool, stem: str) -> None:
    fs.apply_style()
    scores = pd.read_csv(SCORES)
    scores["parent_id"] = scores["parent_id"].astype(str)
    eco = pd.read_csv(ECO)
    eco["parent_id"] = eco["parent_id"].astype(str)
    thresholds = {k: float(v) for k, v in json.load(open(THRESH)).items()}

    # left: v5 buffer dor_knn
    left = {c: {"vals": scores.loc[scores["parent_label"] == c, "dor_knn"].dropna().to_numpy(),
                "n": int((scores["parent_label"] == c).sum())}
            for c in ORDER}
    # right: ecoregion pct_dor (0-100 -> 0-1); n = scored sites
    right = {}
    for c in ORDER:
        v = eco.loc[(eco["parent_label"] == c) & eco["pct_dor"].notna(), "pct_dor"].to_numpy() / 100.0
        right[c] = {"vals": v, "n": int(len(v))}

    # Shared y-axis (both are 0-1 regeneration scores) for direct comparison;
    # each panel still shows its own per-class n in the x-tick labels.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharey=True)
    _panel(axes[0], left, "Regeneration score",
           "a  Local-buffer DoR (v5)", thresholds, colour_by_class)
    # ecoregion panel: 0.5 reference line (percentile midpoint), no calibrated thr
    _panel(axes[1], right, None,
           "b  Ecoregion-percentile DoR", None, colour_by_class)
    axes[1].axhline(0.5, color="#999999", linestyle=":", linewidth=0.9, zorder=1)

    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fs.savefig_dual(fig, PLOTS / stem)
    print(f"wrote {PLOTS / stem}.png / .pdf")


def main() -> None:
    build(colour_by_class=False, stem="dor_v5_vs_ecoregion_grey")
    build(colour_by_class=True, stem="dor_v5_vs_ecoregion_colour")


if __name__ == "__main__":
    main()
