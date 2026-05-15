"""Violin plot of dor_knn scores per parent_label for the combined dataset.

Reads combined/data/test_site_dor_combined.shp and produces
combined/plots/dor_violin_by_category.png — one violin per parent_label
(built_loss, crop_loss, stable_nature, stable_crop, stable_built) with
overlaid points, median, and the unified v3 pooled threshold (t = 0.4859).
"""
from __future__ import annotations

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SHP = os.path.join(BASE_DIR, "combined", "data", "test_site_dor_combined.shp")
OUT_DIR = os.path.join(BASE_DIR, "combined", "plots")
OUT_PNG = os.path.join(OUT_DIR, "dor_violin_by_category.png")

LABEL_ORDER = ["built_loss", "crop_loss",
               "stable_nature", "stable_crop", "stable_built"]
LABEL_COLORS = {
    "built_loss":    "#7a4a2c",
    "crop_loss":     "#d2a13a",
    "stable_nature": "#2c7a3d",
    "stable_crop":   "#c9b06a",
    "stable_built":  "#8a6a55",
}
THRESHOLD = 0.4859


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    gdf = gpd.read_file(SHP)

    data = []
    labels = []
    for lbl in LABEL_ORDER:
        vals = gdf.loc[gdf["lbl"] == lbl, "dor_knn"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(lbl)

    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(1, len(data) + 1)

    parts = ax.violinplot(
        data, positions=positions, widths=0.85,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, lbl in zip(parts["bodies"], labels):
        body.set_facecolor(LABEL_COLORS[lbl])
        body.set_edgecolor("black")
        body.set_alpha(0.7)
        body.set_linewidth(0.8)

    # Jittered points
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, data):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   s=8, color="black", alpha=0.25, zorder=2)

    # Median lines + n labels
    for pos, vals, lbl in zip(positions, data, labels):
        med = float(np.median(vals))
        ax.hlines(med, pos - 0.32, pos + 0.32, colors="white", linewidth=2.5, zorder=3)
        ax.hlines(med, pos - 0.32, pos + 0.32, colors="black", linewidth=1.2, zorder=4)
        ax.text(pos, 1.02, f"n = {len(vals)}", ha="center", va="bottom",
                fontsize=9)

    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1,
               label=f"t = {THRESHOLD}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("dor_knn")
    ax.set_title("dor_knn distribution by parent_label  (v3 loss + v4 stable)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")

    print("\nMedian dor_knn per label:")
    for lbl, vals in zip(labels, data):
        print(f"  {lbl:<16} median={np.median(vals):.3f}  "
              f"q25={np.percentile(vals, 25):.3f}  "
              f"q75={np.percentile(vals, 75):.3f}  n={len(vals)}")


if __name__ == "__main__":
    main()
