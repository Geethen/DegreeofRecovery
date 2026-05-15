"""Distribution of dor_knn for the combined v3 loss + v4 stable dataset.

Produces:
  combined/plots/dor_distribution_by_category.png
    Top panel:    dor_knn histogram by parent_label (5 categories).
    Bottom panel: dor_knn histogram split by unified cat_knn outcome
                  (recovering / indistinguishable / degraded).

Uses combined/data/test_site_dor_combined.shp as input.
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
OUT_PNG = os.path.join(OUT_DIR, "dor_distribution_by_category.png")

LABEL_ORDER = ["built_loss", "crop_loss",
               "stable_nature", "stable_crop", "stable_built"]
LABEL_COLORS = {
    "built_loss":    "#7a4a2c",
    "crop_loss":     "#d2a13a",
    "stable_nature": "#2c7a3d",
    "stable_crop":   "#c9b06a",
    "stable_built":  "#8a6a55",
}
CAT_ORDER = ["recovering", "indistinguishable", "degraded"]
CAT_COLORS = {
    "recovering": "#2c7a3d",
    "indistinguishable": "#b8b8b8",
    "degraded": "#a82a2a",
}
THRESHOLD = 0.4859


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    gdf = gpd.read_file(SHP)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # --- Top: histogram by parent_label ------------------------------------
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    for lbl in LABEL_ORDER:
        sub = gdf[gdf["lbl"] == lbl]["dor_knn"].dropna()
        if len(sub) == 0:
            continue
        ax.hist(
            sub,
            bins=bins,
            histtype="stepfilled",
            alpha=0.55,
            color=LABEL_COLORS[lbl],
            edgecolor=LABEL_COLORS[lbl],
            linewidth=1.2,
            label=f"{lbl}  (n={len(sub)})",
        )
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1,
               label=f"t = {THRESHOLD}")
    ax.set_xlabel("dor_knn")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.set_title("dor_knn distribution by parent_label  (v3 loss + v4 stable)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # --- Bottom: histogram by unified category -----------------------------
    ax = axes[1]
    for cat in CAT_ORDER:
        sub = gdf[gdf["cat_knn"] == cat]["dor_knn"].dropna()
        if len(sub) == 0:
            continue
        ax.hist(
            sub,
            bins=bins,
            histtype="stepfilled",
            alpha=0.7,
            color=CAT_COLORS[cat],
            edgecolor=CAT_COLORS[cat],
            linewidth=1.2,
            label=f"{cat}  (n={len(sub)})",
        )
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1,
               label=f"t = {THRESHOLD}")
    ax.set_xlabel("dor_knn")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.set_title(
        "dor_knn distribution by unified cat_knn  "
        f"(threshold {THRESHOLD}, CI-based decision)"
    )
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")

    # Print summary counts for the README.
    print("\nCounts by label x cat_knn:")
    print(
        gdf.groupby("lbl")["cat_knn"].value_counts().unstack(fill_value=0)
        .reindex(LABEL_ORDER).to_string()
    )


if __name__ == "__main__":
    main()
