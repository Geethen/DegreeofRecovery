"""Per-site agreement scatter: v5 local-buffer DoR vs ecoregion-percentile DoR.

Companion to build_v5_ecoregion_comparison.py (the box-and-strip figure). Where
the box plots compare class-level distributions, this figure shows the *per-site*
relationship: each point is one site placed by its v5
buffer DoR (x) and its ecoregion DoR percentile (y), with the 1:1 line and
marginal distributions. It reveals that the two views are only weakly correlated
per site (they capture complementary information).

Output (greyscale + colour):
  v5/plots/dor_v5_vs_ecoregion_scatter_grey.{png,pdf}
  v5/plots/dor_v5_vs_ecoregion_scatter_colour.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

import figstyle as fs

BASE = Path("/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery")
V5_DATA = BASE / "v5" / "data"
PLOTS = BASE / "v5" / "plots"
SCORES = V5_DATA / "test_site_dor_v5.csv"
ECO = V5_DATA / "test_site_ecoregion_percentile.csv"

ORDER = ["stable_nature", "built_loss", "stable_built"]
PRETTY = {
    "stable_nature": "Stable natural land",
    "built_loss":    "Artificial land reversion",
    "stable_built":  "Stable artificial land",
}
GREY = {"stable_nature": "#9a9a9a", "built_loss": "#444444", "stable_built": "#c0c0c0"}
MARKER = {"stable_nature": "o", "built_loss": "^", "stable_built": "s"}


def load() -> pd.DataFrame:
    s = pd.read_csv(SCORES); s["parent_id"] = s["parent_id"].astype(str)
    e = pd.read_csv(ECO); e["parent_id"] = e["parent_id"].astype(str)
    m = s.merge(e[["parent_id", "pct_dor"]], on="parent_id", how="inner")
    m = m[m["parent_label"].isin(ORDER)].copy()
    m["eco"] = m["pct_dor"] / 100.0
    return m.dropna(subset=["dor_knn", "eco"])


def build(colour: bool, stem: str) -> None:
    fs.apply_style()
    m = load()
    cmap = fs.CLASS_COLORS if colour else GREY

    fig = plt.figure(figsize=(6.6, 6.2))
    gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                  hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax)
    axright = fig.add_subplot(gs[1, 1], sharey=ax)

    # 1:1 and midpoint guides
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0,
            zorder=1, label="1:1")
    ax.axvline(0.5, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axhline(0.5, color="#dddddd", linewidth=0.8, zorder=0)

    bins = np.linspace(0, 1, 26)
    for cls in ORDER:
        sub = m[m["parent_label"] == cls]
        c = cmap[cls]
        ax.scatter(sub["dor_knn"], sub["eco"], s=22, marker=MARKER[cls],
                   facecolor=c, edgecolor="white", linewidth=0.4, alpha=0.8,
                   zorder=3, label=f"{PRETTY[cls]} (n={len(sub)})")
        axtop.hist(sub["dor_knn"], bins=bins, color=c, alpha=0.55,
                   edgecolor="none")
        axright.hist(sub["eco"], bins=bins, color=c, alpha=0.55,
                     edgecolor="none", orientation="horizontal")

    rho_all = spearmanr(m["dor_knn"], m["eco"]).correlation
    ax.text(0.03, 0.97, f"Spearman ρ = {rho_all:+.2f} (all)\n"
                        f"per-class ρ: nat {spearmanr(*_xy(m,'stable_nature')).correlation:+.2f}, "
                        f"rev {spearmanr(*_xy(m,'built_loss')).correlation:+.2f}, "
                        f"art {spearmanr(*_xy(m,'stable_built')).correlation:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Regeneration score — local buffer DoR (v5)")
    ax.set_ylabel("Regeneration score — ecoregion percentile")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7)

    for a in (axtop, axright):
        a.tick_params(labelbottom=False, labelleft=False, length=0)
        for s in a.spines.values():
            s.set_visible(False)
    axtop.set_title("Per-site agreement: local-buffer vs ecoregion DoR", pad=8)

    PLOTS.mkdir(parents=True, exist_ok=True)
    fs.savefig_dual(fig, PLOTS / stem)
    print(f"wrote {PLOTS / stem}.png / .pdf")


def _xy(m, cls):
    sub = m[m["parent_label"] == cls]
    return sub["dor_knn"], sub["eco"]


def main() -> None:
    build(colour=False, stem="dor_v5_vs_ecoregion_scatter_grey")
    build(colour=True, stem="dor_v5_vs_ecoregion_scatter_colour")


if __name__ == "__main__":
    main()
