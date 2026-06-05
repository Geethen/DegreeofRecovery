"""Journal-quality per-class heatmaps for separability (MCC) and within-pool
spatial autocorrelation, INCLUDING the loss-site classes (built_loss, crop_loss).

Reads the 5-class summary CSVs produced by separability_sweep.py --candidate and
spatial_autocorr_sweep.py --candidate, and renders one figure per metric with a
panel per class, ordered stable -> loss. Uses the shared figstyle.

Outputs (PNG 600 dpi + vector PDF):
  v5/plots/separability_mcc_byclass
  v5/plots/spatial_autocorr_byclass
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fst

V5_DATA  = Path(__file__).resolve().parents[3] / "v5" / "data"
V5_PLOTS = Path(__file__).resolve().parents[3] / "v5" / "plots"

CLASS_ORDER = ["stable_nature", "stable_crop", "stable_built",
               "built_loss", "crop_loss"]


def _km(vals):
    return [f"{int(v)/1000:g}" for v in vals]


def panel_grid(summary: pd.DataFrame, value: str, title: str, subtitle: str,
               out_stem: Path, cmap: str, fmt: str = "{:.2f}",
               reverse_good: bool = False, best=(3000, 8000)) -> None:
    """One heatmap panel per class. reverse_good=True means LOWER is better
    (autocorrelation), so we use a reversed map and invert the text-contrast test."""
    fst.apply_style()
    labels = [l for l in CLASS_ORDER if l in set(summary["parent_label"])]
    n = len(labels)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 3.35 * nrow),
                             squeeze=False)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.05, right=0.93,
                        wspace=0.18)
    axes = axes.ravel()

    vmin = float(np.nanmin(summary[value]))
    vmax = float(np.nanmax(summary[value]))
    cm = fst.cmap_with_bad(cmap)

    for ax, lbl in zip(axes, labels):
        sub = summary[summary["parent_label"] == lbl]
        piv = sub.pivot(index="inner_m", columns="outer_m", values=value)
        im = ax.imshow(np.ma.masked_invalid(piv.values), origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax, cmap=cm)
        fst.style_heatmap_axis(ax, _km(piv.columns), _km(piv.index),
                               "Outer ceiling (km)",
                               "Inner exclusion (km)" if ax is axes[0] else "",
                               fst.CLASS_LABELS.get(lbl, lbl))
        ax.title.set_color(fst.CLASS_COLORS.get(lbl, "#222"))
        span = (vmax - vmin) or 1.0
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isfinite(v):
                    continue
                frac = (v - vmin) / span
                dark = frac < 0.55 if not reverse_good else frac > 0.45
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=6.0, color="white" if dark else "#0a0a0a")
        try:
            yi = list(piv.index).index(best[0]); xi = list(piv.columns).index(best[1])
            ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False,
                                       edgecolor=fst.HILITE, linewidth=2.2, zorder=5))
        except ValueError:
            pass

    for ax in axes[len(labels):]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes[:len(labels)], fraction=0.014, pad=0.012)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=7)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.975)
    fig.text(0.49, 0.885, subtitle, ha="center", va="center", fontsize=8.5,
             color="#555")
    fst.savefig_dual(fig, out_stem)


def main() -> None:
    sep = pd.read_csv(V5_DATA / "separability_summary.csv")
    ac = pd.read_csv(V5_DATA / "spatial_autocorr_summary.csv")

    panel_grid(
        sep, "mcc",
        "Good–bad reference separability by buffer, per class",
        "leave-one-out MCC (higher = pools more separable)  ·  red ring = chosen buffer 3–8 km",
        V5_PLOTS / "separability_mcc_byclass", cmap=fst.SEQ_CMAP, fmt="{:.2f}")
    print(f"  Wrote {V5_PLOTS / 'separability_mcc_byclass'}.png/.pdf")

    panel_grid(
        ac, "mean_sim_all",
        "Within-pool reference spatial autocorrelation by buffer, per class",
        "mean pairwise embedding similarity (lower = more independent)  ·  red ring = chosen buffer 3–8 km",
        V5_PLOTS / "spatial_autocorr_byclass", cmap="magma_r", fmt="{:.3f}",
        reverse_good=True)
    print(f"  Wrote {V5_PLOTS / 'spatial_autocorr_byclass'}.png/.pdf")


if __name__ == "__main__":
    main()
