"""Plot reliability diagrams for dor_median vs dor_knn.

Reads v3/data/reliability_dor.csv and saves v3/plots/reliability_dor.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="v3/data/reliability_dor.csv")
    p.add_argument("--summary", default="v3/data/reliability_summary.csv")
    p.add_argument("--out", default="v3/plots/reliability_dor.png")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    sm = pd.read_csv(args.summary)

    scorers = [("dor_median", "#4C72B0"), ("dor_knn", "#DD8452")]
    labels  = {"dor_median": "dor_median  (all refs, cosine median)",
               "dor_knn":   "dor_knn  (k=5 nearest, cosine mean)"}

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1.6], wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── left & centre: reliability diagrams ──────────────────────────────
    for ax, (scorer, colour) in zip(axes[:2], scorers):
        sub = df[df["scorer"] == scorer].copy()
        row = sm[sm["scorer"] == scorer].iloc[0]

        x = sub["mean_score"].to_numpy()
        y = sub["frac_good"].to_numpy()
        n = sub["n"].to_numpy()

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect calibration")

        # Shaded overconfident / underconfident regions
        ax.fill_between([0, 0.5], [0, 0.5], [0, 0],   alpha=0.05, color="red",
                        label="Over-confident (good)")
        ax.fill_between([0.5, 1], [0.5, 1], [1, 1],   alpha=0.05, color="blue",
                        label="Over-confident (bad)")

        # Reliability curve — line + scatter sized by n
        ax.plot(x, y, "-o", color=colour, lw=1.8, ms=0, zorder=3)
        sizes = 40 + 120 * (n / n.max())
        ax.scatter(x, y, s=sizes, color=colour, zorder=4, edgecolors="white", lw=0.8)

        # Gap bars (thin vertical lines from diagonal to observed)
        for xi, yi in zip(x, y):
            ax.plot([xi, xi], [xi, yi], color=colour, lw=1, alpha=0.5, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted score", fontsize=10)
        ax.set_ylabel("Fraction of good probes", fontsize=10)
        ax.set_title(
            f"{labels[scorer]}\n"
            f"Brier={row['brier']:.3f}  ECE={row['ece']:.3f}  MCE={row['mce']:.3f}",
            fontsize=9, pad=6,
        )
        ax.grid(True, alpha=0.25, lw=0.5)

    # ── right: gap bar chart comparing the two scorers ───────────────────
    ax = axes[2]
    bins = df["bin"].unique()
    w = 0.35
    for i, (scorer, colour) in enumerate(scorers):
        sub = df[df["scorer"] == scorer].sort_values("bin")
        gaps = sub["gap"].to_numpy()
        centres = sub["mean_score"].to_numpy()
        x_pos = np.arange(len(bins)) + i * w
        ax.barh(x_pos, gaps, height=w * 0.85, color=colour,
                alpha=0.8, label=labels[scorer].split("  ")[0])
        # Label each bar with the score range midpoint
    ax.axvline(0, color="k", lw=1, alpha=0.5)
    ax.set_yticks(np.arange(len(bins)) + w / 2)
    ax.set_yticklabels(
        [f"bin {b}" for b in bins], fontsize=8
    )
    ax.set_xlabel("Gap  (mean score − frac_good)", fontsize=9)
    ax.set_title("Calibration gap\nper bin", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.5, axis="x")
    ax.set_xlim(-0.38, 0.38)

    fig.suptitle(
        "Reliability diagrams — dor_median vs dor_knn\n"
        "Both scorers overconfident-good below threshold, overconfident-bad above.\n"
        "dor_knn's lower Brier is from sharper scores (refinement), not better calibration (ECE ≈ equal).",
        fontsize=9, y=1.01,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
