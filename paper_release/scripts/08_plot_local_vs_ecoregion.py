"""Step 8 — Local vs Ecoregion DoR comparison figure (horizontal grouped boxes).

Adapts the styling of v5/scripts/analysis/build_v5_ecoregion_comparison.py (box +
jittered strip + white-diamond mean, figstyle palette) into the single-panel
horizontal layout: per transition group, the Local-buffer DoR and the
Ecoregion-percentile DoR are drawn as a paired box+strip, coloured by reference
state type (blue = Local, orange = Ecoregion), with a shared "Degree of
regeneration score" x-axis and a top legend.

Data (latest run): outputs/data/test_site_scores_combined.csv
  Local     = dor_knn                (0-1)
  Ecoregion = pct_dor / 100          (0-1)
Groups shown (matching the reference figure): artificial_reversion, stable_artificial.

Outputs: outputs/figures/dor_local_vs_ecoregion.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUT_DATA, OUT_FIGURES, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import figstyle as fs  # noqa: E402  (reuse the report's style)

COMBINED = OUT_DATA / "test_site_scores_combined.csv"
OUT = OUT_FIGURES / "dor_local_vs_ecoregion"

# Groups shown, top-to-bottom (as in the reference figure).
ORDER = ["artificial_reversion", "stable_artificial"]
PRETTY = {
    "artificial_reversion": "Artificial land reversion",
    "stable_artificial":    "Stable artificial land",
    "stable_natural":       "Stable natural land",
}

# Reference-state-type colours (blue = Local, orange = Ecoregion). Muted,
# colour-blind-safe hues in the Okabe-Ito family, as used across the report.
REF_TYPES = ["Local", "Ecoregion"]
REF_COLOR = {"Local": "#3B7EA1", "Ecoregion": "#D9822B"}
REF_FILL = {"Local": "#3B7EA1", "Ecoregion": "#D9822B"}
OFFSET = {"Local": -0.19, "Ecoregion": +0.19}   # within-group vertical offset
BOX_H = 0.30
INK = "#1a1a1a"


def _lighten(hex_color: str, amount: float) -> tuple:
    """Blend a hex colour toward white by `amount` (0=orig, 1=white)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _hbox(ax, y, vals, colour, rng):
    """Horizontal box, then jittered strip ON TOP, then white-diamond mean.

    Nature-style: hairline box with a pale tinted fill so the overlaid points
    (drawn above the box) carry the distribution; median line and mean diamond
    sit topmost."""
    if len(vals) == 0:
        return
    # 1) box + whiskers underneath (pale tint, hairline edge)
    ax.boxplot(
        vals, positions=[y], vert=False, widths=BOX_H, showfliers=False,
        patch_artist=True, zorder=2,
        medianprops=dict(color=INK, linewidth=1.1),
        whiskerprops=dict(color=colour, linewidth=0.9),
        capprops=dict(color=colour, linewidth=0.9),
        boxprops=dict(facecolor=_lighten(colour, 0.82), edgecolor=colour,
                      linewidth=0.9),
    )
    # 2) jittered points ON TOP of the box
    jit = (rng.random(len(vals)) - 0.5) * (BOX_H * 0.80)
    ax.scatter(vals, np.full(len(vals), y) + jit, s=6.5, color=colour,
               alpha=0.60, edgecolors="white", linewidths=0.15,
               zorder=4, rasterized=True)
    # 3) median redrawn above the points so it stays legible
    med = float(np.median(vals))
    ax.plot([med, med], [y - BOX_H / 2, y + BOX_H / 2], color=INK,
            linewidth=1.2, zorder=5, solid_capstyle="butt")
    # 4) white-diamond mean, topmost
    ax.scatter([float(np.mean(vals))], [y], marker="D", s=42,
               facecolor="white", edgecolor=INK, linewidth=1.0, zorder=6)


def main() -> None:
    fs.apply_style()
    df = pd.read_csv(COMBINED, dtype={"PLOTID": str})
    rng = np.random.default_rng(42)

    # 180 mm two-column width; compact height for a 2-group horizontal panel.
    fig, ax = plt.subplots(figsize=(7.09, 3.2))

    # faint vertical reference gridlines behind everything (Nature-clean)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)

    yticks, yticklabels = [], []
    for gi, grp in enumerate(ORDER):
        base_y = (len(ORDER) - 1 - gi)   # top group at largest y
        sub = df[df["group"] == grp]
        series = {
            "Local": sub["dor_knn"].dropna().to_numpy(),
            "Ecoregion": (sub["pct_dor"].dropna().to_numpy() / 100.0),
        }
        for rt in REF_TYPES:
            _hbox(ax, base_y + OFFSET[rt], series[rt], REF_COLOR[rt], rng)
        # one n per reference type (they differ slightly), shown under the label
        n_loc, n_eco = len(series["Local"]), len(series["Ecoregion"])
        n_txt = (f"n = {n_loc}" if n_loc == n_eco
                 else f"n = {n_loc} / {n_eco}")
        yticks.append(base_y)
        yticklabels.append(f"{PRETTY[grp]}\n{n_txt}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-0.62, len(ORDER) - 0.38)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Degree of regeneration score")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="#333333")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#333333")

    # top legend: reference state type (swatch chips, matching the box hue)
    handles = [mpatches.Patch(facecolor=REF_COLOR[rt], edgecolor="none",
                              label=rt) for rt in REF_TYPES]
    leg = ax.legend(handles=handles, title="Reference state type",
                    loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=2, frameon=False, handlelength=1.1, handleheight=1.1,
                    handletextpad=0.5, columnspacing=1.8)
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fs.savefig_dual(fig, OUT)
    print(f"wrote {OUT}.png / .pdf")
    for grp in ORDER:
        sub = df[df["group"] == grp]
        print(f"  {grp:22} local n={sub.dor_knn.notna().sum():4} "
              f"eco n={sub.pct_dor.notna().sum():4} | "
              f"local med={sub.dor_knn.median():.3f} "
              f"eco med={sub.pct_dor.median()/100:.3f}")


if __name__ == "__main__":
    main()
