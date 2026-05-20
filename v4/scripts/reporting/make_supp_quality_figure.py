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

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_PNG = ROOT / "v4" / "plots" / "supp_dor_quality_metrics.png"
OUT_PDF = ROOT / "v4" / "plots" / "supp_dor_quality_metrics.pdf"

V2_NEFF = ROOT / "v2" / "data" / "neff_calibration_mask_on" / "neff_calibration_summary.csv"
V3_K_SWEEP = ROOT / "v3" / "data" / "k_sweep_summary.csv"
V4_REFS = ROOT / "v4" / "data" / "v4_stable_refs_alphaearth.parquet"
V4_SUMMARY = ROOT / "v4" / "data" / "within_parent_summary_v4.csv"
V4_SCORES = ROOT / "v4" / "data" / "test_site_dor_v4.csv"

CLASS_ORDER = ["stable_nature", "stable_crop", "stable_built"]
CLASS_LABELS = {
    "stable_nature": "Stable\nnear-natural",
    "stable_crop": "Stable\ncropland",
    "stable_built": "Stable\nbuilt-up",
}
# Shared palette with the main-text figure
CATEGORY_ORDER = ["recovering", "indistinguishable", "degraded", "no_data"]
CATEGORY_COLORS = {
    "recovering": "#009E73",
    "degraded": "#D55E00",
    "indistinguishable": "#999999",
    "no_data": "#E5E5E5",
}
CLASS_COLORS = {
    "stable_nature": "#009E73",   # green
    "stable_crop":   "#0072B2",   # blue
    "stable_built":  "#E69F00",   # orange
}
BLUE = "#0072B2"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GREY = "#999999"
BLACK = "#222222"


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


def main() -> None:
    set_journal_style()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

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

    # 180 mm × 130 mm — matches main-text Figure 1 proportions
    fig, axes = plt.subplots(2, 3, figsize=(7.09, 5.1), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # A. Sample-size diagnostics.
    ax_a.plot(neff["size"], neff["median_ci_width"], marker="o", color=BLUE, label="Median")
    ax_a.plot(neff["size"], neff["p90_ci_width"], marker="s", color=ORANGE, label="90th percentile")
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
    ax_b.bar([str(x) for x in buffer_order], counts.values, color=BLUE, alpha=0.85)
    ax_b.set_title("Adaptive buffer radius used")
    ax_b.set_xlabel("Radius covering selected refs (km)")
    ax_b.set_ylabel("Parents")
    clean_axis(ax_b)
    panel_label(ax_b, "b")

    # C. k choice.
    mean_cos = k_sweep[k_sweep["variant"] == "mean_cos"].sort_values("k")
    ax_c.plot(mean_cos["k"], mean_cos["bal_err"] * 100, marker="o", color=BLUE, label="Balanced error (%)")
    ax_c.plot(mean_cos["k"], mean_cos["brier"] * 100, marker="s", color=ORANGE, label="Brier × 100")
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
    ax_d.bar(x - width / 2, val_rates["error_rate"] * 100, width, label="Error", color=PURPLE)
    ax_d.bar(
        x + width / 2,
        val_rates["abstain_rate"] * 100,
        width,
        label="Indistinguishable",
        color=GREY,
    )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
    ax_d.set_title("Internal validation")
    ax_d.set_ylabel("Held-out references (%)")
    ax_d.legend(frameon=False, loc="upper left", handlelength=1.2)
    clean_axis(ax_d)
    colour_class_ticks(ax_d, CLASS_ORDER, axis="x")
    panel_label(ax_d, "d")

    # E. Threshold transfer from v3 pooled threshold to v4 per-class thresholds.
    transfer = (
        scores.assign(changed=scores["category_knn"] != scores["category_knn_v3t"])
        .groupby("parent_label")["changed"]
        .agg(["sum", "count"])
        .reindex(CLASS_ORDER)
    )
    pct_changed = transfer["sum"] / transfer["count"] * 100
    ax_e.bar(x, pct_changed, color=SKY)
    for i, cls in enumerate(CLASS_ORDER):
        ax_e.text(i, pct_changed.loc[cls] + 0.12, f"{int(transfer.loc[cls, 'sum'])}", ha="center", fontsize=6)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
    ax_e.set_title("Threshold transfer check")
    ax_e.set_ylabel("Sites changing class (%)")
    ax_e.set_ylim(0, max(4.0, float(pct_changed.max()) + 1.0))
    clean_axis(ax_e)
    colour_class_ticks(ax_e, CLASS_ORDER, axis="x")
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
        ax_f.bar(
            x,
            vals,
            bottom=bottom,
            color=CATEGORY_COLORS[category],
            label=category.replace("_", " "),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals
    ax_f.set_xticks(x)
    ax_f.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER])
    ax_f.set_title("Stable-site DoR outcomes")
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
    panel_label(ax_f, "f")

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
