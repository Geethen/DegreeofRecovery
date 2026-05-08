"""Generate v4 summary charts.

Outputs to v4/plots/:
  1. dor_distribution_by_class.png   — dor_knn histogram per stable_class
  2. category_breakdown_by_class.png — recovering/indistinguishable/degraded/no_data by class
  3. threshold_transfer.png          — per-class vs v3 pooled threshold (category_knn vs category_knn_v3t)
  4. validation_error_abstain.png    — error% and abstain% by stable_class and step (steps 1-4)

Usage:
  python v4/scripts/reporting/make_v4_charts.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
V4_DATA = os.path.join(BASE_DIR, "v4", "data")
V4_PLOTS = os.path.join(BASE_DIR, "v4", "plots")

CLASSES = ["stable_nature", "stable_crop", "stable_built"]
CAT_ORDER = ["recovering", "indistinguishable", "degraded", "no_data"]
CAT_COLORS = {
    "recovering": "#2c7a3d",
    "indistinguishable": "#b8b8b8",
    "degraded": "#a82a2a",
    "no_data": "#444444",
}
CLASS_COLORS = {
    "stable_nature": "#2c7a3d",
    "stable_crop": "#d2a13a",
    "stable_built": "#7a4a2c",
}


def plot_dor_distribution(scores: pd.DataFrame, thresholds: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        sub = scores[scores["parent_label"] == cls]
        vals = sub["dor_knn"].dropna().to_numpy()
        ax.hist(vals, bins=40, range=(0, 1), color=CLASS_COLORS[cls],
                edgecolor="white", alpha=0.85)
        t = thresholds.get(cls)
        if t is not None:
            ax.axvline(t, color="black", linestyle="--", linewidth=1,
                       label=f"t = {t:.4f}")
        ax.set_title(f"{cls}  (n = {len(vals)})")
        ax.set_xlabel("dor_knn")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylabel("Count")
    fig.suptitle("v4 — dor_knn distribution by stable_class")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_category_breakdown(scores: pd.DataFrame, out: str) -> None:
    counts = (
        scores.groupby(["parent_label", "category_knn"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(index=CLASSES, columns=CAT_ORDER, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(CLASSES))
    for cat in CAT_ORDER:
        vals = pct[cat].to_numpy()
        bars = ax.bar(CLASSES, vals, bottom=bottom, label=cat,
                      color=CAT_COLORS[cat], edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 4:
                ax.text(i, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9, color="white")
        bottom += vals
    for i, cls in enumerate(CLASSES):
        n = int(counts.loc[cls].sum())
        ax.text(i, 101, f"n = {n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of sites")
    ax.set_title("v4 — category breakdown by stable_class (per-class threshold)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_threshold_transfer(scores: pd.DataFrame, out: str) -> None:
    """Compare per-class threshold (category_knn) to v3 pooled (category_knn_v3t)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, col, title in [
        (axes[0], "category_knn", "v4 per-class threshold"),
        (axes[1], "category_knn_v3t", "v3 pooled threshold (transfer check)"),
    ]:
        counts = scores.groupby(["parent_label", col]).size().unstack(fill_value=0)
        counts = counts.reindex(index=CLASSES, columns=CAT_ORDER, fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(CLASSES))
        for cat in CAT_ORDER:
            vals = pct[cat].to_numpy()
            ax.bar(CLASSES, vals, bottom=bottom, label=cat,
                   color=CAT_COLORS[cat], edgecolor="white")
            bottom += vals
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.set_ylabel("% of sites")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.0, -0.08),
                   ncol=4, frameon=False)
    fig.suptitle("v4 — threshold-transfer comparison")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_validation_metrics(summary: pd.DataFrame, out: str) -> None:
    """Per-class error% and abstain% by step from within_parent_summary_v4.csv."""
    steps_keep = [
        "step1_calibrated_t",
        "step2_deadband",
        "step3_deadband_effect",
        "step4_knn",
    ]
    df = summary[summary["step"].isin(steps_keep)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for row, probe in enumerate(["good", "bad"]):
        for col, metric in enumerate(["error_rate", "abstain_rate"]):
            ax = axes[row, col]
            sub = df[df["probe_state"] == probe]
            x = np.arange(len(steps_keep))
            width = 0.25
            for i, cls in enumerate(CLASSES):
                vals = []
                for s in steps_keep:
                    rec = sub[(sub["step"] == s) & (sub["parent_label"] == cls)]
                    vals.append(float(rec[metric].iloc[0]) * 100
                                if not rec.empty else np.nan)
                ax.bar(x + (i - 1) * width, vals, width=width,
                       label=cls, color=CLASS_COLORS[cls], edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace("step", "s").replace("_", "\n", 1)
                                for s in steps_keep], fontsize=9)
            ax.set_title(f"{probe} probe — {metric.replace('_', ' ')} (%)")
            ax.set_ylabel("%")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("v4 within-parent validation — error% and abstain% by class and step")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=os.path.join(V4_DATA, "test_site_dor_v4.csv"))
    parser.add_argument("--summary", default=os.path.join(V4_DATA, "within_parent_summary_v4.csv"))
    parser.add_argument("--thresholds", default=os.path.join(V4_DATA, "calibrated_thresholds_v4.json"))
    parser.add_argument("--out-dir", default=V4_PLOTS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scores = pd.read_csv(args.scores)
    summary = pd.read_csv(args.summary)

    import json
    with open(args.thresholds) as fh:
        thresholds = {k: float(v) for k, v in json.load(fh).items()
                      if k.startswith("stable_")}

    p1 = os.path.join(args.out_dir, "dor_distribution_by_class.png")
    p2 = os.path.join(args.out_dir, "category_breakdown_by_class.png")
    p3 = os.path.join(args.out_dir, "threshold_transfer.png")
    p4 = os.path.join(args.out_dir, "validation_error_abstain.png")

    plot_dor_distribution(scores, thresholds, p1)
    plot_category_breakdown(scores, p2)
    plot_threshold_transfer(scores, p3)
    plot_validation_metrics(summary, p4)

    for p in (p1, p2, p3, p4):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
