"""Distance-stratified DoR sensitivity analysis.

Re-scores each test site using only reference points beyond a minimum spatial
distance from the parent centroid. If DoR scores are stable as nearby refs are
excluded, spatial proximity is not driving the scores. If DoR drops
systematically, proximity inflation is real.

Thresholds tested: 0 m (baseline), 500 m, 1000 m, 2000 m, 3000 m.

Outputs:
  v4/data/distance_stratified_dor.csv  — per-site DoR at each threshold
  v4/plots/distance_stratified_dor.png — summary figure
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
V4_DATA   = BASE_DIR / "v4" / "data"
V4_PLOTS  = BASE_DIR / "v4" / "plots"

REFS_PATH            = V4_DATA / "v4_stable_refs_alphaearth.parquet"
CAND_REFS_PATH       = BASE_DIR / "v2" / "data" / "recover_reference_samples_v2_mask_on_large_alphaearth.parquet"
TEST_SITES_PATH      = BASE_DIR / "v1" / "data" / "test_site_alphaearth_2024.parquet"
TEST_SITES_PATH_V4   = V4_DATA / "test_site_alphaearth_2024_v4.parquet"

STRATEGY   = "random_100"
KNN_K      = 5
N_BOOT     = 500   # lighter than production — enough for shape comparison
SEED       = 42
EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS        = 1e-12

MIN_DIST_THRESHOLDS = [0, 500, 1000, 2000, 3000]
MIN_REFS_PER_CLASS  = 5   # skip scoring if fewer than this survive the filter


# ---------------------------------------------------------------------------
# Core scoring helpers (inline to avoid import dependency on src/)
# ---------------------------------------------------------------------------

def cosine_dists(x: np.ndarray, pts: np.ndarray) -> np.ndarray:
    nx  = np.linalg.norm(x) + EPS
    nps = np.linalg.norm(pts, axis=1) + EPS
    return 1.0 - (pts @ x) / (nps * nx)


def knn_score(d_g: np.ndarray, d_b: np.ndarray, k: int = KNN_K) -> float:
    mg = float(np.mean(np.partition(d_g, min(k - 1, len(d_g) - 1))[:k]))
    mb = float(np.mean(np.partition(d_b, min(k - 1, len(d_b) - 1))[:k]))
    return mb / (mg + mb + EPS)


def bootstrap_dor(d_g: np.ndarray, d_b: np.ndarray,
                  rng: np.random.Generator, n_boot: int) -> float:
    ig = rng.integers(0, len(d_g), size=(n_boot, len(d_g)))
    ib = rng.integers(0, len(d_b), size=(n_boot, len(d_b)))
    scores = np.array([
        knn_score(d_g[ig[i]], d_b[ib[i]]) for i in range(n_boot)
    ])
    return float(np.std(scores))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_refs_with_dist(path: str, strategy: str | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state", "dist_m"] + EMBED_COLS)
    if strategy:
        df = con.execute(
            f"SELECT {cols} FROM read_parquet(?) WHERE strategy = ?",
            [str(path), strategy],
        ).df()
    else:
        df = con.execute(
            f"SELECT {cols} FROM read_parquet(?)", [str(path)]
        ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return df[df["ref_state"].isin(["good", "bad"])].dropna(
        subset=["parent_id", "ref_state", "dist_m"] + EMBED_COLS
    ).reset_index(drop=True)


def load_test_sites(path: str) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    cols = ", ".join(["parent_id"] + EMBED_COLS)
    df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [str(path)]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=["parent_id"] + EMBED_COLS).reset_index(drop=True)
    return {
        pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
        for pid, g in df.groupby("parent_id", sort=False)
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def score_at_threshold(
    refs: pd.DataFrame,
    test_emb: dict[str, np.ndarray],
    min_dist_m: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for pid, x_obs in test_emb.items():
        sub = refs[refs["parent_id"] == pid]
        if sub.empty:
            continue
        sub_filt = sub[sub["dist_m"] >= min_dist_m]
        good = sub_filt.loc[sub_filt["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
        bad  = sub_filt.loc[sub_filt["ref_state"] == "bad",  EMBED_COLS].to_numpy(dtype=float)
        if len(good) < MIN_REFS_PER_CLASS or len(bad) < MIN_REFS_PER_CLASS:
            continue
        d_g = cosine_dists(x_obs, good)
        d_b = cosine_dists(x_obs, bad)
        dor = knn_score(d_g, d_b)
        parent_label = str(sub["parent_label"].iloc[0])
        rows.append({
            "parent_id":    pid,
            "parent_label": parent_label,
            "min_dist_m":   int(min_dist_m),
            "n_good":       len(good),
            "n_bad":        len(bad),
            "dor_knn":      dor,
        })
    return pd.DataFrame(rows)


def main() -> None:
    V4_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading stable refs …")
    refs_stable = load_refs_with_dist(str(REFS_PATH), strategy=STRATEGY)
    print(f"  {len(refs_stable):,} rows, {refs_stable['parent_id'].nunique()} parents")

    print("Loading candidate (loss-site) refs …")
    refs_cand = load_refs_with_dist(str(CAND_REFS_PATH), strategy=None)
    print(f"  {len(refs_cand):,} rows, {refs_cand['parent_id'].nunique()} parents")

    print("Loading candidate (loss-site) test site embeddings …")
    test_emb_cand = load_test_sites(str(TEST_SITES_PATH))
    print(f"  {len(test_emb_cand)} candidate test sites")

    print("Loading stable test site embeddings …")
    test_emb_stable = load_test_sites(str(TEST_SITES_PATH_V4))
    print(f"  {len(test_emb_stable)} stable test sites")

    rng = np.random.default_rng(SEED)

    all_frames = []
    # Candidate sites: test embeddings from v1, refs from v2
    for thr in MIN_DIST_THRESHOLDS:
        print(f"  Scoring candidates at min_dist = {thr} m …")
        df = score_at_threshold(refs_cand, test_emb_cand, thr, rng)
        all_frames.append(df)
        print(f"    {len(df)} sites scored")

    # Stable sites: test embeddings from v4, refs from v4
    for thr in MIN_DIST_THRESHOLDS:
        print(f"  Scoring stable sites at min_dist = {thr} m …")
        df = score_at_threshold(refs_stable, test_emb_stable, thr, rng)
        all_frames.append(df)
        print(f"    {len(df)} sites scored")

    results = pd.concat(all_frames, ignore_index=True)
    out_csv = V4_DATA / "distance_stratified_dor.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    # Pivot: one row per (parent_id, parent_label), columns = thresholds
    pivot = results.pivot_table(
        index=["parent_id", "parent_label"],
        columns="min_dist_m",
        values="dor_knn",
    ).reset_index()

    labels_present = sorted(results["parent_label"].unique())

    PALETTE = {
        "stable_nature": "#009E73",
        "stable_crop":   "#0072B2",
        "stable_built":  "#E69F00",
        "built_loss":    "#D55E00",
        "crop_loss":     "#CC79A7",
    }
    LABEL_NAMES = {
        "stable_nature": "Stable near-natural",
        "stable_crop":   "Stable cropland",
        "stable_built":  "Stable built-up",
        "built_loss":    "Built-loss (candidate)",
        "crop_loss":     "Crop-loss (candidate)",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })

    thrs = MIN_DIST_THRESHOLDS

    # Panel A: median DoR per threshold per class (line plot)
    # Panel B: violin of DoR change (thr - baseline) per class
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(wspace=0.38)

    ax = axes[0]
    for lbl in labels_present:
        sub = pivot[pivot["parent_label"] == lbl]
        medians = []
        xs = []
        for t in thrs:
            if t in sub.columns:
                col = sub[t].dropna()
                if len(col) >= 3:
                    medians.append(col.median())
                    xs.append(t)
        if len(xs) >= 2:
            ax.plot(
                [x / 1000 for x in xs], medians,
                marker="o", markersize=3.5,
                color=PALETTE.get(lbl, "#333333"),
                label=LABEL_NAMES.get(lbl, lbl),
            )
    ax.set_xlabel("Minimum exclusion radius (km)")
    ax.set_ylabel("Median DoR (knn5)")
    ax.set_title("(a) Median DoR vs exclusion radius")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=5.5, framealpha=0.85)

    # Panel B: box of (DoR_thr - DoR_0) per threshold, pooled across sites
    ax2 = axes[1]
    deltas_by_thr = {}
    for t in thrs[1:]:
        if t not in pivot.columns or 0 not in pivot.columns:
            continue
        d = (pivot[t] - pivot[0]).dropna()
        deltas_by_thr[t] = d.values

    positions = list(range(len(deltas_by_thr)))
    xlabels = [f"+{t//1000} km" for t in deltas_by_thr]
    bp = ax2.boxplot(
        list(deltas_by_thr.values()),
        positions=positions,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#CCCCCC")
        patch.set_alpha(0.7)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(xlabels)
    ax2.set_xlabel("Exclusion radius (vs 0 m baseline)")
    ax2.set_ylabel("Delta DoR (threshold - baseline)")
    ax2.set_title("(b) DoR shift from excluding nearby refs")

    out_png = V4_PLOTS / "distance_stratified_dor.png"
    out_pdf = V4_PLOTS / "distance_stratified_dor.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    # -------------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------------
    print("\nMedian DoR by class and exclusion threshold:")
    summary = results.groupby(["parent_label", "min_dist_m"])["dor_knn"].median().unstack()
    print(summary.round(4).to_string())

    print("\nSites surviving each threshold:")
    counts = results.groupby(["parent_label", "min_dist_m"])["parent_id"].count().unstack()
    print(counts.to_string())

    if 0 in pivot.columns:
        print("\nMedian DoR shift from baseline (all classes pooled):")
        for t in thrs[1:]:
            if t in pivot.columns:
                d = (pivot[t] - pivot[0]).dropna()
                print(f"  {t:>5} m: median delta = {d.median():+.4f}  (n={len(d)})")


if __name__ == "__main__":
    main()
