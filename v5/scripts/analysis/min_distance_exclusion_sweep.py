"""v5 minimum-exclusion-radius sweep.

Question: how large can we make the minimum distance between the test-site
centroid and ref points (the exclusion radius) before per-site sample counts
collapse below the v4 operational floor of 30 per pool?

Background:
  - v4 sampler has no minimum exclusion radius between the parent centroid and
    its ref pool. Per-site inspection (v4/scripts/analysis/inspect_built_loss_distance.py)
    found bad refs landing 3-40 m from the test-site centroid for built-loss
    parents, with cosine distances of 0.004-0.011 (effectively identical
    embeddings). That is sampler pseudoreplication, not geostatistical
    autocorrelation: it pulls m_b toward zero and inflates DoR.
  - v4 distance-stratified analysis (distance_stratified_dor.py) confirmed the
    signal at the 500/1000/2000/3000 m steps but those thresholds discard most
    of the pool. We need to find the smallest defensible floor that eliminates
    the contamination without hollowing out sample sizes.

The 100 m candidate clears the AlphaEarth 10 m pixel and a ~3x3 pixel
neighbourhood. This script tests whether 100 m is feasible per-class, or
whether 30/50 m is the largest survivable floor.

Inputs (read-only, no re-sampling):
  v4/data/v4_stable_refs_alphaearth.parquet                                  -- stable refs (v4 sampler)
  v2/data/recover_reference_samples_v2_mask_on_large_alphaearth.parquet      -- candidate (loss-site) refs (v2 sampler)

Outputs:
  v5/data/min_distance_survival_per_site.csv      -- per-(site, threshold, pool) n surviving
  v5/data/min_distance_survival_summary.csv       -- per-(class, threshold) summary stats
  v5/data/min_distance_built_loss_closest.csv     -- built-loss per-site closest-bad-ref distance + cosine
  v5/plots/min_distance_survival.png/.pdf         -- per-class survival curves vs threshold
  v5/plots/min_distance_built_loss_closest.png    -- built-loss deep-dive figure

Thresholds tested: 0, 30, 50, 100, 200, 300, 500, 1000 m.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
V5_DATA  = BASE_DIR / "v5" / "data"
V5_PLOTS = BASE_DIR / "v5" / "plots"

STABLE_REFS_PATH = BASE_DIR / "v4" / "data" / "v4_stable_refs_alphaearth.parquet"
CAND_REFS_PATH   = BASE_DIR / "v2" / "data" / "recover_reference_samples_v2_mask_on_large_alphaearth.parquet"
TEST_SITES_CAND  = BASE_DIR / "v1" / "data" / "test_site_alphaearth_2024.parquet"
TEST_SITES_STAB  = BASE_DIR / "v4" / "data" / "test_site_alphaearth_2024_v4.parquet"

STRATEGY    = "random_100"   # the strategy v4 scoring uses
MIN_DIST_THRESHOLDS = [0, 30, 50, 100, 200, 300, 500, 1000]
EMBED_COLS  = [f"A{i:02d}" for i in range(64)]
EPS         = 1e-12

FLOORS = (5, 30, 100)        # report survival at each
BUILT_LOSS_LABEL = "built_loss"

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_refs(path: Path, strategy: str | None) -> pd.DataFrame:
    """Load ref points with dist_m. Stable refs use strategy filter; candidate
    refs in v2 don't all carry one, so accept strategy=None to skip."""
    con = duckdb.connect()
    cols = ["parent_id", "parent_label", "ref_state", "dist_m"]
    if strategy:
        df = con.execute(
            f"SELECT {', '.join(cols)} FROM read_parquet(?) WHERE strategy = ?",
            [str(path), strategy],
        ).df()
    else:
        df = con.execute(
            f"SELECT {', '.join(cols)} FROM read_parquet(?)", [str(path)]
        ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])]
    df = df.dropna(subset=["parent_id", "ref_state", "dist_m"]).reset_index(drop=True)
    return df


def load_refs_with_embeddings(path: Path, parent_ids: list[str],
                              strategy: str | None) -> pd.DataFrame:
    """Heavier load used only for the built-loss deep-dive."""
    con = duckdb.connect()
    cols = ["parent_id", "parent_label", "ref_state", "dist_m"] + EMBED_COLS
    if strategy:
        df = con.execute(
            f"SELECT {', '.join(cols)} FROM read_parquet(?) "
            f"WHERE strategy = ? AND parent_id IN (SELECT unnest(?))",
            [str(path), strategy, parent_ids],
        ).df()
    else:
        df = con.execute(
            f"SELECT {', '.join(cols)} FROM read_parquet(?) "
            f"WHERE parent_id IN (SELECT unnest(?))",
            [str(path), parent_ids],
        ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])]
    df = df.dropna(subset=["parent_id", "ref_state", "dist_m"] + EMBED_COLS).reset_index(drop=True)
    return df


def load_test_centroids_embedding(path: Path) -> dict[str, np.ndarray]:
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
# Survival sweep
# ---------------------------------------------------------------------------

def per_site_survival(refs: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """For each (parent_id, parent_label, threshold), count surviving good and
    bad refs. Returns long form."""
    rows = []
    for (pid, lbl), grp in refs.groupby(["parent_id", "parent_label"], sort=False):
        dist = grp["dist_m"].to_numpy()
        state = grp["ref_state"].to_numpy()
        for thr in thresholds:
            keep = dist >= thr
            n_good = int(((state == "good") & keep).sum())
            n_bad  = int(((state == "bad")  & keep).sum())
            rows.append({
                "parent_id":    pid,
                "parent_label": lbl,
                "min_dist_m":   int(thr),
                "n_good":       n_good,
                "n_bad":        n_bad,
                "n_min":        min(n_good, n_bad),
            })
    return pd.DataFrame(rows)


def summarise_survival(per_site: pd.DataFrame, floors: tuple[int, ...]) -> pd.DataFrame:
    """Per-(parent_label, threshold) summary: median/p10/p25 of n_min, plus
    fraction passing each floor."""
    rows = []
    for (lbl, thr), grp in per_site.groupby(["parent_label", "min_dist_m"], sort=True):
        n = len(grp)
        nmin = grp["n_min"]
        ngood = grp["n_good"]
        nbad  = grp["n_bad"]
        rec = {
            "parent_label": lbl,
            "min_dist_m":   int(thr),
            "n_sites":      n,
            "median_n_good": float(ngood.median()),
            "p25_n_good":    float(ngood.quantile(0.25)),
            "p10_n_good":    float(ngood.quantile(0.10)),
            "median_n_bad":  float(nbad.median()),
            "p25_n_bad":     float(nbad.quantile(0.25)),
            "p10_n_bad":     float(nbad.quantile(0.10)),
            "median_n_min":  float(nmin.median()),
            "p25_n_min":     float(nmin.quantile(0.25)),
            "p10_n_min":     float(nmin.quantile(0.10)),
        }
        for f in floors:
            rec[f"frac_pass_{f}"] = float((nmin >= f).mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["parent_label", "min_dist_m"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Built-loss deep-dive
# ---------------------------------------------------------------------------

def cosine_dist(x: np.ndarray, pts: np.ndarray) -> np.ndarray:
    nx = np.linalg.norm(x) + EPS
    nps = np.linalg.norm(pts, axis=1) + EPS
    return 1.0 - (pts @ x) / (nps * nx)


def built_loss_closest(refs_emb: pd.DataFrame,
                       test_emb: dict[str, np.ndarray]) -> pd.DataFrame:
    """For each built-loss parent: distance + cosine of the closest bad ref,
    and how many bad refs sit within {30, 50, 100, 200, 300} m."""
    band_radii = [30, 50, 100, 200, 300]
    rows = []
    for pid, grp in refs_emb.groupby("parent_id", sort=False):
        if pid not in test_emb:
            continue
        bad = grp[grp["ref_state"] == "bad"]
        if bad.empty:
            continue
        x = test_emb[pid]
        emb = bad[EMBED_COLS].to_numpy(dtype=float)
        dist_m = bad["dist_m"].to_numpy()
        cdist = cosine_dist(x, emb)
        i_min = int(np.argmin(dist_m))
        rec = {
            "parent_id":            pid,
            "n_bad_total":          int(len(bad)),
            "closest_bad_m":        float(dist_m[i_min]),
            "closest_bad_cosine":   float(cdist[i_min]),
            "min_cosine_any_bad":   float(np.min(cdist)),
            "median_bad_m":         float(np.median(dist_m)),
        }
        for r in band_radii:
            rec[f"n_bad_within_{r}m"] = int((dist_m < r).sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("closest_bad_m").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_survival(per_site: pd.DataFrame, summary: pd.DataFrame,
                  out_png: Path, out_pdf: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })

    labels = sorted(summary["parent_label"].unique())
    thrs = sorted(per_site["min_dist_m"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    fig.subplots_adjust(wspace=0.32)

    # Panel A: median n_min (per pool floor) vs threshold, per class
    axA = axes[0]
    for lbl in labels:
        sub = summary[summary["parent_label"] == lbl].sort_values("min_dist_m")
        color = PALETTE.get(lbl, "#333333")
        axA.plot(sub["min_dist_m"], sub["median_n_min"], marker="o", markersize=3.5,
                 color=color, label=LABEL_NAMES.get(lbl, lbl))
        axA.fill_between(sub["min_dist_m"], sub["p10_n_min"], sub["p25_n_min"],
                         color=color, alpha=0.15, linewidth=0)
    axA.axhline(30, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    axA.text(thrs[-1], 31, "v4 floor (30)", ha="right", fontsize=5.5, color="black")
    axA.axhline(5, color="grey", linewidth=0.7, linestyle=":", alpha=0.6)
    axA.text(thrs[-1], 6, "knn5 minimum (5)", ha="right", fontsize=5.5, color="grey")
    axA.set_xlabel("Minimum exclusion radius (m)")
    axA.set_ylabel("Per-pool surviving count (min of good/bad)")
    axA.set_title("(a) Per-pool surviving sample count")
    axA.legend(fontsize=5.5, framealpha=0.85)

    # Panel B: fraction of sites passing the 30 floor
    axB = axes[1]
    for lbl in labels:
        sub = summary[summary["parent_label"] == lbl].sort_values("min_dist_m")
        axB.plot(sub["min_dist_m"], sub["frac_pass_30"], marker="o", markersize=3.5,
                 color=PALETTE.get(lbl, "#333333"),
                 label=LABEL_NAMES.get(lbl, lbl))
    axB.axhline(0.90, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    axB.text(thrs[-1], 0.905, "90% target", ha="right", fontsize=5.5, color="black")
    axB.set_xlabel("Minimum exclusion radius (m)")
    axB.set_ylabel("Fraction of sites with n_good and n_bad >= 30")
    axB.set_title("(b) Survival of the v4 operational floor")
    axB.set_ylim(-0.02, 1.02)
    axB.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axB.legend(fontsize=5.5, framealpha=0.85)

    # Panel C: fraction of sites still scorable (n_min >= 5)
    axC = axes[2]
    for lbl in labels:
        sub = summary[summary["parent_label"] == lbl].sort_values("min_dist_m")
        axC.plot(sub["min_dist_m"], sub["frac_pass_5"], marker="o", markersize=3.5,
                 color=PALETTE.get(lbl, "#333333"),
                 label=LABEL_NAMES.get(lbl, lbl))
    axC.set_xlabel("Minimum exclusion radius (m)")
    axC.set_ylabel("Fraction of sites still scorable (n_min >= 5)")
    axC.set_title("(c) Scorability floor")
    axC.set_ylim(-0.02, 1.02)
    axC.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axC.legend(fontsize=5.5, framealpha=0.85)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_built_loss_closest(per_site: pd.DataFrame,
                            closest_df: pd.DataFrame,
                            out_png: Path, out_pdf: Path) -> None:
    """Built-loss deep-dive:
      (a) ECDF of closest-bad-ref distance across built-loss parents
      (b) Per-site stacked counts of bad refs within {30,50,100,200,300} m
      (c) Joint scatter: closest-bad-ref distance vs cosine
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    fig.subplots_adjust(wspace=0.34)

    # Panel A: ECDF of closest_bad_m
    axA = axes[0]
    x = np.sort(closest_df["closest_bad_m"].to_numpy())
    if len(x) > 0:
        y = np.arange(1, len(x) + 1) / len(x)
        axA.step(x, y, color=PALETTE["built_loss"], where="post")
        for thr in (30, 50, 100, 200, 300):
            frac_below = float((x < thr).mean())
            axA.axvline(thr, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)
            axA.text(thr, 0.02, f"  {thr}m: {frac_below*100:.0f}% < ",
                     fontsize=5.5, rotation=90, va="bottom", ha="left", color="grey")
    axA.set_xscale("symlog", linthresh=10)
    axA.set_xlabel("Distance from test site to closest bad ref (m)")
    axA.set_ylabel("Empirical CDF (built-loss parents)")
    axA.set_title("(a) How close does the closest bad ref sit?")
    axA.set_ylim(-0.02, 1.02)

    # Panel B: number of bad refs within bands, sorted by closest_bad_m
    axB = axes[1]
    band_radii = [30, 50, 100, 200, 300]
    sub = closest_df.copy().sort_values("closest_bad_m").reset_index(drop=True)
    if len(sub) > 0:
        x_pos = np.arange(len(sub))
        for r, color in zip(band_radii,
                            ["#7B1E1E", "#B7402F", "#D87C32", "#E6B73B", "#F2DD7A"]):
            axB.plot(x_pos, sub[f"n_bad_within_{r}m"],
                     color=color, linewidth=0.8, label=f"< {r} m")
        axB.set_xlabel("Built-loss parents (sorted by closest-bad-ref distance)")
        axB.set_ylabel("Number of bad refs within radius")
        axB.set_title("(b) Pseudoreplication near the test site")
        axB.legend(fontsize=5.5, framealpha=0.85, title="band", title_fontsize=5.5)

    # Panel C: closest_bad_m vs closest_bad_cosine
    axC = axes[2]
    if len(closest_df) > 0:
        axC.scatter(closest_df["closest_bad_m"], closest_df["closest_bad_cosine"],
                    s=8, alpha=0.5, color=PALETTE["built_loss"], edgecolors="none")
    axC.set_xscale("symlog", linthresh=10)
    axC.axhline(0.05, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)
    axC.text(axC.get_xlim()[1] if len(closest_df) else 1, 0.052,
             "0.05 cosine", ha="right", fontsize=5.5, color="grey")
    axC.set_xlabel("Distance to closest bad ref (m)")
    axC.set_ylabel("Cosine distance to closest bad ref")
    axC.set_title("(c) Near refs are also near in embedding space")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    V5_DATA.mkdir(parents=True, exist_ok=True)
    V5_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading stable refs ...")
    refs_stable = load_refs(STABLE_REFS_PATH, strategy=STRATEGY)
    print(f"  {len(refs_stable):,} rows, {refs_stable['parent_id'].nunique()} parents")

    print("Loading candidate refs ...")
    refs_cand = load_refs(CAND_REFS_PATH, strategy=None)
    print(f"  {len(refs_cand):,} rows, {refs_cand['parent_id'].nunique()} parents")

    all_refs = pd.concat([refs_stable, refs_cand], ignore_index=True)

    print("\nComputing per-site survival ...")
    per_site = per_site_survival(all_refs, MIN_DIST_THRESHOLDS)
    per_site_out = V5_DATA / "min_distance_survival_per_site.csv"
    per_site.to_csv(per_site_out, index=False)
    print(f"  Wrote {per_site_out}")

    summary = summarise_survival(per_site, FLOORS)
    summary_out = V5_DATA / "min_distance_survival_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"  Wrote {summary_out}")

    print("\nSurvival summary (median per-pool count, fraction passing the 30-floor):")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[summary["parent_label"] == lbl]
        print(f"\n  {lbl}  (n = {int(sub['n_sites'].iloc[0])} sites)")
        for _, r in sub.iterrows():
            print(f"    {int(r['min_dist_m']):>5} m: "
                  f"median n_min = {r['median_n_min']:>5.1f}  "
                  f"p25 = {r['p25_n_min']:>5.1f}  "
                  f"p10 = {r['p10_n_min']:>5.1f}  "
                  f"frac>=30 = {r['frac_pass_30']*100:>5.1f}%  "
                  f"frac>=5 = {r['frac_pass_5']*100:>5.1f}%")

    print("\nWriting per-class survival figure ...")
    plot_survival(per_site, summary,
                  V5_PLOTS / "min_distance_survival.png",
                  V5_PLOTS / "min_distance_survival.pdf")
    print(f"  Wrote {V5_PLOTS / 'min_distance_survival.png'}")

    # -----------------------------------------------------------------------
    # Built-loss deep-dive
    # -----------------------------------------------------------------------
    print("\nLoading built-loss ref embeddings for deep-dive ...")
    built_pids = sorted(per_site.loc[per_site["parent_label"] == BUILT_LOSS_LABEL,
                                     "parent_id"].unique().tolist())
    print(f"  {len(built_pids)} built-loss parents")

    refs_built_emb = load_refs_with_embeddings(CAND_REFS_PATH, built_pids, strategy=None)
    print(f"  Loaded {len(refs_built_emb):,} ref rows with embeddings")

    print("Loading candidate test-site embeddings ...")
    test_emb_cand = load_test_centroids_embedding(TEST_SITES_CAND)
    print(f"  {len(test_emb_cand)} candidate test sites")

    closest_df = built_loss_closest(refs_built_emb, test_emb_cand)
    closest_out = V5_DATA / "min_distance_built_loss_closest.csv"
    closest_df.to_csv(closest_out, index=False)
    print(f"  Wrote {closest_out}")

    print("\nBuilt-loss closest-bad-ref distance distribution:")
    if not closest_df.empty:
        for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
            v = float(closest_df["closest_bad_m"].quantile(q))
            print(f"  q{int(q*100):02d} = {v:>8.1f} m")
        for r in (30, 50, 100, 200, 300):
            frac = float((closest_df["closest_bad_m"] < r).mean())
            print(f"  fraction of built-loss parents with closest bad ref < {r} m: "
                  f"{frac*100:5.1f}%")
        # Cosine of the closest bad ref
        for q in (0.05, 0.10, 0.25, 0.50):
            v = float(closest_df["closest_bad_cosine"].quantile(q))
            print(f"  closest_bad_cosine q{int(q*100):02d} = {v:.4f}")

    print("\nWriting built-loss deep-dive figure ...")
    plot_built_loss_closest(per_site, closest_df,
                            V5_PLOTS / "min_distance_built_loss_closest.png",
                            V5_PLOTS / "min_distance_built_loss_closest.pdf")
    print(f"  Wrote {V5_PLOTS / 'min_distance_built_loss_closest.png'}")

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    print("\nLargest threshold preserving frac_pass_30 >= 0.90 per class:")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[(summary["parent_label"] == lbl)
                      & (summary["frac_pass_30"] >= 0.90)].sort_values("min_dist_m")
        if sub.empty:
            print(f"  {lbl:<16}: never reaches 90% (max frac = "
                  f"{summary.loc[summary['parent_label']==lbl,'frac_pass_30'].max()*100:.1f}%)")
        else:
            print(f"  {lbl:<16}: <= {int(sub['min_dist_m'].max())} m")

    print("\nLargest threshold preserving frac_pass_30 >= 0.95 per class:")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[(summary["parent_label"] == lbl)
                      & (summary["frac_pass_30"] >= 0.95)].sort_values("min_dist_m")
        if sub.empty:
            print(f"  {lbl:<16}: never reaches 95%")
        else:
            print(f"  {lbl:<16}: <= {int(sub['min_dist_m'].max())} m")


if __name__ == "__main__":
    main()
