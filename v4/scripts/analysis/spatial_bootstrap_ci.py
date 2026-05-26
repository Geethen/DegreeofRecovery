"""Spatial-block bootstrap vs IID bootstrap: CI width comparison.

The IID bootstrap in score_test_sites_v4.py treats each reference distance
value as independent. If reference points are spatially autocorrelated, the
effective sample size is < n, and the IID bootstrap underestimates variance
-> CIs are too narrow -> site classifications are overconfident.

This script compares:
  IID bootstrap  : resample n_g distances with replacement from d_g (current method)
  Block bootstrap: assign ref points to spatial blocks (radius = fitted corr range),
                   resample BLOCKS with replacement, then draw all points within
                   each selected block. Block size is set to the variogram-fitted
                   practical range (per-parent for candidates, median for stable).

The ratio CI_block / CI_IID is the inflation factor. Values > 1 mean the
IID CI is too narrow; values near 1 mean autocorrelation is not inflating
confidence.

Outputs:
  v4/data/spatial_bootstrap_ci_comparison.csv
  v4/plots/spatial_bootstrap_ci_comparison.png
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
V4_DATA  = BASE_DIR / "v4" / "data"
V4_PLOTS = BASE_DIR / "v4" / "plots"

REFS_PATH          = V4_DATA / "v4_stable_refs_alphaearth.parquet"
CAND_REFS_PATH     = BASE_DIR / "v2" / "data" / "recover_reference_samples_v2_mask_on_large_alphaearth.parquet"
TEST_SITES_PATH    = BASE_DIR / "v1" / "data" / "test_site_alphaearth_2024.parquet"
TEST_SITES_V4_PATH = V4_DATA / "test_site_alphaearth_2024_v4.parquet"
CORR_RANGE_CSV     = BASE_DIR / "v2" / "data" / "v2_corr_range_by_parent_random100.csv"

STRATEGY   = "random_100"
KNN_K      = 5
N_BOOT     = 2000
SEED       = 42
EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS        = 1e-12

# Fallback correlation range for stable sites (median from candidate variogram)
FALLBACK_CORR_RANGE_M = 4379.0

EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def haversine_matrix_m(lat_lon: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_lon[:, 0])
    lon = np.radians(lat_lon[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)
    return 2.0 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1 - a)))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def cosine_dists(x: np.ndarray, pts: np.ndarray) -> np.ndarray:
    nx  = np.linalg.norm(x) + EPS
    nps = np.linalg.norm(pts, axis=1) + EPS
    return 1.0 - (pts @ x) / (nps * nx)


def knn_score(d_g: np.ndarray, d_b: np.ndarray, k: int = KNN_K) -> float:
    if len(d_g) < k or len(d_b) < k:
        return float("nan")
    mg = float(np.mean(np.partition(d_g, k - 1)[:k]))
    mb = float(np.mean(np.partition(d_b, k - 1)[:k]))
    return mb / (mg + mb + EPS)


# ---------------------------------------------------------------------------
# IID bootstrap (current production method)
# ---------------------------------------------------------------------------

def iid_bootstrap_ci(
    d_g: np.ndarray,
    d_b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    point = knn_score(d_g, d_b)
    if not np.isfinite(point):
        return float("nan"), float("nan"), float("nan")
    ig = rng.integers(0, len(d_g), size=(n_boot, len(d_g)))
    ib = rng.integers(0, len(d_b), size=(n_boot, len(d_b)))
    boots = np.array([knn_score(d_g[ig[i]], d_b[ib[i]]) for i in range(n_boot)])
    lo, hi = float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Spatial block bootstrap
# ---------------------------------------------------------------------------

def assign_blocks(lat_lon: np.ndarray, block_radius_m: float) -> np.ndarray:
    """Greedy block assignment: seed a new block at each unassigned point.

    Returns integer block IDs (0..n_blocks-1) for each point.
    Block radius defines the maximum distance from seed to include a point.
    This is equivalent to a grid tessellation but works on irregular point sets.
    """
    n = len(lat_lon)
    block_ids = -np.ones(n, dtype=int)
    block_id  = 0
    for i in range(n):
        if block_ids[i] >= 0:
            continue
        # Compute distances from point i to all unassigned points
        lats = lat_lon[:, 0]
        lons = lat_lon[:, 1]
        dlat = np.radians(lats - lat_lon[i, 0])
        dlon = np.radians(lons - lat_lon[i, 1])
        phi1 = np.radians(lat_lon[i, 0])
        phi2 = np.radians(lats)
        a = np.sin(dlat / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
        dist_m = 2.0 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1 - a)))
        within = (dist_m <= block_radius_m) & (block_ids < 0)
        block_ids[within] = block_id
        block_id += 1
    return block_ids


def block_bootstrap_ci(
    d_g: np.ndarray,
    d_b: np.ndarray,
    ll_g: np.ndarray,   # (n_g, 2) lat/lon
    ll_b: np.ndarray,   # (n_b, 2) lat/lon
    corr_range_m: float,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    min_pts: int = KNN_K,
) -> tuple[float, float, float]:
    """Block bootstrap CI.

    Assigns points to spatial blocks of size ~corr_range_m, then resamples
    blocks (with replacement) rather than individual points. Each bootstrap
    replicate collects all points from the resampled blocks.
    """
    point = knn_score(d_g, d_b)
    if not np.isfinite(point):
        return float("nan"), float("nan"), float("nan")

    blk_g = assign_blocks(ll_g, corr_range_m)
    blk_b = assign_blocks(ll_b, corr_range_m)
    unique_g = np.unique(blk_g)
    unique_b = np.unique(blk_b)

    # If only 1 block per class, block bootstrap degenerates — return IID result
    if len(unique_g) < 2 or len(unique_b) < 2:
        return iid_bootstrap_ci(d_g, d_b, rng, n_boot)

    boots = []
    for _ in range(n_boot):
        # Resample blocks with replacement
        sel_g = rng.choice(unique_g, size=len(unique_g), replace=True)
        sel_b = rng.choice(unique_b, size=len(unique_b), replace=True)
        idx_g = np.concatenate([np.where(blk_g == b)[0] for b in sel_g])
        idx_b = np.concatenate([np.where(blk_b == b)[0] for b in sel_b])
        if len(idx_g) < min_pts or len(idx_b) < min_pts:
            continue
        boots.append(knn_score(d_g[idx_g], d_b[idx_b]))

    if len(boots) < 50:
        return iid_bootstrap_ci(d_g, d_b, rng, n_boot)

    boots = np.array(boots)
    lo = float(np.nanpercentile(boots, 2.5))
    hi = float(np.nanpercentile(boots, 97.5))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_refs(path: str, strategy: str | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state"] + EMBED_COLS
                     + ["geo.coordinates[1] AS lon", "geo.coordinates[2] AS lat"])
    q = f"SELECT {cols} FROM read_parquet(?)"
    if strategy:
        q += " WHERE strategy = ?"
        df = con.execute(q, [str(path), strategy]).df()
    else:
        df = con.execute(q, [str(path)]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return (df[df["ref_state"].isin(["good", "bad"])]
            .dropna(subset=["parent_id", "ref_state", "lat", "lon"] + EMBED_COLS)
            .reset_index(drop=True))


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


def load_corr_ranges(csv_path: str) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["parent_id"].astype(str), df["corr_range_m"].astype(float)))


# ---------------------------------------------------------------------------
# Per-site comparison
# ---------------------------------------------------------------------------

def compare_site(
    pid: str,
    x_obs: np.ndarray,
    sub: pd.DataFrame,
    corr_range_m: float,
    rng: np.random.Generator,
) -> dict | None:
    good = sub[sub["ref_state"] == "good"]
    bad  = sub[sub["ref_state"] == "bad"]
    if len(good) < KNN_K or len(bad) < KNN_K:
        return None

    emb_g = good[EMBED_COLS].to_numpy(dtype=float)
    emb_b = bad[EMBED_COLS].to_numpy(dtype=float)
    ll_g  = good[["lat", "lon"]].to_numpy(dtype=float)
    ll_b  = bad[["lat", "lon"]].to_numpy(dtype=float)

    d_g = cosine_dists(x_obs, emb_g)
    d_b = cosine_dists(x_obs, emb_b)

    dor, lo_iid, hi_iid = iid_bootstrap_ci(d_g, d_b, rng)
    if not np.isfinite(dor):
        return None

    _, lo_blk, hi_blk = block_bootstrap_ci(d_g, d_b, ll_g, ll_b, corr_range_m, rng)

    ci_iid  = hi_iid  - lo_iid
    ci_blk  = hi_blk  - lo_blk
    n_blk_g = len(np.unique(assign_blocks(ll_g, corr_range_m)))
    n_blk_b = len(np.unique(assign_blocks(ll_b, corr_range_m)))

    # Classification under each CI
    T = 0.4859  # pooled Youden threshold
    def classify(lo, hi):
        if lo > T:   return "recovering"
        if hi < T:   return "degraded"
        return "indistinguishable"

    cat_iid = classify(lo_iid, hi_iid)
    cat_blk = classify(lo_blk, hi_blk)

    return {
        "parent_id":    pid,
        "parent_label": str(sub["parent_label"].iloc[0]),
        "dor":          dor,
        "n_good":       len(good),
        "n_bad":        len(bad),
        "n_blocks_good": n_blk_g,
        "n_blocks_bad":  n_blk_b,
        "corr_range_m": corr_range_m,
        "ci_iid":       ci_iid,
        "ci_block":     ci_blk,
        "ci_ratio":     ci_blk / max(ci_iid, 1e-6),
        "lo_iid":       lo_iid, "hi_iid": hi_iid,
        "lo_blk":       lo_blk, "hi_blk": hi_blk,
        "cat_iid":      cat_iid,
        "cat_blk":      cat_blk,
        "category_changed": cat_iid != cat_blk,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    V4_PLOTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    corr_ranges = load_corr_ranges(str(CORR_RANGE_CSV))

    print("Loading candidate refs …")
    refs_cand = load_refs(str(CAND_REFS_PATH))
    test_cand = load_test_sites(str(TEST_SITES_PATH))
    print(f"  {refs_cand['parent_id'].nunique()} candidate parents")

    print("Loading stable refs …")
    refs_stab = load_refs(str(REFS_PATH), strategy=STRATEGY)
    test_stab = load_test_sites(str(TEST_SITES_V4_PATH))
    print(f"  {refs_stab['parent_id'].nunique()} stable parents")

    rows = []

    # Candidate sites — use per-parent fitted corr range
    print("Computing candidate site CIs …")
    refs_cand_grp = dict(tuple(refs_cand.groupby("parent_id", sort=False)))
    for i, (pid, x_obs) in enumerate(test_cand.items()):
        sub = refs_cand_grp.get(pid)
        if sub is None:
            continue
        cr = corr_ranges.get(pid, FALLBACK_CORR_RANGE_M)
        result = compare_site(pid, x_obs, sub, cr, rng)
        if result:
            rows.append(result)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_cand)}")

    # Stable sites — use fallback median range (variogram only run on candidates)
    print("Computing stable site CIs …")
    refs_stab_grp = dict(tuple(refs_stab.groupby("parent_id", sort=False)))
    for i, (pid, x_obs) in enumerate(test_stab.items()):
        sub = refs_stab_grp.get(pid)
        if sub is None:
            continue
        result = compare_site(pid, x_obs, sub, FALLBACK_CORR_RANGE_M, rng)
        if result:
            rows.append(result)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(test_stab)}")

    df = pd.DataFrame(rows)
    out_csv = V4_DATA / "spatial_bootstrap_ci_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("CI WIDTH COMPARISON: IID vs SPATIAL BLOCK BOOTSTRAP")
    print("="*60)

    for lbl in sorted(df["parent_label"].unique()):
        sub = df[df["parent_label"] == lbl]
        ratio = sub["ci_ratio"]
        changed = sub["category_changed"].sum()
        print(f"\n{lbl}  (n={len(sub)})")
        print(f"  Blocks per site (good/bad): "
              f"median {sub['n_blocks_good'].median():.0f} / {sub['n_blocks_bad'].median():.0f}")
        print(f"  CI ratio (block/IID): "
              f"median {ratio.median():.2f}  p25={ratio.quantile(0.25):.2f}  "
              f"p75={ratio.quantile(0.75):.2f}  max={ratio.max():.2f}")
        print(f"  IID CI width:   median {sub['ci_iid'].median():.4f}")
        print(f"  Block CI width: median {sub['ci_block'].median():.4f}")
        print(f"  Category changed (IID->block): {changed}/{len(sub)} = {100*changed/len(sub):.1f}%")

    # Global
    print(f"\nAll sites pooled (n={len(df)}):")
    print(f"  CI ratio: median {df['ci_ratio'].median():.2f}  "
          f"p90={df['ci_ratio'].quantile(0.9):.2f}")
    total_changed = df["category_changed"].sum()
    print(f"  Category changed: {total_changed}/{len(df)} = {100*total_changed/len(df):.1f}%")

    # Detailed breakdown of changed sites
    changed_df = df[df["category_changed"]]
    if len(changed_df) > 0:
        print(f"\nChanged site transitions:")
        transitions = changed_df.groupby(["cat_iid", "cat_blk"]).size()
        print(transitions.to_string())

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
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
        "built_loss":    "Built-loss",
        "crop_loss":     "Crop-loss",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    fig.subplots_adjust(wspace=0.38)

    labels_present = sorted(df["parent_label"].unique())

    # Panel A: scatter IID CI vs block CI, coloured by class
    ax = axes[0]
    for lbl in labels_present:
        sub = df[df["parent_label"] == lbl]
        ax.scatter(sub["ci_iid"], sub["ci_block"],
                   s=4, alpha=0.35, color=PALETTE.get(lbl, "#333333"),
                   label=LABEL_NAMES.get(lbl, lbl), rasterized=True)
    lim = max(df["ci_iid"].quantile(0.99), df["ci_block"].quantile(0.99)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("IID bootstrap CI width")
    ax.set_ylabel("Block bootstrap CI width")
    ax.set_title("(a) CI width: IID vs block")
    ax.legend(fontsize=5.5, framealpha=0.85, markerscale=2.5)

    # Panel B: CI ratio distribution by class (violin)
    ax2 = axes[1]
    ratio_data = [df[df["parent_label"] == lbl]["ci_ratio"].values for lbl in labels_present]
    vp = ax2.violinplot(ratio_data, positions=range(len(labels_present)),
                        showmedians=True, showextrema=False)
    for i, (body, lbl) in enumerate(zip(vp["bodies"], labels_present)):
        body.set_facecolor(PALETTE.get(lbl, "#333333"))
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.2)
    ax2.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xticks(range(len(labels_present)))
    ax2.set_xticklabels([LABEL_NAMES.get(l, l).replace(" ", "\n") for l in labels_present],
                        fontsize=5.5)
    ax2.set_ylabel("CI ratio (block / IID)")
    ax2.set_title("(b) CI inflation by class")

    # Panel C: n_blocks histogram (effective independence per pool)
    ax3 = axes[2]
    for lbl in labels_present:
        sub = df[df["parent_label"] == lbl]
        ax3.hist(sub["n_blocks_good"], bins=20, alpha=0.45,
                 color=PALETTE.get(lbl, "#333333"),
                 label=LABEL_NAMES.get(lbl, lbl), density=True)
    ax3.axvline(5, color="black", lw=0.8, ls="--", alpha=0.7, label="k=5 (min knn)")
    ax3.set_xlabel("Independent spatial blocks (good pool)")
    ax3.set_ylabel("Density")
    ax3.set_title("(c) Effective independence of good refs")
    ax3.legend(fontsize=5.5, framealpha=0.85)

    out_png = V4_PLOTS / "spatial_bootstrap_ci_comparison.png"
    out_pdf = V4_PLOTS / "spatial_bootstrap_ci_comparison.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
