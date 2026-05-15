"""Spatial autocorrelation check for v3 knn5.

For each test site, the knn5 scorer selects the 5 nearest good and 5 nearest
bad reference embeddings (cosine distance) within the same parent_id group.

Concern: are those nearest neighbours also the spatially closest refs?
If so, spatial autocorrelation — not ecological similarity — may drive scores.

This script:
  1. Loads refs (random_100 strategy) and test sites.
  2. For each (test_site, parent_id) pair, computes cosine distance from the
     test-site embedding to every good ref and every bad ref.
  3. Identifies the knn5 selected refs (5 smallest cosine distances).
  4. Computes haversine distance (km) from the test site to every ref.
  5. Produces scatter plots: cosine rank vs spatial distance, coloured by label.
"""
from __future__ import annotations

import os
import math

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
REFS_PATH = os.path.join(
    BASE_DIR, "v2", "data", "v2real_mask_on_corr300_exhaustive",
    "sampling_strategy_selected_points.parquet",
)
TEST_SITES_PATH = os.path.join(BASE_DIR, "v1", "data",
                               "test_site_alphaearth_2024.parquet")
OUT_DIR = os.path.join(BASE_DIR, "v3", "plots")
STRATEGY = "random_100"
K = 5
EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cosine_dists(x: np.ndarray, pts: np.ndarray) -> np.ndarray:
    nx = np.linalg.norm(x) + EPS
    np_pts = np.linalg.norm(pts, axis=1, keepdims=False) + EPS
    return 1.0 - (pts @ x) / (np_pts * nx)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_refs(path: str, strategy: str) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state", "lat", "lon"] + EMBED_COLS)
    df = con.execute(
        f"SELECT {cols} FROM read_parquet(?) WHERE strategy = ?",
        [path, strategy],
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return df[df["ref_state"].isin(["good", "bad"])].dropna(
        subset=["parent_id", "ref_state", "lat", "lon"] + EMBED_COLS
    ).reset_index(drop=True)


def load_test_sites(path: str) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id"] + EMBED_COLS + [
        "geo.coordinates[1] AS lon",
        "geo.coordinates[2] AS lat",
    ])
    df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [path]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return df.dropna(subset=["parent_id", "lat", "lon"] + EMBED_COLS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build per-ref distance table
# ---------------------------------------------------------------------------
def build_distance_table(refs: pd.DataFrame, test_sites: pd.DataFrame) -> pd.DataFrame:
    # One centroid per test-site parent_id (mean embedding, mean lat/lon)
    ts_agg = (
        test_sites.groupby("parent_id", sort=False)
        .agg(
            ts_lat=("lat", "mean"),
            ts_lon=("lon", "mean"),
            **{c: (c, "mean") for c in EMBED_COLS},
        )
        .reset_index()
    )

    rows = []
    for _, ts in ts_agg.iterrows():
        pid = ts["parent_id"]
        sub = refs[refs["parent_id"] == pid]
        if sub.empty:
            continue

        x = ts[EMBED_COLS].to_numpy(dtype=float)
        pts = sub[EMBED_COLS].to_numpy(dtype=float)
        cos_d = cosine_dists(x, pts)

        ts_lat, ts_lon = float(ts["ts_lat"]), float(ts["ts_lon"])

        for i, (_, ref_row) in enumerate(sub.iterrows()):
            spat_km = haversine_km(ts_lat, ts_lon,
                                   float(ref_row["lat"]), float(ref_row["lon"]))
            rows.append({
                "parent_id":    pid,
                "ref_state":    ref_row["ref_state"],
                "cosine_dist":  float(cos_d[i]),
                "spatial_km":   spat_km,
            })

    df = pd.DataFrame(rows)

    # Rank cosine distance within (parent_id, ref_state) — rank 1 = closest
    df["cosine_rank"] = df.groupby(["parent_id", "ref_state"])["cosine_dist"].rank(
        method="first", ascending=True
    )
    # Flag whether this ref is among the k nearest in embedding space
    df["is_knn5"] = df["cosine_rank"] <= K

    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plots(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Scatter: cosine_dist vs spatial_km, coloured by is_knn5 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(
        "Spatial distance vs cosine distance for v3 knn5 reference selection\n"
        "(strategy = random_100, k = 5)",
        fontsize=11,
    )

    for ax, state, colour_sel, colour_rest in [
        (axes[0], "good", "#2ca02c", "#aec7a0"),
        (axes[1], "bad",  "#d62728", "#f5a8a8"),
    ]:
        sub = df[df["ref_state"] == state]
        rest = sub[~sub["is_knn5"]]
        sel  = sub[sub["is_knn5"]]

        ax.scatter(rest["cosine_dist"], rest["spatial_km"],
                   s=6, alpha=0.3, color=colour_rest, label="not selected", rasterized=True)
        ax.scatter(sel["cosine_dist"], sel["spatial_km"],
                   s=25, alpha=0.85, color=colour_sel,
                   edgecolors="black", linewidths=0.4,
                   label=f"knn5 selected (n={len(sel)})", zorder=3)

        # Pearson r
        valid = sub.dropna(subset=["cosine_dist", "spatial_km"])
        r = np.corrcoef(valid["cosine_dist"], valid["spatial_km"])[0, 1]
        ax.set_title(f"{state} refs  (r = {r:.3f})", fontsize=10)
        ax.set_xlabel("Cosine distance to test site")
        ax.set_ylabel("Spatial distance to test site (km)")
        ax.legend(fontsize=8, markerscale=1.4)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "spatial_autocorr_knn5_scatter.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ---- 2. Box: spatial_km for knn5-selected vs all other refs ---------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    fig.suptitle(
        "Spatial distance distribution: knn5-selected vs remaining refs\n"
        "(strategy = random_100, k = 5)",
        fontsize=11,
    )

    for ax, state in [(axes[0], "good"), (axes[1], "bad")]:
        sub = df[df["ref_state"] == state]
        groups = [
            sub[sub["is_knn5"]]["spatial_km"].dropna().values,
            sub[~sub["is_knn5"]]["spatial_km"].dropna().values,
        ]
        bp = ax.boxplot(groups, patch_artist=True, notch=False,
                        labels=["knn5\nselected", "remaining\nrefs"],
                        widths=0.5)
        colors = ["#2ca02c", "#aec7a0"] if state == "good" else ["#d62728", "#f5a8a8"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        med_sel  = np.median(groups[0]) if len(groups[0]) else np.nan
        med_rest = np.median(groups[1]) if len(groups[1]) else np.nan
        ax.set_title(
            f"{state} refs\nmedian: selected={med_sel:.0f} km, rest={med_rest:.0f} km",
            fontsize=10,
        )
        ax.set_ylabel("Spatial distance to test site (km)")

    plt.tight_layout()
    p2 = os.path.join(out_dir, "spatial_autocorr_knn5_boxplot.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # ---- 3. Cosine rank vs spatial distance (rank 1..N on x-axis) -------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Cosine-distance rank vs spatial distance\n"
        "(rank 1 = embedding-nearest; knn5 uses ranks 1–5)",
        fontsize=11,
    )

    for ax, state, color in [(axes[0], "good", "#2ca02c"), (axes[1], "bad", "#d62728")]:
        sub = df[df["ref_state"] == state].dropna(subset=["cosine_rank", "spatial_km"])
        # Aggregate: median spatial_km per rank (many parent_ids)
        agg = (
            sub.groupby("cosine_rank")["spatial_km"]
            .agg(median="median", q25=lambda x: np.percentile(x, 25),
                 q75=lambda x: np.percentile(x, 75))
            .reset_index()
        )
        agg = agg[agg["cosine_rank"] <= 50]  # show first 50 ranks
        ax.fill_between(agg["cosine_rank"], agg["q25"], agg["q75"],
                        alpha=0.25, color=color)
        ax.plot(agg["cosine_rank"], agg["median"], color=color, lw=2, label="median")
        ax.axvspan(0.5, K + 0.5, alpha=0.12, color="gold", label=f"knn5 selection zone (1–{K})")
        ax.set_title(f"{state} refs", fontsize=10)
        ax.set_xlabel("Cosine-distance rank (within parent_id)")
        ax.set_ylabel("Spatial distance to test site (km)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    p3 = os.path.join(out_dir, "spatial_autocorr_knn5_rank_vs_spatial.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("SPATIAL AUTOCORRELATION SUMMARY  (v3 knn5, k=5)")
    print(f"{'='*60}")
    print(f"Total ref observations: {len(df):,}")
    for state in ["good", "bad"]:
        sub = df[df["ref_state"] == state]
        sel = sub[sub["is_knn5"]]
        rest = sub[~sub["is_knn5"]]
        r = np.corrcoef(
            sub["cosine_dist"].values, sub["spatial_km"].values
        )[0, 1]
        print(f"\n  {state.upper()} refs ({len(sub):,} total, {len(sel):,} knn5-selected):")
        print(f"    Pearson r(cosine_dist, spatial_km) = {r:.4f}")
        print(f"    Median spatial km — selected: {np.median(sel['spatial_km']):.1f}  "
              f"remaining: {np.median(rest['spatial_km']):.1f}")
        print(f"    Mean   spatial km — selected: {np.mean(sel['spatial_km']):.1f}  "
              f"remaining: {np.mean(rest['spatial_km']):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading refs …")
    refs = load_refs(REFS_PATH, STRATEGY)
    print(f"  {len(refs):,} refs, {refs['parent_id'].nunique()} parents")

    print("Loading test sites …")
    ts = load_test_sites(TEST_SITES_PATH)
    print(f"  {ts['parent_id'].nunique()} unique test-site parent_ids")

    print("Computing distances …")
    df = build_distance_table(refs, ts)
    print(f"  {len(df):,} (test_site × ref) pairs")

    print_summary(df)
    make_plots(df, OUT_DIR)


if __name__ == "__main__":
    main()
