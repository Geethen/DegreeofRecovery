"""Build summary deliverables for the RECOVER DoR pipeline.

Produces:
  - data/test_site_dor.shp                  (point shapefile for QGIS)
  - data/dor_summary_by_label.csv           (counts by label x category)
  - plots/dor_world_map.png                 (global scatter coloured by category)
  - plots/dor_distribution.png              (DoR histograms by label)
  - plots/dor_category_breakdown.png        (stacked bar by label)
  - plots/method_benchmark_summary.png      (AUC bar chart)

Reads:
  - data/test_site_dor.csv                  (per-site DoR)
  - data/test_site_alphaearth_<year>.parquet (parent embeddings — not used,
                                              just for parent_id parity)
  Parent point geometries are fetched once from the source EE asset.

Usage:
  python scripts/reporting/build_summary.py
  python scripts/reporting/build_summary.py --year 2024
"""

import argparse
import os

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

# ── Paths / constants ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

YEAR = 2024
DOR_CSV = os.path.join(DATA_DIR, "test_site_dor.csv")


def default_test_site_parquet(year):
    return os.path.join(DATA_DIR, f"test_site_alphaearth_{year}.parquet")


def category(row):
    """Classify each site by where its 95 % CI sits relative to 0.5."""
    if pd.isna(row["dor_median"]):
        return "no_data"
    if row["dor_median_ci_low"] > 0.5:
        return "recovering"
    if row["dor_median_ci_high"] < 0.5:
        return "degraded"
    return "indistinguishable"


# ── Geometry from cache ─────────────────────────────────────────────
def load_parent_coords(parquet_path):
    """Pull (parent_id, lon, lat) from the test-site parquet's `geo` col.

    The cache stores parent points as GeoJSON-like dicts in `geo`; we just
    project out the coordinate pair to avoid a redundant EE round-trip.
    """
    df = duckdb.connect().execute(f"SELECT parent_id, geo FROM '{parquet_path}'").df()
    df["parent_id"] = df["parent_id"].astype(str).str.zfill(20)
    coords = df["geo"].apply(
        lambda g: g["coordinates"] if isinstance(g, dict) else (None, None)
    )
    df["lon"] = coords.apply(lambda c: c[0] if c else None)
    df["lat"] = coords.apply(lambda c: c[1] if c else None)
    return df[["parent_id", "lon", "lat"]]


# ── Plots ───────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "recovering": "#2ca02c",
    "indistinguishable": "#bdbd22",
    "degraded": "#d62728",
    "no_data": "#999999",
}
CATEGORY_ORDER = ["recovering", "indistinguishable", "degraded", "no_data"]


def plot_world_map(gdf, out_path):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": proj})
        ax.set_global()
        ax.set_extent([-180, 180, -60, 80], crs=proj)
        ax.add_feature(cfeature.LAND, facecolor="#f4f1ec", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#dfe9f2", zorder=0)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, edgecolor="#888888", linewidth=0.4, zorder=1)
        gl = ax.gridlines(
            crs=proj, draw_labels=True, alpha=0.25, linewidth=0.4, color="#888888"
        )
        gl.top_labels = False
        gl.right_labels = False
        scatter_kw = {"transform": proj, "zorder": 3}
    except ModuleNotFoundError:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
        ax.grid(True, alpha=0.3)
        scatter_kw = {"zorder": 3}

    for cat in CATEGORY_ORDER:
        sub = gdf[gdf["category"] == cat]
        if not len(sub):
            continue
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=42,
            c=CATEGORY_COLORS[cat],
            edgecolors="black",
            linewidth=0.4,
            alpha=0.9,
            label=f"{cat} (n={len(sub)})",
            **scatter_kw,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("RECOVER test sites — DoR category (95 % CI vs 0.5)")
    ax.legend(
        loc="lower left", fontsize=9, frameon=True, facecolor="white", framealpha=0.9
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_distribution(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    labels = sorted(df["parent_label"].dropna().unique())
    for ax, lbl in zip(axes, labels):
        sub = df[(df["parent_label"] == lbl) & df["dor_median"].notna()]
        ax.hist(
            sub["dor_median"], bins=20, color="#1f77b4", edgecolor="black", alpha=0.85
        )
        ax.axvline(0.5, color="grey", ls="--", lw=1)
        ax.set_title(f"{lbl}  (n={len(sub)}, mean={sub['dor_median'].mean():.2f})")
        ax.set_xlabel("dor_median  (1 = natural, 0 = degraded)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Count")
    fig.suptitle("DoR distribution by parent label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_category_breakdown(df, out_path):
    counts = (
        df.groupby(["parent_label", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(counts))
    for cat in CATEGORY_ORDER:
        vals = counts[cat].to_numpy()
        ax.bar(
            counts.index,
            vals,
            bottom=bottom,
            color=CATEGORY_COLORS[cat],
            label=cat,
            edgecolor="black",
            linewidth=0.4,
        )
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(
                    i,
                    bottom[i] + v / 2,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        bottom += vals
    ax.set_ylabel("Test sites")
    ax.set_title("Sites by parent label and DoR category")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_method_benchmark(out_path):
    """Best-effort method-AUC bar chart from the recovery_score.py run.

    recovery_score.py prints per-method AUC but doesn't save them; we
    re-derive them by running the same scoring on the same input. Cheap
    enough at 158 parents and 1 in/out per method.
    """
    from scripts.analysis.recovery_score import (
        METHODS,
        load,
        overall_auc,
        score_all,
    )

    refs_path = os.path.join(DATA_DIR, "recover_reference_samples_alphaearth.parquet")
    if not os.path.exists(refs_path):
        return None
    df = load(refs_path)
    scored = score_all(df)
    aucs = {m: overall_auc(scored, m) for m in METHODS}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = list(aucs)
    vals = [aucs[m] for m in methods]
    bars = ax.bar(methods, vals, color="#4c72b0", edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Pooled ROC AUC (good vs bad refs)")
    ax.set_title("Scoring-method benchmark on labelled references")
    ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return aucs


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--dor_csv", default=DOR_CSV)
    parser.add_argument(
        "--out_shp", default=os.path.join(DATA_DIR, "test_site_dor.shp")
    )
    parser.add_argument(
        "--summary_csv",
        default=os.path.join(DATA_DIR, "dor_summary_by_label.csv"),
    )
    parser.add_argument("--plot_prefix", default="dor")
    parser.add_argument(
        "--test_sites",
        default=None,
        help="Test-site parquet (for parent coords). "
        "Default: data/test_site_alphaearth_<year>.parquet",
    )
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Loading {args.dor_csv}")
    dor = pd.read_csv(args.dor_csv)
    dor["parent_id"] = dor["parent_id"].astype(str).str.zfill(20)
    dor["category"] = dor.apply(category, axis=1)
    print(f"  {len(dor)} sites")

    test_sites_path = args.test_sites or default_test_site_parquet(args.year)
    print(f"Loading parent coords from {test_sites_path}")
    coords = load_parent_coords(test_sites_path)
    merged = dor.merge(coords, on="parent_id", how="left")
    n_missing = merged[["lon", "lat"]].isna().any(axis=1).sum()
    if n_missing:
        print(f"  [warn] {n_missing} sites missing coords (will be dropped)")
    merged = merged.dropna(subset=["lon", "lat"]).reset_index(drop=True)

    # ── Shapefile ──
    geom = [Point(xy) for xy in zip(merged["lon"], merged["lat"])]
    shp_cols = [
        "parent_id",
        "parent_label",
        "category",
        "n_good",
        "n_bad",
        "applicable",
        "corr_range_m",
        "n_eff_good",
        "n_eff_bad",
        "n_eff_min",
        "cos_dist_good",
        "cos_dist_bad",
        "dor_median",
        "dor_median_ci_low",
        "dor_median_ci_high",
        "dor_normalised",
        "dor_percentile",
        "dor_cosine",
        "lon",
        "lat",
    ]
    shp_cols = [c for c in shp_cols if c in merged.columns]
    gdf = gpd.GeoDataFrame(merged[shp_cols], geometry=geom, crs="EPSG:4326")

    # Shapefile field names are limited to 10 chars — rename.
    rename = {
        "parent_label": "par_label",
        "corr_range_m": "corr_r_m",
        "n_eff_good": "n_eff_g",
        "n_eff_bad": "n_eff_b",
        "n_eff_min": "n_eff_min",
        "cos_dist_good": "cos_d_good",
        "cos_dist_bad": "cos_d_bad",
        "dor_median": "dor_med",
        "dor_median_ci_low": "dor_ci_lo",
        "dor_median_ci_high": "dor_ci_hi",
        "dor_normalised": "dor_norm",
        "dor_percentile": "dor_pct",
        "dor_cosine": "dor_cos",
    }
    gdf = gdf.rename(columns=rename)
    shp_path = args.out_shp
    gdf.to_file(shp_path)
    print(f"  wrote {shp_path}  ({len(gdf)} features)")

    # ── Summary CSV ──
    pivot = (
        gdf.groupby(["par_label", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot.loc["ALL"] = pivot.sum(axis=0)
    summary_csv = args.summary_csv
    pivot.to_csv(summary_csv)
    print(f"  wrote {summary_csv}")
    print(pivot.to_string())

    # ── Plots ──
    out_world = os.path.join(PLOTS_DIR, f"{args.plot_prefix}_world_map.png")
    plot_world_map(gdf, out_world)
    print(f"  wrote {out_world}")

    out_dist = os.path.join(PLOTS_DIR, f"{args.plot_prefix}_distribution.png")
    plot_distribution(
        gdf.rename(columns={"par_label": "parent_label", "dor_med": "dor_median"}),
        out_dist,
    )
    print(f"  wrote {out_dist}")

    out_cat = os.path.join(PLOTS_DIR, f"{args.plot_prefix}_category_breakdown.png")
    plot_category_breakdown(gdf.rename(columns={"par_label": "parent_label"}), out_cat)
    print(f"  wrote {out_cat}")

    out_bench = os.path.join(
        PLOTS_DIR, f"{args.plot_prefix}_method_benchmark_summary.png"
    )
    aucs = plot_method_benchmark(out_bench)
    if aucs is not None:
        print(f"  wrote {out_bench}")
        print(
            "  method AUCs: "
            + ", ".join(
                f"{m}={a:.3f}" for m, a in sorted(aucs.items(), key=lambda x: -x[1])
            )
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, BASE_DIR)
    main()
