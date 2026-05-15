"""Combine v3 loss-site scores and v4 stable-site scores into one shapefile.

Sources:
  - v3/data/test_site_dor_v3.csv          (built_loss, crop_loss; 158 sites)
  - v3/data/test_site_knn5_scores.shp     (v3 centroid geometry)
  - v4/data/test_site_dor_v4.csv          (stable_nature/crop/built; 1471 sites)
  - v4/data/stable_state_classification.csv (v4 centroids + country)

Scoring consistency
-------------------
Both v3 and v4 use the same scorer (mean-cosine kNN, k=5; bootstrap CI with
n_boot=2000, seed=42; identical `_knn_score` / `_median_score` /
`cosine_dists_to_set`). Only the bad-reference pool differs (per disturbance
label in v3 vs per stable class in v4). The unified `cat_knn` column applies
v3's pooled threshold `t_knn = 0.4859` to BOTH halves:
  - v3 rows: native `category_knn` (already calibrated against 0.4859).
  - v4 rows: `category_knn_v3t` (v3-threshold transfer category).
The v4 per-class category is preserved as `cat_knn_pc` for reference.

Output: combined/data/test_site_dor_combined.{shp,shx,dbf,prj,cpg}
"""
from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
V3_DATA = os.path.join(BASE_DIR, "v3", "data")
V4_DATA = os.path.join(BASE_DIR, "v4", "data")
OUT_DIR = os.path.join(BASE_DIR, "combined", "data")
OUT_SHP = os.path.join(OUT_DIR, "test_site_dor_combined.shp")

V3_SCORES = os.path.join(V3_DATA, "test_site_dor_v3.csv")
V3_SHP = os.path.join(V3_DATA, "test_site_knn5_scores.shp")
V4_SCORES = os.path.join(V4_DATA, "test_site_dor_v4.csv")
V4_CLASS = os.path.join(V4_DATA, "stable_state_classification.csv")

V3_KNN_THRESHOLD = 0.4859


def load_v3_centroids(path: str) -> pd.DataFrame:
    g = gpd.read_file(path)[["parent_id", "geometry"]]
    g["parent_id"] = g["parent_id"].astype(str)
    g["lon"] = g.geometry.x
    g["lat"] = g.geometry.y
    return g[["parent_id", "lon", "lat"]]


def load_v3() -> pd.DataFrame:
    scores = pd.read_csv(V3_SCORES)
    scores["parent_id"] = scores["parent_id"].astype(str)
    cents = load_v3_centroids(V3_SHP)
    df = scores.merge(cents, on="parent_id", how="inner")
    df["source"] = "v3_loss"
    df["stable_class"] = ""
    df["country_iso3"] = ""
    df["t_knn_used"] = V3_KNN_THRESHOLD
    df["category_knn_pc"] = df["category_knn"]  # same threshold as unified
    return df


def load_v4() -> pd.DataFrame:
    scores = pd.read_csv(V4_SCORES)
    scores["parent_id"] = scores["parent_id"].astype(str)
    cls = pd.read_csv(V4_CLASS)[["parent_id", "lon", "lat", "country_iso3"]]
    cls["parent_id"] = cls["parent_id"].astype(str)
    df = scores.merge(cls, on="parent_id", how="inner")
    df["source"] = "v4_stable"
    # Unified category: v3-threshold version (transfer column).
    df["category_knn_pc"] = df["category_knn"]            # per-class
    df["category_knn"] = df["category_knn_v3t"]           # unified t=0.4859
    df = df.drop(columns=["category_knn_v3t"])
    return df


def main() -> None:
    print("Loading v3 loss sites …")
    v3 = load_v3()
    print(f"  {len(v3)} v3 rows (labels: {v3['parent_label'].value_counts().to_dict()})")

    print("Loading v4 stable sites …")
    v4 = load_v4()
    print(f"  {len(v4)} v4 rows (labels: {v4['parent_label'].value_counts().to_dict()})")

    cols = [
        "parent_id", "source", "parent_label", "stable_class", "country_iso3",
        "n_good", "n_bad",
        "dor_knn", "dor_knn_ci_low", "dor_knn_ci_high",
        "t_knn_used", "category_knn", "category_knn_pc",
        "dor_median", "dor_median_ci_low", "dor_median_ci_high",
        "category_median",
        "lon", "lat",
    ]
    merged = pd.concat([v3[cols], v4[cols]], ignore_index=True)
    print(f"  {len(merged)} combined rows")

    # Sort highest dor_knn first; rank = exploration order.
    merged = (
        merged.sort_values("dor_knn", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    merged.insert(0, "rank", merged.index + 1)

    rename = {
        "parent_id":          "parent_id",
        "source":             "source",
        "parent_label":       "lbl",
        "stable_class":       "stab_class",
        "country_iso3":       "country",
        "rank":               "rank",
        "n_good":             "n_good",
        "n_bad":              "n_bad",
        "dor_knn":            "dor_knn",
        "dor_knn_ci_low":     "knn_ci_lo",
        "dor_knn_ci_high":    "knn_ci_hi",
        "t_knn_used":         "t_knn",
        "category_knn":       "cat_knn",
        "category_knn_pc":    "cat_knn_pc",
        "dor_median":         "dor_med",
        "dor_median_ci_low":  "med_ci_lo",
        "dor_median_ci_high": "med_ci_hi",
        "category_median":    "cat_med",
    }
    merged = merged.rename(columns=rename)

    geometry = [Point(r.lon, r.lat) for r in merged.itertuples()]
    gdf = gpd.GeoDataFrame(
        merged.drop(columns=["lon", "lat"]),
        geometry=geometry,
        crs="EPSG:4326",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    gdf.to_file(OUT_SHP)
    print(f"\nWrote {len(gdf)} features -> {OUT_SHP}")

    print("\nUnified cat_knn breakdown (t = 0.4859) by source / label:")
    print(
        gdf.groupby(["source", "lbl"])["cat_knn"]
        .value_counts()
        .unstack(fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    main()
