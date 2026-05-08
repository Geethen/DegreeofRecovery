"""Export v4 stable-site DoR scores as a point shapefile.

Joins:
  - v4/data/test_site_dor_v4.csv         (per-site DoR scores)
  - v4/data/stable_state_classification.csv (parent_id → lon/lat/country/stable_class/votes)

Outputs v4/data/test_site_dor_v4.shp (Point, EPSG:4326) with DBF-safe field
names (≤10 chars). Rows sorted highest dor_knn first; rank column added.

Usage:
  python v4/scripts/reporting/export_v4_shp.py
"""
from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
V4_DATA = os.path.join(BASE_DIR, "v4", "data")

SCORES_CSV = os.path.join(V4_DATA, "test_site_dor_v4.csv")
CLASS_CSV = os.path.join(V4_DATA, "stable_state_classification.csv")
OUT_SHP = os.path.join(V4_DATA, "test_site_dor_v4.shp")

# CSV → SHP DBF-safe field renames (DBF limit: 10 chars).
RENAME = {
    "parent_id":          "parent_id",
    "parent_label":       "lbl",
    "stable_class":       "stab_class",
    "rank":               "rank",
    "n_good":             "n_good",
    "n_bad":              "n_bad",
    "dor_knn":            "dor_knn",
    "dor_knn_ci_low":     "knn_ci_lo",
    "dor_knn_ci_high":    "knn_ci_hi",
    "t_knn_used":         "t_knn",
    "category_knn":       "cat_knn",
    "category_knn_v3t":   "cat_knn_v3",
    "dor_median":         "dor_med",
    "dor_median_ci_low":  "med_ci_lo",
    "dor_median_ci_high": "med_ci_hi",
    "category_median":    "cat_med",
    "country_iso3":       "country",
}


def main() -> None:
    print(f"Loading scores: {SCORES_CSV}")
    scores = pd.read_csv(SCORES_CSV)
    scores["parent_id"] = scores["parent_id"].astype(str)
    print(f"  {len(scores)} scored sites")

    print(f"Loading classification: {CLASS_CSV}")
    cls = pd.read_csv(CLASS_CSV)
    cls["parent_id"] = cls["parent_id"].astype(str)
    keep = ["parent_id", "lon", "lat", "country_iso3"]
    cls = cls[keep]
    print(f"  {len(cls)} classified parents")

    merged = scores.merge(cls, on="parent_id", how="inner")
    print(f"  {len(merged)} after join")

    merged = merged.sort_values("dor_knn", ascending=False, na_position="last").reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)

    # Drop the duplicate stable_class column (v4 score CSV already has one;
    # keep it and drop classification's not — but classification didn't bring it).
    merged = merged.rename(columns=RENAME)

    geometry = [Point(r.lon, r.lat) for r in merged.itertuples()]
    gdf = gpd.GeoDataFrame(
        merged.drop(columns=["lon", "lat"]),
        geometry=geometry,
        crs="EPSG:4326",
    )

    os.makedirs(V4_DATA, exist_ok=True)
    gdf.to_file(OUT_SHP)
    print(f"\nWrote {len(gdf)} features -> {OUT_SHP}")

    print("\nCategory breakdown by stab_class:")
    print(gdf.groupby("stab_class")["cat_knn"].value_counts().to_string())


if __name__ == "__main__":
    main()
