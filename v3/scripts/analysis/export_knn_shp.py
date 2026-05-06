"""Export v3 knn5 test-site scores as a point shapefile.

One point per parent_id (centroid of its test observations).
Attributes include dor_knn score, CI, category, dor_median, and counts.
Rows are sorted highest dor_knn first so field-exploration rank == FID order.

Output: v3/data/test_site_knn5_scores.shp  (+ .dbf/.prj/.shx)
"""
from __future__ import annotations

import os
import duckdb
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
TEST_SITES_PATH = os.path.join(BASE_DIR, "degreeRecover", "data",
                               "test_site_alphaearth_2024.parquet")
SCORES_CSV = os.path.join(BASE_DIR, "v3", "data", "test_site_dor_v3.csv")
OUT_DIR = os.path.join(BASE_DIR, "v3", "data")
OUT_SHP = os.path.join(OUT_DIR, "test_site_knn5_scores.shp")


def load_centroids(path: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute("""
        SELECT
            parent_id,
            AVG(geo.coordinates[1]) AS lon,
            AVG(geo.coordinates[2]) AS lat
        FROM read_parquet(?)
        GROUP BY parent_id
    """, [path]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return df


def main() -> None:
    print("Loading test-site centroids …")
    centroids = load_centroids(TEST_SITES_PATH)
    print(f"  {len(centroids)} parent_ids")

    print("Loading knn5 scores …")
    scores = pd.read_csv(SCORES_CSV)
    scores["parent_id"] = scores["parent_id"].astype(str)
    print(f"  {len(scores)} scored sites")

    merged = centroids.merge(scores, on="parent_id", how="inner")
    print(f"  {len(merged)} after join")

    # Sort highest dor_knn first — FID order = exploration rank
    merged = merged.sort_values("dor_knn", ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)  # 1-based rank

    # Truncate column names to ≤10 chars for shapefile DBF compatibility
    rename = {
        "parent_id":           "parent_id",
        "parent_label":        "lbl",
        "rank":                "rank",
        "n_good":              "n_good",
        "n_bad":               "n_bad",
        "dor_knn":             "dor_knn",
        "dor_knn_ci_low":      "knn_ci_lo",
        "dor_knn_ci_high":     "knn_ci_hi",
        "category_knn":        "cat_knn",
        "dor_median":          "dor_med",
        "dor_median_ci_low":   "med_ci_lo",
        "dor_median_ci_high":  "med_ci_hi",
        "category_median":     "cat_med",
    }
    merged = merged.rename(columns=rename)

    geometry = [Point(row.lon, row.lat) for row in merged.itertuples()]
    gdf = gpd.GeoDataFrame(
        merged.drop(columns=["lon", "lat"]),
        geometry=geometry,
        crs="EPSG:4326",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    gdf.to_file(OUT_SHP)
    print(f"\nWrote {len(gdf)} features -> {OUT_SHP}")
    print("\nTop 10 by dor_knn:")
    print(gdf[["rank", "parent_id", "lbl", "dor_knn", "cat_knn"]].head(10).to_string(index=False))
    print("\nBottom 10 by dor_knn:")
    print(gdf[["rank", "parent_id", "lbl", "dor_knn", "cat_knn"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
