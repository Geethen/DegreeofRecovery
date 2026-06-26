"""Export the 90 v5 built-loss test sites as a point shapefile.

Mirrors the schema of v4/scripts/reporting/export_v4_shp.py (Point, EPSG:4326,
DBF-safe <=10-char field names, rank column) but for the v5 built-loss scores.

Differences from v4 export:
  * Source = v5/data/test_site_dor_v5_built_loss.csv (k=5, 4 km/5 km annulus).
  * Coordinates from v5/data/built_loss_coords.csv (local geo column).
  * Ecoregion-percentile DoR (pct_dor) joined in from
    v5/data/test_site_ecoregion_percentile.csv.
    * Sorted DESCENDING by dor_knn (rank 1 = most regenerated), per request.
  * Writes a companion metadata text file describing every field.

Outputs:
  v5/data/test_site_dor_v5_built_loss.shp  (+ .dbf/.shx/.prj/.cpg)
  v5/data/test_site_dor_v5_built_loss.metadata.txt
"""
from __future__ import annotations

import datetime as dt
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE = "/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery"
V5_DATA = os.path.join(BASE, "v5", "data")

SCORES_CSV = os.path.join(V5_DATA, "test_site_dor_v5_built_loss.csv")
COORDS_CSV = os.path.join(V5_DATA, "built_loss_coords.csv")
ECO_CSV = os.path.join(V5_DATA, "test_site_ecoregion_percentile.csv")
OUT_SHP = os.path.join(V5_DATA, "test_site_dor_v5_built_loss.shp")
OUT_META = os.path.join(V5_DATA, "test_site_dor_v5_built_loss.metadata.txt")

# CSV -> SHP DBF-safe field renames (DBF limit: 10 chars).
RENAME = {
    "parent_id":          "parent_id",
    "parent_label":       "lbl",
    "stable_class":       "class",
    "rank":               "rank",
    "n_good":             "n_good",
    "n_bad":              "n_bad",
    "dor_knn":            "dor_knn",
    "dor_knn_ci_low":     "knn_ci_lo",
    "dor_knn_ci_high":    "knn_ci_hi",
    "t_knn_used":         "t_knn",
    "category_knn":       "cat_knn",
    "dor_median":         "dor_med",
    "dor_median_ci_low":  "med_ci_lo",
    "dor_median_ci_high": "med_ci_hi",
    "category_median":    "cat_med",
    "eco_id":             "eco_id",
    "pct_vs_good":        "pct_good",
    "pct_vs_bad":         "pct_bad",
    "pct_dor":            "pct_dor",
}

META_FIELDS = {
    "rank":      "Descending rank by dor_knn (1 = most regenerated / highest DoR).",
    "parent_id": "GEE system:index of the built-loss parent site.",
    "lbl":       "Upstream abandonment label (always 'built_loss' here).",
    "class":     "DoR scoring class (built_loss).",
    "n_good":    "Number of near-natural (good) reference pixels used.",
    "n_bad":     "Number of built-up (bad) reference pixels used.",
    "dor_knn":   "Primary Degree of Regeneration: mean cosine distance to 5 nearest "
                 "bad refs / (5-NN nat dist + 5-NN bad dist). 1=natural, 0=built.",
    "knn_ci_lo": "Lower bound, 95% bootstrap CI of dor_knn (2,000 resamples).",
    "knn_ci_hi": "Upper bound, 95% bootstrap CI of dor_knn.",
    "t_knn":     "Operating threshold applied (stable_built = 0.4948).",
    "cat_knn":   "Category from dor_knn CI vs t_knn: regenerating / degraded / "
                 "indistinguishable.",
    "dor_med":   "Secondary DoR using median (not 5-NN) reference distances.",
    "med_ci_lo": "Lower bound, 95% bootstrap CI of dor_median.",
    "med_ci_hi": "Upper bound, 95% bootstrap CI of dor_median.",
    "cat_med":   "Category from dor_median vs 0.5.",
    "eco_id":    "RESOLVE 2017 ecoregion ECO_ID the site falls in.",
    "pct_good":  "Ecoregion percentile rank of similarity to the ecoregion's "
                 "natural-reference cloud (0-100; high = natural-like).",
    "pct_bad":   "Ecoregion percentile rank of similarity to the ecoregion's "
                 "built-reference cloud (0-100; high = built-like).",
    "pct_dor":   "Directional ecoregion DoR percentile = (pct_good + (100-pct_bad))/2. "
                 "NaN for 2 sites whose ecoregion lacks an FSCS reference file.",
    "geometry":  "Point, EPSG:4326 (WGS84), parent-site centroid (lon, lat).",
}


def main() -> None:
    scores = pd.read_csv(SCORES_CSV)
    scores["parent_id"] = scores["parent_id"].astype(str)

    coords = pd.read_csv(COORDS_CSV)
    coords["parent_id"] = coords["parent_id"].astype(str)
    coords = coords[["parent_id", "lon", "lat"]]

    eco = pd.read_csv(ECO_CSV)
    eco["parent_id"] = eco["parent_id"].astype(str)
    eco = eco[["parent_id", "eco_id", "pct_vs_good", "pct_vs_bad", "pct_dor"]]

    merged = scores.merge(coords, on="parent_id", how="inner")
    merged = merged.merge(eco, on="parent_id", how="left")
    print(f"{len(merged)} built-loss sites after join "
          f"(coords {merged['lon'].notna().sum()}, eco pct {merged['pct_dor'].notna().sum()})")

    merged = merged.sort_values("dor_knn", ascending=False, na_position="last").reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)

    merged = merged.rename(columns=RENAME)

    geometry = [Point(r.lon, r.lat) for r in merged.itertuples()]
    gdf = gpd.GeoDataFrame(
        merged.drop(columns=["lon", "lat"]),
        geometry=geometry,
        crs="EPSG:4326",
    )
    gdf.to_file(OUT_SHP)
    print(f"Wrote {len(gdf)} features -> {OUT_SHP}")

    # Metadata sidecar
    n_reg = int((gdf["cat_knn"] == "regenerating").sum())
    n_deg = int((gdf["cat_knn"] == "degraded").sum())
    n_ind = int((gdf["cat_knn"] == "indistinguishable").sum())
    lines = [
        "Metadata - v5 built-loss Degree of Regeneration point shapefile",
        "===============================================================",
        f"Generated:    {dt.date.today().isoformat()}",
        f"File:         {os.path.basename(OUT_SHP)}",
        f"Features:     {len(gdf)} built-loss abandonment test sites",
        "Geometry:     Point",
        "CRS:          EPSG:4326 (WGS84)",
        "Sort order:   DESCENDING by dor_knn (rank 1 = most regenerated)",
        "",
        "Scoring (v5):",
        "  - 2024 AlphaEarth 64-band embedding, mean per site.",
        "  - Reference annulus: inner exclusion 4 km, outer ceiling 5 km,",
        "    expanding to 8 km when the per-pool count target is unmet.",
        "  - Near-natural (good) pool = non-crop/non-built/non-water WorldCover,",
        "    minus HABLOSS-loss pixels; built-up (bad) pool = built-up pixels.",
        "  - DoR = d_bad5 / (d_nat5 + d_bad5), 5-nearest-neighbour cosine distance.",
        "  - 2,000-bootstrap 95% CI; threshold 0.4948 (stable_built calibration).",
        "",
        f"Category breakdown (cat_knn): regenerating {n_reg}, "
        f"indistinguishable {n_ind}, degraded {n_deg}.",
        "",
        "Fields:",
    ]
    for col in list(gdf.columns):
        desc = META_FIELDS.get(col, "")
        lines.append(f"  {col:<10} {desc}")
    with open(OUT_META, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote metadata -> {OUT_META}")


if __name__ == "__main__":
    main()
