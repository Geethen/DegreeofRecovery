# `test_site_knn5_scores` dataset metadata

This README documents both:

- `test_site_knn5_scores.csv` (via `v3/data/test_site_dor_v3.csv` — full column names), and
- `test_site_knn5_scores.shp` (DBF-safe abbreviated field names).

## File set

| File | Purpose |
|---|---|
| `test_site_knn5_scores.shp` | Point geometry layer for GIS |
| `test_site_knn5_scores.shx` | Shapefile index |
| `test_site_knn5_scores.dbf` | Shapefile attributes |
| `test_site_knn5_scores.prj` | CRS definition (EPSG:4326) |
| `test_site_knn5_scores.cpg` | DBF text encoding |

## Geometry (SHP)

- Type: Point
- CRS: EPSG:4326 (WGS84 lon/lat)
- One feature per parent site.
- Sorted by `dor_knn` descending — `rank` 1 is the most-recovering site.

## Column descriptions (CSV: `test_site_dor_v3.csv`)

| Column | Type | Description |
|---|---|---|
| `parent_id` | string | Parent site id (zero-padded). |
| `parent_label` | string | Disturbance label (`built_loss` or `crop_loss`). |
| `n_good` | int | Number of good (natural-state) references used. |
| `n_bad` | int | Number of bad (degraded-state) references used. |
| `dor_knn` | float | **Primary v3 score.** Mean cosine distance ratio using the 5 nearest good and bad references: `mean_cos(x → k nearest bad) / (mean_cos(x → k nearest good) + mean_cos(x → k nearest bad))`. Range [0, 1]; 1 = recovering, 0 = degraded. |
| `dor_knn_ci_low` | float | Lower 95 % bootstrap bound for `dor_knn` (n_boot = 2000). |
| `dor_knn_ci_high` | float | Upper 95 % bootstrap bound for `dor_knn`. |
| `category_knn` | string | Classification from `dor_knn` CI (see below). |
| `dor_median` | float | Secondary score (v1/v2 method). Median pairwise cosine distance ratio using all references. |
| `dor_median_ci_low` | float | Lower 95 % bootstrap bound for `dor_median`. |
| `dor_median_ci_high` | float | Upper 95 % bootstrap bound for `dor_median`. |
| `category_median` | string | Classification from `dor_median` CI (simple 0.5 threshold). |

## Category field

| Value | Rule |
|---|---|
| `recovering` | 95 % CI lies entirely above threshold + deadband |
| `degraded` | 95 % CI lies entirely below threshold − deadband |
| `indistinguishable` | CI straddles threshold or deadband |
| `no_data` | score not available (parent centroid on no-coverage pixel) |

`dor_knn` threshold = 0.4859 (Youden-J calibrated), deadband half-width = 0.05.  
`dor_median` threshold = 0.5 (fixed).

## SHP field mapping

Shapefile DBF fields are limited to short names:

| CSV column | SHP field |
|---|---|
| `parent_label` | `lbl` |
| `dor_knn` | `dor_knn` |
| `dor_knn_ci_low` | `knn_ci_lo` |
| `dor_knn_ci_high` | `knn_ci_hi` |
| `category_knn` | `cat_knn` |
| `dor_median` | `dor_med` |
| `dor_median_ci_low` | `med_ci_lo` |
| `dor_median_ci_high` | `med_ci_hi` |
| `category_median` | `cat_med` |
| *(derived)* | `rank` |

## Provenance

- Scoring: `v3/scripts/analysis/score_test_sites_v3.py` (k=5, n_boot=2000, seed=42)
- Export: `v3/scripts/analysis/export_knn_shp.py`
- Product: v3 DoR outputs, AlphaEarth 2024 embeddings, `random_100` reference strategy.
