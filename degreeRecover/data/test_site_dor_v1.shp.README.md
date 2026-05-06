# `test_site_dor_v1` dataset metadata

This README documents both:

- `test_site_dor_v1.csv` (full column names), and
- `test_site_dor_v1.shp` (DBF-safe abbreviated field names).

## File set

| File | Purpose |
|---|---|
| `test_site_dor_v1.csv` | Per-site DoR table with full column names |
| `test_site_dor_v1.shp` | Point geometry layer for GIS |
| `test_site_dor_v1.shx` | Shapefile index |
| `test_site_dor_v1.dbf` | Shapefile attributes |
| `test_site_dor_v1.prj` | CRS definition (EPSG:4326) |
| `test_site_dor_v1.cpg` | DBF text encoding |

## Geometry (SHP)

- Type: Point
- CRS: EPSG:4326 (WGS84 lon/lat)
- One feature per parent site.

## Column descriptions (CSV)

| Column | Type | Description |
|---|---|---|
| `parent_id` | string | Parent site id (zero-padded). |
| `parent_label` | string | Disturbance label (`built_loss` or `crop_loss`). |
| `n_good` | int | Number of good (natural-state) references used. |
| `n_bad` | int | Number of bad (degraded-state) references used. |
| `corr_range_m` | float | Spatial correlation range used for effective sample size calculation (meters). |
| `n_eff_good` | float | Effective sample size estimate for good references. |
| `n_eff_bad` | float | Effective sample size estimate for bad references. |
| `n_eff_min` | float | Minimum of `n_eff_good` and `n_eff_bad`. |
| `cos_dist_good` | float | Cosine distance from site embedding to good-reference centroid. |
| `cos_dist_bad` | float | Cosine distance from site embedding to bad-reference centroid. |
| `dor_median` | float | Primary DoR score using median pairwise-distance ratio. |
| `dor_median_ci_low` | float | Lower 95% bootstrap bound for `dor_median`. |
| `dor_median_ci_high` | float | Upper 95% bootstrap bound for `dor_median`. |
| `dor_normalised` | float | Normalized distance-based diagnostic score. |
| `dor_percentile` | float | Percentile-based diagnostic score. |
| `dor_cosine` | float | Centroid-based cosine DoR diagnostic. |
| `applicable` | bool | True when the site can be scored from available refs. |

## Category field (SHP only)

Shapefiles exported by `build_summary.py` include a `category` field derived from CI overlap with 0.5:

- `recovering`: `dor_ci_lo > 0.5`
- `degraded`: `dor_ci_hi < 0.5`
- `indistinguishable`: CI overlaps 0.5
- `no_data`: score not available

## CSV -> SHP field mapping

Shapefile DBF fields are limited to short names, so several columns are renamed:

| CSV column | SHP field |
|---|---|
| `parent_label` | `par_label` |
| `corr_range_m` | `corr_r_m` |
| `n_eff_good` | `n_eff_g` |
| `n_eff_bad` | `n_eff_b` |
| `cos_dist_good` | `cos_d_good` |
| `cos_dist_bad` | `cos_d_bad` |
| `dor_median` | `dor_med` |
| `dor_median_ci_low` | `dor_ci_lo` |
| `dor_median_ci_high` | `dor_ci_hi` |
| `dor_normalised` | `dor_norm` |
| `dor_percentile` | `dor_pct` |
| `dor_cosine` | `dor_cos` |

## Provenance

- Scoring: `degreeRecover/scripts/analysis/score_test_sites.py`
- Export: `degreeRecover/scripts/reporting/build_summary.py`
- Product: v1 DoR outputs.
