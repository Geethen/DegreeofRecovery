# `test_site_dor_v4` dataset metadata

This README documents both:

- `test_site_dor_v4.csv` (full column names), and
- `test_site_dor_v4.shp` (DBF-safe abbreviated field names).

v4 scores `stable_stable` parents that were architecturally excluded from
v1–v3 (no disturbance label → no parent-specific bad pool). Each parent is
classified as `stable_class ∈ {nature, crop, built}` and routed to a
class-specific bad reference pool; per-class kNN thresholds are refit by
within-parent cross-validation.

## File set

| File | Purpose |
|---|---|
| `test_site_dor_v4.csv` | Per-site DoR table with full column names |
| `test_site_dor_v4.shp` | Point geometry layer for GIS |
| `test_site_dor_v4.shx` | Shapefile index |
| `test_site_dor_v4.dbf` | Shapefile attributes |
| `test_site_dor_v4.prj` | CRS definition (EPSG:4326) |
| `test_site_dor_v4.cpg` | DBF text encoding |

## Geometry (SHP)

- Type: Point
- CRS: EPSG:4326 (WGS84 lon/lat)
- One feature per parent site.
- Coordinates come from `stable_state_classification.csv` (parent centroids).

## Stable-class routing

Each parent is classified using majority vote over: WorldCover v100/v200,
Dynamic World annual mode 2018-2024, WorldCereal `temporarycrops`, VIDA
buildings, MS Buildings. Parents with no clear majority (`stable_class =
ambiguous`) are dropped before scoring.

| stable_class | Bad reference pool (WorldCover v200 codes) | Expected outcome |
|---|---|---|
| `nature` | crop ∪ built (40 ∪ 50) | mixed; primary dataset of interest |
| `crop` | crop only (40) — site sits in crop | sanity check: should mostly be `degraded` |
| `built` | built only (50) — site sits in built | sanity check: should mostly be `degraded` |

Good pool is the same for all classes: WorldCover ≠ {40 crop, 50 built, 80
water} and not in the v2 loss-trend mask.

## Column descriptions (CSV)

| Column | Type | Description |
|---|---|---|
| `parent_id` | string | Parent site id (zero-padded). |
| `parent_label` | string | Encoded stable class — `stable_nature`/`stable_crop`/`stable_built`. |
| `stable_class` | string | `nature`/`crop`/`built` (= `parent_label` minus the `stable_` prefix). |
| `n_good` | int | Number of good (natural-state) references used. |
| `n_bad` | int | Number of bad (per-class state) references used. |
| `dor_knn` | float | Primary DoR score: kNN-5 cosine ratio, `mean(d_bad_top5) / (mean(d_good_top5) + mean(d_bad_top5))`. |
| `dor_knn_ci_low` | float | Lower 95% bootstrap bound for `dor_knn`. |
| `dor_knn_ci_high` | float | Upper 95% bootstrap bound for `dor_knn`. |
| `t_knn_used` | float | Per-class kNN threshold applied (refit by within-parent CV; see below). |
| `category_knn` | string | Decision under per-class threshold: `recovering`/`indistinguishable`/`degraded`/`no_data`. |
| `category_knn_v3t` | string | Transfer-check decision under v3's pooled threshold (`t_knn = 0.4859`). |
| `dor_median` | float | Secondary DoR score: median pairwise distance ratio. |
| `dor_median_ci_low` | float | Lower 95% bootstrap bound for `dor_median`. |
| `dor_median_ci_high` | float | Upper 95% bootstrap bound for `dor_median`. |
| `category_median` | string | Decision under fixed `t_med = 0.5`. |

## Calibrated thresholds (v4)

Per-class kNN thresholds (Youden-J, within-parent 5-fold CV) — see
`calibrated_thresholds_v4.json`:

| stable_class | t_knn |
|---|---|
| stable_nature | 0.4861 |
| stable_crop   | 0.4823 |
| stable_built  | 0.4948 |

v3 pooled threshold (transfer check): `t_knn = 0.4859`.

## Decision rule

For each row the category is derived from the score's bootstrap CI vs the
applicable threshold:

- `recovering`: `ci_low > t`
- `degraded`: `ci_high < t`
- `indistinguishable`: CI overlaps `t`
- `no_data`: score not available (insufficient refs)

`category_knn` uses the per-class threshold; `category_knn_v3t` re-applies
the same CI against v3's pooled threshold for comparability.

## CSV → SHP field mapping

Shapefile DBF field names are limited to ≤10 characters; the export renames:

| CSV column | SHP field |
|---|---|
| `parent_label` | `lbl` |
| `stable_class` | `stab_class` |
| `country_iso3` | `country` |
| `n_good` | `n_good` |
| `n_bad` | `n_bad` |
| `dor_knn` | `dor_knn` |
| `dor_knn_ci_low` | `knn_ci_lo` |
| `dor_knn_ci_high` | `knn_ci_hi` |
| `t_knn_used` | `t_knn` |
| `category_knn` | `cat_knn` |
| `category_knn_v3t` | `cat_knn_v3` |
| `dor_median` | `dor_med` |
| `dor_median_ci_low` | `med_ci_lo` |
| `dor_median_ci_high` | `med_ci_hi` |
| `category_median` | `cat_med` |

The SHP also adds `rank` (1-based, ordered by `dor_knn` descending) and
`country` (ISO3 from classification CSV).

## Provenance

- Classification: `v4/scripts/classification/classify_stable_state.py`
- Sampling: `v4/scripts/sampling/sample_stable_references_v4.py`
- Embedding extraction: `v4/scripts/extraction/extract_stable_refs_alphaearth.py`
- Validation/calibration: `v4/scripts/analysis/validate_steps_within_parent_v4.py`
- Scoring: `v4/scripts/analysis/score_test_sites_v4.py`
- Shapefile export: `v4/scripts/reporting/export_v4_shp.py`
- Embeddings source: AlphaEarth Satellite Embedding V1 (annual, 2024).
