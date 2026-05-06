# test_site_knn5_scores — Shapefile metadata

**File:** `v3/data/test_site_knn5_scores.shp`  
**Created by:** `v3/scripts/analysis/export_knn_shp.py`  
**Source scores:** `v3/data/test_site_dor_v3.csv`  
**CRS:** EPSG:4326 (WGS 84)  
**Geometry:** Point (one centroid per parent site)  
**Features:** 158  
**Sorted:** descending `dor_knn` — rank 1 is the most-recovering site

---

## Attribute schema

| Field | Type | Description |
|---|---|---|
| `rank` | int | Exploration rank: 1 = highest `dor_knn` score |
| `parent_id` | str | Earth Engine `system:index` of the source parent feature |
| `lbl` | str | Site type: `built_loss` or `crop_loss` |
| `n_good` | int | Number of good-state reference embeddings used |
| `n_bad` | int | Number of bad-state reference embeddings used |
| `dor_knn` | float | **Primary v3 score.** Mean cosine distance to 5 nearest bad refs / (good + bad). Range [0, 1]. Closer to 1 = recovering; closer to 0 = degraded |
| `knn_ci_lo` | float | 2.5th percentile of 2000-resample bootstrap CI on `dor_knn` |
| `knn_ci_hi` | float | 97.5th percentile of bootstrap CI on `dor_knn` |
| `cat_knn` | str | Classification based on `dor_knn`: `recovering` / `indistinguishable` / `degraded` / `no_data` |
| `dor_med` | float | Secondary score (v1/v2 method). Median cosine distance to all bad refs / (good + bad) |
| `med_ci_lo` | float | 2.5th percentile bootstrap CI on `dor_med` |
| `med_ci_hi` | float | 97.5th percentile bootstrap CI on `dor_med` |
| `cat_med` | str | Classification based on `dor_med` |

---

## Scoring method

Each test site is represented by the mean of its per-pixel AlphaEarth embeddings (64-band, annual 2024 composite). References are drawn from the same parent site's buffer (~1 km), sampled from WorldCover 2021.

**dor_knn formula:**

```
dor_knn = mean_cos(x → 5 nearest bad refs)
        / (mean_cos(x → 5 nearest good refs) + mean_cos(x → 5 nearest bad refs))
```

where `mean_cos` is mean cosine distance to the k=5 nearest neighbours in each reference pool.

**Classification thresholds (Youden-J calibrated, within-parent LOO):**

| Score | Threshold | Deadband (±hw) |
|---|---|---|
| `dor_knn` | 0.4859 | ±0.05 |
| `dor_med` | 0.5000 | n/a (simple threshold) |

A site is classified as `recovering` only if its entire 95 % bootstrap CI lies above the threshold + deadband. `indistinguishable` means the CI is uncertain relative to the threshold. `no_data` indicates the parent centroid had no AlphaEarth coverage.

---

## Validation summary

Validated via within-parent 5-fold cross-validation (31,600 probes, 158 parents):

| Metric | Value |
|---|---|
| False-degraded rate (good refs misclassified) | 1.8 % |
| False-recovering rate (bad refs misclassified) | 0.9 % |
| Abstention (indistinguishable) | ~40 % |
| AUC (dor_knn vs dor_median) | 0.954 vs 0.912 |

---

## Spatial autocorrelation note

The 5 nearest neighbours used in `dor_knn` are selected in **embedding space**, not geographic space. Analysis (`v3/scripts/analysis/spatial_autocorr_knn5.py`) found a weak positive correlation between cosine rank and spatial distance (r ≈ 0.13–0.16): the nearest embedding neighbours are on average ~0.5 km closer spatially than the remaining refs. Since all refs are within the ~1 km parent buffer, the absolute effect is small, but scores should be interpreted as reflecting spectral-ecological similarity rather than purely spatial proximity. This is an active area of investigation (v4).

---

## Files in this shapefile set

| Extension | Contents |
|---|---|
| `.shp` | Geometry (point coordinates) |
| `.dbf` | Attribute table |
| `.shx` | Shape index |
| `.prj` | Coordinate reference system (EPSG:4326 WKT) |
| `.cpg` | Character encoding (UTF-8) |

---

## Provenance

| Step | Script | Output |
|---|---|---|
| Reference sampling | `degreeRecover/scripts/sampling/sample_reference_states.py` | GEE asset |
| Embedding extraction | `degreeRecover/scripts/extraction/extract_alphaearth_embeddings.py` | `*.parquet` |
| Test-site embeddings | `degreeRecover/scripts/extraction/extract_test_site_embeddings.py` | `*.parquet` |
| Scoring | `v3/scripts/analysis/score_test_sites_v3.py` | `test_site_dor_v3.csv` |
| Shapefile export | `v3/scripts/analysis/export_knn_shp.py` | this file |
