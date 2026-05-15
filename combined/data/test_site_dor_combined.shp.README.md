# `test_site_dor_combined` dataset metadata

A single Point shapefile containing **both** the v3 loss-site DoR scores
(`built_loss`, `crop_loss`) and the v4 stable-site DoR scores
(`stable_nature`, `stable_crop`, `stable_built`). Replaces the need to load
`v3/data/test_site_knn5_scores.shp` and `v4/data/test_site_dor_v4.shp`
separately when an analyst wants a global view of all scored sites.

## File set

| File | Purpose |
|---|---|
| `test_site_dor_combined.shp` | Point geometry layer for GIS |
| `test_site_dor_combined.shx` | Shapefile index |
| `test_site_dor_combined.dbf` | Shapefile attributes |
| `test_site_dor_combined.prj` | CRS definition (EPSG:4326) |
| `test_site_dor_combined.cpg` | DBF text encoding |

## Geometry

- Type: Point
- CRS: EPSG:4326 (WGS84 lon/lat)
- One feature per `parent_id` (v3 = 158, v4 = 1471, total = 1629).
- Sorted by `dor_knn` descending — `rank` 1 is the most-recovering site.
- v3 coordinates are parent-site centroids carried over from
  `v3/data/test_site_knn5_scores.shp`. v4 coordinates are parent centroids
  from `v4/data/stable_state_classification.csv`.

## Source datasets

| `source` value | Origin | Labels | n |
|---|---|---|---|
| `v3_loss` | `v3/data/test_site_dor_v3.csv` | `built_loss`, `crop_loss` | 158 |
| `v4_stable` | `v4/data/test_site_dor_v4.csv` | `stable_nature`, `stable_crop`, `stable_built` | 1471 |

Parent-id namespaces do not overlap between v3 and v4.

## Scoring consistency (v3 ↔ v4)

Both versions use **the same scorer**: mean-cosine kNN with k = 5,
percentile bootstrap CI (`n_boot = 2000`, `seed = 42`), and the cosine
distance helper defined identically in both code paths. The functions
`_knn_score`, `_median_score`, `cosine_dists_to_set`, `bootstrap_ci` and
`classify` in `v3/scripts/analysis/score_test_sites_v3.py` and
`v4/scripts/analysis/score_test_sites_v4.py` are character-identical apart
from whitespace, so the numeric `dor_knn` / `dor_median` columns from the
two halves are directly comparable.

The only methodological difference is the **bad-reference pool**:
- v3 uses per-disturbance-label bad refs (`built_loss` → built bad refs,
  `crop_loss` → crop bad refs).
- v4 uses per-stable-class bad refs (`stable_nature` → crop ∪ built,
  `stable_crop` → crop, `stable_built` → built).

For category assignment, the two versions originally used different
thresholds (v3: pooled `t = 0.4859`; v4: per-class refit thresholds). To
keep `cat_knn` directly comparable across the combined dataset, **the
unified `cat_knn` column always applies the v3 pooled threshold
`t_knn = 0.4859`**:
- v3 rows: native `category_knn` (already calibrated against 0.4859).
- v4 rows: `category_knn_v3t` (the v3-threshold transfer column from the v4
  scoring run).

The original v4 per-class category is preserved in `cat_knn_pc` for users
who need the v4-native decision rule.

## Column descriptions (DBF fields)

| SHP field | Type | Description |
|---|---|---|
| `rank` | int | 1-based rank by `dor_knn` descending across the combined dataset. |
| `parent_id` | string | Parent site id (zero-padded). Unique across the union. |
| `source` | string | `v3_loss` (158 features) or `v4_stable` (1471 features). |
| `lbl` | string | Site label: `built_loss`, `crop_loss`, `stable_nature`, `stable_crop`, or `stable_built`. |
| `stab_class` | string | v4 stable class (`nature` / `crop` / `built`); empty for v3 rows. |
| `country` | string | ISO3 country code from v4 classification; empty for v3 rows. |
| `n_good` | int | Number of good (natural-state) references used. |
| `n_bad` | int | Number of bad references used (per-label for v3, per-class for v4). |
| `dor_knn` | float | **Primary DoR score.** Mean kNN (k=5) cosine distance ratio: `mean(d_bad_top5) / (mean(d_good_top5) + mean(d_bad_top5))`. Range [0, 1]; 1 = recovering, 0 = degraded. |
| `knn_ci_lo` | float | Lower 95 % bootstrap bound for `dor_knn`. |
| `knn_ci_hi` | float | Upper 95 % bootstrap bound for `dor_knn`. |
| `t_knn` | float | Pooled v3 kNN threshold (0.4859) applied to derive `cat_knn`. Identical for all rows. |
| `cat_knn` | string | **Unified classification** under `t_knn = 0.4859`. `recovering` / `indistinguishable` / `degraded` / `no_data`. |
| `cat_knn_pc` | string | Per-version-native classification. v3 rows: same as `cat_knn`. v4 rows: classification under v4's per-class refit thresholds (nature 0.4861, crop 0.4823, built 0.4948). |
| `dor_med` | float | Secondary score: median pairwise cosine distance ratio. |
| `med_ci_lo` | float | Lower 95 % bootstrap bound for `dor_median`. |
| `med_ci_hi` | float | Upper 95 % bootstrap bound for `dor_median`. |
| `cat_med` | string | Median-score classification under fixed `t_med = 0.5`. |
| `geometry` | Point | EPSG:4326 lon/lat. |

## Decision rule

`cat_knn` is derived from the bootstrap CI vs the threshold:

- `recovering`: `knn_ci_lo > t_knn`
- `degraded`: `knn_ci_hi < t_knn`
- `indistinguishable`: CI overlaps `t_knn`
- `no_data`: score not available (insufficient refs at the parent)

## Category breakdown

Unified `cat_knn` (threshold = 0.4859):

| source | label | degraded | indistinguishable | recovering | no_data |
|---|---|---:|---:|---:|---:|
| v3_loss   | built_loss     |  19 |  42 |  29 | 0 |
| v3_loss   | crop_loss      |  20 |  33 |  15 | 0 |
| v4_stable | stable_nature  |  84 | 264 | 358 | 7 |
| v4_stable | stable_crop    | 289 | 191 |  43 | 0 |
| v4_stable | stable_built   | 151 |  71 |  13 | 0 |

## Plots

- `combined/plots/dor_distribution_by_category.png`
  Top: `dor_knn` histogram per `lbl` (5 labels overlaid).
  Bottom: `dor_knn` histogram split by unified `cat_knn` decision.

## Provenance

- Export script: `combined/scripts/export_combined_shp.py`
- Distribution chart: `combined/scripts/plot_combined_distribution.py`
- v3 scoring: `v3/scripts/analysis/score_test_sites_v3.py` (k=5, n_boot=2000, seed=42)
- v4 scoring: `v4/scripts/analysis/score_test_sites_v4.py` (k=5, n_boot=2000, seed=42)
- Embeddings source: AlphaEarth Satellite Embedding V1 (annual, 2024).
- Reference strategy: `random_100` (100 good, 100 bad refs per parent).
