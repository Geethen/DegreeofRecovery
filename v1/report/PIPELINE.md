# RECOVER — Degree-of-Recovery (DoR) pipeline

End-to-end pipeline for scoring how recovered each RECOVER test site is, using
AlphaEarth Satellite Embedding V1 (64-band annual composite) and labelled
"good" / "bad" reference points sampled from WorldCover.

## Data flow

```
samples_recover_w_ref_label  (EE asset, 1766 parent sites)
            │
            ▼
  scripts/sampling/sample_reference_states.py --export
            │
            ▼
  recover_reference_samples  (EE asset, ~31 k stratified ref points)
            │
            ▼
  scripts/extraction/extract_alphaearth_embeddings.py
            │
            ▼
  data/recover_reference_samples_alphaearth.parquet  (31,178 rows × 64 bands)
            │
            ├── scripts/extraction/extract_test_site_embeddings.py
            │           │
            │           ▼
            │   data/test_site_alphaearth_<year>.parquet  (158 parent rows)
            │
            ▼
  scripts/analysis/score_test_sites.py
            │
            ▼
  data/test_site_dor.csv  (per-site DoR + 95 % CI)
            │
            ▼
  scripts/reporting/build_summary.py
            │
            ▼
  data/test_site_dor.shp  +  data/dor_summary_by_label.csv
  plots/dor_world_map.png  + dor_distribution.png + dor_category_breakdown.png
  plots/method_benchmark_summary.png
```

A separate diagnostic script, `scripts/analysis/recovery_score.py`, benchmarks
five DoR scoring methods on the labelled refs (good vs bad ROC AUC).

## Run order

Each script has a `--help` listing all flags. The defaults below assume the
configured project (`ee-gsingh`), the parent asset
`projects/nina/RECOVER/samples_recover_w_ref_label`, and AlphaEarth year 2024.

### 1. Stratified-sample good/bad reference points around each parent

```
python scripts/sampling/sample_reference_states.py --export --verbose
```

- **What it does**: for every `built_loss`/`crop_loss` parent in the source
  asset, buffers it by 1 km, builds a two-class WorldCover class band
  (good = natural; bad = built or crop, depending on the parent label), and
  stratified-samples 100 of each class. Output is exported as an EE asset
  `projects/ee-gsingh/assets/recover_reference_samples`.
- **Run time**: ~1 minute as an EE batch task.
- **Throughput**: 158 parents × 200 ref points → 31,178 reference points
  (a few parents return slightly fewer than 200 due to small natural areas
  in their buffer).

### 2. Extract AlphaEarth embeddings at every reference point

```
python scripts/extraction/extract_alphaearth_embeddings.py \
    --samples_asset projects/ee-gsingh/assets/recover_reference_samples \
    --verbose
```

- **What it does**: shards the FC by random column (~750 pts/shard,
  auto-sized), then for each shard runs `image.reduceRegions` against the
  AlphaEarth annual composite and streams results into a DuckDB buffer.
  On completion, copies the buffer to a ZSTD parquet.
- **Resilience**:
  - Retries with **escalating tileScale** (`2 → 4 → 8 → 16`) on transient/
    memory errors.
  - **Schema lock**: first non-empty shard is asserted to contain all 64
    bands `A00..A63`; refuses to write a partially-banded parquet.
  - **Resumable**: a checkpoint file
    (`data/recover_reference_samples_alphaearth.parquet.checkpoint.json`)
    records `done` and `failed` shard indices. Re-running picks up only
    incomplete shards. Failed shards are auto-retried on the next run.
- **Run time**: ~1 minute for 42 shards × ~750 pts at `MAX_WORKERS=40`.

### 3. Cache test-site (parent point) embeddings

```
python scripts/extraction/extract_test_site_embeddings.py --overwrite
```

- **What it does**: pulls the unique `parent_id`s from the references
  parquet, samples AlphaEarth at each parent's centroid, writes the cache
  to `data/test_site_alphaearth_<year>.parquet`.
- **Important**: uses `fc.map(reduceRegion)` per-feature rather than
  `image.reduceRegions(fc)`. The batched version returns null bands
  silently when the FC spans a very large bounding box with sparse
  points (which is exactly the test-site case — ~158 globally-distributed
  parents).
- **Skip if exists**: idempotent unless `--overwrite` is passed.

### 4. Score every test site

```
python scripts/analysis/score_test_sites.py
```

- **What it does**: for each parent, computes the primary DoR score
  (`dor_median = median(d_bad) / (median(d_good) + median(d_bad))` in
  cosine-distance space) plus a 95 % bootstrap CI on it (n_boot = 2000),
  and three diagnostic scores (`dor_normalised`, `dor_percentile`,
  `dor_cosine`).
- **Parity check**: hard-errors if more than 10 % of the refs parents
  are missing from the test-site parquet (catches stale caches).
- **Outputs**:
  - `data/test_site_dor.csv` — one row per parent.
  - `plots/test_site_dor_with_ci.png` — primary plot, error bars per site.
  - `plots/test_site_recovery_axis.png` — diagnostic: where each parent
    sits on the good→bad embedding axis.

### 5. Build deliverables

```
python scripts/reporting/build_summary.py
```

- **What it does**: classifies each site as
  `recovering` (CI fully > 0.5) / `indistinguishable` (CI straddles 0.5) /
  `degraded` (CI fully < 0.5) / `no_data` (parent landed on a no-data
  pixel), pulls coords from the cache, and writes the QGIS shapefile,
  summary CSV, and summary plots. Also reruns the five-method benchmark
  to refresh `plots/method_benchmark_summary.png`.

### (Diagnostic) Benchmark scoring methods on the labelled refs

```
python scripts/analysis/recovery_score.py
```

- Fits five methods (median pairwise, Euclidean centroid LOO, cosine
  centroid LOO, k-NN margin, logistic regression CV) on the labelled
  references and reports pooled and per-parent ROC AUC, plus PCA.
- NaN ref points (water/edge pixels) are dropped on load.

## Inputs / outputs cheat sheet

| Path | Producer | Consumer |
|---|---|---|
| `projects/nina/RECOVER/samples_recover_w_ref_label` (EE) | (external) | `sample_reference_states.py`, `extract_test_site_embeddings.py` |
| `projects/ee-gsingh/assets/recover_reference_samples` (EE) | `sample_reference_states.py --export` | `extract_alphaearth_embeddings.py` |
| `data/recover_reference_samples_alphaearth.parquet` | `extract_alphaearth_embeddings.py` | `score_test_sites.py`, `recovery_score.py`, `extract_test_site_embeddings.py` |
| `data/test_site_alphaearth_<year>.parquet` | `extract_test_site_embeddings.py` | `score_test_sites.py`, `build_summary.py` |
| `data/test_site_dor.csv` | `score_test_sites.py` | `build_summary.py` |
| `data/test_site_dor.{shp,shx,dbf,prj,cpg}` | `build_summary.py` | (QGIS) |
| `data/dor_summary_by_label.csv` | `build_summary.py` | (report) |
| `plots/test_site_dor_with_ci.png` | `score_test_sites.py` | (report) |
| `plots/test_site_recovery_axis.png` | `score_test_sites.py` | (report) |
| `plots/dor_world_map.png` | `build_summary.py` | (report) |
| `plots/dor_distribution.png` | `build_summary.py` | (report) |
| `plots/dor_category_breakdown.png` | `build_summary.py` | (report) |
| `plots/method_benchmark_summary.png` | `build_summary.py` | (report) |
| `plots/recovery_score_hists.png`, `recovery_roc.png`, `recovery_per_parent_auc.png`, `recovery_pca.png` | `recovery_score.py` | (diagnostic) |

## Shapefile schema (`data/test_site_dor.shp`)

| Field | Type | Description |
|---|---|---|
| `parent_id` | str | EE `system:index` of the source parent feature |
| `par_label` | str | `built_loss` or `crop_loss` |
| `category` | str | `recovering` / `indistinguishable` / `degraded` / `no_data` |
| `n_good` | int | count of natural-state ref points used |
| `n_bad` | int | count of degraded-state ref points used |
| `applicable` | bool | True if the site has both ≥ 1 good and ≥ 1 bad ref |
| `cos_d_good` | float | cosine distance from parent embedding to good centroid |
| `cos_d_bad` | float | cosine distance from parent embedding to bad centroid |
| `dor_med` | float | **primary DoR score** — median pairwise (1 = natural, 0 = degraded) |
| `dor_ci_lo` | float | 2.5 % bootstrap CI on `dor_med` |
| `dor_ci_hi` | float | 97.5 % bootstrap CI on `dor_med` |
| `dor_norm` | float | diagnostic: linear-normalised median score |
| `dor_pct` | float | diagnostic: percentile rank of obs DoR within ref-DoR distribution |
| `dor_cos` | float | diagnostic: centroid-based cosine DoR |
| `lon`, `lat` | float | parent centroid (EPSG:4326) |

(Field names are abbreviated because shapefile DBF caps them at 10 chars.)

## Results (year = 2024)

```
category    recovering  indistinguishable  degraded  no_data  total
par_label
built_loss          32                 27        30        1     90
crop_loss           21                 14        29        4     68
ALL                 53                 41        59        5    158
```

**Method benchmark (pooled ROC AUC, 31 k labelled refs):**

| Method | AUC |
|---|---|
| `knn_margin` | **0.948** |
| `logreg` | 0.921 |
| `euclidean` (centroid LOO) | 0.902 |
| `cosine` (centroid LOO) | 0.901 |
| `median_pairwise` (current primary) | 0.883 |

The current primary score (`median_pairwise`) ranks lowest. `knn_margin` is
the best on this dataset — worth considering as the new primary, especially
because its per-parent AUC std (0.057) is also tighter than
`median_pairwise` (0.092).

## Known caveats

1. **5 sites with `no_data`** — their parent centroid landed on a pixel
   with no AlphaEarth coverage (typically narrow rivers / coastline).
   `applicable=True` in the CSV but every DoR score is NaN. They are
   plotted in grey on the world map and counted in the `no_data` column
   of the summary.
2. **Diagnostic `dor_cos` is unstable** — centroid-based cosine DoR has
   wide negative tails (mean ≈ −0.8 to −1.2 across labels). This is the
   failure mode the docstring of `dor_cosine_centroid` warns about
   (multimodal good clouds break centroid summaries) — the primary
   `dor_med` is well-behaved.
3. **`reduceRegions` quirk** — `image.reduceRegions(fc, ...)` silently
   returned null bands when called over the 158 globally-distributed
   parent points. The workaround `fc.map(reduceRegion)` is used in
   `extract_test_site_embeddings.py`. The refs extractor in
   `extract_alphaearth_embeddings.py` is unaffected because each shard
   contains ~750 spatially-clustered points (only ~1 % NaN, all
   attributable to water/edge pixels).
4. **Year is set per-script** — `extract_alphaearth_embeddings.py`,
   `extract_test_site_embeddings.py`, `score_test_sites.py`, and
   `build_summary.py` each default to `YEAR = 2024`. To process a
   different year, pass `--year` on each.

## Scripts

- `scripts/sampling/sample_reference_states.py`
- `scripts/extraction/extract_alphaearth_embeddings.py`
- `scripts/extraction/extract_test_site_embeddings.py`
- `scripts/analysis/score_test_sites.py`
- `scripts/analysis/recovery_score.py`
- `scripts/reporting/build_summary.py`

## Dependencies

`earthengine-api`, `duckdb`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`,
`tqdm`, `geopandas`, `shapely`, `pyarrow`.
