# Degree of Recovery (DoR)

Satellite-based framework for classifying whether RECOVER restoration test sites are **recovering**, **degraded**, or **indistinguishable**, using 64-dimensional AlphaEarth spectral embeddings and human-labelled reference points.

**Current version:** v3 (kNN cosine scorer, `dor_knn`)  
**Test sites:** 158 parent sites (built_loss / crop_loss) · **Year:** 2024  
**Embedding:** AlphaEarth Annual Composite V1, 64 bands (A00–A63), EPSG:4326

---

## Results summary (v3, 2024)

| Category | built_loss | crop_loss | Total |
|---|---|---|---|
| recovering | 32 | 21 | **53** |
| indistinguishable | 27 | 14 | **41** |
| degraded | 30 | 29 | **59** |
| no_data | 1 | 4 | **5** |

**Validation (within-parent 5-fold, 31,600 probes):**  
Step 4 (kNN cosine, k=5): false-degraded = 1.8 %, false-recovering = 0.9 %, abstain = 40 %

---

## Repository layout

```
degreeRecover/          # v1 production pipeline
  data/                 # scored shapefiles and CSVs
  plots/
  report/
  scripts/
    sampling/           # GEE reference point export
    extraction/         # AlphaEarth embedding extraction
    analysis/           # scoring and benchmarking
    reporting/          # shapefile + plot generation

v2/                     # v2 experiments (sampling strategy comparison)
  data/
  plots/
  scripts/

v3/                     # v3 experiments (kNN cosine scorer)
  data/
    test_site_dor_v3.csv              # 158 sites, dor_knn + dor_median scores
    test_site_knn5_scores.shp         # GIS-ready, ranked by dor_knn
    within_parent_site_scores.csv     # 31,600 validation probes
    sensitivity_sweep.csv             # hw × delta grid
    reliability_dor.csv               # calibration decomposition
    k_sweep_knn.csv                   # kNN performance vs k (1–75)
  plots/
  report/
    METHOD.md                         # full method and validation writeup
  scripts/analysis/
```

---

## Scoring method

For each test site, a **degree-of-recovery score** is computed by comparing its embedding to two pools of within-parent reference points:

- **good** refs (n=100): intact / recovering vegetation (WorldCover natural classes)
- **bad** refs (n=100): degraded state (built or crop cover, matching parent label)

**Primary score — `dor_knn` (v3):**

```
dor_knn = mean_cos(x_obs → k nearest bad refs)
        / (mean_cos(x_obs → k nearest good refs) + mean_cos(x_obs → k nearest bad refs))
```

Score → 1: test site is closer to bad refs (recovering signal).  
Score → 0: test site is closer to good refs (degraded signal).  
k = 5, cosine distance, threshold = 0.4859 (Youden-J calibrated).

**Classification (step 4 recommended):**

| Output | Rule |
|---|---|
| `recovering` | 95 % bootstrap CI entirely above threshold + deadband |
| `degraded` | 95 % bootstrap CI entirely below threshold + deadband |
| `indistinguishable` | CI straddles threshold or deadband |

Default hyperparameters: deadband half-width `hw = 0.05`, effect-size gate `delta = 0.03`.

See [`v3/report/METHOD.md`](v3/report/METHOD.md) for full method, calibration details, sensitivity analysis, and design rationale.

---

## Data pipeline

```
GEE asset: samples_recover_w_ref_label (1766 parent sites)
  └─► sample_reference_states.py        # stratified ref sampling → GEE asset
        └─► extract_alphaearth_embeddings.py   # pull 64-band embeddings → parquet
              ├─► extract_test_site_embeddings.py  # test-site centroids → parquet
              └─► score_test_sites_v3.py           # dor_knn + dor_median → CSV
                    └─► export_knn_shp.py           # ranked shapefile
```

Parquet files (`*.parquet`) and archives (`*.7z`) are excluded from git — re-extract via the scripts above or copy from `P:\155020_recover\WP1\degree_of_recovery\`.

---

## Setup

Requires Python 3.11+, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run pre-commit install   # optional: code-quality hooks
```

**Key dependencies:** `earthengine-api`, `duckdb`, `pandas`, `numpy`, `geopandas`, `shapely`, `scikit-learn`, `matplotlib`, `tqdm`

**Google Earth Engine:** authenticate once with `earthengine authenticate`. Default project: `ee-gsingh`.

---

## Run order (v3)

```bash
# 1. Sample reference points (GEE batch task, ~1 min)
uv run degreeRecover/scripts/sampling/sample_reference_states.py --export

# 2. Extract embeddings at reference points (~1 min, resumable)
uv run degreeRecover/scripts/extraction/extract_alphaearth_embeddings.py

# 3. Extract embeddings at test-site centroids
uv run degreeRecover/scripts/extraction/extract_test_site_embeddings.py --overwrite

# 4. Score test sites (v3 kNN cosine)
uv run v3/scripts/analysis/score_test_sites_v3.py

# 5. Export ranked shapefile
uv run v3/scripts/analysis/export_knn_shp.py
```

---

## Key outputs

| File | Description |
|---|---|
| `v3/data/test_site_knn5_scores.shp` | 158 test sites, ranked by `dor_knn` (rank 1 = most recovering) |
| `v3/data/test_site_dor_v3.csv` | Raw scores, CIs, and categories for all 158 sites |
| `v3/data/within_parent_site_scores.csv` | 31,600 validation probe scores |
| `degreeRecover/data/test_site_dor.shp` | v1 production shapefile (dor_median) |
| `v2/data/test_site_dor_v2.shp` | v2 shapefile |

---

## Network storage

Large data files not tracked in git are stored at:  
`P:\155020_recover\WP1\degree_of_recovery\`  
(NINA network drive — `\\nina.no\Prosjekter\155020_recover\WP1\degree_of_recovery`)

---

## References

- AlphaEarth Satellite Embedding V1 (2024)
- ESA WorldCover 2021 v200 — land cover reference classification
- RECOVER project — `projects/nina/RECOVER/samples_recover_w_ref_label`
