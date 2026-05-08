# Methods (v1)

## Overview

We quantify post-disturbance vegetation recovery as a *Degree of Recovery* (DoR) score that compares each test site's satellite-derived embedding against parent-specific reference pools of "good" (intact natural) and "bad" (degraded) pixels. v1 is the original end-to-end implementation: a fixed-radius reference sampler, AlphaEarth embedding extraction, and a robust median-based DoR with a bootstrap confidence interval.

The pipeline runs in five steps, all on a single year (default = 2024):

1. Sample good and bad reference points around each parent site (Earth Engine).
2. Extract the AlphaEarth 64-band embedding at every reference point.
3. Extract the AlphaEarth embedding at every test-site (parent) point.
4. Score each test site against its own parent's references and produce a 95 % bootstrap CI on the DoR.
5. Classify and export to a shapefile and summary plots.

## Earth-observation inputs

All sampling and feature extraction is performed in Google Earth Engine (GEE). The pixel-level feature representation is the **AlphaEarth Annual Composite V1**, a 64-dimensional spectral embedding (bands `A00–A63`) sampled at 10 m. Reference-pool construction uses **ESA WorldCover v200** (class 40 = cropland, 50 = built, 80 = water).

## Stage 1 — Sampling good/bad reference points

For every parent site flagged as `built_loss` or `crop_loss` in the source RECOVER asset, the sampler:

1. Draws a **1 km buffer** around the parent point.
2. Builds a two-class WorldCover class-band image:
   - `good` = any terrestrial cover other than cropland (40), built (50), or water (80) — i.e. the natural state;
   - `bad` = built (50) for `built_loss` parents, cropland (40) for `crop_loss` parents — i.e. the parent's labelled degraded state.
3. Stratified-samples **100 good + 100 bad** reference points within the buffer (target 200 per parent, 10 m scale, fixed seed).

Parents labelled `stable_stable` are excluded — their bad reference state is undefined under v1's labelling assumption (no transition occurred). The output is exported as a GEE FeatureCollection (~31,200 reference points across 158 parents; a few parents return slightly fewer than 200 where natural cover is sparse within their buffer).

## Stages 2–3 — AlphaEarth feature extraction

Stage 2 extracts the 64-band AlphaEarth 2024 embedding at every reference point (output: `recover_reference_samples_alphaearth.parquet`). The extractor shards the FeatureCollection (~750 points/shard), runs `image.reduceRegions` per shard with escalating `tileScale` retries on memory errors, asserts the schema (all 64 bands present) on the first non-empty shard, and is checkpoint-resumable.

Stage 3 extracts the same embedding at the parent (test-site) points (output: `test_site_alphaearth_<year>.parquet`). It uses `fc.map(reduceRegion)` per feature rather than `image.reduceRegions(fc)` because the batched form silently returns null bands on the spatially-sparse, globally-distributed parent set.

## Stage 4 — Scoring each test site

### The intuition

Each parent site has two reference clouds in 64-dimensional embedding space: a "good" cloud (intact natural vegetation, sampled within 1 km of the parent) and a "bad" cloud (the parent's labelled degraded state — built or cropland — also within 1 km). To decide whether a test site looks recovering or degraded, we ask: *is the site's embedding closer to the good cloud or the bad cloud?*

We summarise the answer as a single number between 0 and 1, the **Degree of Recovery (DoR) score**:

- a score near **1** means the site sits much closer to good than to bad (looks recovering),
- a score near **0** means it sits much closer to bad than to good (looks degraded),
- a score near **0.5** means it is roughly equidistant from both clouds (ambiguous).

### How the score is computed

For each test site we compute pairwise **cosine distances** to every good reference and every bad reference (cosine distance treats two pixels as similar when their 64-band embeddings point in the same direction, regardless of overall brightness). The primary v1 score is the **median pairwise score**:

```
DoR = median(d_bad) / (median(d_good) + median(d_bad))
```

where `d_good` and `d_bad` are the cosine distances from the test-site embedding to the good and bad reference points, respectively. Using the **median** rather than the mean (or a centroid-to-centroid distance) is deliberate: the good cloud is often heterogeneous (multiple vegetation expressions, structural states) within a 1 km buffer, and a single centroid is not a reliable summary of a multimodal cloud. The median is robust to outliers and to multimodality without imposing a Gaussian assumption.

### Quantifying uncertainty

For each site we compute a **95 % bootstrap confidence interval** on the DoR by resampling the good and bad reference pools with replacement (n = 2,000 bootstrap draws). The CI captures uncertainty driven by the finite reference pool, not measurement noise in the test-site embedding itself.

### Categorical call

Each test site is assigned one of four outcomes from its bootstrap CI relative to the natural midpoint **0.5**:

- **`recovering`** — the entire 95 % CI lies above 0.5 (confidently more good-like than bad-like);
- **`degraded`** — the entire 95 % CI lies below 0.5 (confidently more bad-like than good-like);
- **`indistinguishable`** — the CI straddles 0.5 (the site cannot be confidently placed on either side);
- **`no_data`** — the AlphaEarth embedding is missing at the parent point (typically narrow rivers or coastline pixels).

### Diagnostic scores

Three additional scores are written alongside the primary DoR for diagnostic inspection: a linearly-normalised median variant (`dor_normalised`), the percentile rank of the observed DoR within the reference DoR distribution (`dor_percentile`), and a centroid-based cosine score (`dor_cosine`) that is informative when the good cloud is unimodal but unstable when it is not. These are not used for the operational classification.

## Stage 5 — Reporting

The final stage produces:

- **Shapefile** (`test_site_dor.shp`) — every scored test site with parent ID, label, categorical call, point-estimate DoR, 95 % CI bounds, reference counts, and parent coordinates (EPSG:4326), ready for spatial inspection in QGIS/ArcGIS.
- **Summary CSV** (`dor_summary_by_label.csv`) — category counts and mean DoR by `built_loss` / `crop_loss`.
- **Figures** (PNG) — per-site DoR with error bars, recovery-axis diagnostic, world map of categorical calls, distribution histograms, and a five-method scoring benchmark (median pairwise, Euclidean centroid LOO, cosine centroid LOO, kNN margin, logistic regression CV — pooled and per-parent ROC AUC on the labelled references).

## Reproducibility

All Earth Engine stages require `earthengine authenticate` (default GEE project: `ee-gsingh`). Random seeds are fixed (sampling seed = 234, bootstrap seed = 42) so reruns are deterministic. Year is set per-script and defaults to 2024; each extraction and scoring step accepts `--year` for other years.

## Known caveats

1. **5 sites with `no_data`** — their parent centroid lands on a pixel without AlphaEarth coverage (narrow rivers, coastline). The site is reported with `applicable=True` but NaN DoR.
2. **Diagnostic `dor_cosine` is unstable** — centroid-based cosine DoR has wide negative tails when the good cloud is multimodal. The primary `dor_median` is robust to this; `dor_cosine` is retained for diagnostic comparison only.
3. **`stable_stable` parents are not scored** — v1 has no defined bad reference pool for them. (Versions 4+ extend the framework to score these sites by inferring each parent's current land-cover state.)
