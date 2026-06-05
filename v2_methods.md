# Methods (v2)

## Overview

v2 keeps the v1 scoring concept — a per-parent Degree of Recovery (DoR) score that compares each test site's AlphaEarth embedding to its own good/bad reference clouds — but **redesigns how the references are chosen**. The motivating ecological problem with v1 was that some reference points were drawn from areas that were themselves undergoing crop or built expansion (or had recently been lost), polluting the "good" cloud and inflating the "bad" cloud unevenly across parents. v2 addresses this with three changes to sampling, plus diagnostic tooling for spatial autocorrelation.

The scoring math (median pairwise cosine distance, bootstrap CI, ±0.5 categorical call) is **unchanged from v1**. All differences in the v2 results come from cleaner, more standardised reference clouds.

## Earth-observation inputs

Same backbone as v1: AlphaEarth Annual Composite V1 (64 bands `A00–A63`, 10 m) for embeddings; ESA WorldCover v200 for the natural/cropland/built basis. v2 additionally uses several recent-loss layers to mask candidate references:

- **ESA WorldCereal `temporarycrops` (2021)** — cropland presence;
- **GLAD GLCmap2019** — cropland (class 252) and urban (classes 240–249);
- **GLC-FCS30D** — 2018 cropland (≤ 20) and urban (190);
- **HABLOSS crop-loss and built-loss trends (2018–2024)** — Sentinel-derived multi-year loss-trend rasters for cropland and built cover;
- **HABLOSS building-loss trend** — building-footprint loss signal.

These are combined into a **loss-exclusion mask** so that pixels with evidence of recent crop, built, or building loss within the analysis window are removed from the candidate reference pool before sampling.

## Stage 1 — v2 reference sampling

The v2 sampler differs from v1 in three ecologically meaningful ways:

### 1. Loss-region exclusion

Candidate reference pixels are intersected with the loss-exclusion mask described above. Any pixel flagged as recently lost crop, built, or building cover (loss-trend below `−7`, a conservative threshold) is removed from both the good and the bad candidate pools. This stops the pipeline from treating actively-degrading pixels as stable "good" references and stops it from mixing degrading-into-bad pixels with established-bad references.

### 2. Dynamic local buffering

v1 used a fixed 1 km buffer around every parent. v2 instead samples within a **larger maximum radius and selects nearest candidates first**, equivalent to expanding the buffer through a sequence of steps:

```
1 km → 1.5 km → 2 km → 3 km → 5 km → 8 km
```

The maximum buffer is clipped to the **RESOLVE 2017 ecoregion containing the parent**, so candidate references are local both in distance and in ecological setting. This avoids drawing references that are physically nearby but fall across a major ecoregion boundary. The output schema records, per reference point, the actual `dist_m` to the parent, the smallest `buffer_m_used` covering the selected set, and the parent ecoregion identifier `eco_id`, so users can filter to a fixed radius or inspect ecoregion membership post-hoc.

### 3. Balanced class targets and oversample-then-select

v1 used a fixed `stratifiedSample(N=100)` per class. v2 oversamples by a factor of 5 within the maximum radius, then selects the nearest valid candidates per class up to a target of 100 good + 100 bad per parent (minimum 30 per class; cap of 200 total). This produces more consistent realised counts across parents, particularly where natural cover is patchy.

The output GEE FeatureCollection extends v1's schema with `dist_m`, `buffer_m_used`, `target_good`, and `target_bad` fields.

## Stages 2–3 — AlphaEarth feature extraction

Identical to v1 (sharded `reduceRegions` for refs; `fc.map(reduceRegion)` for parent points). The same AlphaEarth extraction script is used; only the input FeatureCollection changes.

## Stage 4 — Scoring each test site

### The intuition (unchanged from v1)

Each parent has a good cloud and a bad cloud in 64-dimensional embedding space. The DoR asks: *is the test site closer to the good cloud or the bad cloud?* A score near 1 means good-like (recovering), near 0 means bad-like (degraded), near 0.5 means in between.

### How the score is computed (unchanged from v1)

The primary v2 score is the **median pairwise cosine score**:

```
DoR = median(d_bad) / (median(d_good) + median(d_bad))
```

where `d_good` and `d_bad` are cosine distances from the test-site embedding to each good and bad reference, respectively. The median is used because the good cloud is typically heterogeneous and a single centroid is not a robust summary.

### Quantifying uncertainty (unchanged from v1)

A 95 % bootstrap confidence interval on the DoR is computed by resampling the good and bad reference pools with replacement (n = 2,000 bootstrap draws).

### Categorical call (unchanged from v1)

Each site is classified relative to the midpoint **0.5**:

- **`recovering`** — entire 95 % CI above 0.5;
- **`degraded`** — entire 95 % CI below 0.5;
- **`indistinguishable`** — CI straddles 0.5;
- **`no_data`** — the AlphaEarth embedding is missing at the parent point.

## Stage 5 — Sampling-strategy comparison (v2 only)

A v2-specific diagnostic stage benchmarks several sampling strategies (random subsets, nearest-N, balanced-by-distance, etc.) against the *actual test-site embedding* — not synthetic probes — to quantify how much score and CI width each strategy delivers per parent. The operational v2 product uses the `random_100` strategy: 100 good + 100 bad randomly drawn from the v2 candidate pool. It was selected for the strongest balance of precision (narrow CIs), site-level stability (small score change under different random draws), and simplicity.

## Stage 6 — Effective-sample-size diagnostics

Reference points sampled within a 1–8 km buffer are spatially autocorrelated; treating the bootstrap n as the raw point count over-states the effective independent sample. v2 adds two diagnostics that quantify this:

- **Variogram-derived correlation range**, fitted per parent on the random_100 references, summarising the distance over which embeddings remain correlated.
- **Effective sample size (`n_eff`) calibration**, which translates the raw n into an autocorrelation-adjusted equivalent.

These are reported alongside the v2 DoR table as flags rather than applied to the bootstrap directly. They identify parents whose nominal CI is likely too narrow because the references are not independent — a known caveat to interpret site-level confidence in those parents conservatively.

## Reproducibility and outputs

All Earth Engine stages require `earthengine authenticate`. Random seeds are fixed (sampling seed = 234, bootstrap seed = 42). Outputs:

- `test_site_dor_v2.csv` / `.shp` — per-site DoR, 95 % CI, categorical call, parent label, parent coordinates;
- `dor_summary_by_label_v2.csv` — category counts by `built_loss` / `crop_loss`;
- `sampling_strategy_comparison.csv` — strategy benchmark results;
- `v2_corr_range_by_parent_random100.csv` — per-parent correlation ranges.

## What changed from v1, and why it matters ecologically

v1 and v2 give a near-identical **mean DoR** across all 158 sites (≈0.482), but **60 of 158 sites change category** between versions. The shift is not a global recovery- or degradation-direction trend; it is a **redistribution out of the `indistinguishable` middle** as cleaner, more standardised reference clouds give individual sites tighter CIs and more decisive categorical calls. The ecological interpretation: v2 does not change the verdict on landscapes that are unambiguously recovering or unambiguously degraded; it changes the verdict on borderline sites where v1's references were contaminated by recent loss or by a buffer too small (or too large) for the local landscape.

## Known caveats

1. **Spatial autocorrelation is reported, not corrected.** The bootstrap CI uses raw reference counts; the `n_eff` and variogram diagnostics flag parents where the nominal CI is likely narrower than the autocorrelation-adjusted CI would be. Future versions could feed `n_eff` into the bootstrap directly.
2. **Loss-exclusion thresholds (`−7`) are conservative defaults** — they were chosen to remove obvious loss without aggressively shrinking small parent buffers. Sensitivity to this threshold has not been characterised.
3. **`stable_stable` parents are still excluded** for the same reason as in v1 (no defined bad pool). Versions 4+ extend the framework to handle them.
