# Methods

## Overview

We quantify post-disturbance vegetation recovery as a *Degree of Recovery* (DoR) score that compares each test site's satellite-derived embedding against parent-specific reference pools of "good" (intact natural) and "bad" (degraded) pixels. The pipeline is implemented as seven stages and executed end-to-end by `run_v4_pipeline.py`. Local stages are fully reproducible (seed = 42).

Version 4 (v4) extends earlier versions of the framework to include `stable_stable` parent sites — sites whose disturbance label asserts that no land-cover transition occurred over the observation window. Earlier versions excluded these sites because they lack a disturbance-defined "bad" pool. v4 instead infers each parent's *current* land-cover state from independent map products and routes the bad-reference pool accordingly.

## Earth-observation inputs

All sampling and feature extraction are performed in Google Earth Engine (GEE). The pixel-level feature representation is the **AlphaEarth** 64-band annual embedding (year 2024), sampled at 10 m. Auxiliary land-cover and cropland sources used for state classification and reference-pool construction are:

- ESA WorldCover v100 (2020) and v200 (2021) — class 40 (cropland), 50 (built), 80 (water);
- Dynamic World V1 — annual modal class for 2018–2024 (class 4 = crops, 6 = built);
- ESA WorldCereal `temporarycrops` (2021);
- VIDA Combined building footprints (Google + Microsoft, country shards) and Microsoft Buildings (country shards), joined to each parent point via a 30 m proximity test using the geoBoundaries ADM0 country layer for shard selection.

## Stage 1 — Stable-state classification of `stable_stable` parents

For each parent point flagged as `stable_stable` in the source RECOVER sample, we infer a `stable_class ∈ {nature, crop, built, ambiguous}` by majority vote across six independent sources: WorldCover v100, WorldCover v200, Dynamic World annual mode (2018–2024), WorldCereal temporary-crops, VIDA building proximity, and MS Buildings proximity. Each source casts one of {`nature`, `crop`, `built`, `unknown`}. Raster sources are sampled in a single GEE round-trip via `sampleRegions`; building footprint sources are evaluated per-country, sharded by ISO3 code, with a 30 m proximity radius. A parent is assigned the modal class only if at least `MIN_AGREE` sources agree on the winner; otherwise it is labelled `ambiguous` and excluded from downstream analysis. Because the parent label asserts no transition occurred over the observation window, the use of single-snapshot building footprints (VIDA, MSBuildings) is acceptable for stable parents.

## Stage 2 — Reference sampling with class-specific bad pools

For each non-ambiguous parent we draw two reference pools within an adaptive search radius (1, 1.5, 2, 3, 5, or 8 km — expanded until a per-class minimum is met, capped at 100 points per class and 200 total per parent):

- **Good pool (shared across classes).** Pixels in WorldCover classes other than {40, 50, 80} (i.e. neither cropland, built, nor water), additionally masked by a v2 multi-year loss-trend mask so that the good pool excludes pixels with evidence of recent loss.
- **Bad pool — routed by `stable_class`** so that each class's bad pool corresponds to its own non-natural state:
  - `stable_nature` → combined cropland ∪ built (WorldCover {40, 50});
  - `stable_crop` → cropland only (WorldCover {40});
  - `stable_built` → built only (WorldCover {50}).

The `stable_crop` and `stable_built` configurations function as **end-to-end sanity checks**: the test site itself sits inside its own bad state, so its embedding should resolve closer to the bad pool than to the natural-good pool, yielding an expected `degraded` score. Both classes are retained in the final dataset to validate the framework.

The sampler exports a GEE `FeatureCollection` of reference points; `parent_label` encodes the stable class (`stable_nature` / `stable_crop` / `stable_built`) so downstream calibration can group by class without schema changes.

## Stages 3–4 — AlphaEarth feature extraction

Stage 3 extracts the 64-band AlphaEarth 2024 embedding at every reference point (output: `v4_stable_refs_alphaearth.parquet`). Stage 4 extracts the same embedding at the parent (test-site) points (output: `test_site_alphaearth_2024_v4.parquet`). Both stages run server-side in GEE and persist results as Parquet for fast local re-use.

## Stage 5 — Calibrating the decision threshold for each land-cover class

### The intuition

For every parent site we now have two reference clouds in 64-dimensional embedding space: a "good" cloud (intact natural vegetation nearby) and a "bad" cloud (the parent's own degraded state — cropland, built, or both, depending on `stable_class`). To decide whether any new pixel looks recovering or degraded, we ask a simple question: *is this pixel closer to the good cloud or the bad cloud?*

We summarise the answer as a single number between 0 and 1, the **Degree of Recovery (DoR) score**:

- a score near **1** means the pixel sits much closer to good than to bad (looks recovering),
- a score near **0** means it sits much closer to bad than to good (looks degraded),
- a score near **0.5** means it is roughly equidistant from both clouds (ambiguous).

We compute this score in two complementary ways, both based on **cosine distance** between embeddings (cosine distance treats two pixels as similar if their 64-band embeddings point in similar directions, ignoring overall brightness):

- **Median score** — uses the *median* cosine distance to all good pixels and to all bad pixels. Robust to outliers, but smooths over local structure.
- **kNN score (k = 5)** — uses the *average* cosine distance to only the **5 nearest** good pixels and the **5 nearest** bad pixels. More sensitive to local structure in the reference clouds, which matters when "good" or "bad" is heterogeneous.

Both scores are computed as `distance_to_bad / (distance_to_good + distance_to_bad)`, which is why they fall in [0, 1] and increase as the pixel becomes more good-like.

### Choosing the cut-off

The DoR is a continuous score, but to say "recovering" vs "degraded" we need a cut-off. The cut-off is not assumed — it is **learned from the reference data itself**, separately for each `stable_class` (nature, crop, built).

To learn it, we play a controlled game on the reference pixels (whose true label is known):

1. Hold out one reference pixel at a time (leave-one-out within its parent), removing it from the good and bad pools.
2. Score the held-out pixel using the remaining references.
3. Repeat for every reference pixel, building a distribution of scores for known-good and known-bad references.
4. Sweep candidate cut-offs across that distribution and pick the one that **maximises Youden's J** — the cut-off that jointly maximises the true-positive rate (good correctly called recovering) and minimises the false-positive rate (bad wrongly called recovering).

This is done independently for each `stable_class` because the embedding geometry of "natural vs cropland", "natural vs built", and "natural vs (cropland ∪ built)" differs, and a single global cut-off would systematically miscalibrate one or more classes. The calibrated thresholds are stored in `calibrated_thresholds_v4.json`.

We additionally retain the v3 pooled thresholds (`t_med = 0.4728`, `t_knn = 0.4859`) and apply them as a parallel "transfer-check" column so we can see how much our per-class refit changes the call. A within-parent 5-fold cross-validation is run alongside calibration to characterise how stable the threshold is across subsets of each parent's references.

## Stage 6 — Scoring each test site

With the per-class cut-offs fixed, each `stable_stable` test site is scored against its parent's reference clouds. Every site receives one of four outcomes:

- **`recovering`** — the site's embedding sits clearly closer to the good cloud than to the bad cloud (DoR above the cut-off, with confident separation between the two clouds);
- **`degraded`** — the site's embedding sits clearly closer to the bad cloud (DoR below the cut-off);
- **`indistinguishable`** — the site lies in a *deadband* around the cut-off where the call would be unreliable, **or** the good and bad clouds for this parent are too similar to each other for the comparison to be meaningful (an effect-size gate). We deliberately abstain rather than force a noisy call;
- **`no_data`** — the AlphaEarth embedding is missing or invalid for the site.

Two safeguards keep the categorical call honest:

1. **Deadband around the threshold.** Scores within a small margin of the cut-off are returned as `indistinguishable`. A score of 0.51 should not be reported as confidently "recovering" when 0.49 would have been "degraded".
2. **Effect-size gate.** If the good and bad reference clouds are not appreciably separated for this parent — i.e. the references themselves don't establish a meaningful contrast — we return `indistinguishable` rather than rely on a cut-off applied to a near-degenerate score distribution.

Uncertainty is quantified with **bootstrap confidence intervals** (2,000 resamples of the reference pools), so each test site is reported with a DoR point estimate, a 95 % CI, and the categorical call.

### Sanity checks built into the design

For sites whose parent is `stable_crop` or `stable_built`, the **expected** outcome is `degraded`. The site itself sits inside its own non-natural stable state, so the framework should recognise that the embedding looks more like the bad cloud (cropland or built pixels) than like the natural good cloud. These two classes are therefore retained in the final dataset as **end-to-end validation**: if the framework calls a stable-cropland site `recovering`, something is wrong upstream.

## Stage 7 — Reporting

The final stage produces:

- **Figures** (PNG) — per-class DoR distributions, calibration diagnostics, parent-level summaries, and visual checks that the sanity-check classes behave as expected.
- **Shapefile** (`test_site_dor_v4.shp`) — every scored test site with its DoR score, confidence interval, categorical call, and `stable_class`, ready for spatial inspection in QGIS / ArcGIS alongside other ecological layers.

## Reproducibility

Stages 1–4 require a Google Earth Engine account (`earthengine authenticate`); Stage 2 publishes a GEE asset asynchronously and the pipeline waits for that export before Stage 3. Stages 5–7 are local-only and deterministic with `seed = 42`. The orchestrator (`run_v4_pipeline.py`) supports `--only`, `--skip`, and `--dry-run` to re-run any subset of stages from cached intermediate Parquet/CSV/JSON outputs.
