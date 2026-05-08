# Methods (v3)

## Overview

v3 reuses the v2 reference clouds (good and bad reference points sampled per parent with loss-region exclusion and dynamic buffering) and changes the **scoring rule** that turns those clouds into a categorical call. The motivating problem with v1/v2 was that a sizeable fraction of sites landed in the `indistinguishable` middle even when the good and bad clouds were well-separated, because the median-based score smoothed over the local geometry of the embedding clouds. v3 introduces a stepwise decision pipeline built around a **k-nearest-neighbour (kNN) cosine score**, calibrated thresholds, a deadband, and an effect-size gate, validated by **within-parent 5-fold cross-validation**.

Reference sampling and feature extraction are unchanged — v3 reads the same v2 references parquet (`strategy = random_100`, 100 good + 100 bad per parent). All differences are downstream of feature extraction.

## Earth-observation inputs and reference sampling

Identical to v2:

- AlphaEarth Annual Composite V1 (64-band, 10 m) for embeddings;
- ESA WorldCover v200 (class 40 = cropland, 50 = built, 80 = water) plus loss-region exclusion (HABLOSS crop/built/building-loss trends, GLAD, GLC-FCS30D, WorldCereal);
- Dynamic per-parent buffering (1 → 8 km), `random_100` strategy: 100 good + 100 bad per parent;
- Cosine distance throughout, on raw 64-band embeddings (no whitening, no PCA).

## Scoring functions

v3 computes two scores for each test site, both expressed in the same 0–1 frame ("near 1 = good-like = recovering"):

### Median score (`dor_median`) — diagnostic in v3

The same robust median pairwise score used in v1/v2:

```
s_med = median_cos(x_obs, bad) / (median_cos(x_obs, good) + median_cos(x_obs, bad))
```

Retained as a diagnostic for continuity with v1/v2 results.

### kNN-margin score (`dor_knn`) — primary in v3

Instead of the median over *all* references, the kNN score uses only the **k = 5 nearest** good neighbours and the **k = 5 nearest** bad neighbours of the test-site embedding:

```
s_knn = mean_kcos(x_obs, bad) / (mean_kcos(x_obs, good) + mean_kcos(x_obs, bad))
```

Both v1's `dor_median` and v3's `dor_knn` answer the same question — *is the site closer to good or bad?* — but they answer it at different scales of the cloud:

- **`dor_median`** smooths over the entire reference cloud. Robust to noise, but blurs the call when the cloud is multimodal: a site that is near a *part* of the good cloud is reported as only weakly good-like.
- **`dor_knn`** zooms in on the local neighbourhood. A site that lands inside a tight cluster of good references gets a strongly good-like score even when other parts of the good cloud are far away. This matches the ecological reality that a parent's "natural state" is rarely a single point in feature space — it is several plausible expressions of natural cover, and a recovering site only needs to look like *one* of them.

A scorer-comparison sweep over six variants and a k-sweep across `k = 1–75` confirmed that:

- Performance peaks at **k = 5–7** and degrades monotonically toward the full-pool median as k grows.
- Cosine and Euclidean distance give near-identical AUC and balanced error; cosine gives consistently lower Brier (better-calibrated scores), so cosine is preferred.
- Mean aggregation of the k nearest distances marginally outperforms median at small k.
- Increasing the reference pool size from n = 40 to n = 200 gives the median scorer essentially no gain (AUC 0.903 → 0.912), confirming that the bottleneck is the aggregation method, not sample size.

**v3 default: `k = 5`, mean cosine distance, primary score = `dor_knn`.**

## Threshold calibration

v3 fits operating thresholds from the labelled reference data instead of using the natural midpoint 0.5. For each scoring rule we:

1. Compute leave-one-out (LOO) scores within each parent (each ref scored against the other 199 of its own parent).
2. Pool LOO scores across all 158 parents.
3. Pick the threshold that **maximises Youden's J** — the cut-off that jointly maximises the true-positive rate (good called recovering) and minimises the false-positive rate (bad called recovering).

Calibrated values (`random_100`, 158 parents):

- `t_med = 0.4728` — median scorer;
- `t_knn = 0.4859` — kNN scorer.

Both sit slightly below 0.5, reflecting a mild good-side bias in the embedding geometry. **Per-label calibration** (separate fits for `built_loss` and `crop_loss`) was tested and discarded: the change in overall error was < 0.1 percentage points, and the per-label fits redistributed errors between labels rather than reducing them. The pooled threshold is used.

A sample-size sanity check refit the threshold using only the 80 fold-training references (rather than the full 199-ref LOO pool); per-fold drift was ≤ ±0.012 and aggregate validation metrics were unchanged, so the threshold is not biased by the fitting / operating sample-size gap.

## Stepwise decision pipeline

Each test site is classified by a stepwise rule that adds one safeguard at a time. The categorical outcome is one of {`recovering`, `degraded`, `indistinguishable`, `no_data`}. **Step 4 with default hyperparameters is the operational v3 default.**

| Step | Scorer | Rule | Operating point |
|---|---|---|---|
| Baseline | median | score ≥ 0.5 → recovering, else degraded | natural midpoint |
| Step 1 | median | score ≥ `t_med` → recovering, else degraded | calibrated threshold |
| Step 2 | median | 95 % CI entirely outside a deadband around `t_med` | + deadband |
| Step 3 | median | step 2 + point score exceeds threshold by `delta` | + effect-size gate |
| **Step 4** | **kNN** | **kNN 95 % CI entirely outside deadband + effect-size gate** | **`t_knn`, deadband, delta** |

Defaults: deadband half-width `hw = 0.05`, effect-size gate `delta = 0.03`, `k = 5`.

**Why these safeguards matter ecologically:**

- The **calibrated threshold** corrects the slight asymmetry in the embedding space — under the natural midpoint 0.5, the baseline scorer is biased toward false-degraded calls.
- The **deadband** says: *don't return a confident call when the site sits within `hw` of the threshold*. A score of 0.51 vs 0.49 should not flip the verdict.
- The **effect-size gate** says: *don't return a confident call when the good and bad clouds for this parent are not appreciably separated*. If the references themselves don't establish a meaningful contrast, the site is reported `indistinguishable` rather than relying on a cut-off applied to a near-degenerate distribution.
- **Step 4 (kNN)** uses the local-neighbourhood score that handles multimodal good clouds correctly.

Confidence intervals are computed by **bootstrap resampling of the reference pools** (n = 400 in validation, n = 2,000 in production), giving each site a point estimate plus a 95 % CI on the DoR. The CI is the test the deadband and effect-size gate apply against.

## Validation: within-parent 5-fold

The validation question is: *if we held out some of a parent's own references and treated them as if they were test sites, would the v3 pipeline classify them correctly?*

For each of the 158 parents we split its 100 good and 100 bad references into 5 stratified folds (~20 per fold per state). For each fold the held-out probes are scored against the parent's remaining 80 + 80 retained references. This produces 158 × 200 = **31,600 probes** with known true labels.

**Probe-direction notation:**

- *good probe → degraded* = false-degraded error (a known-good ref wrongly called degraded);
- *bad probe → recovering* = false-recovering error (a known-bad ref wrongly called recovering).

### Aggregate results

| Step | false-degraded | false-recovering | abstain |
|---|---:|---:|---:|
| Baseline (`t = 0.5`) | 15.0 % | 4.7 % | ~17 % |
| Step 1: calibrated `t` | 11.1 % | 7.4 % | ~16 % |
| Step 2: + deadband | 6.2 % | 3.2 % | ~35 % |
| Step 3: + effect-size gate | 6.1 % | 3.2 % | ~35 % |
| **Step 4: kNN cosine** | **1.8 %** | **0.9 %** | **~40 %** |

Step 4 is the lowest-error operating point and the production default. It correctly classifies a known-good reference as recovering 97.3 % of the time it issues a confident call, and a known-bad reference as degraded 99.1 % of the time.

### Why **within-parent** validation, not leave-one-parent-out?

A leave-one-parent-out (LOPO) design would score each held-out reference against ~31,000 references from *other* parents, not against its own ~80 retained references. That inflates the bootstrap CI width by roughly a factor of 20 (proportional to √n) and forces steps 2–4 to abstain 100 % of the time — the deadband would never fit inside such a wide CI. LOPO was tried in preliminary analysis and produced misleading abstention rates. Within-parent k-fold is the structurally correct test for a per-parent reference framework: it validates the design that is actually deployed.

### Reliability and calibration

A reliability decomposition on the 31,600 probes (10 equal-frequency bins) attributes Step 4's Brier-score improvement (0.158 → 0.126) primarily to **refinement** (sharper, more decisive scores) rather than calibration. Both `dor_median` and `dor_knn` are slightly overconfident-good in the lower half of the score range and overconfident-bad in the upper half; the operational threshold sits near the crossover where both scorers are well-calibrated locally, so this does not undermine deployed accuracy.

## Hyperparameter sensitivity

A grid sweep over `hw` and `delta` (`k = 5` fixed) on the 31,600 probes shows that error decreases **monotonically** with both hyperparameters at the cost of higher abstention. There is no saddle or sweet spot — operating-point choice is a deliberate trade-off between error rate and the fraction of sites returned as `indistinguishable`.

| `hw` (deadband) | false-degraded | false-recovering | abstain |
|---:|---:|---:|---:|
| 0.02 | 14.6 % | 11.1 % | 6.9 % |
| **0.05** | **10.6 %** | **7.2 %** | **18.8 %** |
| 0.08 | 7.6 % | 4.7 % | 31.3 % |
| 0.12 | 4.9 % | 2.5 % | 46.1 % |

Suggested alternative operating points:

- **Lower abstention** — Step 3 (`hw = 0.05, delta = 0.03`): ~6.1 % / ~3.2 % error, ~35 % abstain;
- **Lower error** — Step 4 with `hw = 0.08`: ~0.7 % / ~0.3 % error, ~57 % abstain;
- **No abstention** — Step 1 (calibrated threshold, no deadband): ~11.1 % / ~7.4 % error, all sites classified.

## Reproducibility and outputs

All scoring is local (no Earth Engine) given the v2 references parquet. Random seeds are fixed (bootstrap seed = 42, fold seed = 42).

Key outputs:

- `test_site_dor_v3.csv` — 158 sites with `dor_knn` (primary), `dor_median` (diagnostic), 95 % CIs, and Step 4 categorical call;
- `test_site_knn5_scores.shp` — GIS-ready shapefile, ranked by `dor_knn` (rank 1 = most recovering);
- `within_parent_site_scores.csv` — 31,600 validation probes with per-step classifications;
- `within_parent_summary.csv` / `within_parent_per_parent.csv` — aggregated and per-parent metrics;
- `sensitivity_sweep.csv` — hyperparameter grid;
- `k_sweep_knn.csv`, `scorer_compare_summary.csv`, `reliability_dor.csv` — design-decision diagnostics.

## What changed from v2, and why it matters ecologically

v2 → v3 keeps the same reference clouds and the same ecological question, but switches the scoring rule from a single robust median to a **stepwise, calibrated, locally-sensitive** decision. The ecological consequence is that v3 is decisively better at correctly classifying both intact-natural and clearly-degraded sites (Step 4: 1.8 % / 0.9 % error), at the cost of explicitly marking a larger fraction of sites as `indistinguishable` (~40 %) when the references and the site geometry do not support a confident call. v3 trades blanket coverage for honest uncertainty — the sites it classifies as recovering or degraded are the sites where the underlying embedding geometry actually supports that verdict.

## Known caveats

1. **`stable_stable` parents are still excluded** — v3 inherits v2's reference design, which has no defined bad pool for them. Versions 4+ extend the framework to score these sites.
2. **Threshold drift across folds is small but non-zero** (≤ ±0.012). The deployed threshold is fitted globally; small per-region biases are possible but bounded well within the operating deadband.
3. **Abstention is feature, not bug.** A 40 % `indistinguishable` rate at Step 4 is the framework refusing to over-claim confidence on sites where the embedding evidence is genuinely weak. Operational users should treat `indistinguishable` calls as "needs more evidence" rather than "model failure".
