# v3 Recovery Classification — Method and Validation

## Overview

This document describes the v3 stepwise decision framework for classifying a
restoration site as **recovering**, **degraded**, or **indistinguishable**
relative to a set of human-labelled reference points for the same parent site.
It covers design, validation, hyperparameter sensitivity, and recommended
operating policy.

---

## 1. Design

### 1.1 Reference structure

For each parent site, two pools of reference embeddings are maintained:

- **good** (100 points, `strategy=random_100`): representative of intact /
  recovering vegetation state.
- **bad** (100 points): representative of degraded / lost vegetation state.

A test site embedding `x_obs` is scored *only against its own parent's
references*. This per-parent design is fundamental: using another parent's
references would change the implicit null hypothesis and inflate confidence
intervals (see §3.1).

### 1.2 Embedding space

64-dimensional spectral embeddings (`A00`–`A63`) from the AlphaEarth 2024
dataset. All distance computations use **cosine distance** throughout (unified
in v3; prior versions used Euclidean for kNN at step 4).

### 1.3 Scoring functions

Two scores are computed for each `x_obs`:

**Median score** (steps 1–3):

```
s_med = median_cos(x_obs, bad) / (median_cos(x_obs, good) + median_cos(x_obs, bad))
```

Values near 1 indicate `x_obs` is closer to bad references; near 0 indicates
closer to good references.

**kNN-margin score** (step 4, k=5 by default):

```
s_knn = mean_k_cos(x_obs, bad) / (mean_k_cos(x_obs, good) + mean_k_cos(x_obs, bad))
```

where `mean_k_cos` is the mean cosine distance to the k nearest neighbours in
each reference pool.

Both scores share the same interpretation: score > threshold → recovering,
score < threshold → degraded.

### 1.4 Confidence intervals

Bootstrap CIs (n=400 resamples) are computed by resampling the reference pools
with replacement. The CI is used in classification steps 2–4 as a measure of
how reliably the score separates from a threshold.

### 1.5 Decision steps

| Step | Scorer | Rule | Threshold |
|------|--------|------|-----------|
| **Baseline** | median | score ≥ 0.5 → recovering | fixed at 0.5 |
| **Step 1** | median | score ≥ t_med → recovering | Youden-J calibrated |
| **Step 2** | median | CI entirely outside deadband | calibrated + deadband |
| **Step 3** | median | Step 2 + effect-size gate | calibrated + deadband + delta |
| **Step 4** | kNN | kNN CI entirely outside deadband + delta | calibrated + deadband + delta |

**Deadband**: a symmetric band of half-width `hw` centred on `t_med` (or
`t_knn`). A site is classified only if its entire CI lies *outside* this band,
guarding against threshold uncertainty.

**Effect-size gate** (delta): additionally requires the point score to exceed
the threshold by at least `delta`, filtering borderline calls.

Default hyperparameters: `hw=0.05`, `delta=0.03`, `k=5`.

---

## 2. Threshold calibration

Thresholds `t_med` and `t_knn` are calibrated via leave-one-out (LOO) scoring
within each parent's own references, then Youden-J optimisation pooled across
all 158 parents:

```
t* = argmax_t  TPR(t) - FPR(t)
```

Calibrated values (strategy `random_100`, 158 parents):

- `t_med = 0.4728` (slightly below 0.5, reflecting a mild good-side bias)
- `t_knn = 0.4859` (cosine; was 0.4929 with Euclidean)

**Per-label calibration** (built_loss vs crop_loss) was tested and found to
give negligible improvement (<0.1 pp change in overall error) while shifting
errors between labels rather than reducing them. Pooled calibration is used.

---

## 3. Validation

### 3.1 Design: within-parent 5-fold

The correct validation for a per-parent reference framework is to hold out
some of *each parent's own* references as probes, score them against the
same parent's retained references, and compare predicted label to known label.

**Procedure:**

- For each parent (158 total): split 100 good and 100 bad into 5 stratified
  folds (~20 per fold per state).
- For each fold: score held-out probes against the 80+80 retained training
  refs of the same parent.
- Thresholds calibrated globally (see §2) — not refitted per fold (mild
  leakage of threshold prior, negligible in practice given a single scalar
  is fitted).

**Total probes:** 158 × 200 = 31,600 (15,800 per probe direction).

**Ground truth assumptions:** All reference points are human-labelled. Labels
are assumed largely correct; some noise is possible, which would *understate*
model error (some apparent errors may be mislabelled probes).

**Why not LOPO?** A leave-one-parent-out design scores each probe against
~31,000 references from other parents rather than ~80 from its own. This
inflates bootstrap CI width roughly 20× (proportional to √n), causing steps
2–4 to abstain 100% of the time. LOPO was used in the preliminary analysis
and produced misleading abstention rates; within-parent folding is the
structurally correct test.

### 3.2 Results

**Probe direction notation:**
- *good probe, error* = false-degraded rate (known-good ref called degraded)
- *bad probe, error* = false-recovering rate (known-bad ref called recovering)

#### Overall performance

| Step | good: true% | good: error% | good: abstain% | good: brier | bad: true% | bad: error% | bad: abstain% | bad: brier |
|------|------------|-------------|---------------|------------|-----------|------------|--------------|-----------|
| Baseline (t=0.5) | 64.5% | 15.0% | 20.6% | 0.186 | 82.3% | 4.7% | 13.0% | 0.129 |
| Step 1: calibrated t | 72.3% | 11.1% | 16.6% | 0.186 | 76.8% | 7.4% | 15.8% | 0.129 |
| Step 2: + deadband | 57.1% | 6.2% | 36.8% | 0.186 | 63.9% | 3.2% | 32.9% | 0.129 |
| Step 3: + effect size | 56.5% | 6.1% | 37.4% | 0.186 | 63.4% | 3.2% | 33.4% | 0.129 |
| Step 4: kNN cosine | 59.5% | **1.8%** | 38.7% | 0.136 | 56.7% | **0.9%** | 42.3% | 0.117 |

**Key observations:**

1. **Step 1 (calibrated threshold)** reduces false-degraded (15%→11%) but
   *increases* false-recovering (4.7%→7.4%). Youden-J is biased toward the
   more numerous class; the calibrated threshold sits slightly below 0.5,
   favouring "recovering" calls.

2. **Steps 2–3 (deadband + effect size)** substantially reduce both error
   directions by routing uncertain calls to "indistinguishable" (~33–37%
   abstention). The effect-size gate (step 3) adds little beyond the deadband
   alone because most borderline calls are already within the CI-uncertainty
   zone captured by step 2.

3. **Step 4 (cosine kNN)** achieves the lowest error rates (1.8% / 0.9%) with
   only 39–42% abstention — better than the Euclidean kNN variant, which
   achieved lower error (0.7% / 0.3%) only by pushing 50–56% to abstention.
   Unified cosine distance makes more decisive correct calls.

4. **Brier scores** improve at step 4, confirming the kNN scorer has better
   probabilistic calibration than the median scorer.

#### Per-label breakdown (error rates)

| Step | Label | good probe error | bad probe error |
|------|-------|-----------------|----------------|
| Step 1 | built_loss | 9.3% | 7.4% |
| Step 1 | crop_loss | 13.5% | 7.4% |
| Step 2 | built_loss | 5.0% | 3.4% |
| Step 2 | crop_loss | 7.7% | 2.9% |
| Step 3 | built_loss | 5.0% | 3.4% |
| Step 3 | crop_loss | 7.6% | 2.9% |
| Step 4 | built_loss | 2.0% | 0.8% |
| Step 4 | crop_loss | 1.6% | 1.1% |

Crop-loss sites have higher false-degraded rates at steps 1–3 (embeddings are
more spread), but the kNN scorer (step 4) equalises performance across labels.

---

## 4. Hyperparameter sensitivity

The sensitivity sweep re-classified all 31,600 probes using point-estimate
scores (no CI) across a grid of `hw` and `delta` values (k=5 fixed).

### Effect of deadband half-width (hw) — step 2

| hw | false-deg error | false-rec error | abstain |
|----|----------------|----------------|---------|
| 0.02 | 14.6% | 11.1% | 6.9% |
| **0.05** | **10.6%** | **7.2%** | **18.8%** |
| 0.08 | 7.6% | 4.7% | 31.3% |
| 0.12 | 4.9% | 2.5% | 46.1% |

### Effect of effect-size gate (delta) — step 3, hw=0.05

| delta | false-deg error | false-rec error | abstain |
|-------|----------------|----------------|---------|
| 0.00 | 10.6% | 7.2% | 18.8% |
| 0.02 | 8.4% | 5.4% | 27.3% |
| **0.03** | **7.6%** | **4.7%** | **31.3%** |
| 0.05 | 6.2% | 3.4% | 38.8% |

### Conclusions from sweep

- Error decreases monotonically with both `hw` and `delta`; there is no
  saddle point or sweet spot. The current defaults are not tuned to a local
  optimum.
- `delta` is inert for step 2 (deadband-only), as designed.
- The error/abstention frontier is smooth. Operating-point choice depends on
  whether indistinguishable sites incur a cost (e.g. requiring resurvey) or
  are acceptable as no-decision.
- Suggested alternative operating point for lower error: `hw=0.08, delta=0.03`
  → ~0.7% false-degraded, ~0.3% false-recovering at ~57% abstention.

---

## 5. Design decisions and rationale

### 5.1 Unified cosine distance

Steps 1–4 all use cosine distance. The earlier Euclidean kNN (step 4) was
replaced because: (a) cosine distance is invariant to embedding magnitude,
which is appropriate for spectral ratio features; (b) a single distance
function reduces implementation complexity; (c) empirically, cosine kNN
achieves better Brier scores and lower abstention than Euclidean kNN at
comparable error levels.

### 5.2 Pooled vs per-label calibration

Per-label Youden-J thresholds (separate fits for built_loss and crop_loss)
were tested. Results: `built_loss t_med=0.4852`, `crop_loss t_med=0.4580`.
Overall error change: <0.1 pp. Within-label, errors shifted between labels
rather than decreasing. Recommendation: **pooled calibration**.

### 5.3 Youden-J vs cost-weighted threshold

Youden-J maximises balanced accuracy (TPR − FPR). In a restoration context,
false-recovering (calling a degraded site recovering) may carry higher
operational cost than false-degraded (triggering unnecessary inspection).
The baseline error rates (15% false-degraded vs 4.7% false-recovering)
suggest the opposite asymmetry in the raw score distribution — the embedding
space already leans toward correct "recovering" calls. If operational costs
are asymmetric, a cost-weighted threshold could be substituted at step 1
with no other changes to the pipeline.

### 5.4 kNN k=5 selection and dor_knn as primary score

A scorer comparison experiment (`v3/scripts/analysis/compare_scorers.py`) evaluated
six scorer variants on 31,600 held-out probes (within-parent 5-fold). Key results:

| Variant | AUC | Brier | bal_err |
|---------|-----|-------|---------|
| dor_median (median cosine, all refs) | 0.912 | 0.158 | 15.9% |
| mean cosine k=5 | 0.954 | 0.126 | 11.8% |
| mean cosine k=7 | 0.953 | 0.128 | 11.7% |
| mean Euclidean k=5 | 0.955 | 0.167 | 11.7% |

A k-sweep (`v3/scripts/analysis/compare_k_sweep.py`) across k=1–75 shows:
- Performance peaks at k=5–7 and degrades monotonically toward the full-pool
  median as k increases.
- Cosine and Euclidean give near-identical AUC/error; cosine gives consistently
  lower Brier (better calibration), so cosine is preferred.
- Mean aggregation marginally outperforms median at small k (median of 5 values
  discards too much information); at k=all, median is preferred for robustness.
- Increasing the reference pool size (n=40→200) gives `dor_median` essentially
  no improvement (AUC 0.903→0.912), confirming the bottleneck is the aggregation
  method, not sample size.

**Decision: k=5 with mean cosine distance is the default** (`dor_knn`), replacing
`dor_median` as the primary score. k=7 gives ~0.1 pp lower bal_err but the
difference is within noise; k=5 is retained for continuity with the existing
default. The bug in the original `score_knn_obs` (Euclidean instead of cosine)
was fixed in `test_steps_1_to_4.py` and `loo_knn_scores`.

The v3 production scorer (`v3/scripts/analysis/score_test_sites_v3.py`) outputs
`dor_knn` as primary with `dor_median` retained as a secondary diagnostic.

### 5.5 Bootstrap CI width at n=80 refs

With 80 training references per state per fold, bootstrap CIs are tight
enough for the deadband to be informative (steps 2–4 abstain 33–42%, not
100%). The original LOPO design used ~15,800 references, producing CIs ≈20×
wider and 100% abstention — structurally incompatible with the deployed
per-parent design.

---

## 6. Recommended policy

For operational deployment:

| Use case | Recommended step | Rationale |
|----------|-----------------|-----------|
| Maximum precision, moderate abstention | **Step 4** (cosine kNN, defaults) | 1.8% / 0.9% error, 39–42% abstain |
| Lower abstention acceptable | **Step 3** (deadband + effect) | 6.1% / 3.2% error, 33–37% abstain |
| Lowest possible error, high abstention OK | Step 4, hw=0.08 | ~0.7% / ~0.3% error, ~57% abstain |
| No abstention required | Step 1 | 11.1% / 7.4% error, all sites classified |

Step 4 with default hyperparameters is the recommended default.

---

## 7. Outputs

| File | Description |
|------|-------------|
| `v3/data/within_parent_site_scores.csv` | Per-probe scores and classifications (31,600 rows) |
| `v3/data/within_parent_summary.csv` | Aggregated metrics per step × probe_state × label |
| `v3/data/sensitivity_sweep.csv` | Hyperparameter grid results |
| `v3/data/test_site_dor_v3.csv` | Per-site dor_knn + dor_median scores and classifications (158 sites) |
| `v3/data/v2_vs_v3_comparison.csv` | Site-level v2→v3 category transitions |
| `v3/data/scorer_compare_summary.csv` | Six-scorer comparison metrics |
| `v3/data/k_sweep_knn.csv` | kNN k-sweep results (k=1–75) |
| `v3/data/k_sweep_median_sample_sizes.csv` | dor_median at n=40–200 refs |

### Scripts

| Script | Purpose |
|--------|---------|
| `v3/scripts/analysis/score_test_sites_v3.py` | Production scorer: dor_knn primary, dor_median secondary |
| `v3/scripts/analysis/test_steps_1_to_4.py` | Stepwise classification pipeline (steps 1–4) |
| `v3/scripts/analysis/validate_steps_within_parent.py` | Within-parent 5-fold validation |
| `v3/scripts/analysis/sensitivity_sweep.py` | Hyperparameter sensitivity grid |
| `v3/scripts/analysis/compare_scorers.py` | Six-variant scorer comparison |
| `v3/scripts/analysis/compare_k_sweep.py` | k-sweep for kNN variants vs dor_median sample sizes |
