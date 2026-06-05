# Methods (v5)

## Overview

v5 keeps the v4 scoring framework — a per-parent Degree of Recovery (DoR) score that compares each test site's AlphaEarth embedding to its own good/bad reference clouds, with the kNN (k = 5) cosine score, bootstrap CIs, and per-class calibrated thresholds inherited from v3/v4 — and **redesigns the reference sampler and the choice of inclusion/exclusion buffer** that defines which pixels enter those clouds.

The motivating problem with v4 was twofold. First, v4 sampling imposed no minimum exclusion radius between a parent and its references: per-site inspection found "bad" references landing 3–40 m from the test-site centroid with near-identical embeddings (cosine distance 0.004–0.011). This is sampler pseudoreplication — it pulls the distance-to-bad term toward zero and inflates DoR — not geostatistical signal. Second, v4 references could be drawn from anywhere inside the search buffer, including pixels that are physically nearby but ecologically unrelated, and v4 had no principled basis for the buffer's inner and outer radii.

v5 addresses both with three sampler changes (ecoregion constraint, GHM attachment, a two-tier *selected* / *diagnostic* split), and then **derives the operational inner-exclusion and outer-ceiling radii from a normalised multi-objective optimisation** over five sampling goals rather than choosing them by hand.

The scoring math (kNN-5 cosine DoR, `m_b / (m_g + m_b)`, bootstrap CI, per-class threshold) is **unchanged from v4**. All differences in the v5 reference clouds come from the cleaner sampler and the calibrated buffer.

## Earth-observation inputs

Same backbone as v4: the **AlphaEarth** Annual Composite V1 (64 bands `A00–A63`, year 2024, 10 m) for embeddings; ESA WorldCover (class 40 = cropland, 50 = built, 80 = water) plus the v2 multi-year loss-exclusion mask (HABLOSS crop/built/building-loss trends, GLAD, GLC-FCS30D, WorldCereal) for the natural/degraded basis and good-pool masking. v5 additionally uses:

- **RESOLVE 2017 Ecoregions** — the polygon containing each parent, used to constrain the sampling region;
- **Global Human Modification (GHM) v3, 2022, all-threats band (`AA`), 90 m** — sampled at every reference point as an auxiliary covariate (`ghm_aa`).

## Stage 1 — v5 reference sampling

The v5 sampler (`sample_stable_references_v5.py` for `stable_stable` parents, `sample_candidate_references_v5.py` for `built_loss` / `crop_loss` parents) extends the v4 sampler in three ways. Class routing for the bad pool is unchanged from v4 (`stable_nature` → cropland ∪ built; `stable_crop` → cropland; `stable_built` → built; loss-site parents routed by their own loss type), and the good pool is unchanged (WorldCover ∉ {40, 50, 80}, masked by the loss-trend layer).

### 1. Ecoregion intersection

The sampling region is the parent's `buffer(MAX_R)` **intersected with the RESOLVE 2017 ecoregion polygon containing the parent**. References that fall within the buffer but across a major ecoregion boundary are rejected. This keeps references local both in distance and in ecological setting, removing pixels that are physically near but ecologically irrelevant. Each reference carries the parent's `eco_id`.

### 2. GHM attachment

Every reference point carries `ghm_aa` — the GHM 2022 all-threats value at that point, sampled natively at 90 m. This lets downstream analyses condition on, or diagnose against, the human-modification gradient that the AlphaEarth embedding compresses but does not isolate. (GHM is used **diagnostically only** — see Stage 3; it is never an optimisation target.)

### 3. Two-tier selected / diagnostic split

The sampler draws references within the v4 buffer-step ladder (1 → 1.5 → 2 → 3 → 5 → 8 km, ceiling 8 km) and then partitions them by distance to the parent:

- **`selected`** — the nearest `TARGET_PER_CLASS = 100` good + 100 bad references with `dist_m ≥ INNER_EXCLUSION_M`. These are the production-scoring references, structurally insulated from sub-pixel contamination and the inner portion of the variogram correlation range.
- **`diagnostic`** — up to `DIAG_PER_CLASS = 200` additional references per class drawn from the inner disk (`dist_m < INNER_EXCLUSION_M`). These are retained, tagged, and never used in production scoring, but they let the buffer analyses (Stage 3) characterise the contamination and autocorrelation gradient across the full 0–8 km range **without re-sampling**.

The sampler default `INNER_EXCLUSION_M = 4 km` defines the selected/diagnostic boundary; the operational buffer was subsequently refined to 3 km by the Stage 3 analysis. The output schema extends v4 with `eco_id`, `ghm_aa`, `selection`, `inner_exclusion_m`, and per-point `dist_m`. Sampling seed = 234.

## Stage 2 — AlphaEarth feature extraction

Identical to v4: sharded per-parent `reduceRegions` for reference points (`extract_stable_refs_alphaearth_v5.py`, `extract_candidate_refs_alphaearth_v5.py`), and `fc.map(reduceRegion)` over parent centroids for the test-site embeddings. Outputs are cached as Parquet (`v5_stable_refs_alphaearth.parquet`, `v5_candidate_refs_alphaearth.parquet`, `test_site_alphaearth_2024_*.parquet`). Because the v5 sampler exports diagnostic references out to 8 km, the stable-reference parquet is itself the full 0–8 km superset that the buffer analyses sweep over — no separate extraction is needed for the wider radii.

## Stage 3 — Buffer-width calibration (the v5 analysis)

The central v5 contribution: the inclusion/exclusion buffer is **not assumed**, it is chosen by a normalised multi-objective optimisation over the design space `inner ∈ {0, 0.5, 1, 2, 3, 4} km × outer ∈ {1 … 8} km`. Five sampling goals are each scored per `(inner, outer)` cell, per class, then combined.

### The five goals (axes)

1. **Contamination control.** The failure to remove is bad references that are *both physically close and embedding-near-identical* to the test site (the sub-pixel pseudoreplication diagnosed in v4). In the loss-site pool the closest bad reference's distance and its cosine distance are strongly correlated (Spearman ≈ +0.72), so this contaminating mass is concentrated within a few hundred metres. The axis is the fraction of that near-site mass removed by the inner radius; it saturates by ~300–500 m. (`min_distance_exclusion_sweep.py`, `dor_stability_sweep.py`.)

2. **Good–bad separability.** How cleanly the good and bad reference pools can be told apart, measured by leave-one-out (LOO) reference classification (the v4 calibration construction): each reference is scored with the DoR functional `m_b / (m_g + m_b)` against the *other* references in the buffer, scores are pooled per class, and the threshold maximising the **Matthews correlation coefficient (MCC)** is taken. We report **F1 and MCC** at that threshold (plus ROC AUC). Higher is better. (`separability_sweep.py`.)

3. **Spatial independence (autocorrelation).** Within-pool reference autocorrelation, measured as the mean pairwise AlphaEarth cosine similarity among the references a buffer retains, per parent, averaged per class. The empirical variogram of the references shows this similarity decaying with geographic separation and largely plateauing past ~3 km; a wider pool (and, to a lesser extent, a larger inner radius) lowers mean autocorrelation. Lower is better. (`spatial_autocorr_sweep.py`.)

4. **Interval tightness (confidence).** The median width of the per-site 95 % bootstrap DoR confidence interval. Tighter is better. (`buffer_extent_sweep.py`.)

5. **Sample retention.** The count of sites with a computable paired DoR at the buffer, with a hard floor disqualifying any cell that retains fewer than 85 % of the best cell's paired-site count. Higher is better. (`buffer_extent_sweep.py`.)

### GHM is a diagnostic, not an objective

The DoR is *expected* to correlate with the human-modification gradient — a site recovering within a more modified landscape genuinely reads as less recovered — so minimising `|ρ(DoR, GHM)|` would optimise away legitimate ecological signal. `ρ(DoR, GHM)` is therefore reported as a per-class descriptive diagnostic (Spearman and Pearson, with significance), **never optimised**. At the chosen buffer it is moderate and sensible (stable-natural ρ ≈ −0.23, loss-site ρ weak/non-significant), confirming the score tracks recovery rather than only ambient context.

### Combining the axes — two scales, deliberately not min–max

Each `(inner, outer)` cell is scored two complementary ways, weights 1.5 for contamination, separability, and spatial independence and 1.0 for confidence and retention in both.

**Absolute quality `D` (0–1).** Goals already on an interpretable [0,1] scale — contamination fraction, separability MCC, and spatial independence (defined as `1 − within-pool similarity`) — are kept on their *true* scale; only the arbitrary-unit goals (CI width in DoR units, retention as a site count) are min–max rescaled within the sweep. The five are combined by a weighted geometric mean,

```
D = ( ∏ q_i^{w_i} ) ^ ( 1 / Σ w_i )
```

The geometric mean still drives `D → 0` if any single goal collapses, but because the axes are absolute, `D` does **not** saturate at 1 — it tops out near **0.67**, held down by the genuinely modest spatial-independence term.

**Relative ranking `Z` (SD units).** Each goal is z-scored (mean 0, sd 1) across the swept cells and combined by a weighted sum. This spreads the cells over ~3 standard deviations and is the signal that actually discriminates the optimum from mediocre buffers; `Z` is the primary ordering used to pick the buffer.

Min–max desirability — rescaling every axis so its best observed cell = 1 — was rejected on purpose: it manufactures a near-perfect combined score (≈ 0.99) regardless of true quality, saturates so quickly that it cannot separate competing buffers, and hides genuinely weak axes (here, spatial independence) by stretching them to span [0,1]. The absolute `D` answers "how good is this buffer, really?"; the relative `Z` answers "which buffer is best?". They agree closely on the ordering (Spearman ≈ 0.94 between the two scores), which is the robustness check — the recommendation does not depend on the scoring convention. (`buffer_desirability.py`.)

### Result

The optimisation is run across all five classes — the three `stable_stable` sanity classes and the two operational loss classes (`built_loss`, `crop_loss`). The top-ranked cell is **inner = 3 km, outer = 8 km** (`Z = +0.63 SD`; absolute quality `D = 0.67`), on a broad plateau (20 cells within 0.3 SD of the top, spanning inner 0.5–4 km × outer 5–8 km). At the chosen buffer the goal qualities are: contamination removed 1.00, separability (MCC) 0.84, spatial independence 0.22, interval tightness 0.94, retention 1.00 — i.e. every goal is strong except spatial independence, which is intrinsically limited because nearby AlphaEarth references stay ~78 % similar regardless of buffer. The point estimate is nearly invariant to the outer ceiling beyond ~5 km (the k = 5 score uses only the nearest references), so the outer ceiling's real effect is on interval width and spatial independence, both of which favour the wider radius at no bias cost. A v4-reproducible fallback (references reach only 4 km) is **inner = 0.5 km, outer = 4 km** (`Z = +0.29 SD`, `D = 0.64`).

**Operational buffer: references with `3 000 m ≤ dist_m < 8 000 m`.**

## Stage 4 — Scoring each test site

Unchanged from v4. Each test site is scored against its parent's `selected` reference clouds (now drawn from the calibrated buffer) with the kNN-5 cosine DoR; uncertainty is a 95 % bootstrap CI; the categorical call (`recovering` / `degraded` / `indistinguishable` / `no_data`) is the per-class CI-vs-threshold rule from v4. The `stable_crop` and `stable_built` classes remain end-to-end sanity checks (the site sits in its own bad state, so a low DoR is expected).

## Validation behaviour at the chosen buffer

Per-class metrics at `3 km → 8 km` confirm the design:

- The **loss classes are the most separable of all** — `built_loss` MCC = 0.916, F1 = 0.961, ROC AUC = 0.992; `crop_loss` MCC = 0.911, AUC = 0.989 — above the stable sanity classes (MCC 0.76–0.83). The good/bad reference pools around real disturbance sites are cleanly distinguishable.
- The loss-site reference pools also carry the **lowest within-pool autocorrelation**, and their DoR is **not GHM-confounded** (built_loss ρ ≈ +0.20, crop_loss ρ ≈ +0.04, n.s.).
- Loss-site median DoR sits near 0.50–0.55 (vs 0.39–0.42 for the stable bad-state classes), which is ecologically correct: a loss site has *lost* its degraded cover and is partially recovering, so it reads as intermediate rather than fully degraded.

Crucially, adding the operational loss sites to the optimisation did **not** change the recommended buffer or its ranking.

## Reproducibility and outputs

Earth Engine stages (sampling, extraction) require `earthengine authenticate`; the Stage 3 analyses are local and deterministic given the cached Parquet (sampling seed = 234, bootstrap seed = 42). The buffer analyses read only the reference and test-site Parquet — no GEE re-extraction.

Key outputs:

- `v5_stable_refs_alphaearth.parquet`, `v5_candidate_refs_alphaearth.parquet` — reference clouds (0–8 km, with `selection`, `dist_m`, `ghm_aa`, `eco_id`);
- `buffer_desirability.csv` — every `(inner, outer)` cell with each axis and overall `D`;
- `separability_summary.csv`, `spatial_autocorr_summary.csv`, `buffer_extent_summary.csv`, `buffer_extent_ghm_corr.csv` — the per-class axis sweeps (all five classes);
- `v5/report/buffer_decision.html` — self-contained decision report;
- Figures (600-dpi PNG + vector PDF in `v5/plots/`): `buffer_desirability_heatmap` (six-panel hypercube), `buffer_axis_profiles`, `buffer_ghm_scatter` (per-class DoR-vs-GHM diagnostic), `separability_mcc_byclass`, `spatial_autocorr_byclass`.

## What changed from v4, and why it matters ecologically

v4 → v5 keeps the same per-parent scoring and the same ecological question but fixes the references the score is built on. The sampler now excludes the near-field pseudoreplicated references that artificially inflated DoR in v4, constrains references to the parent's ecoregion, and carries a human-modification covariate for honest diagnostics. The inclusion/exclusion buffer is no longer a hand-set radius but the outcome of an explicit five-goal optimisation, and that optimisation lands on a buffer (`3 km → 8 km`) that simultaneously removes contamination, maximises good/bad separability, minimises within-pool autocorrelation, tightens confidence intervals, and retains nearly all sites — verified across both the stable sanity classes and the operational loss sites.

## Known caveats

1. **The buffer optimum is a plateau, not a spike.** A broad region (inner 0.5–4 km × outer 5–8 km, 20 cells) is within 0.3 SD of the top ranking score; `3 km → 8 km` is the top-ranked cell and is reported as the single recommendation, but any cell in the plateau is defensible.
2. **The 8 km outer ceiling relies on the v5 diagnostic references.** The v4-era reference data reaches only 4 km; the `inner = 0.5 km, outer = 4 km` fallback is provided for reproduction on v4-era pools.
3. **GHM is reported, not corrected.** As in v2's autocorrelation diagnostics, the GHM correlation is surfaced to characterise the score, not removed from it — by design, since DoR should co-vary with human modification.
4. **Contamination thresholds inherit v2/v4 defaults** (loss-trend `−7`; the cosine < 0.05 "near-duplicate" cut used to define the contaminating mass). Sensitivity to the cosine cut has not been formally characterised, though the contamination axis saturates well before the operational inner radius regardless.
