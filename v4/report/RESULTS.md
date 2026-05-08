# v4 — Stable-stable parent scoring: results

## Why v4 exists

v1–v3 architecturally exclude `stable_stable` parents because they assume a
parent-specific bad pool tied to a *disturbance label*, which stable_stable
parents lack. v4 closes that gap by classifying each stable_stable parent's
state and routing it to a class-specific bad pool, then refitting kNN
thresholds per class.

## Pipeline

1. **Classify** each stable_stable parent via majority vote across 6
   sources (WorldCover v100/v200, Dynamic World 2024 mode,
   WorldCereal `temporarycrops`, VIDA buildings, MS Buildings) →
   `stable_class ∈ {nature, crop, built, ambiguous}`. A single DW year
   is sufficient because parents are stable_stable by construction;
   per-pixel mode over the full 6-year archive timed out during
   `sampleRegions`.
2. **Sample** good and bad reference points within parent buffers per
   stable_class:
   - `nature`: bad = WC{40, 50}
   - `crop`:   bad = WC{40} *(crop only — sanity check)*
   - `built`:  bad = WC{50} *(built only — sanity check)*
   - good (all classes): WC ≠ {40, 50, 80} and not in v2 loss-trend mask.
3. **Extract** AlphaEarth Satellite Embedding V1 (annual 2024, 64 bands)
   at every reference point.
4. **Validate + calibrate** kNN-5 thresholds per stable_class via
   within-parent 5-fold cross-validation + Youden-J.
5. **Score** test sites (parent centroids) using kNN-5 cosine ratio with
   per-class threshold + bootstrap CI → category.

## Classification counts

| stable_class | parents kept |
|---|---|
| nature | 788 |
| crop   | 528 |
| built  | 236 |
| ambiguous (skipped) | 56 |
| **total stable_stable considered** | **1608** |

Reference parquet covers 1552 parents (a few rows dropped during sampling
where the stratified sampler couldn't hit min targets within the largest
buffer).

## Calibrated thresholds (Youden-J on within-parent 5-fold)

| stable_class | t_knn (v4) | t_knn (v3 pooled, transfer) |
|---|---|---|
| stable_nature | 0.4861 | 0.4859 |
| stable_crop   | 0.4823 | 0.4859 |
| stable_built  | 0.4948 | 0.4859 |

Transfer-check: per-class thresholds are within ±0.01 of v3's pooled value.
Only **13 / 1471** scored sites change category under v3's pooled threshold
(7 built, 5 crop, 1 nature) — the framework transfers cleanly.

## Within-parent validation (steps 1–4)

Error% (false-classification rate among non-abstained sites) at the final
operating step (`step4_knn`):

| stable_class | good probe | bad probe |
|---|---|---|
| stable_nature | 1.5% | 1.0% |
| stable_crop   | 1.5% | 1.3% |
| stable_built  | 2.2% | 0.6% |

Abstain rates at step4_knn range 33–45% across class × probe combinations
(comparable to v3's pooled abstain rate). Full step-by-step numbers are in
`v4/data/within_parent_summary_v4.csv`.

## Test-site scoring

1471 sites scored, 74 skipped for missing good or bad refs (insufficient
candidates after stratified sampling). Median `dor_knn` is well-separated
between classes:

| stable_class | n | median dor_knn | mean | min | max |
|---|---|---|---|---|---|
| stable_nature | 713 | 0.574 | 0.595 | 0.043 | 0.982 |
| stable_crop   | 523 | 0.362 | 0.350 | 0.017 | 0.832 |
| stable_built  | 235 | 0.342 | 0.342 | 0.056 | 0.738 |

### Category breakdown (per-class threshold)

| stable_class | recovering | indistinguishable | degraded | no_data |
|---|---|---|---|---|
| stable_nature (n=713) | **357 (50%)** | 265 (37%) | 84 (12%) | 7 (1%) |
| stable_crop (n=523)   | 44 (8%)       | 194 (37%) | **285 (54%)** | 0 |
| stable_built (n=235)  | 11 (5%)       | 68 (29%)  | **156 (66%)** | 0 |

## Sanity-check outcomes

- `stable_built` ✓ — 66% degraded, 29% indistinguishable, only 5%
  recovering. Sites sit *in* built-up land and the bad pool is built; their
  embeddings cluster near bad references → degraded.
- `stable_crop` ✓ — 54% degraded, 37% indistinguishable, 8% recovering.
  Same logic with cropland.
- `stable_nature` — 50% recovering, 37% indistinguishable, 12% degraded.
  The new `stable_nature` parents (which v1–v3 dropped) are split roughly
  half-and-half between "clearly recovering" and "indistinguishable", which
  matches the framework's expectation that these include a mix of mature
  nature and slowly-changing systems.

A first iteration with the wrong bad-pool routing (`stable_crop` →
built-only, `stable_built` → crop+built) inverted the stable_crop sanity
check (69% recovering). Switching to **bad-pool = parent's own
non-natural state** — corroborated below — resolved it.

## Why the per-class own-state bad pool works

For a stable_crop parent, the test embedding sits *in* cropland. If the bad
pool is also cropland, the test embedding is close to bad and far from
natural good → low `dor_knn` → `degraded`. If the bad pool is built (the
v1 design, which was carried forward by mistake into the first v4 sampler),
the cropland embedding is closer to natural good than to built bad → high
`dor_knn` → spurious `recovering`. The same logic applies to stable_built.

For stable_nature there is no parent-state bad pool (the parent *is* the
good state), so we keep the v2/v3 union pool (crop ∪ built).

## Outputs

| Path | Contents |
|---|---|
| `v4/data/v4_stable_refs_alphaearth.parquet` *(not in repo, 26.7 MB)* | 300,879 reference rows (good+bad), 64 bands. Regenerate with `v4/scripts/extraction/extract_stable_refs_alphaearth.py`. |
| `v4/data/within_parent_summary_v4.csv` | Per-class step1–step4 error/abstain/Brier |
| `v4/data/within_parent_site_scores_v4.csv` *(not in repo, 59.1 MB)* | Per-site CV diagnostics. Regenerate with `v4/scripts/analysis/validate_steps_within_parent_v4.py`. |
| `v4/data/calibrated_thresholds_v4.json` | Per-class kNN thresholds |
| `v4/data/test_site_dor_v4.csv` | 1471 site scores + categories |
| `v4/data/test_site_dor_v4.shp` | Shapefile (Point, EPSG:4326), see README |
| `v4/data/test_site_dor_v4.shp.README.md` | SHP/CSV column map and provenance |
| `v4/plots/dor_distribution_by_class.png` | Histogram of dor_knn per class |
| `v4/plots/category_breakdown_by_class.png` | Stacked bars of categories per class |
| `v4/plots/threshold_transfer.png` | v4 per-class vs v3 pooled threshold |
| `v4/plots/validation_error_abstain.png` | error% / abstain% by class × step |

## Provenance

- Classification: `v4/scripts/classification/classify_stable_state.py`
- Sampling: `v4/scripts/sampling/sample_stable_references_v4.py`
- Embedding extraction: `v4/scripts/extraction/extract_stable_refs_alphaearth.py`
  (thin wrapper around v1 extractor)
- Test-site embedding cache: `degreeRecover/scripts/extraction/extract_test_site_embeddings.py`
- Validation/calibration: `v4/scripts/analysis/validate_steps_within_parent_v4.py`
- Scoring: `v4/scripts/analysis/score_test_sites_v4.py`
- Charts: `v4/scripts/reporting/make_v4_charts.py`
- Shapefile export: `v4/scripts/reporting/export_v4_shp.py`
- Embedding source: AlphaEarth Satellite Embedding V1 (annual 2024).
