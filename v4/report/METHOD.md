# v4 Recovery Classification — Stable-Pixel Extension

## Overview

v4 extends v3 to score `stable_stable` parent sites that v1–v3 architecturally
exclude. The v3 framework requires a parent-specific *bad* reference pool, and
the disturbance label (`built_loss` / `crop_loss`) determines what bad means.
For stable_stable parents the label asserts no transition occurred and gives
no bad pool, so v1–v3 simply drop them at the sampling stage.

v4 classifies each stable_stable parent's *current* state from multi-source
agreement, then routes the bad pool per class. The v3 stepwise classifier
(`dor_knn` k=5 cosine, deadband, effect-size gate) is reused unchanged; only
the thresholds are refit per stable_class.

---

## 1. Stable-class classification

For each stable_stable parent point, votes are sampled at SCALE=10 m from:

| Source | EE asset | Vote rule |
|---|---|---|
| WorldCover v100 (2020) | `ESA/WorldCover/v100` | 40→crop, 50→built, 80→unknown, else→nature |
| WorldCover v200 (2021) | `ESA/WorldCover/v200` | same as above |
| Dynamic World annual mode 2018–2024 | `GOOGLE/DYNAMICWORLD/V1` | 4→crop, 6→built, 0→unknown, else→nature |
| ESA WorldCereal `temporarycrops` (2021) | `ESA/WorldCereal/2021/MODELS/v100` | 1→crop, else→unknown |
| VIDA Combined buildings (per-country) | `projects/sat-io/open-datasets/VIDA_COMBINED/<ISO3>` | footprint within 30 m → built; absent → nature; shard missing → unknown |
| Microsoft Buildings (per-country) | `projects/sat-io/open-datasets/MSBuildings/<country_name>` | same vote rule as VIDA |

Country routing for the building shards: each parent point is joined to its
country via `ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM0")` (ISO3
in the `shapeGroup` property). VIDA shards are keyed by ISO3 directly; the
MSBuildings catalog uses country *names* with mixed conventions
(`UnitedStates`, `CzechRepublic`, …), so an explicit ISO3→name CSV must be
passed via `--msb-country-map`. Without that map, MSBuildings votes
`unknown` for every point and the classifier falls back on the other 5
sources.

Building footprints are temporal snapshots, not time series. For
`stable_stable` parents this is acceptable: the label asserts no transition
across 2018–2024, so the snapshot is a valid summary.

Decision: take the modal class. Require ≥`min_agree` (default 2) votes for
the winner; ties or low agreement → `ambiguous` (skipped). Final label is
`stable_class ∈ {nature, crop, built, ambiguous}`.

Run:
```bash
# raster sources only (fast; reasonable when no MSB country map is available)
python v4/scripts/classification/classify_stable_state.py --skip-buildings

# full multi-source with both building catalogs
python v4/scripts/classification/classify_stable_state.py \
  --msb-country-map v4/data/msb_country_map.csv --verbose
```

Output: `v4/data/stable_state_classification.csv`.

---

## 2. Reference sampling per stable_class

Good pool is identical for all classes (WorldCover ≠ {40, 50, 80}, with the
v2 loss-trend exclusion mask applied). Bad pool routes by `stable_class`:

| `stable_class` | Bad pool | Rationale |
|---|---|---|
| `nature` | WC ∈ {40, 50} (crop ∪ built) | A stable-nature site can degrade to either crop or built |
| `crop` | WC = 50 (built only) | **Sanity check** — included in final dataset; recovery framing implies stable-crop sites should score `degraded` since they have not transitioned toward nature |
| `built` | WC ∈ {40, 50} (crop ∪ built) | **Sanity check** — included in final dataset; same logic, stable-built sites should score `degraded` |

In the output parquet, `parent_label` is set to `stable_<class>` so v3's
downstream tooling can group/calibrate by it without schema changes.

Run (after classification):
```bash
python v4/scripts/sampling/sample_stable_references_v4.py \
  --classification v4/data/stable_state_classification.csv \
  --export
```

---

## 3. Threshold calibration (per stable_class)

`validate_steps_within_parent_v4.py` runs full-LOO Youden-J calibration
*restricted to each stable_class* and writes the per-class kNN thresholds to
`v4/data/calibrated_thresholds_v4.json`. The within-parent 5-fold validation
also reports a `step4_knn_v3t` column applying v3's pooled threshold
(`t_knn=0.4859`) for transfer comparison.

Run:
```bash
python v4/scripts/analysis/validate_steps_within_parent_v4.py \
  --refs v4/data/v4_stable_refs_alphaearth.parquet \
  --strategy random_100
```

Outputs:
- `v4/data/within_parent_site_scores_v4.csv`
- `v4/data/within_parent_summary_v4.csv`
- `v4/data/calibrated_thresholds_v4.json`

---

## 4. Test-site scoring

`score_test_sites_v4.py` mirrors v3's scorer but picks the kNN threshold
from `calibrated_thresholds_v4.json` based on each parent's `parent_label`.
It also writes a `category_knn_v3t` column applying v3's threshold for
transfer comparison.

```bash
python v4/scripts/analysis/score_test_sites_v4.py \
  --refs v4/data/v4_stable_refs_alphaearth.parquet \
  --thresholds v4/data/calibrated_thresholds_v4.json
```

Output: `v4/data/test_site_dor_v4.csv`.

---

## 5. Sanity-check expectation: stable_crop and stable_built

The framework asks "is this site recovering toward natural state?" A stable
*crop* or *built* site is correctly stable (it has not transitioned), but on
the recovery axis both classes sit near the bad pool by construction. The
expected `category_knn` for both stable_crop and stable_built test sites is
therefore `degraded`. They are kept in the final dataset to validate the
framework end-to-end.

If a large fraction of stable_crop or stable_built sites do not score
`degraded`, that flags one of:
1. The classifier mislabelled the parent (real class is nature, or wrong
   crop/built class).
2. The bad-pool sampling failed to populate enough WC pixels in the parent's
   buffer (check `n_bad` in `test_site_dor_v4.csv`).
3. The embedding does not separate the bad pool from nature for that biome —
   inspect the `dor_knn` distribution before reading the categorical output.

Note that for stable_crop, the bad pool is built only (WC=50). A stable-crop
site's embedding is *closer* to crop pixels than to either nature or built,
which is why a built-only bad pool is the right comparison: it asks "is this
site closer to natural-vegetation than to urbanisation?" — and the expected
answer is "no, it's still crop", landing it in the indistinguishable or
degraded bucket. Sites that score `recovering` here likely have meaningfully
shifted toward natural vegetation (real cropland abandonment).

---

## 6. Folder layout

```
v4/
  scripts/
    classification/classify_stable_state.py
    sampling/sample_stable_references_v4.py
    analysis/score_test_sites_v4.py
    analysis/validate_steps_within_parent_v4.py
  data/
    stable_state_classification.csv
    v4_stable_refs_alphaearth.parquet     (after EE export + AlphaEarth join)
    calibrated_thresholds_v4.json
    within_parent_site_scores_v4.csv
    within_parent_summary_v4.csv
    test_site_dor_v4.csv
  plots/
  report/METHOD.md
```

---

## 7. Outstanding work

- AlphaEarth embedding extraction for v4 stable refs (reuse
  `v1/scripts/extraction/extract_alphaearth_embeddings.py`
  pointed at the v4 export asset). The expected output parquet is
  `v4/data/v4_stable_refs_alphaearth.parquet` and the schema must match
  v2's `sampling_strategy_selected_points.parquet` (with a `strategy`
  column = `random_100`).
- Update `MSBuildings` / `VIDA` asset paths if the global aggregations
  used in `classify_stable_state.py` are not available in your project —
  fall back to per-country shards via `ee.data.listAssets`.
- Add reliability + sensitivity sweeps analogous to v3 once the validation
  numbers are stable.
