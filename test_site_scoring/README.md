# Test-site scoring — three land-change groups (buffer + ecoregion DoR)

Scores the test sites in `R/data/for_gee/samples_recover_w_ref_label.shp` with **both**
the v5 local-buffer Degree-of-Regeneration (DoR) and the ecoregion-percentile DoR,
restricted to three transition groups defined by the 2018→2024 land-cover change
(crop classes excluded):

| Group | 2018 → 2024 | n (unique sites) |
|---|---|---:|
| `stable_natural` | nature → nature | 747 |
| `stable_artificial` | artificial → artificial | 386 |
| `artificial_reversion` | artificial → nature | 115 |
| **total** | | **1,248** |

The shp now embeds the prior class in `lc_2018`, so the reversion group is derived
directly from the (`lc_2018`, `lc_2024`) pair — no external buildings/HABLOSS lookup.

## Two caveats this run fixes

1. **Year caveat (ecoregion scoring).** The original ecoregion FSCS references were
   sampled on the **2022** AlphaEarth vintage while test sites are scored on **2024**.
   Step 2 produces **2024** references for every target ecoregion:
   - *Path A* (`02a`): for ecoregions that already had a 2022 file, re-sample 2024 at
     the **same point coordinates** (clean year comparison, no FSCS).
   - *Path B* (`02b`): for ecoregions with no file, run FSCS at 2024.
   Step 4 reads **only** 2024 references and **raises** if a scored site's ecoregion has
   a 2022 file but no 2024 one (no silent vintage fallback). Every scored row carries
   `ref_vintage = 2024`.

2. **Buffer caveat (local DoR).** The v5 `selected` references were cut at inner = 4 km
   (the v4-reproducible fallback), inconsistent with the empirically-optimal
   **inner = 3 km / outer = 8 km** buffer from the desirability analysis. Step 3
   **re-selects** references at `3000 ≤ dist_m < 8000` from the full v5 parquets
   (`selected` ∪ `diagnostic`) — the 3–4 km refs already exist with embeddings, so no
   new Earth Engine ref extraction is needed.

## Cost controls (year-caveat extractions)

- Ecoregion references are capped at **5,000 points/ecoregion** (10× the scorer's
  `N_BASELINE = 500`). The legacy files held up to 3.4 M points but the percentile
  scorer only ever uses ~500 baseline + `MIN_REFS = 10` per pool, so the cap cuts the
  giants by 1–2 orders of magnitude with **no effect on the scores**.
- Path B processes ecoregions **smallest → largest** and emits a **30-min heartbeat**.

## Scripts (run in order)

| Script | EE? | What it does |
|---|---|---|
| `00_build_target_groups.py` | no | shp → `target_groups.parquet` (group, parent_id, eco_id, has_embedding) |
| `01_extract_missing_2024_parents.py` | yes | 2024 embeddings for the 281 sites with no cache, from local shp geometry; tags eco_id |
| `03_score_buffer_dor.py` | no | local-buffer DoR re-selected at 3–8 km (the 894 sites with buffer refs) |
| `02a_eco_refs_2024_resample.py` | yes | 2024 eco refs re-sampled at existing 2022 coords (Path A) |
| `02b_eco_refs_2024_fscs.py` | yes | 2024 eco refs via FSCS for ecoregions with no file (Path B, detached) |
| `06_extract_buffer_refs_281.py` | yes | buffer refs for the 281 sites not in the original DoR sampling (local geometry, reuses v5 sampler), so they get buffer DoR too |
| `04_score_ecoregion_dor.py` | no | ecoregion-percentile DoR vs 2024 refs (all 1,248 sites with an embedding) |
| `05_combine_and_summarise.py` | no | join both scores, per-group summary + buffer-vs-eco agreement |

After `06` lands, **re-run `03`** — it unions the step-6 refs (keyed by PLOTID) and the
step-1 embeddings, so the previously buffer-unscoreable 281 sites get scored too.

`03` (buffer) needs no extraction and can run immediately. `04` (ecoregion) needs `02a`
+ `02b` to finish first.

## Coverage notes

- Buffer DoR scores only sites with a `parent_id` (⇒ v5 buffer refs) and ≥1 good and
  ≥1 bad ref in 3–8 km: **894 sites**. The 281 sites with no buffer refs are
  ecoregion-scored only.
- 4 of the 281 newly-extracted sites are unscoreable (1 null-band 2024 pixel, 3 outside
  any RESOLVE ecoregion); all are `stable_natural` controls.

## Outputs (`data/`)

- `target_groups.parquet` — site manifest
- `test_site_alphaearth_2024_missing.parquet` — 281 new parent embeddings
- `ref_samples_eco{id}_2024.parquet` — 2024 ecoregion references (Path A + B)
- `test_site_dor_buffer_3_8km.csv` — buffer DoR
- `test_site_dor_ecoregion_2024.csv` — ecoregion percentile DoR
- `test_site_scores_combined.csv` — both, joined, per site
