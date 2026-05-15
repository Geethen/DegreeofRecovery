# `test_site_dor.shp` — metadata

Per-site Degree-of-Recovery (DoR) results for the RECOVER test sites,
ready to load into QGIS / ArcGIS / any OGR-compatible tool.

## File set

| File | Purpose |
|---|---|
| `test_site_dor.shp` | geometries (point) |
| `test_site_dor.shx` | spatial index |
| `test_site_dor.dbf` | attribute table |
| `test_site_dor.prj` | CRS definition |
| `test_site_dor.cpg` | character encoding (UTF-8) |

All five files must travel together.

## Geometry

- **Type**: Point (one feature per RECOVER parent site)
- **CRS**: EPSG:4326 (WGS 84, lon/lat in decimal degrees)
- **Source of coordinates**: centroid of the parent polygon in the source
  EE asset `projects/nina/RECOVER/samples_recover_w_ref_label`, computed
  with `Geometry.centroid(maxError=1)`.
- **Feature count**: 158
- **Bounding box**: lon ∈ [−120.89, 148.65], lat ∈ [−35.79, 67.17]

## Attribute schema

Field names are abbreviated to ≤ 10 characters because the shapefile DBF
format does not allow longer names.

| Field | Type | Range / values | Description |
|---|---|---|---|
| `parent_id` | str(20) | hex `system:index` | EE feature id of the source parent in `samples_recover_w_ref_label`. Zero-padded to 20 chars. |
| `par_label` | str | `built_loss`, `crop_loss` | Class of degradation the parent represents. Drives which WorldCover class becomes the "bad" reference state. |
| `category` | str | `recovering`, `indistinguishable`, `degraded`, `no_data` | Verdict from the 95 % bootstrap CI on `dor_med`: CI fully > 0.5 → recovering; fully < 0.5 → degraded; straddles 0.5 → indistinguishable; parent embedding NaN → no_data. |
| `n_good` | int | typically 100 | Number of natural-state reference points used to score this site. |
| `n_bad` | int | 2 – 100 | Number of degraded-state reference points. < 100 means the parent's 1 km buffer didn't contain enough qualifying WorldCover pixels of the bad class. |
| `applicable` | bool | True / False | True if the site has both ≥ 1 good and ≥ 1 bad reference. False sites have NaN scores. |
| `cos_d_good` | float | 0.006 – 0.609 | Cosine distance from the parent embedding to the centroid of the good references (lower = closer to natural). |
| `cos_d_bad` | float | 0.006 – 0.533 | Cosine distance from the parent embedding to the centroid of the bad references (lower = closer to degraded). |
| `dor_med` | float | 0.027 – 0.910 | **Primary DoR score**. Definition: `median(d_bad) / (median(d_good) + median(d_bad))` over cosine distances from the parent embedding to each reference. 1 = identical to natural cloud, 0 = identical to degraded cloud, 0.5 = midpoint. |
| `dor_ci_lo` | float | 0.025 – 0.870 | 2.5 % bootstrap percentile of `dor_med` (n_boot = 2000, references resampled with replacement per class). |
| `dor_ci_hi` | float | 0.031 – 0.923 | 97.5 % bootstrap percentile of `dor_med`. |
| `dor_norm` | float | −3.60 – 0.89 | Diagnostic. `1 − d_obs / d_inter` using medians; 1 at the good cloud, 0 when the obs is as far from good as the bad cloud is, < 0 if the obs sits even further from good than the bad cloud. |
| `dor_pct` | float | 0.0 – 1.0 | Diagnostic. Percentile rank of the parent's `dor_med` within the DoR distribution computed over the parent's own labelled refs (LOO). Values near 1 mean the parent looks more recovered than nearly all of the natural references; values near 0 mean it looks more degraded than nearly all the bad refs. |
| `dor_cos` | float | −21.6 – 0.96 | Diagnostic. Centroid-based cosine DoR. **Unstable** — wide negative tails are expected when the good cloud is multimodal (most of the parents). Use only as a sanity check alongside the median-based primary, not in isolation. |
| `lon` | float | decimal degrees | Parent centroid longitude (EPSG:4326). |
| `lat` | float | decimal degrees | Parent centroid latitude (EPSG:4326). |

## Suggested QGIS styles

- **Categorised on `category`** (recommended for a quick read):
  - `recovering` → green (`#2ca02c`)
  - `indistinguishable` → olive (`#bdbd22`)
  - `degraded` → red (`#d62728`)
  - `no_data` → grey (`#999999`)
- **Graduated on `dor_med`** (continuous view): blue→white→red diverging
  ramp centred on 0.5; mask `applicable = False` (or hide where
  `dor_med IS NULL`).
- **Error-bar inspection**: label features with
  `format_number(dor_med, 2) || ' ['  || format_number(dor_ci_lo, 2) || ', ' ||  format_number(dor_ci_hi, 2) || ']'`.

## Provenance

| Producer | Version |
|---|---|
| Reference asset (parents) | `projects/nina/RECOVER/samples_recover_w_ref_label` (1766 parents; 158 here are the `built_loss` + `crop_loss` subset that have a degraded counterpart) |
| Stratified ref sampling | `scripts/sampling/sample_reference_states.py` → asset `projects/ee-gsingh/assets/recover_reference_samples` (31,178 ref points) |
| Embedding source | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, year **2024**, 64 bands `A00..A63`, 10 m native scale |
| Per-site scoring | `scripts/analysis/score_test_sites.py` (n_boot = 2000, seed = 42) |
| This shapefile | `scripts/reporting/build_summary.py` |
| Generated | 2026-04-29 |

For the full pipeline and run order see [`PIPELINE.md`](../PIPELINE.md).

## Caveats

1. `category = no_data` (5 sites) — the parent centroid landed on a pixel
   with no AlphaEarth coverage (typically a narrow river or coastline).
   `applicable` is still `True` because the references exist, but every
   `dor_*` field is NaN.
2. Parents with `n_bad < 100` had buffers containing fewer qualifying
   degraded WorldCover pixels than requested. Their CIs are wider than
   the 100/100 sites — read `dor_ci_lo`/`dor_ci_hi` accordingly rather
   than the point estimate alone.
3. `dor_cos` should not be used standalone (see schema row above).
4. The shapefile fixes the year. To regenerate for a different year,
   re-run the pipeline with `--year <Y>` and re-export.
