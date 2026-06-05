# Handover — interactive site-inspection UI for v5 DoR scatters

## Goal

Let me click a point in any of the per-class DoR-vs-GHM scatter plots and see
*what that point actually is on the ground*, so I can sanity-check the score
against imagery rather than trusting the number alone. The intent is validation,
not reporting — a quick "look at this site" tool, not a polished product.

For each clicked site, surface:

1. **A Google Earth link** that opens centred on the site's lat/lon with a
   marker (e.g. `https://earth.google.com/web/@LAT,LON,1000a,0d,30y,0h,0t,0r`).
2. **A Wayback (Esri World Imagery time-series) hyperlink** for the same
   coordinates (e.g. `https://livingatlas.arcgis.com/wayback/?ext=LON,LAT,LON,LAT`
   — verify the current URL format).
3. **The site's numbers**: `parent_id`, class, **DoR** point estimate, **CI
   `[dor_lo, dor_hi]`**, **GHM**, and the per-class threshold call
   (`recovering` / `degraded` / `indistinguishable`) if computable.
4. The plotted point should highlight on selection so I can see *which* one
   I clicked when points overlap.

Keep it basic. A single self-contained HTML page (Plotly + a side panel) is
plenty; no build step, no React, no server unless trivially needed.

## Why this matters

The v5 buffer-decision report shows per-class DoR-vs-GHM scatters at the chosen
`3 km → 8 km` buffer (`v5/plots/buffer_ghm_scatter.{png,pdf}`). The figure
confirms the *distributions* (loss-site DoR near 0.5; weak GHM correlation in
loss classes; stable-natural ρ ≈ −0.23) but it cannot answer "is that point at
DoR 0.85 actually a recovering site, or a misclassified field?". The
interactive tool is the bridge between the score and the landscape.

## Inputs available — no new computation needed

All data is local Parquet/CSV. No Earth Engine call required to build the UI.

| What | File | Key columns |
|---|---|---|
| Per-site DoR + 95 % bootstrap CI at every `(inner, outer)` in the sweep, for all 5 classes | `v5/data/buffer_extent_per_site.csv` | `parent_id`, `parent_label`, `inner_m`, `outer_m`, `n_good`, `n_bad`, `m_g`, `m_b`, **`dor`**, **`dor_lo`**, **`dor_hi`** |
| Parent (test-site) lat/lon for the stable classes | `v4/data/test_site_alphaearth_2024_v4.parquet` | `parent_id`, `geo` (struct: `{type, coordinates: [lon, lat]}`) |
| Parent lat/lon for the loss classes | `v5/data/test_site_alphaearth_2024_candidate.parquet` | same schema |
| Per-parent median GHM (already used in the report scatter) | derived from `v5/data/v5_stable_refs_alphaearth.parquet` + `v5/data/v5_candidate_refs_alphaearth.parquet`, column `ghm_aa`; see `_load_parent_ghm()` in `v5/scripts/analysis/buffer_desirability.py` | `parent_id`, median `ghm_aa` |
| Per-class calibrated thresholds (for the categorical call) | `v4/data/calibrated_thresholds_v4.json` if present; otherwise compute from `separability_summary.csv` (`opt_threshold` per class at chosen buffer) | per-class `t_knn` |

The recommended buffer is **inner = 3 000 m, outer = 8 000 m**. Filter
`buffer_extent_per_site.csv` to that single cell and you get **1 703 sites**
across the 5 classes — exactly what the static scatter shows.

Quick load (Python, ~10 lines, mirrors the existing `plot_ghm_scatter`):

```python
import duckdb, pandas as pd
con = duckdb.connect()
ps  = pd.read_csv("v5/data/buffer_extent_per_site.csv")
ps  = ps[(ps.inner_m == 3000) & (ps.outer_m == 8000)].dropna(subset=["dor"])
geo = con.execute("""
    SELECT CAST(parent_id AS VARCHAR) parent_id,
           geo.coordinates[1] AS lon, geo.coordinates[2] AS lat
    FROM read_parquet([
      'v4/data/test_site_alphaearth_2024_v4.parquet',
      'v5/data/test_site_alphaearth_2024_candidate.parquet'])
""").df()
df = ps.merge(geo, on="parent_id", how="left")
# GHM: import _load_parent_ghm from buffer_desirability and map onto parent_id
```

## Reuse cues

- `v5/scripts/analysis/buffer_desirability.py` →
  - `_load_parent_ghm()` (lines ~219–241) — already reads stable + candidate GHM into a `{parent_id: ghm}` dict.
  - `plot_ghm_scatter()` (~lines 446–520) — Matplotlib implementation of the *static* version; the layout (3 stable + 2 loss panels), class colours (`fst.CLASS_COLORS`), and class labels (`fst.CLASS_LABELS`) all live in `v5/scripts/analysis/figstyle.py`. Match these so the interactive view is visually consistent.
- `v5/scripts/analysis/figstyle.py` → palette + class labels.

The new tool should NOT modify those files. It should consume their outputs.

## Suggested design (pick what's simplest)

- **Option A (simplest, recommended):** single self-contained `.html` written
  by a Python script using **Plotly**. Plotly's `click_event` (or `customdata`
  + a tiny JS callback in `plotly.figure_factory.create_scatterplotmatrix`)
  surfaces the selected point's payload; render a side panel via plain JS that
  fills in the parent_id, scores, GHM, and the two map links. Output:
  `v5/report/buffer_inspector.html`. Open in any browser, no server.
- **Option B:** Streamlit / Dash if you prefer Python callbacks; spin up
  locally. More flexible but needs a running process. Only do this if Option A
  feels limiting.

Per-point payload to embed in the figure: `{parent_id, parent_label, lat, lon,
dor, dor_lo, dor_hi, ghm, n_good, n_bad}`.

URL templates (verify against the live tools when wiring):
```
https://earth.google.com/web/@{lat},{lon},1000a,0d,30y,0h,0t,0r
https://livingatlas.arcgis.com/wayback/?ext={lon-0.005},{lat-0.005},{lon+0.005},{lat+0.005}
```
(Wayback's `ext` is a bbox; the small ±0.005° box puts the site at frame
centre. If the link format has moved, capture the current `?` params from the
running tool.)

## Sanity checks for the new agent

- Confirm the click→panel→links round-trip works on **3 sites per class**
  spanning the DoR range (one near 0, one near 0.5, one near 1) before
  declaring it done.
- Compare a high-DoR built-loss site against Google Earth imagery: does it
  look like recovering vegetation? A low-DoR built-loss site: does it look
  like it stayed bare/built? Use these as your validation pass.
- Don't reinvent the static scatter (`buffer_ghm_scatter.png`); the
  interactive view is a *companion* to it.

## Out of scope

- Recomputing DoR. Use the cached `buffer_extent_per_site.csv` values.
- Touching the buffer-decision logic, the figures, or `v5_methods.md`.
- Adding new dependencies beyond Plotly (Option A) or Streamlit (Option B).
- Anything that needs Earth Engine — the project's `ee-gsingh` quota is in
  restricted mode and 429s on init.

## Project context

- **Recommended buffer** (the cell whose sites are inspected): inner 3 km →
  outer 8 km. Z = +0.63 SD ranking, D = 0.67 absolute quality.
- **Scoring** is dual: an absolute `D` on a true 0–1 scale and a relative
  `Z` in SD units (the report explains why min-max was rejected). The
  interactive UI doesn't need to know about either — it just shows the
  per-site numbers.
- **Five classes** with established colours: `stable_nature` (#009E73),
  `stable_crop` (#0072B2), `stable_built` (#E69F00), `built_loss` (#D55E00),
  `crop_loss` (#CC79A7). Use these so the interactive view matches the
  rest of the report.
- **Per-class median DoR** at the chosen buffer for orientation:
  stable_nature ≈ 0.56, stable_crop ≈ 0.42, stable_built ≈ 0.39,
  built_loss ≈ 0.55, crop_loss ≈ 0.50. Anything far from these is a
  candidate for spot-checking.
