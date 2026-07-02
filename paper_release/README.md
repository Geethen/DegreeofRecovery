# Degree of Recovery (DoR) — test-site scoring reproduction package

Reproduces the final test-site scoring results for the RECOVER restoration
paper: a satellite-based **Degree of Recovery / Regeneration (DoR)** score for
1,248 restoration test sites, using 64-dimensional AlphaEarth spectral
embeddings and two independent scoring methods that are cross-checked against
each other.

- **Local-buffer DoR** (`dor_knn`) — compares each site's embedding to
  intact/degraded reference points sampled in a 3–8 km annulus around it.
- **Ecoregion-percentile DoR** (`pct_dor`) — ranks each site against a
  reference distribution of points sampled across its whole RESOLVE
  ecoregion.

Sites are split into three land-cover transition groups (2018 → 2024,
cropland excluded): `stable_natural` (747), `stable_artificial` (386),
`artificial_reversion` (115).

## Quick start

```bash
pip install -e .           # or: pip install -e ".[gee]" for the optional extraction steps
python scripts/00_build_target_groups.py     # rebuilds outputs/data/target_groups.parquet
python scripts/03_score_buffer_dor.py        # local-buffer DoR (no download needed)
python scripts/04_score_ecoregion_dor.py     # ecoregion DoR (sample data — see below)
python scripts/05_combine_and_summarise.py   # joins both scores, prints per-group summary
python scripts/08_plot_local_vs_ecoregion.py # regenerates the comparison figure
python scripts/07_build_combined_inspector.py # interactive HTML site inspector
```

This runs **out of the box** on the sample data shipped in this repo — no
Earth Engine account, no download, no GIS setup beyond the Python packages
in `pyproject.toml`. `outputs/` already contains one worked example of every
file these scripts produce, generated from the shipped sample, so you can
diff your own re-run against it.

## Two run modes

The scripts are the same code either way — what changes is how much
reference data is on disk in `data/ecoregion_refs/`.

| Mode | What you need | What you get |
|---|---|---|
| **Sample run** (default) | Nothing extra — 5 ecoregions ship with the repo | Full buffer-DoR coverage (~1,160/1,248 sites); ecoregion-DoR for the ~100 sites in the sampled ecoregions |
| **Full reproduction** | Download the remaining ecoregion files (~191 MB, see `data/ecoregion_refs/DOWNLOAD.md`) | The exact published coverage: ecoregion-DoR for all scoreable sites |

Local-buffer DoR does **not** depend on `data/ecoregion_refs/` at all — it
already scores at full coverage in the sample run, because its reference
pools ship in full in `data/cached/`. Only the ecoregion-percentile method is
limited by which ecoregions are present. Missing coverage degrades
gracefully: unscoreable sites are reported as `NaN`/unscored, never silently
dropped or estimated, and the scripts print how many sites were affected and
why.

## Pipeline steps

| Step | Script | Requires GEE? | What it does |
|---|---|:---:|---|
| 0 | `00_build_target_groups.py` | no | Derives the 3 transition groups from the raw shapefile; builds the site manifest |
| 1 | `01_extract_missing_2024_parents.py` | yes | Extracts 2024 embeddings for the 281 sites with no cache *(output already shipped — optional)* |
| 2a | `02a_eco_refs_2024_resample.py` | yes | Re-samples 2024 ecoregion refs at existing 2022 coordinates *(optional, from-scratch only)* |
| 2b | `02b_eco_refs_2024_fscs.py` | yes | Samples 2024 ecoregion refs via FSCS for ecoregions with no prior file *(optional, from-scratch only)* |
| 3 | `03_score_buffer_dor.py` | no | Local-buffer DoR (k=5 kNN, cosine distance, 2,000-bootstrap CI) |
| 4 | `04_score_ecoregion_dor.py` | no | Ecoregion-percentile DoR against 2024 references |
| 5 | `05_combine_and_summarise.py` | no | Joins both scores; prints per-group summary + rank agreement |
| 6 | `06_extract_buffer_refs_281.py` | yes | Buffer references for the 281 sites outside the original sampling *(output already shipped — optional)* |
| 7 | `07_build_combined_inspector.py` | no | Interactive HTML: click a point → site score + satellite imagery links |
| 8 | `08_plot_local_vs_ecoregion.py` | no | Regenerates the local-vs-ecoregion comparison figure |

Steps 1, 2a, 2b, and 6 call Google Earth Engine live and require your own EE
project + access to the AlphaEarth annual embedding collection
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) and RESOLVE ecoregions asset. Their
outputs already ship precomputed in `data/cached/`, so **you do not need to
run them** to reproduce the scored result — they exist for full
from-scratch provenance only. Install with `pip install -e ".[gee]"` if you
want to re-run them.

**Recommended order for a full reproduction:** `00 → 03 → 04 → 05 → 07 → 08`.
Steps 1/2a/2b/6 are only needed if you want to regenerate the cached
extraction outputs from Earth Engine yourself.

## Scoring method

For each test site, a **Degree-of-Recovery score** compares its AlphaEarth
embedding to two pools of reference points: **good** refs (intact/recovering
natural land cover) and **bad** refs (degraded state — built-up or, for
buffer scoring, cropland, matching the site's transition class).

```
dor_knn = mean_cos(x → k nearest bad refs)
        / (mean_cos(x → k nearest good refs) + mean_cos(x → k nearest bad refs))
```

Score → 1: closer to bad refs (regenerating signal, since the site has moved
away from its bad-cover state). Score → 0: closer to good refs (degraded
signal). k = 5, cosine distance, 2,000-bootstrap 95% CI, per-class calibrated
thresholds. Classification uses the CI against the threshold:
`regenerating` (CI entirely above), `degraded` (CI entirely below),
`indistinguishable` (CI straddles).

See [`docs/METHODS.md`](docs/METHODS.md) for the full method write-up,
including the buffer-geometry decision (inner=3 km/outer=8 km) and the
ecoregion-percentile scorer.

## Repository layout

```
src/degree_of_recovery/   scoring primitives (kNN score, bootstrap CI, classification)
scripts/                  the 8 pipeline steps (00-08) + shared path config (_paths.py)
scripts/sampling/         reference-point samplers (FSCS ecoregion sampler, buffer sampler)
scripts/analysis/         plot style + interactive HTML report builders
data/raw/                 input shapefile (test sites with 2018/2024 land-cover labels)
data/cached/               small (~100 MB) precomputed embedding/reference caches
data/ecoregion_refs/       ecoregion reference points — 5-ecoregion sample ships here;
                           see DOWNLOAD.md for the full 317-ecoregion set
outputs/                   everything the scripts generate (data, figures, report)
docs/METHODS.md            full method write-up
```

## Input data provenance

- `data/raw/samples_recover_w_ref_label.shp` — RECOVER test-site sample
  points with 2018/2024 land-cover labels (`lc_2018`, `lc_2024`).
- AlphaEarth embeddings and RESOLVE ecoregion references are extracted from
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` and `RESOLVE/ECOREGIONS/2017` via
  Google Earth Engine (steps 1, 2a, 2b, 6 — outputs cached, see above).
- Human modification (GHM) values used in the interactive inspector come
  from the "all-threats" Anthropogenic Amplifier, 2022, 90 m product.

## License

See the parent repository / paper for licensing terms.
