# Degree of Recovery (DoR)

Satellite-based framework for scoring whether RECOVER restoration test sites
are **recovering/regenerating**, **degraded**, or **indistinguishable**,
using 64-dimensional AlphaEarth spectral embeddings and human-labelled
reference points.

This is a working research repository: `v1`–`v5` are successive method
iterations, kept for provenance. The **final, published result** — DoR
scores for all 1,248 test sites, computed two independent ways and
cross-checked — lives in the self-contained package below.

---

## Reproducing the paper's results

**→ [`paper_release/`](paper_release/)**

Ships its own README, source (`src/degree_of_recovery`), the 8 pipeline
scripts, a small sample dataset so it runs out of the box with no Earth
Engine account, and instructions for downloading the full reference set to
reproduce the exact published coverage.

- **Local-buffer DoR** (`dor_knn`) — compares each site's embedding to
  intact/degraded reference points sampled in a 3–8 km annulus around it
  (v5 sampler and buffer geometry).
- **Ecoregion-percentile DoR** (`pct_dor`) — ranks each site against a
  reference distribution sampled across its whole RESOLVE ecoregion.

Sites are split into three 2018→2024 land-cover transition groups (cropland
excluded): `stable_natural` (747), `stable_artificial` (386),
`artificial_reversion` (115).

See [`paper_release/docs/METHODS.md`](paper_release/docs/METHODS.md) for the
full method write-up.

---

## Scoring method (common to all versions)

For each test site, a DoR score compares its embedding to two pools of
reference points: **good** refs (intact/recovering vegetation) and **bad**
refs (degraded state — built or crop cover, matching the site's transition
class).

```
dor_knn = mean_cos(x_obs → k nearest bad refs)
        / (mean_cos(x_obs → k nearest good refs) + mean_cos(x_obs → k nearest bad refs))
```

Score → 1: closer to bad refs (recovering/regenerating signal). Score → 0:
closer to good refs (degraded signal). k = 5, cosine distance, bootstrap CI,
per-class calibrated thresholds.

| Output | Rule |
|---|---|
| `recovering` / `regenerating` | 95 % bootstrap CI entirely above threshold |
| `degraded` | 95 % bootstrap CI entirely below threshold |
| `indistinguishable` | CI straddles the threshold |

What changes release to release is the **reference sampler and buffer
geometry**, not this scoring math — see each version's `report/METHOD.md`
(or `paper_release/docs/METHODS.md` for the final version) for what changed
and why.

---

## Repository layout

```
paper_release/          # ★ final, self-contained reproduction package (see above)
test_site_scoring/      # buffer + ecoregion DoR run across 3 land-change groups (1,248 sites)
                         # — working precursor to paper_release/

v1/                     # v1 production pipeline
v1-ecoregion/           # ecoregion-level reference sampling (FSCS) for v1
v2/                     # sampling-strategy comparison experiments
v3/                     # kNN cosine scorer (dor_knn), 158 built_loss/crop_loss sites
v4/                     # stable-pixel extension, per-class calibrated thresholds
v5/                     # redesigned reference sampler + calibrated buffer geometry
combined/               # combined v3 loss + v4 stable shapefile/plots

  Each vN/ follows the same shape:
  data/                 scored shapefiles and CSVs
  plots/
  report/METHOD.md       method + validation writeup (where present)
  scripts/
    sampling/            GEE reference point export
    extraction/          AlphaEarth embedding extraction
    analysis/             scoring and benchmarking
    reporting/            shapefile + plot generation

src/degree_of_recovery/  # shared scoring primitives (kNN score, bootstrap CI,
                          # classification) imported by v1-v5 and test_site_scoring/
scripts/                 # methods-doc build pipeline (paper_methods.md -> docx)
tests/                   # pytest unit tests for src/degree_of_recovery

scratch/                # local-only scratch (gitignored)
```

---

## Setup

Requires Python 3.11+, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run pre-commit install   # optional: code-quality hooks
```

**Key dependencies:** `earthengine-api`, `duckdb`, `pandas`, `numpy`, `geopandas`, `shapely`, `scikit-learn`, `matplotlib`, `tqdm`

**Google Earth Engine:** authenticate once with `earthengine authenticate`. Default project: `ee-gsingh`.

`paper_release/` has its own `pyproject.toml` and does not require Earth
Engine for the default (sample-data) run — see its README.

---

## Writing workflow

The methods write-up is markdown-first. `paper_methods.md` is the source of
truth; the docx is a regenerated export, **never** hand-edited.

- **Source:** `paper_methods.md` — edit this for all prose, table, and equation changes.
- **Figures:** `scripts/make_dor_results_figure.py` and `v4/scripts/reporting/make_supp_quality_figure.py` write PNGs into `v4/plots/`. The md embeds those PNGs by path.
- **Styling reference:** `scripts/reference.docx` — open in Word and edit to change fonts, margins, or heading styles. Content paragraphs in this file are ignored by pandoc; only style definitions are copied.
- **Build:** `python scripts/build_docx.py` regenerates the figures and writes `build/paper_methods.docx`. Use `--skip-figures` for prose-only rebuilds, `--open` to launch the result in Word.

For coauthor review, share `build/paper_methods.docx`; collect their comments and reconcile them back into `paper_methods.md`, then rebuild.

---

## Network storage

Large data files not tracked in git are stored at:
`P:\155020_recover\WP1\degree_of_recovery\`
(NINA network drive — `\\nina.no\Prosjekter\155020_recover\WP1\degree_of_recovery`)

---

## References

- AlphaEarth Satellite Embedding V1 (2024)
- ESA WorldCover 2021 v200 — land cover reference classification
- RESOLVE Ecoregions 2017
- Global Human Modification (GHM) v3, 2022
- RECOVER project — `projects/nina/RECOVER/samples_recover_w_ref_label`
