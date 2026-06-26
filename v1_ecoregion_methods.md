# Methods (v1-ecoregion)

## Overview

v1-ecoregion is an **ecoregion-level reference sampling layer** that sits upstream of the per-parent DoR scoring pipeline. Rather than drawing reference pixels from within a fixed distance buffer around each parent (the v4/v5 approach), v1-ecoregion first identifies the RESOLVE 2017 ecoregion each parent falls in, then densely samples that ecoregion using Feature Space Coverage Sampling (FSCS) on a 10 km grid. The resulting per-ecoregion reference clouds are the population from which the v5 sampler's ecoregion-constraint step ultimately draws.

The motivation is ecological: pixels within the same RESOLVE ecoregion share a biome-level habitat context (climate regime, vegetation structure, biogeographic history) that a purely distance-based buffer cannot guarantee. An 8 km buffer around a savanna parent may straddle an ecoregion boundary and pull in Mediterranean scrub or grassland pixels whose AlphaEarth embeddings are superficially similar but ecologically irrelevant as recovery references. Restricting the reference pool to the same ecoregion removes that source of ecological misattribution without tightening the spatial radius.

v1-ecoregion does **not** change the DoR scoring function (kNN-5 cosine, `m_b / (m_g + m_b)`, bootstrap CIs, per-class thresholds from v3/v4/v5) — it is a data-preparation stage only.

## Earth-observation inputs

- **AlphaEarth Annual Composite V1, year 2022** (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64 bands `A00–A63`, 10 m) — used both as the FSCS clustering covariates (first 10 bands) and as the extracted embedding for each sampled point (all 64 bands).
- **ESA WorldCover v2** — used to derive the `natural` label (class 40 = cropland, 50 = built, 80 = water are non-natural).
- **HABLOSS multi-year loss-trend layers** (v5 candidate-sampler exclusion mask) — applied on top of WorldCover to exclude pixels actively transitioning to cropland, built, or building-loss even where WorldCover has not yet reclassified them. Mirrors the exclusion used in the v5 `sample_candidate_references_v5.py` so the natural/non-natural definition is identical across both pipelines.
- **RESOLVE 2017 Ecoregions** (`RESOLVE/ECOREGIONS/2017`) — provides the sampling polygon for each ecoregion and the `eco_id` label attached to every output point.

The 2022 AlphaEarth vintage is used here (rather than 2024 as in v5) because 2022 is the most recent full-year composite available at the time the v1-ecoregion layer was built. The embedding space is stable across annual composites; any vintage offset is small relative to inter-ecoregion variation.

## Sampling design

### Ecoregion selection

The target ecoregions are those that **contain at least one DoR loss-site parent** (non-`stable_stable` samples in the `projects/nina/RECOVER/samples_recover_w_ref_label` asset). Each parent centroid is spatially joined to the RESOLVE ecoregion polygon it falls in via `ecoregions.filterBounds(centroid).first()` (a per-point polygon lookup, not `reduceToImage` — the latter renders empty at 10 m scale). The union of matched `ECO_ID` values gives the working ecoregion set. The `--skip_crop` flag restricts this further to ecoregions containing at least one `built_loss` parent (i.e. excludes ecoregions whose parents are exclusively `crop_loss` / `stable_crop`).

### Natural / non-natural label

A sampled pixel is labelled `natural = 1` when:

1. Its ESA WorldCover class is **not** cropland (40), built (50), or water (80), **and**
2. It does **not** fall in the HABLOSS exclusion mask (pixels with a crop-loss, built-loss, or building-loss trend below threshold `−7` in the multi-year HABLOSS trend layers, intersected with a corresponding base-class mask from GLAD / GLC-FCS30D / WorldCereal).

`natural = 0` is the bitwise complement. This two-layer definition matches the good/bad pool logic in the v5 candidate sampler exactly: a pixel must be both WorldCover-clean and HABLOSS-clean to count as a good (natural) reference.

### Grid and FSCS

For each ecoregion:

1. **Grid.** The ecoregion's bounding box is divided into a 10 km `coveringGrid` using the ecoregion's native CRS. The grid is downloaded unpaginated for ecoregions with ≤ 2,000 cells and paginated in 2,000-cell blocks (with a `randomColumn` shuffle for stable ordering) for larger ones. Grid cells outside the ecoregion polygon are skipped cheaply by a `pixel_count < 10` guard, avoiding a `filterBounds` call that OOMs on GEE for large ecoregions (> ~3,000 cells).

2. **FSCS per cell.** Within each cell, FSCS runs on the first 10 AlphaEarth-2022 bands:
   - `N_INIT_POINTS = min(500, pixel_count)` random points are drawn to initialise KMeans.
   - `n_clusters = min(100, max(2, pixel_count // 5))` — adaptive so sparse cells yield fewer clusters rather than failing.
   - The clusterer assigns each pixel to a cluster; for each cluster the pixel with the **minimum Euclidean distance to the cluster centroid** in the 10-band space is selected (one sample per cluster). Three edge cases are guarded:
     - KMeans may return fewer populated clusters than requested → the centroid dictionary is built from `centroid_dict.keys()` (not a fixed `0..n-1` range).
     - Grouped-mean vectors can be incomplete/null under GEE load → an `_is_complete` filter drops incomplete groups before constructing the centroid image.
     - A cluster's difference band can be fully masked over the cell (all its nearest-centroid pixels fall outside the clipped region) → the min is null; `extract_samples` guards this with `ee.Algorithms.If(IsEqual(closest_val, None), empty, masked)`.

3. **Extraction.** A single `reduceRegions` call at the sampled points extracts: all 64 AlphaEarth-2022 bands, the `natural` label, and pixel longitude/latitude. `eco_id` and `cell_index` are appended in Python (setting these from a GEE `reduceToImage` on the ecoregion FC renders empty at 10 m point scale and is avoided).

   GEE call parameters: `scale = 10`, `tileScale = 4`, high-volume endpoint (`earthengine-highvolume.googleapis.com`). Sampling seed = 42.

4. **Parallelism.** Cells run concurrently in a `ThreadPoolExecutor` with up to `MAX_WORKERS = 20` threads. Each GEE call is retried up to 3 times with exponential backoff (2, 4, 8 s) on transient errors. Cells that time out (> 300 s) or fail all retries are logged as failed and do not block the rest.

## Durability and resume design

Storage is a CIFS network mount (`/data/P-Prosjekter2`) with aggressive mount options (`soft`, `retrans=1`, `actimeo=1`). Two bugs in the original sampler caused repeated data loss on the largest ecoregions (eco717, 94,401 cells) and were fixed before the production run:

**Bug 1 (CIFS deadlock).** The original sampler rewrote the entire checkpoint JSON inside a global lock on every cell completion. A truncating CIFS write (`open(f, "w")`) that hung in uninterruptible kernel I/O (`do_truncate`) held the lock indefinitely, deadlocking all worker threads. Fix: the checkpoint is written atomically (`json.dump` to a `.tmp` file, then `os.replace`) with the file I/O **outside** the lock (only the done-set snapshot is taken under the lock), and checkpointing is batched (every `CHECKPOINT_EVERY = 25` cells) to reduce CIFS write frequency.

**Bug 2 (resume data loss).** The original sampler accumulated all rows in an in-memory DuckDB buffer and wrote the parquet only once, at the end. On resume, the buffer started empty — previously extracted rows (not yet flushed) were lost, and the final `COPY` overwrote the parquet with only the current session's subset. Fix: each cell's rows are written immediately to an atomic per-cell parquet part (`data/ref_samples_eco{id}_parts/cell_{idx}.parquet`) before the cell is marked done (done ⇒ data on disk). On resume, the done-set is seeded from the checkpoint, on-disk parts, and `DISTINCT cell_index` values already in the consolidated parquet, so only missed cells are re-extracted. The final consolidation merges `existing parquet ∪ all parts` with `DISTINCT ON (geo)` deduplication.

Both fixes were validated on eco717 (the hardest case, previously producing zero data across four interrupted runs). After the fix, eco717 completed in a single run: 94,401 cells processed, 684 productive, 65,561 rows, zero data loss (GEE point-count sum = parquet row count exactly).

## Data-loss audit

Of the 49 ecoregions processed before the bugs were fixed, **~43 parquets hold every productive cell with no loss** — the low apparent "coverage" (distinct cells ÷ grid total) is explained by the high legitimate skip-rate: the bounding-box grid covers far more area than the ecoregion polygon, and cells with < 10 AlphaEarth pixels inside the polygon are correctly skipped. Only 5 ecoregions lost data through Bug 2 (interrupted mid-run across sessions):

| eco | productive cells (log sum) | parquet distinct cells | cells lost |
|-----|---------------------------|------------------------|------------|
| 717 | ≥ 14,412 | 0 (no parquet) | all |
| 81  | ~5,270 | 3,307 | ~1,963 |
| 653 | ~5,095 | 4,392 | ~703 |
| 500 | ~1,504 | 1,148 | ~356 |
| 233 | ~940 | 731 | ~209 |

eco717 was re-run with the fixed code on a local ext4 disk to bypass the CIFS mount, then moved back to the project directory. The 5 affected ecoregions (717 complete; 81, 653, 500, 233 pending append-fill with the fixed sampler) are the only ones that require re-extraction.

## Output schema

Each `data/ref_samples_eco{eco_id}.parquet` contains one row per sampled point:

| column | description |
|--------|-------------|
| `A00` … `A63` | AlphaEarth-2022 embedding (64 floats) |
| `natural` | 1 = natural, 0 = transformed (WorldCover + HABLOSS definition above) |
| `longitude`, `latitude` | point coordinates (degrees) |
| `eco_id` | RESOLVE 2017 ECO_ID of the sampled ecoregion |
| `cell_index` | grid-cell index within the ecoregion (0-based) |
| `geo` | GEE geometry string (used as the dedup key in `DISTINCT ON (geo)`) |

## Reproducibility

GEE stages require `earthengine authenticate` (project `ee-gsingh`, high-volume endpoint). The pipeline is deterministic given the same GEE state and seed (42), with the caveat that KMeans / `randomPoints` under high-volume GEE load can place a given FSCS representative pixel up to ~665 m from its prior position while preserving point counts exactly (verified on eco266: 0 cells differ in count, ~0.5% of points shifted ≤ 665 m on a fresh re-run of the fixed code).

Run command:
```
python3 scripts/sampling/sample_reference_states.py --all --skip_crop --max_workers 40
```
Working directory: `v1-ecoregion/`. Python: `/home/geethen.singh/.pixi/envs/geo/bin/python3` (pixi `geo` env, `ee` 1.7.28). **Not** the conda env.

## Known caveats

1. **AlphaEarth 2022 vs 2024.** The FSCS clustering and extracted embeddings use the 2022 annual composite. v5 scoring uses 2024 embeddings for test-site features. The inter-year embedding shift is small but non-zero; downstream scoring should use the same vintage for references and test sites.
2. **Grid covers bounding box, not ecoregion polygon.** Cells outside the polygon are skipped by the `< 10 pixel` guard rather than by pre-filtering. This is correct but means the cell-index space is sparse: most indices in the range `[0, total_cells)` correspond to skipped cells.
3. **GHM not yet attached.** Unlike v5 references, v1-ecoregion points do not carry a `ghm_aa` covariate. If GHM diagnostics are needed for ecoregion-level references, a post-hoc extraction step is required.
4. **eco717 run on local disk.** eco717's parquet was produced on a local ext4 filesystem and copied back; all other ecoregions were produced on the CIFS mount. The data itself is identical in schema and format.
5. **5 ecoregions pending re-fill.** eco81, eco500, eco233, eco653 still have partial parquets (Bug-2 losses). The fixed sampler's `--force_resume` flag will append the missed cells without overwriting existing data.
