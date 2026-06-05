"""Ecoregion-level FSCS reference sampling for degree-of-recovery (DoR).

This replicates the RECOVER ecoregion sampling strategy
(`RECOVER/scripts/extraction/sample_reference_points.py`) but adapts it to the
DoR setup in two ways:

  1. Extent — instead of sampling every South African ecoregion, we sample only
     the ecoregions that *contain the DoR loss-site parents* (the unique
     ecoregions of `SAMPLES_ASSET`). Downstream DoR scoring matches a parent to
     references from the same `eco_id`.

  2. Natural / non-natural label — RECOVER labels points with an SBTN/GHM/BII
     mask. Here we use the *DoR datasets* instead: a point is `natural` when its
     WorldCover class is not cropland (40) / built (50) / water (80) AND it does
     not fall in the upstream candidate-recovery (HABLOSS) exclusion mask used by
     the v5 candidate sampler. `natural = 0` is simply the inverse.

Sampling design (per ecoregion, unchanged from RECOVER):
  1. Lay a 10 km grid over the ecoregion polygon.
  2. Within each grid cell, run Feature Space Coverage Sampling (FSCS) on the
     2022 AlphaEarth embeddings: KMeans on the first 10 bands, then pick the
     pixel closest to each cluster centroid. n_clusters is adaptive
     (min(100, max(2, pixels/5))) so sparse cells yield fewer points.
  3. Single-shot `reduceRegions` at the sampled points extracts the 64 AlphaEarth
     bands + natural label + eco_id + lon/lat.
  4. Cells run in parallel; a JSON checkpoint makes the run resumable.

Output: one parquet per ecoregion at data/ref_samples_eco{eco_id}.parquet.

Usage:
  python v1-ecoregion/scripts/sampling/sample_reference_states.py --eco_id 40
  python v1-ecoregion/scripts/sampling/sample_reference_states.py --all
  python v1-ecoregion/scripts/sampling/sample_reference_states.py --all --n_clusters 200
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import duckdb
import ee
from tqdm.auto import tqdm

# ── Constants ───────────────────────────────────────────────────────
PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"
ECOREGIONS_ASSET = "RESOLVE/ECOREGIONS/2017"

GRID_SCALE = 10_000          # 10 km cells (matches RECOVER eco81 density)
N_CLUSTERS = 100             # FSCS clusters per cell
N_INIT_POINTS = 500          # random init points for KMeans
FSCS_BANDS = 10              # first N embedding bands for clustering (speed)
SCALE = 10                   # metres (AlphaEarth native)
SEED = 42
MAX_WORKERS = 20
CELL_TIMEOUT_S = 300        # max seconds per cell before treating as failed

# WorldCover transformed classes (excluded from "natural")
WC_CROP = 40
WC_BUILT = 50
WC_WATER = 80

# HABLOSS exclusion (upstream candidate-recovery areas) — mirrors the v5
# candidate sampler so the natural mask uses the same DoR datasets.
CROP_LOSS_THR = -7
BUILT_LOSS_THR = -7
BUILDING_LOSS_THR = -7

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── GEE helpers ─────────────────────────────────────────────────────
def init_gee(project=PROJECT):
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=project)
    print(f"[OK] GEE initialized (project={project})")


def retry_gee(func, max_retries=3, backoff=2):
    """Call func(), retry on failure with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff ** (attempt + 1)
            print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {e}")
            time.sleep(wait)


def parent_ecoregion_ids():
    """Distinct RESOLVE ECO_IDs containing the DoR loss-site parents.

    Parents are the non-`stable_stable` samples in SAMPLES_ASSET; we take each
    parent centroid, find the ecoregion it falls in, and return the unique set.
    """
    parents = (
        ee.FeatureCollection(SAMPLES_ASSET)
        .filter(ee.Filter.neq("r", "stable_stable"))
        .map(lambda ft: ft.setGeometry(ft.geometry().centroid(5)))
    )
    ecoregions = ee.FeatureCollection(ECOREGIONS_ASSET)

    # Spatial join per parent: reduceToImage on the global ecoregion FC renders
    # empty at point scale, so match each parent to the polygon it falls in.
    def tag(ft):
        match = ecoregions.filterBounds(ft.geometry()).first()
        return ft.set(
            "eco_id",
            ee.Algorithms.If(match, ee.Feature(match).get("ECO_ID"), -1),
        )

    ids = parents.map(tag).aggregate_array("eco_id").distinct().getInfo()
    return sorted({int(i) for i in ids if i is not None and int(i) >= 0})


# ── Natural mask (DoR datasets: WorldCover + HABLOSS exclusion) ─────
def build_exclusion_mask(include_buildings_loss=True):
    """Upstream candidate-recovery (HABLOSS) exclusion — mirrors v5 candidate
    sampler. Pixels here are degrading crop/built/buildings and are *not*
    natural. Returns (exclusion_byte, worldcover_mosaic)."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").mosaic()

    worldcereal = (
        ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
        .filter('product == "temporarycrops"')
        .select("classification")
        .max()
    )
    glad = ee.ImageCollection("projects/glad/GLCmap2019").mosaic()
    glad_crop = glad.eq(252)
    glad_urban = glad.gte(240).And(glad.lte(249))

    glc = ee.ImageCollection(
        "projects/sat-io/open-datasets/GLC-FCS30D/annual").mosaic()
    glc_urb_2018 = glc.select("b19").eq(190)
    glc_crop_2018 = glc.select("b19").lte(20)

    crop_trend = ee.ImageCollection(
        "projects/gee-zander-nina/assets/HABLOSS/crop_trend_2018_2024_v1"
    ).mosaic()
    cropland_base = (
        ee.Image(0).where(worldcereal, 1).where(glad_crop, 1)
        .where(glc_crop_2018, 1)
    )
    crop_trend = crop_trend.where(cropland_base.eq(0), 0)
    crop_loss = crop_trend.lt(CROP_LOSS_THR)

    built_trend = ee.ImageCollection(
        "projects/gee-zander-nina/assets/HABLOSS/grey_trend_2018_2024_v1"
    ).mosaic()
    built_base = ee.Image(0).where(glad_urban, 1).where(glc_urb_2018, 1)
    built_trend = built_trend.where(built_base.eq(0), 0)
    built_loss = built_trend.lt(BUILT_LOSS_THR)

    if include_buildings_loss:
        building_trend = ee.ImageCollection(
            "projects/gee-zander-nina/assets/HABLOSS/buildings_trend_2018_2023_v1"
        ).mosaic()
        buildings_loss = building_trend.lt(BUILDING_LOSS_THR)
    else:
        buildings_loss = ee.Image(0)

    exclusion = crop_loss.Or(built_loss).Or(buildings_loss).unmask(0).toByte()
    return exclusion, wc


def get_natural_mask():
    """natural = WorldCover not in {crop,built,water} AND not HABLOSS-excluded.

    Returns ee.Image band 'natural' (1=natural, 0=transformed), unmasked to 0.
    """
    exclusion, wc = build_exclusion_mask(include_buildings_loss=True)
    wc_natural = wc.neq(WC_CROP).And(wc.neq(WC_BUILT)).And(wc.neq(WC_WATER))
    natural = wc_natural.And(exclusion.Not())
    return natural.unmask(0).rename("natural")


# ── FSCS (Feature Space Coverage Sampling) ──────────────────────────
def fscs(covariates, n_clusters, n_init_points, geometry, scale=10, seed=42):
    """Feature Space Coverage Sampling on AlphaEarth embeddings.

    KMeans → per-cluster centroids → pixel closest to each centroid → 1 sample
    per cluster. Returns (clusters_image, sample_points_fc).
    """
    clipped = covariates.clip(geometry)

    init_points = ee.FeatureCollection.randomPoints(
        region=geometry, points=n_init_points, seed=seed, maxError=scale)
    init_data = clipped.sampleRegions(collection=init_points, scale=scale)

    clusterer = ee.Clusterer.wekaKMeans(
        nClusters=n_clusters, init=1, seed=seed)
    clusterer = clusterer.train(init_data, clipped.bandNames())
    clusters = clipped.cluster(clusterer)

    centroids_result = init_data.cluster(clusterer).reduceColumns(
        selectors=clipped.bandNames().add("cluster"),
        reducer=ee.Reducer.mean().repeat(
            clipped.bandNames().size()).group(
            groupField=clipped.bandNames().size(), groupName="cluster"))

    # Keep only clusters whose centroid is fully populated. KMeans can return
    # fewer populated clusters than requested (empty ids would be absent from
    # the dict -> "Dictionary does not contain key"), and a grouped mean can
    # occasionally come back null/incomplete under load (-> Image.constant
    # null). Dropping incomplete groups up front makes cluster_diff total.
    n_bands = clipped.bandNames().size()

    def _is_complete(g):
        mean = ee.List(ee.Dictionary(g).get("mean", ee.List([])))
        return mean.size().eq(n_bands)

    # map(..., dropNulls=True) removes the None entries for incomplete groups.
    good_groups = ee.List(centroids_result.get("groups")).map(
        lambda g: ee.Algorithms.If(_is_complete(g), g, None), True)

    centroid_dict = ee.Dictionary.fromLists(
        good_groups.map(lambda g: ee.Number(
            ee.Dictionary(g).get("cluster")).format("%d")),
        good_groups.map(lambda g: ee.Dictionary(g).get("mean")))

    cluster_seq = centroid_dict.keys().map(
        lambda k: ee.Number.parse(k).toInt())

    def cluster_diff(cluster_id):
        cluster_id = ee.Number(cluster_id).toInt()
        # Name each band uniquely by cluster id ("c<id>"). After toBands()
        # prefixes the collection index, regexpRename strips it so the band
        # name is exactly "c<id>" — letting sample_mins / select key by id
        # regardless of how many (sparse) clusters survived.
        band_name = ee.String("c").cat(cluster_id.format("%d"))
        means = ee.List(centroid_dict.get(cluster_id.format("%d")))
        centroid_img = ee.Image.constant(means).rename(clipped.bandNames())
        diff = clipped.subtract(centroid_img).pow(2).reduce(
            ee.Reducer.sum()).sqrt()
        return diff.updateMask(clusters.eq(cluster_id)).rename(band_name)

    diffs = ee.ImageCollection(cluster_seq.map(cluster_diff))
    diffs_banded = diffs.toBands().regexpRename("^[0-9]+_", "")

    sample_mins = diffs_banded.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=geometry, scale=scale, maxPixels=1e13, tileScale=4)

    def extract_samples(cluster_id):
        cluster_id = ee.Number(cluster_id).toInt()
        band_name = ee.String("c").cat(cluster_id.format("%d"))
        # A cluster's diff band can be fully masked over the region (its
        # nearest-centroid pixels fall outside / get masked), so the min is
        # null. Skip those — ee.Image.constant(null) would otherwise throw.
        closest_val = sample_mins.get(band_name)
        empty = ee.Image().updateMask(ee.Image(0)).rename("points")
        masked = clusters.updateMask(
            diffs_banded.select([band_name]).eq(
                ee.Image.constant(closest_val))).rename("points")
        return ee.Image(
            ee.Algorithms.If(ee.Algorithms.IsEqual(closest_val, None),
                             empty, masked))

    to_sample = ee.Image(ee.ImageCollection(
        cluster_seq.map(extract_samples)).max()).rename("toSample")
    samples = to_sample.stratifiedSample(
        numPoints=1, classBand="toSample", region=geometry,
        scale=scale, seed=seed, geometries=True)

    return clusters, samples


# ── Extraction stack ────────────────────────────────────────────────
def build_extraction_stack():
    """One combined image for one-shot per-cell extraction:
    AlphaEarth 2022 (64 bands) + natural label + lat/lon.

    `eco_id` is *not* sampled from an image here — reduceToImage on the global
    ecoregion FC renders empty at point scale. It is set per cell from the
    known ecoregion being processed (see run_ecoregion)."""
    aef22 = ee.ImageCollection(
        "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    ).filterDate("2022-01-01", "2023-01-01").reduce(
        ee.Reducer.first()
    ).regexpRename("_first$", "")

    natural_mask = get_natural_mask()

    return ee.Image.cat([
        aef22, natural_mask, ee.Image.pixelLonLat()
    ])


# ── Per-ecoregion pipeline ──────────────────────────────────────────
def run_ecoregion(eco_id, aef22, aef22_fscs, extraction_stack,
                  n_clusters=N_CLUSTERS, max_workers=MAX_WORKERS,
                  grid_scale=GRID_SCALE):
    """FSCS sampling within 10 km grid cells over one ecoregion, then a
    single-shot extraction of embeddings + natural + eco_id at the sampled
    points. Saves to data/ref_samples_eco{eco_id}.parquet.

    aef22, aef22_fscs, and extraction_stack are built once by the caller
    and reused across all ecoregions to avoid repeated OOM-prone stack
    evaluation on GEE."""
    output_path = os.path.join(DATA_DIR, f"ref_samples_eco{eco_id}.parquet")
    checkpoint_file = os.path.join(
        DATA_DIR, f"ref_samples_eco{eco_id}.checkpoint.json")

    if os.path.exists(output_path) and not os.path.exists(checkpoint_file):
        existing = duckdb.sql(
            f"SELECT count(*) FROM '{output_path}'").fetchone()[0]
        n_nat = duckdb.sql(
            f'SELECT count(*) FROM \'{output_path}\' '
            f'WHERE "natural" = 1').fetchone()[0]
        print(f"  [SKIP] {output_path} exists "
              f"({existing:,} rows, {n_nat:,} natural)")
        return

    print(f"\n{'#' * 70}")
    print(f"# ECOREGION {eco_id}")
    print(f"{'#' * 70}")

    ecoregions = ee.FeatureCollection(ECOREGIONS_ASSET)
    ecoregion = ecoregions.filter(ee.Filter.eq("ECO_ID", eco_id))
    eco_name = retry_gee(
        lambda: ecoregion.first().get("ECO_NAME").getInfo())
    print(f"  Name: {eco_name}")

    eco_geom = ecoregion.geometry()

    # Always use the unfiltered bounds grid. filterBounds against a complex
    # ecoregion polygon OOMs on GEE for ecoregions with >~3000 cells, even
    # for the size() call. Empty cells outside the polygon are skipped cheaply
    # by the pixel_count < 10 guard in process_cell (same as RECOVER).
    eco_raw_geom = ecoregion.geometry()
    grid_unfiltered = eco_raw_geom.bounds().coveringGrid(
        eco_raw_geom.projection(), grid_scale)
    total_cells = retry_gee(lambda: grid_unfiltered.size().getInfo())
    print(f"  Grid: {total_cells} cells ({grid_scale/1000:.0f}km) "
          f"[unfiltered bounds], {n_clusters} FSCS clusters/cell")

    if total_cells > 2000:
        print(f"  Paginating grid download ({total_cells} cells)...")
        grid_tagged = grid_unfiltered.randomColumn("_rnd", seed=SEED)
        grid_features = []
        n_pages = max(1, (total_cells + 1999) // 2000)
        for pi in range(n_pages):
            lo, hi = pi / n_pages, (pi + 1) / n_pages
            page = retry_gee(
                lambda l=lo, h=hi: grid_tagged.filter(
                    ee.Filter.And(
                        ee.Filter.gte("_rnd", l),
                        ee.Filter.lt("_rnd", h),
                    )).getInfo()["features"])
            grid_features.extend(page)
            print(f"    Downloaded {len(grid_features)}/{total_cells} cells")
    else:
        grid_features = retry_gee(
            lambda: grid_unfiltered.getInfo()["features"])

    # Checkpoint
    processed_cells = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_cells = set(json.load(f))
        print(f"  Resuming: {len(processed_cells)}/{total_cells} cells done")

    lock = Lock()

    def save_checkpoint():
        with lock:
            with open(checkpoint_file, "w") as f:
                json.dump(sorted(processed_cells), f)

    db_conn = duckdb.connect()
    if os.path.exists(output_path):
        print("  Loading existing parquet into buffer...")
        db_conn.execute(
            f"CREATE TABLE data AS SELECT * FROM '{output_path}'")
        existing = db_conn.execute(
            "SELECT count(*) FROM data").fetchone()[0]
        print(f"  [OK] Loaded {existing:,} existing rows")

    successful = 0
    failed_cells = []

    def process_cell(cell_idx):
        if cell_idx in processed_cells:
            return 0

        cell_geom = ee.Geometry(grid_features[cell_idx]["geometry"])

        try:
            cell_eco = cell_geom.intersection(eco_geom, ee.ErrorMargin(1))
            pixel_count = aef22.select(0).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=cell_eco,
                scale=SCALE,
                maxPixels=1e8,
            ).getNumber(aef22.bandNames().get(0)).getInfo()
        except Exception as e:
            tqdm.write(f"    Cell {cell_idx}: geom error — {e}")
            processed_cells.add(cell_idx)
            save_checkpoint()
            return 0

        if pixel_count is None or pixel_count < 10:
            tqdm.write(f"    Cell {cell_idx}: skip ({pixel_count} pixels)")
            processed_cells.add(cell_idx)
            save_checkpoint()
            return 0

        # Adaptive cluster count: can't have more clusters than pixels
        cell_n_clusters = min(n_clusters, max(2, int(pixel_count // 5)))
        cell_n_init = min(N_INIT_POINTS, int(pixel_count))

        try:
            _, samples = fscs(
                covariates=aef22_fscs,
                n_clusters=cell_n_clusters,
                n_init_points=cell_n_init,
                geometry=cell_eco,
                scale=SCALE,
                seed=SEED,
            )
        except Exception as e:
            tqdm.write(f"    Cell {cell_idx}: FSCS failed — {e}")
            processed_cells.add(cell_idx)
            save_checkpoint()
            return 0

        sampled = extraction_stack.reduceRegions(
            collection=samples,
            reducer=ee.Reducer.first(),
            scale=SCALE,
            tileScale=4,
        )

        df = ee.data.computeFeatures({
            "expression": sampled,
            "fileFormat": "PANDAS_DATAFRAME",
        })

        n_rows = 0 if df is None else len(df)
        tqdm.write(
            f"    Cell {cell_idx}: {n_rows} pts "
            f"(k={cell_n_clusters}, px={pixel_count:.0f})")

        if df is not None and not df.empty:
            df["eco_id"] = eco_id
            df["cell_index"] = cell_idx
            with lock:
                try:
                    db_conn.execute("SELECT 1 FROM data LIMIT 0")
                    table_exists = True
                except duckdb.CatalogException:
                    table_exists = False
                if table_exists:
                    db_conn.execute("INSERT INTO data SELECT * FROM df")
                else:
                    db_conn.execute("CREATE TABLE data AS SELECT * FROM df")
            processed_cells.add(cell_idx)
            save_checkpoint()
            return len(df)

        processed_cells.add(cell_idx)
        save_checkpoint()
        return 0

    effective_workers = min(max_workers, total_cells)
    print(f"  Processing with {effective_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(
            lambda idx=i: retry_gee(lambda: process_cell(idx))
        ): i for i in range(total_cells)}

        for future in tqdm(as_completed(futures),
                           total=total_cells, desc="FSCS cells"):
            cell_idx = futures[future]
            try:
                n = future.result(timeout=CELL_TIMEOUT_S)
                if n and n > 0:
                    successful += 1
            except TimeoutError:
                failed_cells.append(cell_idx)
                tqdm.write(f"  [TIMEOUT] Cell {cell_idx}: exceeded {CELL_TIMEOUT_S}s")
            except Exception as e:
                failed_cells.append(cell_idx)
                tqdm.write(f"  [ERROR] Cell {cell_idx}: {e}")

    if failed_cells:
        print(f"\n  {len(failed_cells)} cells failed after retries: "
              f"{failed_cells}")

    print(f"\n  Cells: {successful} with data, "
          f"{len(failed_cells)} failed, "
          f"{len(processed_cells)} total processed")

    try:
        buf_rows = db_conn.execute(
            "SELECT count(*) FROM data").fetchone()[0]
    except Exception:
        buf_rows = 0

    if buf_rows == 0:
        print(f"  [WARN] No data extracted for eco_id={eco_id}")
        db_conn.close()
        return

    db_conn.execute(f"""
        COPY (
            SELECT DISTINCT ON (geo) *
            FROM data
        ) TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    final_rows = db_conn.execute(
        f"SELECT count(*) FROM '{output_path}'").fetchone()[0]
    n_natural = db_conn.execute(f"""
        SELECT count(*) FROM '{output_path}'
        WHERE "natural" = 1
    """).fetchone()[0]
    file_mb = os.path.getsize(output_path) / 1e6

    print(f"\n[OK] Saved: {output_path}")
    print(f"  Total: {final_rows:,} points "
          f"({n_natural:,} natural, "
          f"{final_rows - n_natural:,} transformed)")
    print(f"  Size: {file_mb:.1f} MB")

    db_conn.close()

    if not failed_cells and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ecoregion-level FSCS reference sampling for DoR")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eco_id", type=int,
                       help="Single ecoregion ECO_ID")
    group.add_argument("--all", action="store_true",
                       help="All ecoregions containing DoR parents")
    parser.add_argument("--n_clusters", type=int, default=N_CLUSTERS,
                        help=f"FSCS clusters per grid cell "
                             f"(default {N_CLUSTERS})")
    parser.add_argument("--grid_scale", type=int, default=GRID_SCALE,
                        help=f"Grid cell size in metres (default {GRID_SCALE})")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()

    init_gee(args.project)

    # Build once — reused across all ecoregions to avoid per-ecoregion OOM
    # on GEE when assembling the heavy multi-source extraction stack.
    print("Building AlphaEarth + extraction stack (once)...")
    aef22 = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
             .filterDate("2022-01-01", "2023-01-01")
             .reduce(ee.Reducer.first())
             .regexpRename("_first$", ""))
    aef22_fscs = aef22.select(aef22.bandNames().slice(0, FSCS_BANDS))
    extraction_stack = build_extraction_stack()
    print("  [OK] stack ready")

    if args.eco_id:
        run_ecoregion(args.eco_id, aef22, aef22_fscs, extraction_stack,
                      args.n_clusters, args.max_workers, args.grid_scale)
    else:
        eco_ids = parent_ecoregion_ids()
        print(f"[OK] {len(eco_ids)} ecoregions contain DoR parents: {eco_ids}")
        for eco_id in eco_ids:
            run_ecoregion(eco_id, aef22, aef22_fscs, extraction_stack,
                          args.n_clusters, args.max_workers, args.grid_scale)


if __name__ == "__main__":
    main()
