"""Extract AlphaEarth embeddings (64 bands) at reference sample points.

Input asset is the FeatureCollection produced by
`scripts/sampling/sample_reference_states.py` — points tagged with
`ref_state`, `parent_id`, and `parent_label`.

For each point we sample the AlphaEarth Satellite Embedding V1 image for the
configured year and write `A00..A63` plus the carry-through metadata to a
parquet file. Sharding via `randomColumn` lets us run many small
`computeFeatures` calls in parallel; a JSON checkpoint records which shard
indices have been written so reruns are resumable.

Usage:
  python scripts/extraction/extract_alphaearth_embeddings.py
  python scripts/extraction/extract_alphaearth_embeddings.py --year 2023 --n_shards 50
  python scripts/extraction/extract_alphaearth_embeddings.py --test_mode
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import duckdb
import ee
from tqdm.auto import tqdm

# ── Constants ───────────────────────────────────────────────────────
PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/ee-gsingh/assets/recover_reference_samples_test1"
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

YEAR = 2024
SCALE = 10                  # AlphaEarth native resolution (m)
TARGET_SHARD_SIZE = 750     # points/shard sweet spot for reduceRegions+computeFeatures
N_SHARDS = None             # None = auto-size from asset point count (#8)
MAX_WORKERS = 40            # high-volume endpoint comfortably handles 40-60 (#7)
TILE_SCALE = 2              # 1-2 is sufficient for point sampling at native scale (#10)
SEED = 42

CARRY_PROPERTIES = ["ref_state", "parent_id", "parent_label"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── GEE helpers ─────────────────────────────────────────────────────
def init_gee(project=PROJECT):
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
        print(f"[OK] GEE initialised with high-volume endpoint "
              f"(project={project})")
    except Exception as e:
        print(f"  High-volume init failed ({e}); falling back to standard.")
        ee.Initialize(project=project)
        print(f"[OK] GEE initialised (project={project})")


def retry_gee(func, max_retries=3, backoff=2):
    """Run `func()` with exponential-backoff retry on transient GEE errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff ** (attempt + 1)
            tqdm.write(f"    retry {attempt + 1}/{max_retries} "
                       f"after {wait}s: {str(e)[:120]}")
            time.sleep(wait)


# ── Image stack ─────────────────────────────────────────────────────
def build_alphaearth_image(year):
    """Annual AlphaEarth embedding image with bands A00..A63.

    `reduce(first)` is preferred over `mosaic()` (RECOVER benchmark gotcha
    #4) — `mosaic` rebuilds the whole tile graph per request.
    """
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(start, end)
        .reduce(ee.Reducer.first())
        .regexpRename("_first$", "")
    )


EXPECTED_BANDS = [f"A{i:02d}" for i in range(64)]


# ── Extraction ──────────────────────────────────────────────────────
def process_shard(shard_fc, image, db_conn, lock, tile_scale=TILE_SCALE,
                  schema_checked=None):
    """Sample `image` at every point in `shard_fc`, append to DuckDB buffer.

    Returns the row count written, or raises so the retry logic can handle
    transient errors. `schema_checked` is a one-shot flag (list with one
    bool); when False, the first non-empty df is asserted to contain all
    64 expected bands.
    """
    sampled = image.reduceRegions(
        collection=shard_fc,
        reducer=ee.Reducer.first(),
        scale=SCALE,
        tileScale=tile_scale,
    )

    df = ee.data.computeFeatures({
        "expression": sampled,
        "fileFormat": "PANDAS_DATAFRAME",
    })

    if df is None or df.empty:
        return 0

    if schema_checked is not None and not schema_checked[0]:
        missing = [b for b in EXPECTED_BANDS if b not in df.columns]
        if missing:
            raise RuntimeError(
                f"Shard df missing AlphaEarth bands {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} "
                f"(got {len(df.columns)} cols). Refusing to write a "
                f"partially-banded parquet.")
        schema_checked[0] = True

    with lock:
        try:
            db_conn.execute("INSERT INTO data BY NAME SELECT * FROM df")
        except duckdb.CatalogException:
            db_conn.execute("CREATE TABLE data AS SELECT * FROM df")

    return len(df)


def process_shard_with_escalation(shard_idx, make_shard, image, db_conn,
                                  lock, schema_checked,
                                  tile_scales=(TILE_SCALE, 4, 8, 16),
                                  backoff=2):
    """Run `process_shard` with tileScale escalation on retry.

    Each retry bumps tileScale, which trades wall-time for headroom
    against GEE memory/timeout limits. Cheap shards still finish at the
    base tileScale on attempt 0; expensive shards get more tiles.
    """
    last_exc = None
    for attempt, ts in enumerate(tile_scales):
        try:
            return process_shard(make_shard(shard_idx), image, db_conn,
                                 lock, tile_scale=ts,
                                 schema_checked=schema_checked)
        except Exception as e:
            last_exc = e
            if attempt == len(tile_scales) - 1:
                raise
            wait = backoff ** (attempt + 1)
            tqdm.write(
                f"    shard {shard_idx} retry "
                f"{attempt + 1}/{len(tile_scales) - 1} "
                f"(tileScale {ts}->{tile_scales[attempt + 1]}) "
                f"after {wait}s: {str(e)[:120]}")
            time.sleep(wait)
    raise last_exc


def run(year, n_shards, max_workers, output_path, samples_asset=SAMPLES_ASSET,
        test_mode=False, verbose=False):
    checkpoint_file = output_path + ".checkpoint.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Reset checkpoint if the parquet was deleted but checkpoint lingered
    if not os.path.exists(output_path) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # ── Inputs ──
    samples = ee.FeatureCollection(samples_asset)
    if test_mode:
        n_shards = 2
        samples = samples.limit(500)
        print("  *** TEST MODE: 500 points × 2 shards ***")

    # Auto-size shard count from asset point count (#8). One round-trip
    # only when the user did not specify --n_shards explicitly.
    if n_shards is None:
        n_points = retry_gee(lambda: samples.size().getInfo())
        n_shards = max(1, math.ceil(n_points / TARGET_SHARD_SIZE))
        print(f"  Points in asset: {n_points:,}  ->  "
              f"n_shards={n_shards} (target {TARGET_SHARD_SIZE}/shard)")
    elif verbose:
        n_points = retry_gee(lambda: samples.size().getInfo())
        print(f"  Points in asset: {n_points:,}")

    # Checkpoint format: {"done": [...], "failed": [...]}. Old format
    # (a bare list of done indices) is still accepted on read.
    processed = set()
    prior_failed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            cp = json.load(f)
        if isinstance(cp, list):
            processed = set(cp)
        else:
            processed = set(cp.get("done", []))
            prior_failed = set(cp.get("failed", []))
        print(f"  Resuming: {len(processed)}/{n_shards} done, "
              f"{len(prior_failed)} previously failed (will retry)")

    image = build_alphaearth_image(year)
    print(f"  AlphaEarth year: {year} (64 bands A00..A63)")

    # Bucket each feature into an integer shard once (#6). Single eq-filter
    # graph per shard instead of compound range filters.
    samples_sharded = (
        samples.randomColumn("shard_rand", seed=SEED)
        .map(lambda f: f.set(
            "shard_idx",
            ee.Number(f.get("shard_rand")).multiply(n_shards).floor().min(
                n_shards - 1)))
    )

    # ── DuckDB buffer (load existing parquet on resume) ──
    db_conn = duckdb.connect()
    if os.path.exists(output_path):
        print(f"  Loading existing parquet into buffer "
              f"({os.path.basename(output_path)})...")
        db_conn.execute(f"CREATE TABLE data AS SELECT * FROM '{output_path}'")
        existing = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
        print(f"  [OK] Loaded {existing:,} existing rows")

    lock = Lock()
    failed = set()

    def save_checkpoint():
        with lock, open(checkpoint_file, "w") as f:
            json.dump({"done": sorted(processed),
                       "failed": sorted(failed)}, f)

    # ── Build shards ──
    shard_indices = [i for i in range(n_shards) if i not in processed]

    def make_shard(i):
        return samples_sharded.filter(ee.Filter.eq("shard_idx", i))

    # First-shard schema lock — flips True after the first non-empty df
    # is verified to contain all 64 AlphaEarth bands.
    schema_checked = [False]

    # ── Parallel shard processing ──
    successful = 0
    total_rows = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                process_shard_with_escalation,
                i, make_shard, image, db_conn, lock, schema_checked,
            ): i
            for i in shard_indices
        }

        with tqdm(total=len(shard_indices),
                  desc="AlphaEarth shards", ncols=100) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    n = future.result()
                    if n is not None:
                        total_rows += n
                        successful += 1
                        processed.add(idx)
                        failed.discard(idx)
                        save_checkpoint()
                        tqdm.write(f"    shard {idx}: {n} rows")
                except Exception as e:
                    failed.add(idx)
                    save_checkpoint()
                    tqdm.write(f"  [ERROR] shard {idx}: {str(e)[:200]}")
                pbar.update(1)

    if failed:
        print(f"\n  {len(failed)} shards failed after retries: "
              f"{sorted(failed)}")
        print(f"  Re-run the same command to retry only the failed shards "
              f"(checkpoint: {os.path.basename(checkpoint_file)})")

    # ── Export buffer to parquet ──
    try:
        buf_rows = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
    except Exception:
        buf_rows = 0

    if buf_rows == 0:
        print("  [WARN] No data extracted")
        db_conn.close()
        return

    tmp = output_path + ".tmp"
    db_conn.execute(
        f"COPY data TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    db_conn.close()

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp, output_path)

    file_mb = os.path.getsize(output_path) / 1e6
    print(f"\n[OK] Saved {output_path}")
    print(f"  Rows: {buf_rows:,}   Size: {file_mb:.1f} MB")
    print(f"  Shards: {successful} ok, {len(failed)} failed, "
          f"{len(processed)} total processed")

    if not failed and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    default_output = os.path.join(
        DATA_DIR, "recover_reference_samples_alphaearth.parquet")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--year", type=int, default=YEAR,
                        help=f"AlphaEarth annual year (default {YEAR})")
    parser.add_argument("--n_shards", type=int, default=N_SHARDS,
                        help=f"Number of randomColumn shards. Default: "
                             f"auto-sized to ~{TARGET_SHARD_SIZE} points/shard "
                             f"from the asset's point count.")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel shard workers (default {MAX_WORKERS})")
    parser.add_argument("--output", default=default_output,
                        help="Output parquet path")
    parser.add_argument("--samples_asset", default=SAMPLES_ASSET,
                        help=f"Source FeatureCollection asset id "
                             f"(default {SAMPLES_ASSET})")
    parser.add_argument("--test_mode", action="store_true",
                        help="Limit to 500 points × 2 shards for a smoke test")
    parser.add_argument("--verbose", action="store_true",
                        help="Print upfront diagnostics (each one is a "
                             "blocking GEE round-trip).")
    args = parser.parse_args()

    init_gee(args.project)
    run(year=args.year,
        n_shards=args.n_shards,
        max_workers=args.max_workers,
        output_path=args.output,
        samples_asset=args.samples_asset,
        test_mode=args.test_mode,
        verbose=args.verbose)


if __name__ == "__main__":
    main()
