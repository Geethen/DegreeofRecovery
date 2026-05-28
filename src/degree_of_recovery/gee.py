"""Earth Engine helpers for AlphaEarth embedding extraction.

These helpers are ported from `v1/scripts/extraction/extract_alphaearth_embeddings.py`,
which is the canonical learning artifact for the extraction pattern. v1's
extractor keeps its own inline copy verbatim and does NOT import from this
module — preserve that. v4 and later extractors should import from here
and only override version-specific constants (asset id, output path).

Public surface:
    init_gee(project)                       — initialise EE with high-volume endpoint
    retry_gee(func, ...)                    — exponential-backoff retry wrapper
    build_alphaearth_image(year)            — annual AlphaEarth embedding image
    process_shard(shard_fc, image, ...)     — sample one shard, append to DuckDB
    process_shard_with_escalation(...)      — same with tileScale escalation on retry
    run(year, n_shards, ..., output_path)   — full sharded extraction with checkpointing

Constants exported for re-use:
    EMBEDDING_COLLECTION, EXPECTED_BANDS, SCALE, TARGET_SHARD_SIZE,
    MAX_WORKERS, TILE_SCALE, SEED, CARRY_PROPERTIES
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import duckdb
import ee
from tqdm.auto import tqdm

PROJECT = "ee-gsingh"
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

SCALE = 10
TARGET_SHARD_SIZE = 750
MAX_WORKERS = 40
TILE_SCALE = 2
SEED = 42

CARRY_PROPERTIES = ["ref_state", "parent_id", "parent_label"]
EXPECTED_BANDS = [f"A{i:02d}" for i in range(64)]


def init_gee(project: str = PROJECT) -> None:
    try:
        ee.Initialize(
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
        print(
            f"[OK] GEE initialised with high-volume endpoint "
            f"(project={project})"
        )
    except Exception as e:
        print(f"  High-volume init failed ({e}); falling back to standard.")
        ee.Initialize(project=project)
        print(f"[OK] GEE initialised (project={project})")


def retry_gee(func, max_retries: int = 3, backoff: int = 2):
    """Run `func()` with exponential-backoff retry on transient GEE errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff ** (attempt + 1)
            tqdm.write(
                f"    retry {attempt + 1}/{max_retries} "
                f"after {wait}s: {str(e)[:120]}"
            )
            time.sleep(wait)


def build_alphaearth_image(year: int):
    """Annual AlphaEarth embedding image with bands A00..A63."""
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(start, end)
        .reduce(ee.Reducer.first())
        .regexpRename("_first$", "")
    )


def process_shard(
    shard_fc, image, db_conn, lock, tile_scale: int = TILE_SCALE, schema_checked=None
) -> int:
    """Sample `image` at every point in `shard_fc`, append to DuckDB buffer."""
    sampled = image.reduceRegions(
        collection=shard_fc,
        reducer=ee.Reducer.first(),
        scale=SCALE,
        tileScale=tile_scale,
    )

    df = ee.data.computeFeatures(
        {
            "expression": sampled,
            "fileFormat": "PANDAS_DATAFRAME",
        }
    )

    if df is None or df.empty:
        return 0

    if schema_checked is not None and not schema_checked[0]:
        missing = [b for b in EXPECTED_BANDS if b not in df.columns]
        if missing:
            raise RuntimeError(
                f"Shard df missing AlphaEarth bands {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} "
                f"(got {len(df.columns)} cols). Refusing to write a "
                f"partially-banded parquet."
            )
        schema_checked[0] = True

    with lock:
        try:
            db_conn.execute("INSERT INTO data BY NAME SELECT * FROM df")
        except duckdb.CatalogException:
            db_conn.execute("CREATE TABLE data AS SELECT * FROM df")

    return len(df)


def process_shard_with_escalation(
    shard_idx,
    make_shard,
    image,
    db_conn,
    lock,
    schema_checked,
    tile_scales=(TILE_SCALE, 4, 8, 16, 32, 64),
    backoff: int = 2,
) -> int:
    """Run `process_shard` with tileScale escalation on retry."""
    last_exc = None
    for attempt, ts in enumerate(tile_scales):
        try:
            return process_shard(
                make_shard(shard_idx),
                image,
                db_conn,
                lock,
                tile_scale=ts,
                schema_checked=schema_checked,
            )
        except Exception as e:
            last_exc = e
            if attempt == len(tile_scales) - 1:
                raise
            wait = backoff ** (attempt + 1)
            tqdm.write(
                f"    shard {shard_idx} retry "
                f"{attempt + 1}/{len(tile_scales) - 1} "
                f"(tileScale {ts}->{tile_scales[attempt + 1]}) "
                f"after {wait}s: {str(e)[:120]}"
            )
            time.sleep(wait)
    raise last_exc


def run(
    year: int,
    n_shards,
    max_workers: int,
    output_path: str,
    samples_asset: str,
    test_mode: bool = False,
    verbose: bool = False,
) -> None:
    """Full sharded AlphaEarth extraction with checkpointing.

    `samples_asset` is required (no default) — each version's extractor
    supplies its own asset id.
    """
    checkpoint_file = output_path + ".checkpoint.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    samples = ee.FeatureCollection(samples_asset)
    if test_mode:
        samples = samples.limit(500)
        print("  *** TEST MODE: 500 points ***")
    else:
        n_points = retry_gee(lambda: samples.size().getInfo())
        print(f"  Points in asset: {n_points:,}")

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
        print(
            f"  Resuming: {len(processed)} done, "
            f"{len(prior_failed)} previously failed (will retry)"
        )

    image = build_alphaearth_image(year)
    print(f"  AlphaEarth year: {year} (64 bands A00..A63)")

    # Parent-based sharding: one shard per parent_id. Each parent's reference
    # points lie within an 8 km buffer around the parent geometry (BUFFER_STEPS_M
    # ceiling in sample_stable_references_v5.py), so GEE only loads AlphaEarth
    # tiles from one small region per request — naturally aligned with the
    # AlphaEarth UTM tile footprint and bounded in memory.
    import random
    parent_ids = retry_gee(
        lambda: samples.aggregate_array("parent_id").distinct().getInfo()
    )
    parent_ids = [str(p) for p in parent_ids]
    random.Random(SEED).shuffle(parent_ids)
    n_shards = len(parent_ids)
    print(f"  Parent-based sharding: {n_shards} parent sites (shuffled)")

    db_conn = duckdb.connect()
    if os.path.exists(output_path):
        print(
            f"  Loading existing parquet into buffer "
            f"({os.path.basename(output_path)})..."
        )
        db_conn.execute(f"CREATE TABLE data AS SELECT * FROM '{output_path}'")
        existing = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
        print(f"  [OK] Loaded {existing:,} existing rows")

    lock = Lock()
    failed = set()
    flush_counter = [0]
    FLUSH_EVERY = 10

    def save_checkpoint():
        with lock, open(checkpoint_file, "w") as f:
            json.dump(
                {"done": sorted(processed), "failed": sorted(failed)}, f
            )

    def flush_parquet():
        # Atomically write in-memory buffer to disk so kills don't lose work.
        try:
            buf_rows = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
        except Exception:
            return
        if buf_rows == 0:
            return
        tmp = output_path + ".tmp"
        db_conn.execute(f"COPY data TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(tmp, output_path)
        tqdm.write(f"  [flush] {buf_rows:,} rows -> {os.path.basename(output_path)}")

    shard_indices = [i for i in range(n_shards) if i not in processed]

    def make_shard(i):
        return samples.filter(ee.Filter.eq("parent_id", parent_ids[i]))

    schema_checked = [False]

    successful = 0
    total_rows = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                process_shard_with_escalation,
                i,
                make_shard,
                image,
                db_conn,
                lock,
                schema_checked,
            ): i
            for i in shard_indices
        }

        with tqdm(
            total=len(shard_indices), desc="AlphaEarth shards", ncols=100
        ) as pbar:
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
                        flush_counter[0] += 1
                        if flush_counter[0] % FLUSH_EVERY == 0:
                            with lock:
                                flush_parquet()
                except Exception as e:
                    failed.add(idx)
                    save_checkpoint()
                    tqdm.write(f"  [ERROR] shard {idx}: {str(e)[:200]}")
                pbar.update(1)

    if failed:
        print(
            f"\n  {len(failed)} shards failed after retries: "
            f"{sorted(failed)}"
        )
        print(
            f"  Re-run the same command to retry only the failed shards "
            f"(checkpoint: {os.path.basename(checkpoint_file)})"
        )

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
        f"COPY data TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    db_conn.close()

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp, output_path)

    file_mb = os.path.getsize(output_path) / 1e6
    print(f"\n[OK] Saved {output_path}")
    print(f"  Rows: {buf_rows:,}   Size: {file_mb:.1f} MB")
    print(
        f"  Shards: {successful} ok, {len(failed)} failed, "
        f"{len(processed)} total processed"
    )

    if not failed and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
