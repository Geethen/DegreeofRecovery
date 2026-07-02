"""Step 2, Path A — re-sample 2024 AlphaEarth at the EXISTING 2022 ecoregion ref points.

Fixes the year caveat cheaply for every ecoregion already covered by a 2022 FSCS file:
the 2022 parquets carry per-point latitude/longitude + the natural label, so we just
re-sample the 2024 embedding at the identical coordinates. No FSCS, no grid, and the
points are identical across vintages -> a clean year comparison.

Scope: the target ecoregions (containing >=1 of the three groups) that ALREADY have a
ref_samples_eco{id}.parquet. The 255 ecoregions with no file are handled by Path B
(full FSCS at 2024).

Output: outputs/data/ref_samples_eco{id}_2024.parquet  (suffix _2024 so the
2022 files are never clobbered). Same columns as the 2022 file: natural, eco_id,
cell_index, latitude, longitude, A00..A63.

NOTE: requires Google Earth Engine access, and the 2022-vintage
ref_samples_eco{id}.parquet source files (not shipped — see
data/ecoregion_refs/DOWNLOAD.md). Not needed for the default reproduction path,
which scores against the already-2024 ref_samples_eco{id}_2024.parquet files.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import duckdb
import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA, OUT_DATA, OUT_LOGS  # noqa: E402

ECO_DIR = DATA / "ecoregion_refs_2022"      # 2022 FSCS source (download only)
TARGETS = OUT_DATA / "target_groups.parquet"
MISSING = OUT_DATA / "test_site_alphaearth_2024_missing.parquet"
OUT_DIR = DATA / "ecoregion_refs"
LOG = OUT_LOGS / "02a_resample.log"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
PROJECT = "ee-gsingh"
SCALE = 10
YEAR = 2024
CHUNK = 1000          # points per computeFeatures call
POINT_CAP = 5000      # max points re-sampled per ecoregion (10x N_BASELINE=500)
SUBSAMPLE_SEED = 42


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def init_gee():
    try:
        ee.Initialize(project=PROJECT,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=PROJECT)
    log(f"GEE initialised (project={PROJECT})")


def alphaearth(year):
    return (ee.ImageCollection(EMBEDDING_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .reduce(ee.Reducer.first())
            .regexpRename("_first$", ""))


def target_path_a_ecos() -> list[int]:
    con = duckdb.connect()
    ecos = set(int(e) for e in con.execute(
        f"SELECT DISTINCT eco_id FROM read_parquet('{TARGETS}') "
        f"WHERE eco_id IS NOT NULL AND CAST(eco_id AS INT) >= 0").df()["eco_id"])
    if MISSING.exists():
        ecos |= set(int(e) for e in con.execute(
            f"SELECT DISTINCT eco_id FROM read_parquet('{MISSING}') "
            f"WHERE CAST(eco_id AS INT) >= 0").df()["eco_id"])
    con.close()
    files = {int(re.search(r"eco(\d+)", f).group(1))
             for f in os.listdir(ECO_DIR)
             if re.match(r"ref_samples_eco\d+\.parquet$", f)}
    return sorted(ecos & files)


def resample_one(eco_id: int, image) -> None:
    out_path = OUT_DIR / f"ref_samples_eco{eco_id}_2024.parquet"
    if out_path.exists():
        log(f"eco {eco_id}: [skip] {out_path.name} exists")
        return

    src = ECO_DIR / f"ref_samples_eco{eco_id}.parquet"
    con = duckdb.connect()
    pts = con.execute(
        f'SELECT latitude, longitude, "natural", eco_id, cell_index '
        f"FROM read_parquet('{src}') "
        f"WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).df()
    con.close()
    pts = pts.rename(columns={"natural": "natural_label"})  # avoid keyword/positional issues

    # Cap at POINT_CAP points/ecoregion (these 2022 files hold up to millions of
    # points but the percentile scorer only uses N_BASELINE=500 + MIN_REFS=10 per
    # pool). Subsample stratified by `natural` so both good and bad pools survive,
    # proportional to their availability, never dropping a pool below MIN_REFS.
    n_total = len(pts)
    if n_total > POINT_CAP:
        rng_frac = POINT_CAP / n_total
        kept = []
        for lab, grp in pts.groupby("natural_label"):
            take = max(min(len(grp), 10), int(round(len(grp) * rng_frac)))
            take = min(take, len(grp))
            kept.append(grp.sample(n=take, random_state=SUBSAMPLE_SEED))
        pts = pd.concat(kept, ignore_index=True)
    pts = pts.reset_index(drop=True)
    log(f"eco {eco_id}: re-sampling {len(pts):,}/{n_total:,} points at 2024 "
        f"(cap={POINT_CAP})")

    frames = []
    n_chunks = (len(pts) + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        chunk = pts.iloc[i * CHUNK:(i + 1) * CHUNK]
        # carry lat/lon/natural/eco/cell through the feature properties so no
        # post-merge is needed (computeFeatures returns only what we set + bands).
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                {"natural": int(r.natural_label), "eco_id": int(r.eco_id),
                 "cell_index": int(r.cell_index),
                 "latitude": float(r.latitude), "longitude": float(r.longitude)})
            for r in chunk.itertuples()
        ]
        fc = ee.FeatureCollection(feats)

        def sample_one(ft):
            vals = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=ft.geometry(), scale=SCALE, tileScale=4)
            return ft.set(vals)

        for attempt in range(4):
            try:
                df = ee.data.computeFeatures({
                    "expression": fc.map(sample_one),
                    "fileFormat": "PANDAS_DATAFRAME"})
                frames.append(df)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                log(f"eco {eco_id} chunk {i+1}/{n_chunks} attempt {attempt+1} "
                    f"failed ({str(e)[:70]}); retry {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"eco {eco_id} chunk {i+1} failed after retries")

    out = pd.concat(frames, ignore_index=True)
    missing_bands = [c for c in EMBED_COLS if c not in out.columns]
    if missing_bands:
        raise RuntimeError(f"eco {eco_id}: missing bands {missing_bands[:3]}")
    keep = ["natural", "eco_id", "cell_index", "latitude", "longitude"] + EMBED_COLS
    out = out[keep]
    n_null = out[EMBED_COLS].isna().any(axis=1).sum()

    tmp = f"{out_path}.tmp.{os.getpid()}"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    log(f"eco {eco_id}: wrote {len(out):,} pts ({n_null} null-band) -> {out_path.name}")


def main() -> None:
    ecos = target_path_a_ecos()
    log(f"Path A: {len(ecos)} target ecoregions with an existing 2022 file")
    init_gee()
    image = alphaearth(YEAR)
    for j, eco_id in enumerate(ecos, 1):
        log(f"=== [{j}/{len(ecos)}] eco {eco_id} ===")
        resample_one(eco_id, image)
    log("Path A complete.")


if __name__ == "__main__":
    main()
