"""Step 1 — extract 2024 AlphaEarth embeddings for the target sites lacking a cache.

281 of the 1,248 target sites have no cached 2024 parent embedding (they were never
extracted because their 2018->2024 transition was outside the original DoR `r` labels).
They are likely NOT in the published EE asset, so we build the FeatureCollection from
the LOCAL shp point coordinates (carried in target_groups.parquet) and sample the 2024
AlphaEarth first-composite at each point. We also tag each point's RESOLVE eco_id in the
same pass so the ecoregion scorer can place sites the v5 refs never mapped.

Reuses the per-feature reduceRegion pattern from
v1/scripts/extraction/extract_test_site_embeddings.py (avoids the null-band failure mode
of a single reduceRegions over a globally-sparse FC).

Output: test_site_scoring/data/test_site_alphaearth_2024_missing.parquet
Columns: PLOTID, eco_id, A00..A63, year
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import ee
import pandas as pd

BASE = Path("/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery")
TARGETS = BASE / "test_site_scoring/data/target_groups.parquet"
OUT = BASE / "test_site_scoring/data/test_site_alphaearth_2024_missing.parquet"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
ECOREGIONS_ASSET = "RESOLVE/ECOREGIONS/2017"
PROJECT = "ee-gsingh"
SCALE = 10
YEAR = 2024
CHUNK = 200          # points per computeFeatures call


def init_gee():
    try:
        ee.Initialize(project=PROJECT,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=PROJECT)
    print(f"[OK] GEE initialised (project={PROJECT})")


def alphaearth(year):
    return (ee.ImageCollection(EMBEDDING_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .reduce(ee.Reducer.first())
            .regexpRename("_first$", ""))


def fetch_chunk(df_chunk: pd.DataFrame, image, ecoregions) -> pd.DataFrame:
    """Sample embedding + eco_id at each local point via per-feature reduceRegion."""
    feats = [
        ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                   {"PLOTID": str(r.PLOTID)})
        for r in df_chunk.itertuples()
    ]
    fc = ee.FeatureCollection(feats)

    def sample_one(ft):
        vals = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=ft.geometry(),
            scale=SCALE,
            tileScale=4,
        )
        match = ecoregions.filterBounds(ft.geometry()).first()
        eco_id = ee.Algorithms.If(match, ee.Feature(match).get("ECO_ID"), -1)
        return ft.set(vals).set("eco_id", eco_id)

    sampled = fc.map(sample_one)
    return ee.data.computeFeatures({
        "expression": sampled,
        "fileFormat": "PANDAS_DATAFRAME",
    })


def main() -> None:
    con = duckdb.connect()
    miss = con.execute(
        f"""SELECT PLOTID, longitude, latitude
            FROM read_parquet('{TARGETS}')
            WHERE has_embedding = FALSE"""
    ).df()
    con.close()
    print(f"Missing-embedding target sites: {len(miss)}")

    init_gee()
    image = alphaearth(YEAR)
    ecoregions = ee.FeatureCollection(ECOREGIONS_ASSET)

    frames = []
    n_chunks = (len(miss) + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        chunk = miss.iloc[i * CHUNK:(i + 1) * CHUNK]
        for attempt in range(4):
            try:
                frames.append(fetch_chunk(chunk, image, ecoregions))
                print(f"  chunk {i + 1}/{n_chunks}: {len(chunk)} pts OK")
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  chunk {i + 1}/{n_chunks} attempt {attempt + 1} failed "
                      f"({str(e)[:80]}); retry in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"chunk {i + 1} failed after retries")

    out = pd.concat(frames, ignore_index=True)
    out["PLOTID"] = out["PLOTID"].astype(str)

    missing_bands = [c for c in EMBED_COLS if c not in out.columns]
    if missing_bands:
        raise RuntimeError(f"output missing bands: {missing_bands[:3]}...")

    out["year"] = YEAR
    out = out[["PLOTID", "eco_id"] + EMBED_COLS + ["year"]]

    n_null = out[EMBED_COLS].isna().any(axis=1).sum()
    print(f"\nExtracted {len(out)} sites; {n_null} have any null band")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}")

    # verification
    assert len(out) == len(miss), f"row count {len(out)} != requested {len(miss)}"
    n_eco = (out["eco_id"].fillna(-1).astype(int) >= 0).sum()
    print(f"with valid eco_id: {n_eco}/{len(out)}")
    print(out[["PLOTID", "eco_id", "A00", "A01"]].head(3).to_string())


if __name__ == "__main__":
    main()
