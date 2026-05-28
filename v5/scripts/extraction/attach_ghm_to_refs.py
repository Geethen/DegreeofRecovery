"""Backfill GHM v3 2022 (90 m, AA band) onto existing ref parquets.

Reads:
  v4/data/v4_stable_refs_alphaearth.parquet           -> stable refs (v4 sampler)
  v2/data/recover_reference_samples_v2_mask_on_large_alphaearth.parquet
                                                       -> candidate refs (v2)

For each ref the geometry is stored in the `geo` column as a GeoJSON Point
dict. We push points to EE in chunks, run `ghm_image.reduceRegions(... mean)`
against each, then merge the per-point AA value back as `ghm_aa`.

Outputs (alongside the originals so v4/v2 stay untouched):
  v5/data/v4_stable_refs_alphaearth_ghm.parquet
  v5/data/recover_reference_samples_v2_mask_on_large_alphaearth_ghm.parquet

Why this matters: the v5 re-sampling adds GHM natively, but you also want GHM
on the *current* refs so you can compare old-sample vs new-sample analyses on
the same human-modification scale (e.g. "does built-loss contamination track
ghm_aa?").

Run:
  python v5/scripts/extraction/attach_ghm_to_refs.py --target stable
  python v5/scripts/extraction/attach_ghm_to_refs.py --target candidate
  python v5/scripts/extraction/attach_ghm_to_refs.py --target both
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import ee
import numpy as np
import pandas as pd

PROJECT = "ee-gsingh"
GHM_ASSET = "projects/sat-io/open-datasets/GHM/HM_2022_90M"
GHM_BAND = "AA"
SCALE_GHM = 90

BASE_DIR = Path(__file__).resolve().parents[3]
V5_DATA  = BASE_DIR / "v5" / "data"

INPUTS = {
    "stable": {
        "src": BASE_DIR / "v4" / "data" / "v4_stable_refs_alphaearth.parquet",
        "dst": V5_DATA / "v4_stable_refs_alphaearth_ghm.parquet",
    },
    "candidate": {
        "src": BASE_DIR / "v2" / "data" / "recover_reference_samples_v2_mask_on_large_alphaearth.parquet",
        "dst": V5_DATA / "recover_reference_samples_v2_mask_on_large_alphaearth_ghm.parquet",
    },
}

DEFAULT_BATCH = 2000   # points per EE getInfo round-trip


def initialize() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=PROJECT)


def load_points(src: Path) -> pd.DataFrame:
    """Return a dataframe with row_idx, lon, lat (and the original columns
    re-attached at the end via row_idx)."""
    con = duckdb.connect()
    # Read everything; we need to write the same table back with one new column.
    df = con.execute(
        "SELECT * FROM read_parquet(?)", [str(src)]
    ).df()
    con.close()

    # Extract lon/lat from the geo dict column. EE points are GeoJSON
    # {type: 'Point', coordinates: [lon, lat]}.
    def _lonlat(g):
        if g is None or not isinstance(g, dict):
            return (None, None)
        coords = g.get("coordinates")
        if not coords or len(coords) < 2:
            return (None, None)
        return (float(coords[0]), float(coords[1]))

    lonlats = df["geo"].apply(_lonlat)
    df["lon"] = [p[0] for p in lonlats]
    df["lat"] = [p[1] for p in lonlats]
    return df


def sample_ghm_batch(lons: np.ndarray, lats: np.ndarray,
                     ghm_img: ee.Image, scale: int = SCALE_GHM) -> np.ndarray:
    """Sample GHM at each (lon, lat) using reduceRegions over a FeatureCollection.

    Returns an array of float (NaN where masked / no overlap).
    """
    feats = [
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"i": int(i)})
        for i, (lon, lat) in enumerate(zip(lons, lats))
    ]
    fc = ee.FeatureCollection(feats)
    sampled = ghm_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=scale,
    )
    # Pull back as a list of dicts; preserve order by `i`.
    info = sampled.getInfo()["features"]
    vals = np.full(len(lons), np.nan, dtype=float)
    for f in info:
        props = f.get("properties", {})
        i = props.get("i")
        v = props.get("first")
        if i is not None and v is not None:
            vals[int(i)] = float(v)
    return vals


def attach_ghm(src: Path, dst: Path, batch_size: int, verbose: bool) -> None:
    print(f"\n[{src.name}]")
    df = load_points(src)
    n_total = len(df)
    print(f"  {n_total:,} rows loaded")

    valid = df[df[["lon", "lat"]].notna().all(axis=1)]
    n_valid = len(valid)
    print(f"  {n_valid:,} rows with valid lon/lat")

    ghm_img = ee.ImageCollection(GHM_ASSET).filter(
        ee.Filter.stringContains("system:index", f"_{GHM_BAND}_90")
    ).first()
    ghm_img = ee.Image(ghm_img).rename("ghm_aa")

    ghm_vals = np.full(n_total, np.nan, dtype=float)
    lons = valid["lon"].to_numpy()
    lats = valid["lat"].to_numpy()
    idx  = valid.index.to_numpy()

    n_batches = (n_valid + batch_size - 1) // batch_size
    t0 = time.time()
    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_valid)
        try:
            vals = sample_ghm_batch(lons[s:e], lats[s:e], ghm_img)
        except Exception as exc:
            print(f"  batch {b+1}/{n_batches} failed ({exc}); retrying with half batch")
            half = (e - s) // 2
            v1 = sample_ghm_batch(lons[s:s+half], lats[s:s+half], ghm_img)
            v2 = sample_ghm_batch(lons[s+half:e], lats[s+half:e], ghm_img)
            vals = np.concatenate([v1, v2])
        ghm_vals[idx[s:e]] = vals
        if verbose or (b % 5 == 0 or b == n_batches - 1):
            done = e
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  batch {b+1}/{n_batches}  ({done:,}/{n_valid:,}, "
                  f"{rate:.0f} pts/s)")

    df["ghm_aa"] = ghm_vals

    n_attached = int(np.isfinite(ghm_vals).sum())
    print(f"  GHM attached to {n_attached:,}/{n_total:,} rows "
          f"({100.0*n_attached/n_total:.1f}%)")

    # Write back via duckdb so we keep duckdb's parquet metadata (and, if you're
    # on duckdb >= 1.5, the native GEOMETRY column round-trips as
    # GeoArrow-compliant parquet metadata).
    dst.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("df_view", df)
    con.execute(f"COPY df_view TO '{dst}' (FORMAT PARQUET)")
    con.close()
    print(f"  Wrote {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", choices=["stable", "candidate", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                    help="Points per EE getInfo round-trip. Larger = faster but "
                         "more prone to EE compute timeouts.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    initialize()

    targets = ["stable", "candidate"] if args.target == "both" else [args.target]
    for t in targets:
        cfg = INPUTS[t]
        if not cfg["src"].exists():
            print(f"  skip {t}: {cfg['src']} not found")
            continue
        attach_ghm(cfg["src"], cfg["dst"], args.batch_size, args.verbose)


if __name__ == "__main__":
    main()
