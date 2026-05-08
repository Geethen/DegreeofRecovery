"""Classify each stable_stable parent point into nature / crop / built / ambiguous.

v1-v3 filter `stable_stable` parents out because they have no parent-specific
"bad" reference pool (the disturbance label encodes the bad pool). v4 keeps
them by classifying each parent's *current* state from multi-source agreement
and routing the bad pool accordingly.

Sources:
  Raster sources (sampled in one EE round-trip via sampleRegions):
    - ESA WorldCover v100 (2020) and v200 (2021): 40=crop, 50=built, 80=water, else=nature
    - Dynamic World V1 annual mode (2018-2024): 4=crops, 6=built, 0=water, else=nature
    - ESA WorldCereal `temporarycrops` (2021): boolean cropland

  Building footprint sources (per-country FeatureCollection shards):
    - VIDA Combined buildings (Google + Microsoft) — country shards under
      `projects/sat-io/open-datasets/VIDA_COMBINED/<ISO3>`
    - MS Buildings — country shards under
      `projects/sat-io/open-datasets/MSBuildings/<country_name>`
    Each parent point is joined to a country via
    `ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM0")`, then
    `featurecollection.distance(BUILDING_RADIUS_M)` is used to test whether
    a footprint sits within `BUILDING_RADIUS_M` of the point. A hit votes
    `built`; no-hit / shard-missing votes `unknown`.

Each source casts a vote for {nature, crop, built, unknown}. Final stable_class
requires >= MIN_AGREE sources to agree on the winner; ties or low agreement
=> 'ambiguous'.

Output: CSV with one row per stable_stable parent:
  parent_id, lon, lat, country_iso3,
  vote_wc100, vote_wc200, vote_dw, vote_worldcereal, vote_vida, vote_msbuildings,
  n_nature, n_crop, n_built, stable_class

Usage:
  python v4/scripts/classification/classify_stable_state.py
  python v4/scripts/classification/classify_stable_state.py --min-agree 3
  # Skip building sources entirely (faster; relies on raster sources only):
  python v4/scripts/classification/classify_stable_state.py --skip-buildings
"""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import ee
import pandas as pd

PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"
SCALE = 10

ADM0_ASSET = "WM/geoLab/geoBoundaries/600/ADM0"
ADM0_ISO3_PROP = "shapeGroup"   # ISO3 code in this geoBoundaries asset

VIDA_PREFIX = "projects/sat-io/open-datasets/VIDA_COMBINED"      # /<ISO3>
MSB_PREFIX = "projects/sat-io/open-datasets/MSBuildings"          # /<country_name>
BUILDING_RADIUS_M = 30.0

VOTE_NATURE = "nature"
VOTE_CROP = "crop"
VOTE_BUILT = "built"
VOTE_UNK = "unknown"

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "stable_state_classification.csv"


def initialize() -> None:
    try:
        ee.Initialize(project=PROJECT,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=PROJECT)


def _wc_vote(wc_img: ee.Image, band: str = "Map") -> ee.Image:
    """WorldCover -> {1=nature, 2=crop, 3=built, 0=unknown(water/etc)}.

    40=crop, 50=built, 80=water are the meaningful classes; everything else
    (10 trees, 20 shrub, 30 grass, 60 bare, 90 wetland, 95 mangrove, 100 moss)
    rolls up to nature.
    """
    m = wc_img.select(band)
    return (
        ee.Image(0)
        .where(m.eq(40), 2)
        .where(m.eq(50), 3)
        .where(m.eq(80), 0)
        .where(m.neq(40).And(m.neq(50)).And(m.neq(80)), 1)
        .rename("vote")
        .toByte()
    )


def build_raster_vote_images() -> dict[str, ee.Image]:
    """Vote images for raster sources only (WC100/WC200/DW/WorldCereal).

    Each is an Int8 image with values 1=nature, 2=crop, 3=built, 0=unknown.
    Building footprints are handled separately via per-country FC shards.
    """
    wc100 = ee.ImageCollection("ESA/WorldCover/v100").mosaic()
    wc200 = ee.ImageCollection("ESA/WorldCover/v200").mosaic()

    # Dynamic World 2024 mode (single year is sufficient — parents are
    # stable_stable by construction, so any year is representative). Mode
    # over the full 6-year archive is too heavy to compute per-pixel during
    # sampleRegions and times out.
    # label band: 0=water 1=trees 2=grass 3=flooded_veg 4=crops 5=shrub_scrub
    # 6=built 7=bare 8=snow_ice
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate("2024-01-01", "2025-01-01")
        .select("label")
        .mode()
    )
    dw_vote = (
        ee.Image(0)
        .where(dw.eq(4), 2)            # crops -> crop
        .where(dw.eq(6), 3)            # built -> built
        .where(dw.eq(0), 0)            # water -> unknown
        .where(
            dw.neq(4).And(dw.neq(6)).And(dw.neq(0)).And(dw.gte(1)),
            1,                          # trees/grass/shrub/flooded/bare/snow -> nature
        )
        .rename("vote")
        .toByte()
    )

    # WorldCereal temporary crops (2021) - boolean. 1 -> crop, 0 -> not-crop.
    # Not-crop here means "no opinion on built vs nature", coded as unknown.
    worldcereal = (
        ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
        .filter('product == "temporarycrops"')
        .select("classification")
        .max()
    )
    wc_cereal_vote = (
        ee.Image(0).where(worldcereal.eq(1), 2).rename("vote").toByte()
    )

    return {
        "wc100": _wc_vote(wc100),
        "wc200": _wc_vote(wc200),
        "dw": dw_vote,
        "worldcereal": wc_cereal_vote,
    }


def attach_country_via_join(samples: ee.FeatureCollection) -> ee.FeatureCollection:
    """Server-side spatial join: attach ISO3 to each parent point.

    Uses ee.Join.saveFirst with ee.Filter.intersects rather than rasterising
    ADM0 — much cheaper for sparse points. Each point gets a 'country_iso3'
    string property; points not in any polygon (e.g. ocean) get '' .
    """
    adm0 = ee.FeatureCollection(ADM0_ASSET)
    join = ee.Join.saveFirst(matchKey="_match", outer=True)
    spatial_filter = ee.Filter.intersects(leftField=".geo", rightField=".geo")
    joined = join.apply(samples, adm0, spatial_filter)

    def attach_iso(ft: ee.Feature) -> ee.Feature:
        match = ft.get("_match")
        iso = ee.Algorithms.If(
            match,
            ee.Feature(match).get(ADM0_ISO3_PROP),
            "",
        )
        return ft.set("country_iso3", ee.String(iso)).set("_match", None)

    return joined.map(attach_iso)


def sample_raster_votes(samples: ee.FeatureCollection,
                        votes: dict[str, ee.Image]) -> ee.FeatureCollection:
    """Stack all vote bands and sample once per parent point."""
    bands = [img.rename(f"v_{name}") for name, img in votes.items()]
    stack = ee.Image.cat(bands)
    return stack.sampleRegions(
        collection=samples,
        scale=SCALE,
        geometries=True,
        tileScale=2,
    )


# ---------------------------------------------------------------------------
# Building-footprint votes — per-country FC shards
# ---------------------------------------------------------------------------

def _shard_exists(asset_id: str) -> bool:
    """Cheap existence check; getInfo() throws if asset is missing."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def _vida_asset_for(iso3: str) -> str | None:
    asset_id = f"{VIDA_PREFIX}/{iso3}"
    return asset_id if _shard_exists(asset_id) else None


def _msb_asset_for(country_name: str) -> str | None:
    """MSBuildings shards use country names (e.g. 'Australia'), not ISO codes."""
    asset_id = f"{MSB_PREFIX}/{country_name}"
    return asset_id if _shard_exists(asset_id) else None


def _iso3_to_msb_country(iso3: str, override_map: dict[str, str] | None = None
                         ) -> str | None:
    """Map ISO3 to the country-name slug used by MSBuildings shards.

    The MSB asset list uses country *names* with mixed conventions
    ('UnitedStates', 'CzechRepublic', 'KoreaSouth'). We don't ship a full
    table here; users can pass `--msb-country-map` to provide one. If the
    mapping is missing, MSBuildings simply votes 'unknown' for that point.
    """
    if override_map and iso3 in override_map:
        return override_map[iso3]
    return None


def building_votes_for_points(
    points_df: pd.DataFrame,
    msb_country_map: dict[str, str] | None,
    radius_m: float = BUILDING_RADIUS_M,
    verbose: bool = False,
) -> pd.DataFrame:
    """Add v_vida and v_msbuildings vote columns by querying per-country shards.

    Strategy:
      - Group points by country_iso3.
      - For each country with at least one parent point:
          * Look up VIDA shard at VIDA_PREFIX/<ISO3>; if present, query each
            point with `shard.filterBounds(point.buffer(radius_m)).size()`.
            Hits => v_vida = 3 (built); else 1 (nature-default; the absence
            of a footprint is a vote against built). Treat shard-missing as
            v_vida = 0 (unknown).
          * Same for MSBuildings using msb_country_map[ISO3] -> country-name.

    Note: 'absence of footprint' is conservatively voted as 'nature' rather
    than 'unknown' — building footprint datasets aim to be globally complete,
    so a verified-no-hit is informative. If the shard itself is missing, that
    is unknown (we have no evidence either way).
    """
    out = points_df.copy()
    out["v_vida"] = 0           # default: unknown
    out["v_msbuildings"] = 0    # default: unknown

    out["country_iso3"] = out["country_iso3"].fillna("").astype(str)
    countries = sorted(c for c in out["country_iso3"].unique() if c)
    if verbose:
        print(f"  Countries to query: {len(countries)}")

    for iso3 in countries:
        idx = out.index[out["country_iso3"] == iso3]

        # VIDA: ISO3 directly indexes the shard.
        vida_asset = _vida_asset_for(iso3)
        if vida_asset is not None:
            vida = ee.FeatureCollection(vida_asset)
            vida_votes = _query_shard_for_points(out.loc[idx], vida, radius_m)
            out.loc[idx, "v_vida"] = vida_votes
        elif verbose:
            print(f"    {iso3}: VIDA shard missing")

        # MSBuildings: needs ISO3->country-name translation.
        msb_country = _iso3_to_msb_country(iso3, msb_country_map)
        if msb_country is not None:
            msb_asset = _msb_asset_for(msb_country)
            if msb_asset is not None:
                msb = ee.FeatureCollection(msb_asset)
                msb_votes = _query_shard_for_points(out.loc[idx], msb, radius_m)
                out.loc[idx, "v_msbuildings"] = msb_votes
            elif verbose:
                print(f"    {iso3} -> {msb_country}: MSBuildings shard missing")

    return out


def _query_shard_for_points(
    sub_df: pd.DataFrame, shard: ee.FeatureCollection, radius_m: float,
) -> list[int]:
    """For each point in `sub_df`, return 3 if any footprint within radius_m, else 1.

    Implementation: build a single FC of all points in this country, map
    `filterBounds(point.buffer(radius_m)).size()` over them, return as a
    list of ints. One getInfo() per country.
    """
    feats = [
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                   {"_idx": int(i)})
        for i, r in sub_df.reset_index(drop=True).iterrows()
    ]
    fc = ee.FeatureCollection(feats)

    def has_footprint(ft: ee.Feature) -> ee.Feature:
        n = shard.filterBounds(ft.geometry().buffer(radius_m)).size()
        vote = ee.Algorithms.If(n.gt(0), 3, 1)
        return ft.set("vote", vote)

    voted = fc.map(has_footprint)
    info = voted.aggregate_array("vote").getInfo()
    return [int(v) for v in info]


def _decode_vote(v: int) -> str:
    return {1: VOTE_NATURE, 2: VOTE_CROP, 3: VOTE_BUILT}.get(int(v), VOTE_UNK)


def aggregate(df: pd.DataFrame, min_agree: int) -> pd.DataFrame:
    vote_cols = [c for c in df.columns if c.startswith("v_")]
    decoded = df[vote_cols].apply(lambda col: col.map(_decode_vote))
    decoded.columns = [c.replace("v_", "vote_") for c in decoded.columns]

    n_nature = (decoded == VOTE_NATURE).sum(axis=1)
    n_crop = (decoded == VOTE_CROP).sum(axis=1)
    n_built = (decoded == VOTE_BUILT).sum(axis=1)

    def decide(nn: int, nc: int, nb: int) -> str:
        winner = max((nn, VOTE_NATURE), (nc, VOTE_CROP), (nb, VOTE_BUILT),
                     key=lambda x: x[0])
        if winner[0] < min_agree:
            return "ambiguous"
        # Tie check: any other class equals the winner count.
        ties = [name for cnt, name in
                ((nn, VOTE_NATURE), (nc, VOTE_CROP), (nb, VOTE_BUILT))
                if cnt == winner[0] and name != winner[1]]
        if ties:
            return "ambiguous"
        return winner[1]

    stable_class = [decide(a, b, c) for a, b, c in zip(n_nature, n_crop, n_built)]

    base_cols = ["parent_id", "lon", "lat"]
    if "country_iso3" in df.columns:
        base_cols.append("country_iso3")
    out = pd.concat([
        df[base_cols].reset_index(drop=True),
        decoded.reset_index(drop=True),
    ], axis=1)
    out["n_nature"] = n_nature.values
    out["n_crop"] = n_crop.values
    out["n_built"] = n_built.values
    out["stable_class"] = stable_class
    return out


# ---------------------------------------------------------------------------
# Sharded extraction (mirrors v1 extract_alphaearth_embeddings.py pattern)
# ---------------------------------------------------------------------------

TARGET_SHARD_SIZE = 200   # points/shard — conservative for multi-band stack
MAX_WORKERS = 20
SEED = 42


def _extract_sharded(
    sampled_fc: ee.FeatureCollection,
    n_points: int,
    max_workers: int = MAX_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """Pull all features from `sampled_fc` using sharded computeFeatures.

    Mirrors the v1 extraction pattern: assign each feature a random shard
    index, then call ee.data.computeFeatures per shard in parallel threads.
    """
    n_shards = max(1, math.ceil(n_points / TARGET_SHARD_SIZE))
    print(f"  {n_points} points  ->  {n_shards} shards "
          f"(target {TARGET_SHARD_SIZE}/shard, {max_workers} workers)")

    sharded = (
        sampled_fc.randomColumn("_shard_rand", seed=SEED)
        .map(lambda f: f.set(
            "_shard_idx",
            ee.Number(f.get("_shard_rand")).multiply(n_shards).floor()
            .min(n_shards - 1),
        ))
    )

    lock = Lock()
    frames: list[pd.DataFrame] = []
    failed: list[int] = []

    def pull_shard(i: int) -> pd.DataFrame:
        shard = sharded.filter(ee.Filter.eq("_shard_idx", i))
        for attempt in range(4):
            try:
                df = ee.data.computeFeatures({
                    "expression": shard,
                    "fileFormat": "PANDAS_DATAFRAME",
                })
                return df if df is not None else pd.DataFrame()
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2 ** (attempt + 1)
                time.sleep(wait)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(pull_shard, i): i for i in range(n_shards)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            done += 1
            try:
                df = future.result()
                if df is not None and not df.empty:
                    with lock:
                        frames.append(df)
                if verbose or done % 5 == 0:
                    print(f"  shard {i} done  ({done}/{n_shards})", flush=True)
            except Exception as e:
                failed.append(i)
                print(f"  [ERROR] shard {i}: {str(e)[:120]}")

    if failed:
        print(f"  {len(failed)} shards failed: {failed}")
    if not frames:
        raise RuntimeError("No data returned from any shard.")
    return pd.concat(frames, ignore_index=True)


def _load_msb_country_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    import csv as _csv
    out: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            out[row["iso3"].strip()] = row["msb_country"].strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples-asset", default=SAMPLES_ASSET)
    parser.add_argument(
        "--out-csv", default=str(DEFAULT_OUT),
        help="Path to write the per-parent classification CSV.",
    )
    parser.add_argument(
        "--min-agree", type=int, default=2,
        help="Minimum number of sources that must vote for the winning class. "
             "Below this -> 'ambiguous'.",
    )
    parser.add_argument(
        "--skip-buildings", action="store_true",
        help="Skip VIDA + MSBuildings; classify from raster sources only.",
    )
    parser.add_argument(
        "--msb-country-map", default=None,
        help="CSV with columns iso3,msb_country mapping ISO3 codes to "
             "MSBuildings asset country names (e.g. AUS,Australia). Without "
             "this, MSBuildings votes 'unknown' for every point.",
    )
    parser.add_argument(
        "--building-radius-m", type=float, default=BUILDING_RADIUS_M,
        help="Radius (m) within which a building footprint counts as 'built'.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initialize()

    samples = (
        ee.FeatureCollection(args.samples_asset)
        .filter(ee.Filter.eq("r", "stable_stable"))
        .map(lambda ft: ee.Feature(ft.geometry().centroid(5))
             .set({"parent_id": ft.id()}))
    )
    if args.limit is not None:
        samples = samples.limit(args.limit)

    n = samples.size().getInfo()
    print(f"stable_stable parents to classify: {n}")

    # Attach ISO3 country code via server-side spatial join (cheap; no
    # ADM0 rasterisation needed).
    print("Attaching country ISO3 via spatial join ...", flush=True)
    samples_with_country = attach_country_via_join(samples)

    # Sample raster vote images at each (now country-tagged) point, and add
    # lon/lat for the optional building-shard step.
    raster_votes = build_raster_vote_images()
    sampled = sample_raster_votes(samples_with_country, raster_votes)
    sampled = sampled.map(
        lambda ft: ft.set({
            "lon": ft.geometry().coordinates().get(0),
            "lat": ft.geometry().coordinates().get(1),
        })
    )

    print("Extracting raster votes via sharded computeFeatures ...")
    df = _extract_sharded(sampled, n, max_workers=args.max_workers, verbose=args.verbose)

    if "country_iso3" not in df.columns:
        df["country_iso3"] = ""
    else:
        df["country_iso3"] = df["country_iso3"].fillna("").astype(str)

    if args.skip_buildings:
        df["v_vida"] = 0
        df["v_msbuildings"] = 0
    else:
        msb_map = _load_msb_country_map(args.msb_country_map)
        if msb_map is None:
            print("[warn] No --msb-country-map provided; MSBuildings will "
                  "vote 'unknown' for all points. VIDA shards (ISO3-keyed) "
                  "will still be queried.")
        df = building_votes_for_points(
            df, msb_country_map=msb_map,
            radius_m=args.building_radius_m, verbose=args.verbose,
        )

    out = aggregate(df, min_agree=args.min_agree)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"\nstable_class breakdown (min_agree={args.min_agree}):")
    print(out["stable_class"].value_counts().to_string())
    print(f"\nWrote {args.out_csv}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
