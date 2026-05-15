"""Extract AlphaEarth embeddings (64 bands) at parent (test-site) points.

Test-site DoR scoring (`scripts/analysis/score_test_sites.py`) needs the
AlphaEarth embedding sampled at each parent point. That sample is
deterministic for a given (year, parent_asset, scale), so we extract it
once here and cache it as a parquet — `score_test_sites.py` then loads
the cache instead of hitting Earth Engine on every run.

Parents to extract are the unique `parent_id`s in the reference samples
parquet (one row per parent), so the cache stays aligned with whatever
sites the references cover.

Usage:
  python scripts/extraction/extract_test_site_embeddings.py
  python scripts/extraction/extract_test_site_embeddings.py --year 2024
"""

import argparse
import os

import duckdb
import ee

# ── Paths / constants ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
PARENT_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
PROJECT = "ee-gsingh"
SCALE = 10
YEAR = 2024

DEFAULT_REFS = os.path.join(
    DATA_DIR, "recover_reference_samples_alphaearth.parquet")


def init_gee(project=PROJECT):
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=project)
    print(f"[OK] GEE initialised (project={project})")


def build_alphaearth_image(year):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(EMBEDDING_COLLECTION)
        .filterDate(start, end)
        .reduce(ee.Reducer.first())
        .regexpRename("_first$", "")
    )


def fetch_parent_embeddings(parent_ids, year, parent_asset):
    """Sample AlphaEarth at each parent point, server-side per-feature.

    `image.reduceRegions(fc, ...)` silently returns null bands when the
    feature collection spans a very large bounding box with sparse points
    (e.g. test-site parents distributed globally). `fc.map(reduceRegion)`
    evaluates per-feature and avoids that failure mode.
    """
    parent_ids_ee = ee.List(list(map(str, parent_ids)))
    fc = (
        ee.FeatureCollection(parent_asset)
        .filter(ee.Filter.inList('system:index', parent_ids_ee))
        .map(lambda ft: ft.set('parent_id', ft.id()))
        .map(lambda ft: ft.setGeometry(ft.geometry().centroid(1)))
    )
    image = build_alphaearth_image(year)

    def sample_one(ft):
        vals = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=ft.geometry(),
            scale=SCALE,
            tileScale=4,
        )
        return ft.set(vals)

    sampled = fc.map(sample_one)
    df = ee.data.computeFeatures({
        'expression': sampled,
        'fileFormat': 'PANDAS_DATAFRAME',
    })
    return df


def default_output_path(year):
    return os.path.join(DATA_DIR, f"test_site_alphaearth_{year}.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", default=DEFAULT_REFS,
                        help="References parquet — parent_ids extracted from this")
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--parent_asset", default=PARENT_ASSET)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--output", default=None,
                        help="Output parquet path "
                             "(default: data/test_site_alphaearth_<year>.parquet)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if the output already exists")
    args = parser.parse_args()

    out_path = args.output or default_output_path(args.year)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {out_path} already exists. Pass --overwrite to refresh.")
        return

    print(f"Loading parent_ids from {args.refs}")
    con = duckdb.connect()
    parent_ids = (
        con.execute(
            f"SELECT DISTINCT CAST(parent_id AS VARCHAR) AS parent_id "
            f"FROM '{args.refs}'"
        )
        .df()['parent_id'].tolist()
    )
    con.close()
    print(f"  {len(parent_ids)} unique parents")

    init_gee(args.project)
    print(f"Sampling AlphaEarth {args.year} at parent points...")
    parents = fetch_parent_embeddings(parent_ids, args.year, args.parent_asset)
    parents['parent_id'] = parents['parent_id'].astype(str)

    missing = [c for c in EMBED_COLS if c not in parents.columns]
    if missing:
        raise RuntimeError(f"Parent embeddings missing bands: {missing[:3]}...")

    keep = ['parent_id'] + EMBED_COLS + [
        c for c in parents.columns
        if c not in EMBED_COLS and c != 'parent_id'
    ]
    parents = parents[keep]
    parents['year'] = args.year

    parents.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}  ({len(parents)} rows)")


if __name__ == "__main__":
    main()
