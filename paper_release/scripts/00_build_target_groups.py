"""Step 0 — build the target-group manifest for test-site scoring.

Derives the three transition groups from the new shapefile (which now carries the
prior class in `lc_2018`), attaches each site's `parent_id` (EE system:index) and
`eco_id`, and flags which sites still need a 2024 parent embedding extracted.

Groups (2018 -> 2024):
  stable_natural        nature      -> nature
  stable_artificial     artificial  -> artificial
  artificial_reversion  artificial  -> nature
Crop classes are excluded entirely (project decision to drop cropland from DoR).

Output: outputs/data/target_groups.parquet  (one row per target site)
Columns: PLOTID, group, lc_2018, lc_2024, longitude, latitude,
         parent_id (nullable), eco_id (nullable), has_embedding (bool)

NOTE: this output already ships precomputed at data/cached/target_groups.parquet
(it is small and fully determined by the raw shapefile + cached embedding
parquets). Re-running this script is only needed to verify/regenerate it from
data/raw/samples_recover_w_ref_label.shp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CACHED, OUT_DATA, SHP  # noqa: E402

# Test-site embedding parquets: source of PLOTID <-> parent_id and the set of
# sites that already have a cached 2024 embedding.
TS_STABLE = CACHED / "v4_test_site_alphaearth_2024.parquet"
TS_CAND = CACHED / "v5_test_site_alphaearth_2024_candidate.parquet"

# v5 buffer ref parquets: source of parent_id <-> eco_id.
STABLE_REFS = CACHED / "v5_stable_refs_alphaearth.parquet"
CAND_REFS = CACHED / "v5_candidate_refs_alphaearth.parquet"

OUT = OUT_DATA / "target_groups.parquet"

NATURE = "nature (not cropland / artificial)"
ARTIFICIAL = "artificial"

GROUP_MAP = {
    (NATURE, NATURE): "stable_natural",
    (ARTIFICIAL, ARTIFICIAL): "stable_artificial",
    (ARTIFICIAL, NATURE): "artificial_reversion",
}


def main() -> None:
    g = gpd.read_file(SHP)
    g["PLOTID"] = g["PLOTID"].astype(str)
    g["group"] = [GROUP_MAP.get((a, b)) for a, b in zip(g["lc_2018"], g["lc_2024"])]
    tgt = g[g["group"].notna()].copy()

    # The shp contains 54 duplicate PLOTIDs, all at identical geometry (verified).
    # One of them — a126-...45.68 — is an exact-duplicate stable_artificial row;
    # the rest pair a target-group row with a (now-excluded) cropland row at the
    # same point. Dropping exact (PLOTID, transition) duplicates collapses the one
    # true double to a single row and leaves no PLOTID mapping to >1 target group,
    # so PLOTID is a safe key downstream. 1249 raw target rows -> 1248 unique sites.
    n_raw = len(tgt)
    tgt = tgt.drop_duplicates(subset=["PLOTID", "lc_2018", "lc_2024"]).copy()
    if n_raw != len(tgt):
        print(f"[dedup] dropped {n_raw - len(tgt)} exact-duplicate target rows "
              f"({n_raw} -> {len(tgt)})")

    # point coordinates for the local-geometry extraction in step 1. These are
    # ~10 m polygons, so a representative interior point is an exact-enough centre
    # (and avoids the geographic-CRS centroid warning).
    rp = tgt.geometry.representative_point()
    tgt["longitude"] = rp.x.values
    tgt["latitude"] = rp.y.values
    tgt = tgt[["PLOTID", "group", "lc_2018", "lc_2024", "longitude", "latitude"]]

    con = duckdb.connect()

    # PLOTID -> parent_id (1:1 via the embedding parquets)
    pl2pid = con.execute(
        f"""SELECT DISTINCT CAST(PLOTID AS VARCHAR) PLOTID,
                   CAST(parent_id AS VARCHAR) parent_id
            FROM read_parquet('{TS_STABLE}')
            UNION
            SELECT DISTINCT CAST(PLOTID AS VARCHAR) PLOTID,
                   CAST(parent_id AS VARCHAR) parent_id
            FROM read_parquet('{TS_CAND}')"""
    ).df()

    # parent_id -> eco_id (from the v5 buffer refs)
    pid2eco = con.execute(
        f"""SELECT DISTINCT CAST(parent_id AS VARCHAR) parent_id,
                   CAST(eco_id AS INTEGER) eco_id
            FROM read_parquet('{STABLE_REFS}') WHERE eco_id IS NOT NULL
            UNION
            SELECT DISTINCT CAST(parent_id AS VARCHAR) parent_id,
                   CAST(eco_id AS INTEGER) eco_id
            FROM read_parquet('{CAND_REFS}') WHERE eco_id IS NOT NULL"""
    ).df()

    # set of PLOTIDs that already have a cached 2024 embedding
    have_emb = set(pl2pid["PLOTID"])
    con.close()

    out = tgt.merge(pl2pid, on="PLOTID", how="left")
    out = out.merge(pid2eco, on="parent_id", how="left")
    out["has_embedding"] = out["PLOTID"].isin(have_emb)
    # collapse any eco_id duplication from the merge (parent_id can appear under
    # >1 stable_class row in refs but eco_id is constant per parent)
    out = out.drop_duplicates(subset=["PLOTID"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    # ── report / verification ───────────────────────────────────────────
    print(f"Wrote {len(out):,} target sites -> {OUT}")
    print("\ngroup counts:")
    print(out["group"].value_counts().to_string())
    assert len(out) == 1248, f"expected 1248 target sites, got {len(out)}"
    assert out.groupby("group")["PLOTID"].nunique().to_dict() == {
        "stable_natural": 747, "stable_artificial": 386, "artificial_reversion": 115
    }, "group counts changed unexpectedly"

    n_pid = out["parent_id"].notna().sum()
    n_eco = (out["eco_id"].notna() & (out["eco_id"].fillna(-1) >= 0)).sum()
    n_emb = out["has_embedding"].sum()
    n_missing = (~out["has_embedding"]).sum()
    print(f"\nwith parent_id:    {n_pid:,}")
    print(f"with eco_id:       {n_eco:,}")
    print(f"with embedding:    {n_emb:,}")
    print(f"MISSING embedding: {n_missing:,}  (need 2024 extraction)")
    print("\nmissing-embedding by group:")
    print(out[~out["has_embedding"]]["group"].value_counts().to_string())

    # ecoregions spanned by the targets (for step 2 scoping)
    ecos = sorted({int(e) for e in out["eco_id"].dropna() if int(e) >= 0})
    print(f"\ntarget ecoregions (mapped so far): {len(ecos)}")


if __name__ == "__main__":
    main()
