"""Step 6 — buffer references for the 281 target sites that have no v5 buffer refs.

These 281 sites were never in the original DoR sampling (not in the EE asset), so the
v5 buffer ref parquets have no rows for them and they got no local-buffer DoR. This
script samples their buffer references using the EXACT v5 sampling logic (imported from
v5/scripts/sampling/sample_stable_references_v5.py — same good/bad WorldCover routing,
buffer∩ecoregion region, two-tier selected/diagnostic split, GHM attach) but driven from
LOCAL shp geometry (the sites aren't in the asset). Sampling + 2024 embedding extraction
are done in one client-side pass per parent (no heavy EE asset export round-trip).

Group -> v5 stable_class routing (controls the bad pool, _classband_for):
  stable_natural        -> "nature"  (bad = WC{40 crop, 50 built})
  stable_artificial     -> "built"   (bad = WC{50})
  artificial_reversion  -> "built"   (bad = WC{50}; matches built_loss bad pool)

Output: outputs/data/v5_extra_refs_281_alphaearth.parquet
  Same schema as the v5 ref parquets (parent_id, ref_state, dist_m, selection,
  eco_id, ... + A00..A63), keyed by PLOTID in parent_id so Step 3 can union it in.
Then re-run Step 3 (03_score_buffer_dor.py) — it already unions all ref sources and
re-selects 3-8 km — to score these sites too.

NOTE: this output already ships precomputed at
data/cached/v5_extra_refs_281_alphaearth.parquet. Re-running this script
requires a Google Earth Engine account and is only needed to regenerate it
from scratch.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import duckdb
import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUT_DATA, OUT_LOGS, ROOT  # noqa: E402

SAMPLER = ROOT / "scripts" / "sampling" / "sample_stable_references_v5.py"
TARGETS = OUT_DATA / "target_groups.parquet"
OUT = OUT_DATA / "v5_extra_refs_281_alphaearth.parquet"
LOG = OUT_LOGS / "06_extra_refs.log"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
PROJECT = "ee-gsingh"
SCALE = 10
YEAR = 2024
CHUNK = 20               # parents per save-checkpoint batch

GROUP_TO_STABLECLASS = {
    "stable_natural": "nature",
    "stable_artificial": "built",
    "artificial_reversion": "built",
}

# import the v5 sampler module to reuse its sampling logic verbatim
spec = importlib.util.spec_from_file_location("srs5", SAMPLER)
srs5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srs5)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
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
            .reduce(ee.Reducer.first()).regexpRename("_first$", ""))


def sample_parent(plotid, lon, lat, sampler, image) -> pd.DataFrame | None:
    """Run the v5 sampler on one local-geometry parent, then attach 2024 embeddings.

    The parent feature's id() is set to PLOTID so the sampler stamps parent_id=PLOTID
    (consistent with how the rest of test_site_scoring keys these sites)."""
    parent = ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {}) \
        .set("system:index", str(plotid))
    refs = ee.FeatureCollection(sampler(parent))   # sampled ref points (no embeddings)
    # attach 2024 embedding per ref point
    embedded = image.reduceRegions(collection=refs,
                                   reducer=ee.Reducer.first(),
                                   scale=SCALE, tileScale=4)
    df = ee.data.computeFeatures({"expression": embedded,
                                  "fileFormat": "PANDAS_DATAFRAME"})
    if df is None or df.empty:
        return None
    df["parent_id"] = str(plotid)   # ensure PLOTID key (sampler set it, but be explicit)
    return df


def main() -> None:
    con = duckdb.connect()
    tgt = con.execute(
        f'SELECT PLOTID, "group", longitude, latitude '
        f"FROM read_parquet('{TARGETS}') WHERE has_embedding = FALSE"
    ).df()
    con.close()
    log(f"buffer-ref extraction for {len(tgt)} sites with no existing buffer refs")

    init_gee()
    image = alphaearth(YEAR)
    ecoregions = srs5.get_ecoregions()
    ghm = srs5.get_ghm_image()
    exclusion = srs5.build_exclusion_mask(include_buildings_loss=True)

    # one sampler per stable_class (classband differs by bad-pool routing)
    samplers = {}
    for sc in set(GROUP_TO_STABLECLASS.values()):
        cb = srs5._classband_for(sc, exclusion)
        samplers[sc] = srs5.make_sampler(
            classband=cb, parent_label=f"stable_{sc}",
            ecoregions=ecoregions, ghm_image=ghm)

    # resume: skip parents already in a partial output
    done = set()
    if OUT.exists():
        done = set(pd.read_parquet(OUT, columns=["parent_id"])["parent_id"].astype(str))
        log(f"resuming: {len(done)} parents already extracted")

    frames = []
    if OUT.exists():
        frames.append(pd.read_parquet(OUT))

    todo = tgt[~tgt["PLOTID"].astype(str).isin(done)].reset_index(drop=True)
    log(f"{len(todo)} parents to do")

    n_done = 0
    for i, r in enumerate(todo.itertuples(), 1):
        sc = GROUP_TO_STABLECLASS[r.group]
        for attempt in range(4):
            try:
                df = sample_parent(r.PLOTID, r.longitude, r.latitude,
                                   samplers[sc], image)
                if df is not None and not df.empty:
                    df["group"] = r.group
                    frames.append(df)
                    ng = int((df.get("ref_state") == "good").sum()) if "ref_state" in df else 0
                    nb = int((df.get("ref_state") == "bad").sum()) if "ref_state" in df else 0
                    log(f"[{i}/{len(todo)}] {r.PLOTID} ({sc}): "
                        f"{len(df)} refs (good={ng} bad={nb})")
                else:
                    log(f"[{i}/{len(todo)}] {r.PLOTID} ({sc}): no refs")
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                log(f"[{i}/{len(todo)}] {r.PLOTID} attempt {attempt+1} failed "
                    f"({str(e)[:70]}); retry {wait}s")
                time.sleep(wait)
        else:
            log(f"[{i}/{len(todo)}] {r.PLOTID}: FAILED after retries")
        n_done += 1
        # checkpoint every CHUNK parents
        if n_done % CHUNK == 0 and frames:
            out = pd.concat(frames, ignore_index=True)
            tmp = f"{OUT}.tmp.{os.getpid()}"
            out.to_parquet(tmp, index=False)
            os.replace(tmp, OUT)
            log(f"  checkpoint: {out['parent_id'].nunique()} parents written")

    if frames:
        out = pd.concat(frames, ignore_index=True)
        tmp = f"{OUT}.tmp.{os.getpid()}"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, OUT)
        log(f"DONE: {out['parent_id'].nunique()} parents, {len(out)} ref rows -> {OUT}")
    else:
        log("DONE: no refs extracted")


if __name__ == "__main__":
    main()
