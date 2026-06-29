"""Step 2, Path B — FSCS ecoregion reference sampling at the 2024 AlphaEarth vintage.

For the target ecoregions (containing >=1 of the three groups) that have NO existing
2022 ref file, we run the RECOVER/DoR FSCS pipeline at the **2024** vintage with two
cost controls the year-caveat fix makes safe:

  * order ecoregions smallest -> largest (by grid-cell count) so cheap ecos finish first;
  * **cap at ~5,000 points per ecoregion** (10x the scorer's N_BASELINE=500): submit grid
    cells in batches and stop once the cap is met (but not before both pools clear
    MIN_REFS=10). FSCS yields <=100 pts/cell, so ~5k points is ~50-100 populated cells —
    a 1-2 order-of-magnitude saving on large ecoregions with no effect on the percentile
    scores (which only ever use ~500 baseline points).

Reuses the FSCS sampler, natural mask, and helpers from
v1-ecoregion/scripts/sampling/sample_reference_states.py (imported, not re-implemented).
The cell loop here is a capped variant of that module's run_ecoregion.

Output: test_site_scoring/data/ref_samples_eco{id}_2024.parquet  (suffix _2024).
Run detached; per-ecoregion outputs are atomic so a kill loses at most the in-flight eco.
A 30-minute heartbeat reports progress + ETA.
"""
from __future__ import annotations

import importlib.util
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import ee
import pandas as pd

BASE = Path("/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery")
SAMPLER = BASE / "v1-ecoregion/scripts/sampling/sample_reference_states.py"
TARGETS = BASE / "test_site_scoring/data/target_groups.parquet"
MISSING = BASE / "test_site_scoring/data/test_site_alphaearth_2024_missing.parquet"
ECO_DIR = BASE / "v1-ecoregion/data"
OUT_DIR = BASE / "test_site_scoring/data"
LOG = BASE / "test_site_scoring/logs/02b_fscs.log"

YEAR = 2024
POINT_CAP = 5000
MIN_REFS = 10            # both pools must clear this before honouring the cap
CELL_BATCH = 24          # cells submitted per parallel batch
MAX_WORKERS = 20
HEARTBEAT_S = 1800       # 30-minute progress heartbeat

spec = importlib.util.spec_from_file_location("srs", SAMPLER)
srs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srs)

EMBED_COLS = [f"A{i:02d}" for i in range(64)]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def build_stack_2024():
    aef = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
           .reduce(ee.Reducer.first()).regexpRename("_first$", ""))
    natural_mask = srs.get_natural_mask()
    return aef, ee.Image.cat([aef, natural_mask, ee.Image.pixelLonLat()])


def target_path_b_ecos() -> list[int]:
    con = duckdb.connect()
    ecos = set(int(e) for e in con.execute(
        f"SELECT DISTINCT eco_id FROM read_parquet('{TARGETS}') "
        f"WHERE eco_id IS NOT NULL AND CAST(eco_id AS INT) >= 0").df()["eco_id"])
    if MISSING.exists():
        ecos |= set(int(e) for e in con.execute(
            f"SELECT DISTINCT eco_id FROM read_parquet('{MISSING}') "
            f"WHERE CAST(eco_id AS INT) >= 0").df()["eco_id"])
    con.close()
    have = {int(re.search(r"eco(\d+)", f).group(1))
            for f in os.listdir(ECO_DIR)
            if re.match(r"ref_samples_eco\d+\.parquet$", f)}
    have |= {int(re.search(r"eco(\d+)", f).group(1))
             for f in os.listdir(OUT_DIR)
             if re.match(r"ref_samples_eco\d+_2024\.parquet$", f)}
    return sorted(ecos - have)


def grid_cells(eco_id, grid_scale=srs.GRID_SCALE, filter_to_poly=False):
    """Covering grid over the ecoregion. If filter_to_poly, keep only cells that
    intersect the (simplified) ecoregion polygon — drops the many empty bounding-box
    cells whose per-cell pixel-count probe would otherwise cost a slow EE round-trip."""
    eco = ee.FeatureCollection(srs.ECOREGIONS_ASSET).filter(
        ee.Filter.eq("ECO_ID", eco_id))
    geom = eco.geometry()
    grid = geom.bounds().coveringGrid(geom.projection(), grid_scale)
    if filter_to_poly:
        simp = geom.simplify(maxError=1000)
        grid = grid.filterBounds(simp)
    n = int(srs.retry_gee(lambda: grid.size().getInfo()))
    return geom, grid, n


TARGET_CELLS = 1500          # aim for <= this many bbox cells; coarsen grid for giants


def adaptive_grid(eco_id):
    """Pick a grid scale near TARGET_CELLS, then drop empty cells via filterBounds.

    Two compounding speedups for giant ecoregions:
      * coarsen the grid (cells ~ area/scale^2) so a giant's tens-of-thousands of cells
        collapse toward TARGET_CELLS — fewer cells to consider, no effect on the capped
        FSCS output (we still FSCS within each, larger, cell);
      * filterBounds to the polygon so the mostly-empty bounding-box cells (ocean / outside
        the ecoregion) are never probed — those empty cells each cost a 12-33s EE call.
    Returns (geom, grid_of_populated_cells, n_cells, scale).
    """
    import math
    base = srs.GRID_SCALE                       # 10 km
    geom, _, n_bbox = grid_cells(eco_id, base)  # cheap bbox count to size the grid
    scale = base
    if n_bbox > TARGET_CELLS:
        scale = int(base * math.sqrt(n_bbox / TARGET_CELLS))
    # build the polygon-filtered grid at the chosen scale
    geom, grid, n = grid_cells(eco_id, scale, filter_to_poly=True)
    return geom, grid, n, scale


def sample_cell(cell_geom, eco_geom, eco_id, cell_idx, aef, aef_fscs, stack):
    """One cell's FSCS sample + 2024 extraction -> DataFrame (or None)."""
    try:
        cell_eco = cell_geom.intersection(eco_geom, ee.ErrorMargin(1))
        px = aef.select(0).reduceRegion(
            reducer=ee.Reducer.count(), geometry=cell_eco,
            scale=srs.SCALE, maxPixels=1e8
        ).getNumber(aef.bandNames().get(0)).getInfo()
    except Exception:
        return None
    if px is None or px < 10:
        return None
    k = min(srs.N_CLUSTERS, max(2, int(px // 5)))
    n_init = min(srs.N_INIT_POINTS, int(px))
    try:
        _, samples = srs.fscs(covariates=aef_fscs, n_clusters=k,
                              n_init_points=n_init, geometry=cell_eco,
                              scale=srs.SCALE, seed=srs.SEED)
        sampled = stack.reduceRegions(collection=samples,
                                      reducer=ee.Reducer.first(),
                                      scale=srs.SCALE, tileScale=4)
        df = ee.data.computeFeatures({"expression": sampled,
                                      "fileFormat": "PANDAS_DATAFRAME"})
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df["eco_id"] = eco_id
    df["cell_index"] = cell_idx
    return df


def iter_cell_pages(grid, n_cells, page_size=2000):
    """Yield grid cells in randomized pages of <=page_size features.

    EE's getInfo() aborts a collection query past ~5000 elements, so large grids
    (giant ecoregions) must be paged. We randomize with randomColumn so each page is
    a spatially-mixed sample of the ecoregion -> the first page alone usually contains
    enough populated cells to hit the point cap, and we never download the whole grid.
    """
    if n_cells <= page_size:
        feats = srs.retry_gee(lambda: grid.getInfo()["features"])
        yield feats
        return
    tagged = grid.randomColumn("_rnd", seed=srs.SEED)
    n_pages = (n_cells + page_size - 1) // page_size
    for pi in range(n_pages):
        lo, hi = pi / n_pages, (pi + 1) / n_pages
        feats = srs.retry_gee(lambda l=lo, h=hi: tagged.filter(
            ee.Filter.And(ee.Filter.gte("_rnd", l),
                          ee.Filter.lt("_rnd", h))).getInfo()["features"])
        yield feats


def run_capped(eco_id, aef, aef_fscs, stack) -> None:
    out_path = OUT_DIR / f"ref_samples_eco{eco_id}_2024.parquet"
    if out_path.exists():
        log(f"eco {eco_id}: [skip] exists")
        return

    geom, grid, n_cells, scale = adaptive_grid(eco_id)
    log(f"eco {eco_id}: {n_cells} cells @ {scale/1000:.0f}km grid; "
        f"collecting up to {POINT_CAP} pts "
        f"({'paged' if n_cells > 2000 else 'single'})")

    collected = []
    n_good = n_bad = 0
    cap_met = False
    global_idx = 0
    for feats in iter_cell_pages(grid, n_cells):
        if cap_met:
            break
        pos = 0
        while pos < len(feats):
            page_batch = feats[pos:pos + CELL_BATCH]
            base = global_idx + pos
            pos += CELL_BATCH
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futs = {pool.submit(srs.retry_gee, lambda f=f, ci=base + k:
                                    sample_cell(ee.Geometry(f["geometry"]), geom,
                                                eco_id, ci, aef, aef_fscs, stack)):
                        k for k, f in enumerate(page_batch)}
                for fut in as_completed(futs):
                    try:
                        df = fut.result(timeout=srs.CELL_TIMEOUT_S)
                    except Exception:
                        df = None
                    if df is not None and not df.empty:
                        collected.append(df)
                        if "natural" in df.columns:
                            n_good += int((df["natural"] == 1).sum())
                            n_bad += int((df["natural"] == 0).sum())
            # stop once cap met AND both pools clear MIN_REFS
            total = n_good + n_bad
            if total >= POINT_CAP and n_good >= MIN_REFS and n_bad >= MIN_REFS:
                log(f"eco {eco_id}: cap met ({total} pts, good={n_good} "
                    f"bad={n_bad}) after ~{base + len(page_batch)} cells")
                cap_met = True
                break
        global_idx += len(feats)

    if not collected:
        log(f"eco {eco_id}: [WARN] no data collected")
        return

    out = pd.concat(collected, ignore_index=True)
    # dedup on geo like the original consolidation
    if "geo" in out.columns:
        out = out.drop_duplicates(subset=["geo"])
    keep = [c for c in ["natural", "eco_id", "cell_index", "latitude",
                        "longitude", "geo"] if c in out.columns] + \
           [c for c in EMBED_COLS if c in out.columns]
    out = out[keep]
    tmp = f"{out_path}.tmp.{os.getpid()}"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    ng = int((out["natural"] == 1).sum()); nb = int((out["natural"] == 0).sum())
    log(f"eco {eco_id}: wrote {len(out)} pts (good={ng} bad={nb}) -> {out_path.name}")


def main() -> None:
    srs.init_gee()
    log(f"Path B: FSCS at {YEAR}, point cap={POINT_CAP}")
    aef, stack = build_stack_2024()
    aef_fscs = aef.select(aef.bandNames().slice(0, srs.FSCS_BANDS))

    ecos = target_path_b_ecos()
    log(f"{len(ecos)} target ecoregions need FSCS (no 2022 file, no _2024 yet)")

    log("Ranking ecoregions by grid-cell count (smallest first)...")
    sized = []
    for e in ecos:
        try:
            _, _, n = grid_cells(e)
        except Exception as ex:
            log(f"  eco {e}: size probe failed ({str(ex)[:50]}); deferring")
            n = 10 ** 9
        sized.append((e, n))
    sized.sort(key=lambda x: x[1])
    ordered = [e for e, _ in sized]
    cells_map = dict(sized)
    log(f"order (smallest first): {ordered[:12]}... ({len(ordered)} total)")

    state = {"done": 0, "total": len(ordered), "current": None, "stop": False}
    t0 = time.time()

    def heartbeat():
        while not state["stop"]:
            time.sleep(HEARTBEAT_S)
            if state["stop"]:
                break
            el = (time.time() - t0) / 60
            d, tot = state["done"], state["total"]
            eta = (el / d * (tot - d)) if d else float("nan")
            log(f"[HEARTBEAT] {d}/{tot} ecos done, current=eco {state['current']}, "
                f"elapsed={el:.0f}min, ETA~{eta:.0f}min")

    threading.Thread(target=heartbeat, daemon=True).start()

    for j, eco_id in enumerate(ordered, 1):
        state["current"] = eco_id
        log(f"=== [{j}/{len(ordered)}] eco {eco_id} (cells~{cells_map.get(eco_id)}) ===")
        try:
            run_capped(eco_id, aef, aef_fscs, stack)
        except Exception as ex:
            log(f"eco {eco_id}: [ERROR] {str(ex)[:120]}")
        state["done"] = j

    state["stop"] = True
    log(f"Path B complete: {state['done']}/{state['total']} ecoregions.")


if __name__ == "__main__":
    main()
