"""Step 3 — local-buffer DoR for the three target groups, at the empirical 3-8 km buffer.

Fixes the buffer caveat: the v5 `selected` refs were cut at inner=4 km (the
v4-reproducible fallback), inconsistent with the desirability-analysis optimum of
inner=3 km / outer=8 km. The 3-4 km refs already exist WITH embeddings in the
`diagnostic` pool, so we simply RE-SELECT every ref with 3000 <= dist_m < 8000 from the
full v5 ref parquets (selected + diagnostic) — no new Earth Engine extraction.

Scoring math (k=5 kNN, 2000-bootstrap CI, cosine distance) is reused unchanged from
degree_of_recovery.core, exactly as v5/scripts/analysis/score_test_sites_v5.py.

Per-group bad-pool threshold (calibrated_thresholds_v4.json):
  stable_natural        -> stable_nature
  stable_artificial     -> stable_built
  artificial_reversion  -> stable_built   (built-up bad pool, as score_built_loss_v5.py)

Coverage: only sites with a parent_id (=> v5 buffer refs) AND >=1 good and >=1 bad ref in
the 3-8 km band are scored. The 281 sites with no parent_id (no buffer refs) are
ecoregion-scored only (step 4).

Output: test_site_scoring/data/test_site_dor_buffer_3_8km.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE = Path("/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery")
sys.path.insert(0, str(BASE / "src"))
from degree_of_recovery.core import (  # noqa: E402
    bootstrap_ci,
    classify,
    cosine_dists_to_set,
    knn_score as _knn_score,
    median_score as _median_score,
)

TARGETS = BASE / "test_site_scoring/data/target_groups.parquet"
STABLE_REFS = BASE / "v5/data/v5_stable_refs_alphaearth.parquet"
CAND_REFS = BASE / "v5/data/v5_candidate_refs_alphaearth.parquet"
# Step 6: buffer refs for the 281 sites that were not in the original DoR sampling.
# These key parent_id == PLOTID (set by the local-geometry sampler in step 6).
EXTRA_REFS = BASE / "test_site_scoring/data/v5_extra_refs_281_alphaearth.parquet"
# parent (test-site) embeddings: union of v4 stable + v5 candidate + step-1 missing.
# The v4/v5 parquets key parent_id (EE index); the missing parquet keys PLOTID.
TS_STABLE = BASE / "v4/data/test_site_alphaearth_2024_v4.parquet"
TS_CAND = BASE / "v5/data/test_site_alphaearth_2024_candidate.parquet"
TS_MISSING = BASE / "test_site_scoring/data/test_site_alphaearth_2024_missing.parquet"
THRESH_JSON = BASE / "v4/data/calibrated_thresholds_v4.json"
OUT = BASE / "test_site_scoring/data/test_site_dor_buffer_3_8km.csv"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
INNER_M, OUTER_M = 3000, 8000        # empirical desirability-analysis optimum
KNN_K, N_BOOT, SEED, MEDIAN_THRESHOLD = 5, 2000, 42, 0.5

GROUP_THRESHOLD = {
    "stable_natural": "stable_nature",
    "stable_artificial": "stable_built",
    "artificial_reversion": "stable_built",
}


def load_band_refs(con, parquet: Path) -> pd.DataFrame:
    """All refs in the 3-8 km band with a non-null embedding, from selected+diagnostic."""
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)
    df = con.execute(
        f"""SELECT CAST(parent_id AS VARCHAR) parent_id, ref_state, {embed_sql}
            FROM read_parquet('{parquet}')
            WHERE dist_m >= {INNER_M} AND dist_m < {OUTER_M}
              AND A00 IS NOT NULL AND ref_state IN ('good','bad')"""
    ).df()
    return df


def load_parent_embeddings(con) -> dict[str, np.ndarray]:
    """Mean 2024 parent embedding keyed by the ref-key used downstream.

    Original sites: keyed by parent_id (EE index), from v4 stable + v5 candidate.
    The 281 step-6 sites: keyed by PLOTID, from the step-1 missing parquet (their
    step-6 refs also use parent_id == PLOTID, so the keys line up)."""
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)
    frames = []
    for p in (TS_STABLE, TS_CAND):
        frames.append(con.execute(
            f'SELECT CAST(parent_id AS VARCHAR) emb_key, {embed_sql} '
            f"FROM read_parquet('{p}')").df())
    if TS_MISSING.exists():
        frames.append(con.execute(
            f'SELECT CAST(PLOTID AS VARCHAR) emb_key, {embed_sql} '
            f"FROM read_parquet('{TS_MISSING}')").df())
    df = pd.concat(frames, ignore_index=True).dropna(subset=EMBED_COLS)
    return {
        k: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
        for k, g in df.groupby("emb_key", sort=False)
    }


def main() -> None:
    thresholds = json.load(open(THRESH_JSON))
    rng = np.random.default_rng(SEED)
    knn_fn = lambda dg, db: _knn_score(dg, db, KNN_K)
    median_fn = lambda dg, db: _median_score(dg, db)

    con = duckdb.connect()
    targets = con.execute(
        f'SELECT PLOTID, "group", parent_id FROM read_parquet(\'{TARGETS}\')'
    ).df()
    targets["PLOTID"] = targets["PLOTID"].astype(str)
    # ref-key = EE parent_id for original sites, else PLOTID (the step-6 281 sites,
    # whose step-6 refs + step-1 embeddings both key on PLOTID).
    targets["ref_key"] = targets["parent_id"].astype("string").fillna(
        targets["PLOTID"])

    ref_sources = [load_band_refs(con, STABLE_REFS), load_band_refs(con, CAND_REFS)]
    if EXTRA_REFS.exists():
        ref_sources.append(load_band_refs(con, EXTRA_REFS))
        print(f"including step-6 extra refs: {EXTRA_REFS.name}")
    refs = pd.concat(ref_sources, ignore_index=True)
    emb = load_parent_embeddings(con)
    con.close()

    refs_groups = dict(tuple(refs.groupby("parent_id", sort=False)))
    print(f"target sites: {len(targets)}")
    print(f"parents with 3-8 km refs:   {len(refs_groups)}")

    rows = []
    skipped = {"no_emb": 0, "no_refs": 0, "missing_pool": 0}
    for r in targets.itertuples():
        pid, group = r.ref_key, r.group
        x = emb.get(pid)
        sub = refs_groups.get(pid)
        if x is None:
            skipped["no_emb"] += 1
            continue
        if sub is None:
            skipped["no_refs"] += 1
            continue
        good = sub.loc[sub["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
        bad = sub.loc[sub["ref_state"] == "bad", EMBED_COLS].to_numpy(dtype=float)
        if len(good) == 0 or len(bad) == 0:
            skipped["missing_pool"] += 1
            continue

        tkey = GROUP_THRESHOLD[group]
        thr = float(thresholds[tkey])
        d_g = cosine_dists_to_set(x, good)
        d_b = cosine_dists_to_set(x, bad)
        s_knn, lo_knn, hi_knn = bootstrap_ci(knn_fn, d_g, d_b, N_BOOT, rng)
        s_med, lo_med, hi_med = bootstrap_ci(median_fn, d_g, d_b, N_BOOT, rng)
        rows.append({
            "PLOTID": r.PLOTID,
            "parent_id": pid,
            "group": group,
            "n_good": int(len(good)),
            "n_bad": int(len(bad)),
            "inner_m": INNER_M,
            "outer_m": OUTER_M,
            "dor_knn": s_knn,
            "dor_knn_ci_low": lo_knn,
            "dor_knn_ci_high": hi_knn,
            "t_knn_used": thr,
            "threshold_key": tkey,
            "category_knn": classify(s_knn, lo_knn, hi_knn, thr),
            "dor_median": s_med,
            "dor_median_ci_low": lo_med,
            "dor_median_ci_high": hi_med,
            "category_median": classify(s_med, lo_med, hi_med, MEDIAN_THRESHOLD),
        })

    df = pd.DataFrame(rows).sort_values(["group", "PLOTID"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\nWrote {len(df)} buffer-scored sites -> {OUT}")
    print(f"skipped: {skipped}")
    print("\nn scored by group:")
    print(df.groupby("group").size().to_string())
    print("\nmedian dor_knn by group (expect stable_natural > reversion > stable_artificial):")
    print(df.groupby("group")["dor_knn"].median().round(4).to_string())
    print("\ncategory_knn by group:")
    print(df.groupby("group")["category_knn"].value_counts().to_string())

    # verification: every scored site had both pools populated
    assert (df["n_good"] > 0).all() and (df["n_bad"] > 0).all()


if __name__ == "__main__":
    main()
