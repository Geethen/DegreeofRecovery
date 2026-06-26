"""Score the 90 built-loss test sites with the v5 buffer (per-site DoR, k=5).

The v4 scorer (`v4/scripts/analysis/score_test_sites_v4.py`) only produced the
stable-control scores. The built-loss test sites were scored against the v5
candidate reference pool (operational annulus, inner exclusion 4 km, outer ceiling
5 km expanding to 8 km when the count target is unmet). This script reproduces
that score per built-loss parent so the result can be exported as a shapefile.

Bad pool for built-loss = built-up pixels, so the operating threshold is the
stable_built calibrated kNN threshold (0.4948, from calibrated_thresholds_v4.json).

Inputs (all local, no Earth Engine):
  v5/data/v5_candidate_refs_alphaearth.parquet        (selection='selected' refs)
  v5/data/test_site_alphaearth_2024_candidate.parquet (parent 2024 embeddings)
  v4/data/calibrated_thresholds_v4.json               (per-class kNN thresholds)

Output:
  v5/data/test_site_dor_v5_built_loss.csv
"""
from __future__ import annotations

import json
import os

import duckdb
import numpy as np
import pandas as pd

from degree_of_recovery.core import (
    bootstrap_ci,
    classify,
    cosine_dists_to_set,
    knn_score as _knn_score,
    median_score as _median_score,
)

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
BASE = "/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery"
CAND_REFS = f"{BASE}/v5/data/v5_candidate_refs_alphaearth.parquet"
CAND_TS = f"{BASE}/v5/data/test_site_alphaearth_2024_candidate.parquet"
THRESH_JSON = f"{BASE}/v4/data/calibrated_thresholds_v4.json"
OUT = f"{BASE}/v5/data/test_site_dor_v5_built_loss.csv"

KNN_K = 5
N_BOOT = 2000
SEED = 42
MEDIAN_THRESHOLD = 0.5


def main() -> None:
    with open(THRESH_JSON) as fh:
        t_built = float(json.load(fh)["stable_built"])
    print(f"built-loss threshold (stable_built): {t_built:.4f}")

    con = duckdb.connect()
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)

    # selected refs for built_loss parents only
    refs = con.execute(
        f"""SELECT parent_id, ref_state, {embed_sql}
            FROM read_parquet('{CAND_REFS}')
            WHERE selection = 'selected' AND parent_label = 'built_loss'
              AND ref_state IN ('good','bad')""",
    ).df()
    refs["parent_id"] = refs["parent_id"].astype(str)
    refs = refs.dropna(subset=EMBED_COLS).reset_index(drop=True)

    # built_loss parent ids (from refs) + their 2024 mean embeddings
    bl_ids = set(refs["parent_id"].unique())
    ts = con.execute(
        f"SELECT parent_id, {embed_sql} FROM read_parquet('{CAND_TS}')"
    ).df()
    con.close()
    ts["parent_id"] = ts["parent_id"].astype(str)
    ts = ts[ts["parent_id"].isin(bl_ids)].dropna(subset=EMBED_COLS)
    test_emb = {
        pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
        for pid, g in ts.groupby("parent_id", sort=False)
    }
    print(f"built_loss parents with refs: {len(bl_ids)}; with embeddings: {len(test_emb)}")

    refs_groups = dict(tuple(refs.groupby("parent_id", sort=False)))
    rng = np.random.default_rng(SEED)
    knn_fn = lambda dg, db: _knn_score(dg, db, KNN_K)
    median_fn = lambda dg, db: _median_score(dg, db)

    rows = []
    for pid, x_obs in test_emb.items():
        sub = refs_groups.get(pid)
        if sub is None:
            continue
        good = sub.loc[sub["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
        bad = sub.loc[sub["ref_state"] == "bad", EMBED_COLS].to_numpy(dtype=float)
        if len(good) == 0 or len(bad) == 0:
            continue
        d_g = cosine_dists_to_set(x_obs, good)
        d_b = cosine_dists_to_set(x_obs, bad)
        s_knn, lo_knn, hi_knn = bootstrap_ci(knn_fn, d_g, d_b, N_BOOT, rng)
        s_med, lo_med, hi_med = bootstrap_ci(median_fn, d_g, d_b, N_BOOT, rng)
        rows.append({
            "parent_id": pid,
            "parent_label": "built_loss",
            "stable_class": "built_loss",
            "n_good": int(len(good)),
            "n_bad": int(len(bad)),
            "dor_knn": s_knn,
            "dor_knn_ci_low": lo_knn,
            "dor_knn_ci_high": hi_knn,
            "t_knn_used": t_built,
            "category_knn": classify(s_knn, lo_knn, hi_knn, t_built),
            "dor_median": s_med,
            "dor_median_ci_low": lo_med,
            "dor_median_ci_high": hi_med,
            "category_median": classify(s_med, lo_med, hi_med, MEDIAN_THRESHOLD),
        })

    df = pd.DataFrame(rows).sort_values("parent_id").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} built-loss sites -> {OUT}")
    print("\ncategory_knn breakdown:")
    print(df["category_knn"].value_counts().to_string())
    print(f"\ndor_knn median: {df['dor_knn'].median():.4f}  "
          f"mean: {df['dor_knn'].mean():.4f}")


if __name__ == "__main__":
    main()
