"""Score all v5 test/control sites with the v5 buffer (per-site DoR, k=5).

Generalises score_built_loss_v5.py to every class:
  built_loss      -> candidate refs, threshold = stable_built (built-up bad pool)
  stable_built    -> stable refs,    threshold = stable_built
  stable_nature   -> stable refs,    threshold = stable_nature
  stable_crop     -> stable refs,    threshold = stable_crop   (scored but
                     EXCLUDED from the headline DoR analysis per project decision)

Reference pool = the v5 'selected' annulus (inner exclusion 4 km, outer ceiling
5 km expanding to 8 km when the per-pool count target is unmet) from the
v5 stable/candidate ref parquets. Scoring function, k, bootstrap, and per-class
thresholds are carried over from v4 unchanged.

Inputs (local, no Earth Engine):
  v5/data/v5_stable_refs_alphaearth.parquet
  v5/data/v5_candidate_refs_alphaearth.parquet
  v4/data/test_site_alphaearth_2024_v4.parquet        (stable parent embeddings)
  v5/data/test_site_alphaearth_2024_candidate.parquet (loss parent embeddings)
  v4/data/calibrated_thresholds_v4.json

Output:
  v5/data/test_site_dor_v5.csv   (all classes, one row per scored parent)
"""
from __future__ import annotations

import json

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
STABLE_REFS = f"{BASE}/v5/data/v5_stable_refs_alphaearth.parquet"
CAND_REFS = f"{BASE}/v5/data/v5_candidate_refs_alphaearth.parquet"
TS_STABLE = f"{BASE}/v4/data/test_site_alphaearth_2024_v4.parquet"
TS_CAND = f"{BASE}/v5/data/test_site_alphaearth_2024_candidate.parquet"
THRESH_JSON = f"{BASE}/v4/data/calibrated_thresholds_v4.json"
OUT = f"{BASE}/v5/data/test_site_dor_v5.csv"

KNN_K, N_BOOT, SEED, MEDIAN_THRESHOLD = 5, 2000, 42, 0.5

# label -> (refs parquet, threshold key)
CONFIG = {
    "built_loss":    (CAND_REFS, "stable_built"),
    "stable_built":  (STABLE_REFS, "stable_built"),
    "stable_nature": (STABLE_REFS, "stable_nature"),
    "stable_crop":   (STABLE_REFS, "stable_crop"),
}


def load_selected_refs(parquet: str, label: str) -> pd.DataFrame:
    con = duckdb.connect()
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)
    df = con.execute(
        f"""SELECT parent_id, ref_state, {embed_sql}
            FROM read_parquet('{parquet}')
            WHERE selection='selected' AND parent_label=? AND ref_state IN ('good','bad')""",
        [label],
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return df.dropna(subset=EMBED_COLS).reset_index(drop=True)


def load_parent_embeddings(parquet: str, label: str | None) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)
    where = "WHERE r=?" if label else ""
    params = [label] if label else []
    df = con.execute(
        f"SELECT parent_id, {embed_sql} FROM read_parquet('{parquet}') {where}", params
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=EMBED_COLS)
    return {
        pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
        for pid, g in df.groupby("parent_id", sort=False)
    }


def main() -> None:
    thresholds = json.load(open(THRESH_JSON))
    rng = np.random.default_rng(SEED)
    knn_fn = lambda dg, db: _knn_score(dg, db, KNN_K)
    median_fn = lambda dg, db: _median_score(dg, db)

    all_rows = []
    for label, (refs_pq, tkey) in CONFIG.items():
        thr = float(thresholds[tkey])
        refs = load_selected_refs(refs_pq, label)
        # parent embeddings: candidate parents keyed by r=label; stable parents
        # are all in the stable TS file -> filter by the parents present in refs.
        if label == "built_loss":
            emb = load_parent_embeddings(TS_CAND, "built_loss")
        else:
            emb = load_parent_embeddings(TS_STABLE, None)
        ref_ids = set(refs["parent_id"].unique())
        groups = dict(tuple(refs.groupby("parent_id", sort=False)))

        n_before = len(all_rows)
        total = len(ref_ids)
        for i, pid in enumerate(ref_ids, 1):
            x = emb.get(pid)
            sub = groups.get(pid)
            if x is None or sub is None:
                continue
            good = sub.loc[sub["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
            bad = sub.loc[sub["ref_state"] == "bad", EMBED_COLS].to_numpy(dtype=float)
            if len(good) == 0 or len(bad) == 0:
                continue
            d_g = cosine_dists_to_set(x, good)
            d_b = cosine_dists_to_set(x, bad)
            s_knn, lo_knn, hi_knn = bootstrap_ci(knn_fn, d_g, d_b, N_BOOT, rng)
            s_med, lo_med, hi_med = bootstrap_ci(median_fn, d_g, d_b, N_BOOT, rng)
            all_rows.append({
                "parent_id": pid,
                "parent_label": label,
                "stable_class": label.replace("stable_", "") if label.startswith("stable_") else label,
                "n_good": int(len(good)),
                "n_bad": int(len(bad)),
                "dor_knn": s_knn,
                "dor_knn_ci_low": lo_knn,
                "dor_knn_ci_high": hi_knn,
                "t_knn_used": thr,
                "category_knn": classify(s_knn, lo_knn, hi_knn, thr),
                "dor_median": s_med,
                "dor_median_ci_low": lo_med,
                "dor_median_ci_high": hi_med,
                "category_median": classify(s_med, lo_med, hi_med, MEDIAN_THRESHOLD),
            })
            if i % 100 == 0:
                print(f"{label:14s} progress {i}/{total}", flush=True)
        print(f"{label:14s} thr={thr:.4f}  scored {len(all_rows)-n_before} parents", flush=True)

    df = pd.DataFrame(all_rows).sort_values(["parent_label", "parent_id"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} sites -> {OUT}")
    print("\ncategory_knn by class:")
    print(df.groupby("parent_label")["category_knn"].value_counts().to_string())
    print("\nmedian dor_knn by class:")
    print(df.groupby("parent_label")["dor_knn"].median().round(4).to_string())


if __name__ == "__main__":
    main()
