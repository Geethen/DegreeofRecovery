"""Decompose distance-stratified DoR into its m_good and m_bad components.

For each exclusion threshold, record the mean cosine distance to the 5 nearest
good refs (m_g) and the 5 nearest bad refs (m_b) separately. This lets us see
which side of the ratio is responsible for any DoR shift, rather than guessing.

DoR = m_b / (m_g + m_b), so:
  - DoR up + m_g down       -> good refs got closer (unlikely if excluding nearby)
  - DoR up + m_b up         -> bad refs got farther (lost similar bad refs)
  - DoR up + both up        -> m_b grew faster than m_g
  - DoR up + both down      -> m_g shrunk faster than m_b
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
V4_DATA  = BASE_DIR / "v4" / "data"

REFS_PATH        = V4_DATA / "v4_stable_refs_alphaearth.parquet"
CAND_REFS_PATH   = BASE_DIR / "v2" / "data" / "recover_reference_samples_v2_mask_on_large_alphaearth.parquet"
TEST_SITES_PATH  = BASE_DIR / "v1" / "data" / "test_site_alphaearth_2024.parquet"
TEST_SITES_V4    = V4_DATA / "test_site_alphaearth_2024_v4.parquet"

STRATEGY = "random_100"
KNN_K    = 5
EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS = 1e-12
THRESHOLDS = [0, 500, 1000, 2000]
MIN_REFS = 5


def cosine_dists(x: np.ndarray, pts: np.ndarray) -> np.ndarray:
    nx  = np.linalg.norm(x) + EPS
    nps = np.linalg.norm(pts, axis=1) + EPS
    return 1.0 - (pts @ x) / (nps * nx)


def knn_mean(d: np.ndarray, k: int = KNN_K) -> float:
    if len(d) < k:
        return float("nan")
    return float(np.mean(np.partition(d, k - 1)[:k]))


def load_refs(path: str, strategy: str | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state", "dist_m"] + EMBED_COLS)
    if strategy:
        df = con.execute(f"SELECT {cols} FROM read_parquet(?) WHERE strategy = ?",
                         [str(path), strategy]).df()
    else:
        df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [str(path)]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return (df[df["ref_state"].isin(["good", "bad"])]
            .dropna(subset=["parent_id", "ref_state", "dist_m"] + EMBED_COLS)
            .reset_index(drop=True))


def load_test_sites(path: str) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    cols = ", ".join(["parent_id"] + EMBED_COLS)
    df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [str(path)]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=["parent_id"] + EMBED_COLS).reset_index(drop=True)
    return {pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
            for pid, g in df.groupby("parent_id", sort=False)}


def score_all(refs: pd.DataFrame, test_emb: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    refs_grp = dict(tuple(refs.groupby("parent_id", sort=False)))
    for pid, x_obs in test_emb.items():
        sub = refs_grp.get(pid)
        if sub is None:
            continue
        parent_label = str(sub["parent_label"].iloc[0])
        for thr in THRESHOLDS:
            sf = sub[sub["dist_m"] >= thr]
            good = sf[sf["ref_state"] == "good"][EMBED_COLS].to_numpy(dtype=float)
            bad  = sf[sf["ref_state"] == "bad" ][EMBED_COLS].to_numpy(dtype=float)
            if len(good) < MIN_REFS or len(bad) < MIN_REFS:
                continue
            d_g = cosine_dists(x_obs, good)
            d_b = cosine_dists(x_obs, bad)
            m_g = knn_mean(d_g)
            m_b = knn_mean(d_b)
            rows.append({
                "parent_id":    pid,
                "parent_label": parent_label,
                "min_dist_m":   thr,
                "n_good":       len(good),
                "n_bad":        len(bad),
                "m_g":          m_g,
                "m_b":          m_b,
                "dor":          m_b / (m_g + m_b + EPS),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading data ...")
    refs_cand = load_refs(str(CAND_REFS_PATH))
    refs_stab = load_refs(str(REFS_PATH), strategy=STRATEGY)
    test_cand = load_test_sites(str(TEST_SITES_PATH))
    test_stab = load_test_sites(str(TEST_SITES_V4))

    df_cand = score_all(refs_cand, test_cand)
    df_stab = score_all(refs_stab, test_stab)
    df = pd.concat([df_cand, df_stab], ignore_index=True)

    out_csv = V4_DATA / "dor_component_decomposition.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    print("\nMedian m_g (good distance) by class and threshold:")
    print(df.groupby(["parent_label","min_dist_m"])["m_g"].median().unstack().round(4).to_string())
    print("\nMedian m_b (bad distance) by class and threshold:")
    print(df.groupby(["parent_label","min_dist_m"])["m_b"].median().unstack().round(4).to_string())
    print("\nMedian DoR by class and threshold:")
    print(df.groupby(["parent_label","min_dist_m"])["dor"].median().unstack().round(4).to_string())

    print("\n" + "="*60)
    print("Per-site change from baseline (median across sites):")
    print("="*60)
    pivot_g = df.pivot_table(index=["parent_id","parent_label"],
                             columns="min_dist_m", values="m_g")
    pivot_b = df.pivot_table(index=["parent_id","parent_label"],
                             columns="min_dist_m", values="m_b")
    pivot_d = df.pivot_table(index=["parent_id","parent_label"],
                             columns="min_dist_m", values="dor")

    for lbl in sorted(df["parent_label"].unique()):
        mask = pivot_g.index.get_level_values("parent_label") == lbl
        print(f"\n{lbl}:")
        for t in [500, 1000, 2000]:
            if t not in pivot_g.columns or 0 not in pivot_g.columns:
                continue
            dg = (pivot_g.loc[mask, t] - pivot_g.loc[mask, 0]).dropna()
            db = (pivot_b.loc[mask, t] - pivot_b.loc[mask, 0]).dropna()
            dd = (pivot_d.loc[mask, t] - pivot_d.loc[mask, 0]).dropna()
            print(f"  thr={t:>4}m  n={len(dg):>4}  "
                  f"delta_m_g={dg.median():+.4f}  "
                  f"delta_m_b={db.median():+.4f}  "
                  f"delta_DoR={dd.median():+.4f}")


if __name__ == "__main__":
    main()
