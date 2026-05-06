"""v3 test-site scorer: dor_knn (mean cosine, k=5) as primary score.

Identical structure to degreeRecover/scripts/analysis/score_test_sites.py
but replaces dor_median with dor_knn as the primary score. dor_median is
retained as a secondary score for comparison.

Key differences from production (degreeRecover):
  - PRIMARY_SCORE = dor_knn  (mean cosine k=5, threshold=0.4859)
  - dor_median retained as SECONDARY_SCORE
  - Classification uses KNN_THRESHOLD (Youden-J calibrated) not 0.5

Usage:
  python v3/scripts/analysis/score_test_sites_v3.py
"""
from __future__ import annotations

import argparse
import os

import duckdb
import numpy as np
import pandas as pd

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS = 1e-12

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_DIR = os.path.join(BASE_DIR, "degreeRecover", "data")
V3_DATA_DIR = os.path.join(BASE_DIR, "v3", "data")
DEFAULT_REFS = os.path.join(
    BASE_DIR,
    "v2", "data", "v2real_mask_on_corr300_exhaustive",
    "sampling_strategy_selected_points.parquet",
)
DEFAULT_TEST_SITES = os.path.join(DATA_DIR, "test_site_alphaearth_2024.parquet")

KNN_K = 5
KNN_THRESHOLD = 0.4859   # Youden-J calibrated on random_100 (v3 experiments)
MEDIAN_THRESHOLD = 0.5
N_BOOT = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def cosine_dists_to_set(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    nx = np.linalg.norm(x) + EPS
    np_pts = np.linalg.norm(points, axis=1) + EPS
    return 1.0 - (points @ x) / (np_pts * nx)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def _knn_score(d_g: np.ndarray, d_b: np.ndarray, k: int) -> float:
    if len(d_g) < k or len(d_b) < k:
        return float("nan")
    m_g = float(np.mean(np.partition(d_g, k - 1)[:k]))
    m_b = float(np.mean(np.partition(d_b, k - 1)[:k]))
    return m_b / (m_g + m_b + EPS)


def _median_score(d_g: np.ndarray, d_b: np.ndarray) -> float:
    return float(np.median(d_b) / (np.median(d_g) + np.median(d_b) + EPS))


def bootstrap_ci(
    score_fn, d_g: np.ndarray, d_b: np.ndarray,
    n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    point = score_fn(d_g, d_b)
    if not np.isfinite(point):
        return float("nan"), float("nan"), float("nan")
    n_g, n_b = len(d_g), len(d_b)
    ig = rng.integers(0, n_g, size=(n_boot, n_g))
    ib = rng.integers(0, n_b, size=(n_boot, n_b))
    boots = np.array([score_fn(d_g[ig[i]], d_b[ib[i]]) for i in range(n_boot)])
    return float(point), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(score: float, ci_lo: float, ci_hi: float, threshold: float) -> str:
    if not (np.isfinite(score) and np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return "no_data"
    if ci_lo > threshold:
        return "recovering"
    if ci_hi < threshold:
        return "degraded"
    return "indistinguishable"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_refs(parquet_path: str, strategy: str) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state"] + EMBED_COLS)
    df = con.execute(
        f"SELECT {cols} FROM read_parquet(?) WHERE strategy = ?",
        [parquet_path, strategy],
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    return df[df["ref_state"].isin(["good", "bad"])].dropna(
        subset=["parent_id", "ref_state"] + EMBED_COLS
    ).reset_index(drop=True)


def load_test_sites(parquet_path: str) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    cols = ", ".join(["parent_id"] + EMBED_COLS)
    df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [parquet_path]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=["parent_id"] + EMBED_COLS).reset_index(drop=True)
    return {
        pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
        for pid, g in df.groupby("parent_id", sort=False)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refs",       default=DEFAULT_REFS)
    parser.add_argument("--test-sites", default=DEFAULT_TEST_SITES)
    parser.add_argument("--strategy",   default="random_100")
    parser.add_argument("--out-dir",    default=V3_DATA_DIR)
    parser.add_argument("--n-boot",     type=int, default=N_BOOT)
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--knn-k",      type=int, default=KNN_K)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading refs ({args.strategy})...")
    refs = load_refs(args.refs, args.strategy)
    print(f"  {len(refs):,} rows, {refs['parent_id'].nunique()} parents")

    print("Loading test site embeddings...")
    test_emb = load_test_sites(args.test_sites)
    print(f"  {len(test_emb)} test sites")

    refs_groups = dict(tuple(refs.groupby("parent_id", sort=False)))

    knn_fn    = lambda dg, db: _knn_score(dg, db, args.knn_k)
    median_fn = lambda dg, db: _median_score(dg, db)

    rows = []
    for pid, x_obs in test_emb.items():
        sub = refs_groups.get(pid)
        if sub is None:
            continue
        good = sub.loc[sub["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
        bad  = sub.loc[sub["ref_state"] == "bad",  EMBED_COLS].to_numpy(dtype=float)
        if len(good) == 0 or len(bad) == 0:
            continue

        d_g = cosine_dists_to_set(x_obs, good)
        d_b = cosine_dists_to_set(x_obs, bad)

        s_knn, lo_knn, hi_knn       = bootstrap_ci(knn_fn,    d_g, d_b, args.n_boot, rng)
        s_med, lo_med, hi_med       = bootstrap_ci(median_fn, d_g, d_b, args.n_boot, rng)

        rows.append({
            "parent_id":          pid,
            "parent_label":       str(sub["parent_label"].iloc[0]),
            "n_good":             int(len(good)),
            "n_bad":              int(len(bad)),
            # Primary: dor_knn
            "dor_knn":            s_knn,
            "dor_knn_ci_low":     lo_knn,
            "dor_knn_ci_high":    hi_knn,
            "category_knn":       classify(s_knn, lo_knn, hi_knn, KNN_THRESHOLD),
            # Secondary: dor_median
            "dor_median":         s_med,
            "dor_median_ci_low":  lo_med,
            "dor_median_ci_high": hi_med,
            "category_median":    classify(s_med, lo_med, hi_med, MEDIAN_THRESHOLD),
        })

    df = pd.DataFrame(rows).sort_values("parent_id").reset_index(drop=True)

    out_path = os.path.join(args.out_dir, "test_site_dor_v3.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"V3 TEST SITE SCORES  (primary = dor_knn, k={args.knn_k}, threshold={KNN_THRESHOLD})")
    print(f"{'='*60}")
    print(f"Sites scored: {len(df)}")

    for scorer, col, t in [("dor_knn",    "category_knn",    KNN_THRESHOLD),
                            ("dor_median", "category_median", MEDIAN_THRESHOLD)]:
        counts = df[col].value_counts(dropna=False)
        print(f"\n  {scorer} (threshold={t}):")
        for cat in ["recovering", "indistinguishable", "degraded", "no_data"]:
            print(f"    {cat:<20} {int(counts.get(cat, 0)):>4}")

    print(f"\n  Per-label (dor_knn):")
    for lbl, sub in df.groupby("parent_label"):
        c = sub["category_knn"].value_counts(dropna=False)
        print(f"    {lbl}: recovering={c.get('recovering',0)}  "
              f"indist={c.get('indistinguishable',0)}  "
              f"degraded={c.get('degraded',0)}")

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
