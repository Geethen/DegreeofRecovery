"""v4 test-site scorer for stable_stable parents (per-stable-class thresholds).

Mirrors v3/scripts/analysis/score_test_sites_v3.py, with two differences:

  1. Refs come from the v4 stable-pixel parquet (parent_label encodes
     stable_class as 'stable_nature' / 'stable_crop' / 'stable_built').
  2. The kNN threshold is selected per stable_class from
     CALIBRATED_KNN_THRESHOLDS_V4 (refit by `validate_steps_within_parent_v4.py`).
     A v3-thresholds-applied transfer-check column is also written.

Stable-crop and stable-built parents are scored against built-only and
crop+built bad pools respectively, as sanity checks. The framework's framing
("recovery toward natural state") implies that both classes should score
'degraded' — they are stable, not recovering toward nature. This is the
expected outcome and both classes are kept in the final dataset to validate
the framework end-to-end.

Usage:
  python v4/scripts/analysis/score_test_sites_v4.py
"""
from __future__ import annotations

import argparse
import os

import duckdb
import numpy as np
import pandas as pd

from degree_of_recovery.core import (
    EPS,
    bootstrap_ci,
    classify,
    cosine_dists_to_set,
    knn_score as _knn_score,
    median_score as _median_score,
)

EMBED_COLS = [f"A{i:02d}" for i in range(64)]

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
V1_DATA_DIR = os.path.join(BASE_DIR, "v1", "data")
V4_DATA_DIR = os.path.join(BASE_DIR, "v4", "data")
DEFAULT_REFS = os.path.join(
    V4_DATA_DIR, "v4_stable_refs_alphaearth.parquet"
)
DEFAULT_TEST_SITES = os.path.join(V1_DATA_DIR, "test_site_alphaearth_2024.parquet")

KNN_K = 5
N_BOOT = 2000
SEED = 42

# Defaults — overwritten when calibration JSON is present (see --thresholds).
# Initial seed values copied from v3 for the transfer-check column; the
# per-class operational threshold is loaded from disk if available.
V3_KNN_THRESHOLD = 0.4859
CALIBRATED_KNN_THRESHOLDS_V4 = {
    "stable_nature": 0.4859,   # placeholder — refit via validator
    "stable_crop":   0.4859,
    "stable_built":  0.4859,
}
MEDIAN_THRESHOLD = 0.5


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


def load_thresholds(json_path: str | None) -> dict[str, float]:
    """Read per-class kNN thresholds from JSON.

    Supports both schemas:
      - new (v4 ≥ per_fold + t_med fix):
          {"stable_nature": {"t_med": ..., "t_knn": ...}, ...}
      - old (initial v4):
          {"stable_nature": 0.4861, ...}
    """
    if not json_path or not os.path.exists(json_path):
        return dict(CALIBRATED_KNN_THRESHOLDS_V4)
    import json
    with open(json_path) as fh:
        payload = json.load(fh)
    out = dict(CALIBRATED_KNN_THRESHOLDS_V4)
    for k, v in payload.items():
        if not k.startswith("stable_"):
            continue
        if isinstance(v, dict):
            out[k] = float(v["t_knn"])
        else:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refs",       default=DEFAULT_REFS)
    parser.add_argument("--test-sites", default=DEFAULT_TEST_SITES)
    parser.add_argument("--strategy",   default="random_100")
    parser.add_argument("--out-dir",    default=V4_DATA_DIR)
    parser.add_argument("--n-boot",     type=int, default=N_BOOT)
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--knn-k",      type=int, default=KNN_K)
    parser.add_argument(
        "--thresholds",
        default=os.path.join(V4_DATA_DIR, "calibrated_thresholds_v4.json"),
        help="JSON with {'stable_nature': t, 'stable_crop': t, 'stable_built': t}.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t_per_class = load_thresholds(args.thresholds)

    print(f"Loading refs ({args.strategy})...")
    refs = load_refs(args.refs, args.strategy)
    print(f"  {len(refs):,} rows, {refs['parent_id'].nunique()} parents")
    print(f"  parent_label split: "
          f"{refs.groupby('parent_label')['parent_id'].nunique().to_dict()}")

    print("Loading test site embeddings...")
    test_emb = load_test_sites(args.test_sites)
    print(f"  {len(test_emb)} test sites available")

    refs_groups = dict(tuple(refs.groupby("parent_id", sort=False)))

    knn_fn    = lambda dg, db: _knn_score(dg, db, args.knn_k)
    median_fn = lambda dg, db: _median_score(dg, db)

    rows = []
    skipped_no_parent: list[str] = []
    skipped_no_refs: list[str] = []
    for pid, x_obs in test_emb.items():
        sub = refs_groups.get(pid)
        if sub is None:
            skipped_no_parent.append(pid)
            continue
        good = sub.loc[sub["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
        bad  = sub.loc[sub["ref_state"] == "bad",  EMBED_COLS].to_numpy(dtype=float)
        if len(good) == 0 or len(bad) == 0:
            skipped_no_refs.append(pid)
            continue

        d_g = cosine_dists_to_set(x_obs, good)
        d_b = cosine_dists_to_set(x_obs, bad)

        s_knn, lo_knn, hi_knn = bootstrap_ci(knn_fn,    d_g, d_b, args.n_boot, rng)
        s_med, lo_med, hi_med = bootstrap_ci(median_fn, d_g, d_b, args.n_boot, rng)

        parent_label = str(sub["parent_label"].iloc[0])
        t_knn_class = t_per_class.get(parent_label, V3_KNN_THRESHOLD)

        rows.append({
            "parent_id":          pid,
            "parent_label":       parent_label,
            "stable_class":       parent_label.replace("stable_", ""),
            "n_good":             int(len(good)),
            "n_bad":              int(len(bad)),
            # Primary: dor_knn with per-class threshold
            "dor_knn":            s_knn,
            "dor_knn_ci_low":     lo_knn,
            "dor_knn_ci_high":    hi_knn,
            "t_knn_used":         t_knn_class,
            "category_knn":       classify(s_knn, lo_knn, hi_knn, t_knn_class),
            # Transfer check: v3's pooled threshold applied to v4 scores
            "category_knn_v3t":   classify(s_knn, lo_knn, hi_knn, V3_KNN_THRESHOLD),
            # Secondary: dor_median
            "dor_median":         s_med,
            "dor_median_ci_low":  lo_med,
            "dor_median_ci_high": hi_med,
            "category_median":    classify(s_med, lo_med, hi_med, MEDIAN_THRESHOLD),
        })

    df = pd.DataFrame(rows).sort_values("parent_id").reset_index(drop=True)
    out_path = os.path.join(args.out_dir, "test_site_dor_v4.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"V4 STABLE-SITE SCORES  (k={args.knn_k})")
    print(f"{'='*60}")
    print(f"Sites scored:                    {len(df)}")
    print(f"Sites skipped (no parent in refs): {len(skipped_no_parent)}")
    print(f"Sites skipped (no good or bad):    {len(skipped_no_refs)}")

    print("\nThresholds applied per stable_class:")
    for cls, t in t_per_class.items():
        print(f"  {cls}: {t:.4f}")
    print(f"  (transfer check uses v3 t_knn = {V3_KNN_THRESHOLD:.4f})")

    print("\nCategory breakdown by stable_class (per-class threshold):")
    by_cls = df.groupby("parent_label")["category_knn"].value_counts().unstack(fill_value=0)
    print(by_cls.to_string())

    for sanity_label in ("stable_crop", "stable_built"):
        if sanity_label not in df["parent_label"].values:
            continue
        print(f"\nSanity check — {sanity_label} sites should mostly be 'degraded':")
        sb = df[df["parent_label"] == sanity_label]
        cnt = sb["category_knn"].value_counts()
        n = len(sb)
        for cat in ("degraded", "indistinguishable", "recovering"):
            print(f"  {cat:<20} {int(cnt.get(cat, 0)):>4}/{n}")

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
