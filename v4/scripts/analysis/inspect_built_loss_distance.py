"""Inspect actual distance distributions for built-loss sites.

Test the hypothesis that bad refs (all WC=50 buildings) should be spectrally
homogeneous regardless of spatial distance, so m_b should be roughly flat
as nearby refs are excluded.

For 5 sample built-loss sites, print:
  - cosine distance distribution of bad refs (sorted by spatial distance)
  - what the 5 spectrally-nearest bad refs are, and where they sit spatially
  - the same for good refs
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
CAND_REFS_PATH  = BASE_DIR / "v2" / "data" / "recover_reference_samples_v2_mask_on_large_alphaearth.parquet"
TEST_SITES_PATH = BASE_DIR / "v1" / "data" / "test_site_alphaearth_2024.parquet"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS = 1e-12


def cosine_dists(x, pts):
    nx = np.linalg.norm(x) + EPS
    nps = np.linalg.norm(pts, axis=1) + EPS
    return 1.0 - (pts @ x) / (nps * nx)


def main() -> None:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state", "dist_m"] + EMBED_COLS)
    refs = con.execute(
        f"SELECT {cols} FROM read_parquet(?) WHERE parent_label = 'built_loss'",
        [str(CAND_REFS_PATH)]
    ).df()
    refs["parent_id"] = refs["parent_id"].astype(str)

    test_cols = ", ".join(["parent_id"] + EMBED_COLS)
    test = con.execute(f"SELECT {test_cols} FROM read_parquet(?)",
                       [str(TEST_SITES_PATH)]).df()
    test["parent_id"] = test["parent_id"].astype(str)
    test_emb = {pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
                for pid, g in test.groupby("parent_id")}
    con.close()

    # Pick 5 built-loss parents that have refs at varying spatial distances
    built_pids = refs["parent_id"].unique()
    sample_pids = sorted(built_pids)[:5]

    for pid in sample_pids:
        if pid not in test_emb:
            continue
        x_obs = test_emb[pid]
        sub = refs[refs["parent_id"] == pid]
        good = sub[sub["ref_state"] == "good"].copy()
        bad  = sub[sub["ref_state"] == "bad" ].copy()
        if len(good) < 5 or len(bad) < 5:
            continue

        print("\n" + "="*70)
        print(f"Parent {pid}  (n_good={len(good)}, n_bad={len(bad)})")
        print("="*70)

        for label, pool in [("BAD (buildings, WC=50)", bad), ("GOOD (natural)", good)]:
            emb = pool[EMBED_COLS].to_numpy(dtype=float)
            d = cosine_dists(x_obs, emb)
            sp = pool["dist_m"].to_numpy()
            print(f"\n{label}:")
            print(f"  cosine dist:   median={np.median(d):.4f}  "
                  f"min={d.min():.4f}  max={d.max():.4f}  std={d.std():.4f}")
            print(f"  spatial dist:  median={np.median(sp)/1000:.2f} km  "
                  f"range=[{sp.min()/1000:.2f}, {sp.max()/1000:.2f}] km")

            # 5 spectrally nearest — where are they spatially?
            order = np.argsort(d)
            knn_idx = order[:5]
            print(f"  5 spectrally nearest:  cosine={d[knn_idx].round(4).tolist()}")
            print(f"                         spatial_km={(sp[knn_idx]/1000).round(2).tolist()}")

            # What happens at exclusion radii
            print(f"  Exclusion sensitivity:")
            for thr in [0, 500, 1000, 2000]:
                keep = sp >= thr
                if keep.sum() < 5:
                    continue
                d_k = d[keep]
                m_k = float(np.mean(np.partition(d_k, 4)[:5]))
                print(f"    >= {thr:>4}m  n={keep.sum():>3}  m={m_k:.4f}  "
                      f"new_knn_spatial={np.sort(sp[keep][np.argsort(d_k)[:5]]/1000).round(2).tolist()}")


if __name__ == "__main__":
    main()
