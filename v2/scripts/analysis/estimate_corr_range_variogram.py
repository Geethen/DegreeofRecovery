"""Estimate parent-specific spatial correlation ranges from embedding semivariograms.

The script fits an exponential semivariogram model on pairwise points within each
parent and writes a parent_id -> corr_range_m table for downstream n_eff usage.

Model:
  gamma(h) = nugget + sill * (1 - exp(-h / a))
Practical range (95% sill) = 3a.

Input parquet must include:
  - parent_id
  - ref_state in {'good','bad'}
  - geo (GeoJSON-like Point dict)
  - A00..A63 embedding bands

Usage:
  python v2/scripts/analysis/estimate_corr_range_variogram.py \
      --input v2/data/recover_reference_samples_v2_mask_on_large_alphaearth.parquet \
      --out-csv v2/data/v2_corr_range_by_parent.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EARTH_RADIUS_M = 6371000.0


def load_refs(path: str, strategy: str | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute("SELECT * FROM read_parquet(?)", [path]).df()
    con.close()
    missing = [c for c in EMBED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing embedding bands in input: {missing[:5]}")

    df["parent_id"] = df["parent_id"].astype(str)
    if strategy and "strategy" in df.columns:
        df = df[df["strategy"] == strategy].copy()
    df = df[df["ref_state"].isin(["good", "bad"])].copy()
    df = df.dropna(subset=EMBED_COLS + ["geo", "parent_id", "ref_state"]).reset_index(
        drop=True
    )

    coords = df["geo"].map(
        lambda g: g.get("coordinates") if isinstance(g, dict) else None
    )
    df["lon"] = coords.map(lambda c: float(c[0]) if c else np.nan)
    df["lat"] = coords.map(lambda c: float(c[1]) if c else np.nan)
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    return df


def haversine_matrix(lat_lon: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_lon[:, 0])
    lon = np.radians(lat_lon[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return (
        2.0
        * EARTH_RADIUS_M
        * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))
    )


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    sim = (x @ x.T) / (n @ n.T)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def _pairwise_upper(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def binned_empirical_semivariogram(
    spatial_d: np.ndarray,
    embed_d: np.ndarray,
    max_dist_m: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # gamma(h) ~= 0.5 * E[(z_i-z_j)^2]. Using embedding cosine distance as
    # dissimilarity proxy: gamma ~= 0.5 * mean(cosine_distance).
    d = _pairwise_upper(spatial_d)
    g = 0.5 * _pairwise_upper(embed_d)

    keep = np.isfinite(d) & np.isfinite(g) & (d > 0) & (d <= max_dist_m)
    d = d[keep]
    g = g[keep]
    if len(d) == 0:
        return np.array([]), np.array([]), np.array([])

    edges = np.linspace(0, max_dist_m, n_bins + 1)
    bins = np.digitize(d, edges) - 1
    x = []
    y = []
    n = []
    for b in range(n_bins):
        m = bins == b
        if not np.any(m):
            continue
        x.append(float(np.median(d[m])))
        y.append(float(np.mean(g[m])))
        n.append(int(np.sum(m)))
    return np.array(x), np.array(y), np.array(n)


def fit_exponential_range(
    h: np.ndarray,
    gamma: np.ndarray,
    w: np.ndarray,
    min_range_m: float,
    max_range_m: float,
) -> tuple[float, float, float, float]:
    # Coarse grid search over nugget ratio and practical range; cheap and robust.
    if len(h) < 4:
        return (np.nan, np.nan, np.nan, np.nan)

    sill_obs = float(np.nanquantile(gamma, 0.9))
    sill_obs = max(sill_obs, 1e-6)

    nugget_ratios = np.linspace(0.0, 0.5, 21)
    practical_ranges = np.linspace(min_range_m, max_range_m, 120)

    best = (math.inf, np.nan, np.nan, np.nan)
    for nr in nugget_ratios:
        nugget = nr * sill_obs
        sill = max(sill_obs - nugget, 1e-9)
        for r95 in practical_ranges:
            a = r95 / 3.0
            pred = nugget + sill * (1.0 - np.exp(-h / max(a, 1e-9)))
            sse = float(np.sum(w * (gamma - pred) ** 2))
            if sse < best[0]:
                best = (sse, r95, nugget, sill)

    sse, r95, nugget, sill = best
    # weighted R^2
    pred = nugget + sill * (1.0 - np.exp(-h / max(r95 / 3.0, 1e-9)))
    y_bar = float(np.average(gamma, weights=w))
    sst = float(np.sum(w * (gamma - y_bar) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-12)
    return float(r95), float(nugget), float(sill), float(r2)


def estimate_for_parent(
    sub: pd.DataFrame,
    max_dist_m: float,
    n_bins: int,
    min_pairs_per_bin: int,
    min_range_m: float,
    max_range_m: float,
) -> dict[str, float | int | str]:
    x = sub[EMBED_COLS].to_numpy(dtype=float)
    ll = sub[["lat", "lon"]].to_numpy(dtype=float)
    n_points = len(sub)

    sd = haversine_matrix(ll)
    ed = cosine_distance_matrix(x)
    h, g, n = binned_empirical_semivariogram(
        sd, ed, max_dist_m=max_dist_m, n_bins=n_bins
    )

    keep = n >= min_pairs_per_bin
    h = h[keep]
    g = g[keep]
    n = n[keep]

    if len(h) < 4:
        return {
            "corr_range_m": np.nan,
            "nugget": np.nan,
            "sill": np.nan,
            "r2": np.nan,
            "n_points": int(n_points),
            "n_bins": int(len(h)),
            "status": "insufficient_bins",
        }

    r95, nugget, sill, r2 = fit_exponential_range(
        h,
        g,
        w=n.astype(float),
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )
    return {
        "corr_range_m": r95,
        "nugget": nugget,
        "sill": sill,
        "r2": r2,
        "n_points": int(n_points),
        "n_bins": int(len(h)),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--max-dist-m", type=float, default=8000.0)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--min-pairs-per-bin", type=int, default=50)
    parser.add_argument("--min-range-m", type=float, default=75.0)
    parser.add_argument("--max-range-m", type=float, default=1200.0)
    parser.add_argument("--default-range-m", type=float, default=300.0)
    args = parser.parse_args()

    df = load_refs(args.input, strategy=args.strategy)
    rows = []
    for pid, sub in df.groupby("parent_id", sort=False):
        est = estimate_for_parent(
            sub,
            max_dist_m=args.max_dist_m,
            n_bins=args.n_bins,
            min_pairs_per_bin=args.min_pairs_per_bin,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
        )
        est["parent_id"] = pid
        rows.append(est)

    out = pd.DataFrame(rows)
    # Fill failed fits with a stable default so downstream scoring can run.
    out["corr_range_m"] = out["corr_range_m"].fillna(args.default_range_m)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("parent_id").to_csv(out_path, index=False)

    ok = out[out["status"] == "ok"]
    print(f"Wrote {out_path}")
    print(f"Parents: {len(out)} | successful fits: {len(ok)}")
    print(
        out["corr_range_m"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
