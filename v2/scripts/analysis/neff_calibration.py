"""Empirical calibration of effective sample size (n_eff) targets.

Sweeps subsample size per parent and reports how DoR uncertainty (bootstrap
CI width), DoR point stability across seeds, and pooled AUC behave.

The output curves let you defensibly pick `n_eff_min` / `n_eff_target` rather
than asserting them. The recommendation is the smallest n where:
  * median CI width has plateaued (slope < 0.01 per +10 points), and
  * cross-seed DoR std is below an operational tolerance (e.g. 0.02).

Usage:
  python v2/scripts/analysis/neff_calibration.py \
      --input v2/data/recover_reference_samples_v2_mask_on_large_alphaearth.parquet \
      --out-dir v2/data/neff_calibration_mask_on \
      --plot-dir v2/plots/neff_calibration_mask_on
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from degree_of_recovery.core import cosine_dists_to_set

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EARTH_RADIUS_M = 6371000.0


def load_refs(parquet_path: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute("SELECT * FROM read_parquet(?)", [parquet_path]).df()
    con.close()
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
    df["parent_id"] = df["parent_id"].astype(str)
    return df


def load_test_emb(parquet_path: str) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    df = con.execute("SELECT * FROM read_parquet(?)", [parquet_path]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=EMBED_COLS + ["parent_id"]).reset_index(drop=True)
    return {
        pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
        for pid, g in df.groupby("parent_id", sort=False)
    }


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


def compute_neff(dist_m: np.ndarray, corr_range_m: float) -> float:
    if len(dist_m) == 0:
        return 0.0
    if corr_range_m <= 0:
        return float(dist_m.shape[0])
    w = np.exp(-((dist_m / corr_range_m) ** 2))
    return float((dist_m.shape[0] ** 2) / max(np.sum(w), 1e-12))


def dor_with_ci(
    d_g: np.ndarray, d_b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    point = float(np.median(d_b)) / (
        float(np.median(d_g)) + float(np.median(d_b)) + 1e-12
    )
    ig = rng.integers(0, len(d_g), size=(n_boot, len(d_g)))
    ib = rng.integers(0, len(d_b), size=(n_boot, len(d_b)))
    boot = np.median(d_b[ib], axis=1) / (
        np.median(d_g[ig], axis=1) + np.median(d_b[ib], axis=1) + 1e-12
    )
    ci_width = float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5))
    return point, ci_width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--test-embeddings",
        default="v1/data/test_site_alphaearth_2024.parquet",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument(
        "--sample-sizes",
        default="10,20,30,50,75,100,150,200",
        help="Comma-separated per-class sample sizes to evaluate.",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--corr-range-m", type=float, default=300.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_refs(args.input)
    test_emb = load_test_emb(args.test_embeddings)
    sizes = [int(s) for s in args.sample_sizes.split(",") if s.strip()]

    rows = []
    for size in sizes:
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(1000 * size + seed)
            for pid, parent in df.groupby("parent_id", sort=False):
                if pid not in test_emb:
                    continue
                g = parent[parent["ref_state"] == "good"]
                b = parent[parent["ref_state"] == "bad"]
                n_g, n_b = min(size, len(g)), min(size, len(b))
                if n_g < 5 or n_b < 5:
                    continue
                ig = rng.choice(len(g), size=n_g, replace=False)
                ib = rng.choice(len(b), size=n_b, replace=False)
                gs = g.iloc[ig]
                bs = b.iloc[ib]
                x_obs = test_emb[pid]
                d_g = cosine_dists_to_set(x_obs, gs[EMBED_COLS].to_numpy(dtype=float))
                d_b = cosine_dists_to_set(x_obs, bs[EMBED_COLS].to_numpy(dtype=float))
                dor, ciw = dor_with_ci(d_g, d_b, n_boot=args.n_boot, rng=rng)
                neff_g = compute_neff(
                    haversine_matrix(gs[["lat", "lon"]].to_numpy()),
                    args.corr_range_m,
                )
                neff_b = compute_neff(
                    haversine_matrix(bs[["lat", "lon"]].to_numpy()),
                    args.corr_range_m,
                )
                rows.append(
                    {
                        "size": size,
                        "seed": seed,
                        "parent_id": pid,
                        "parent_label": parent["parent_label"].iloc[0],
                        "n_eff_min": float(min(neff_g, neff_b)),
                        "dor": dor,
                        "ci_width": ciw,
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "neff_calibration_raw.csv", index=False)

    # Per-parent stability across seeds at fixed size
    stab = raw.groupby(["size", "parent_id"], as_index=False).agg(
        dor_std=("dor", "std"), ci_width_mean=("ci_width", "mean")
    )
    # Aggregate over parents
    summary = (
        raw.groupby("size", as_index=False)
        .agg(
            median_ci_width=("ci_width", "median"),
            p90_ci_width=("ci_width", lambda x: x.quantile(0.9)),
            median_neff_min=("n_eff_min", "median"),
            mean_dor=("dor", "mean"),
        )
        .sort_values("size")
    )
    seed_stab = stab.groupby("size", as_index=False).agg(
        median_dor_std=("dor_std", "median"),
        p90_dor_std=("dor_std", lambda x: x.quantile(0.9)),
    )
    summary = summary.merge(seed_stab, on="size", how="left")
    summary.to_csv(out_dir / "neff_calibration_summary.csv", index=False)

    # Decision: ci_width slope and dor_std threshold
    summary = summary.sort_values("size").reset_index(drop=True)
    summary["ci_width_slope_per_10"] = summary["median_ci_width"].diff() / (
        summary["size"].diff() / 10.0
    )

    plateau = summary[
        (summary["ci_width_slope_per_10"].abs() < 0.005)
        & (summary["median_dor_std"] < 0.02)
    ]
    pick = (
        int(plateau["size"].iloc[0])
        if not plateau.empty
        else int(summary["size"].max())
    )

    print("\nCalibration summary (per-class size):")
    print(
        summary[
            [
                "size",
                "median_ci_width",
                "p90_ci_width",
                "median_neff_min",
                "median_dor_std",
                "p90_dor_std",
                "ci_width_slope_per_10",
            ]
        ].to_string(index=False)
    )
    print(f"\nRecommended n_eff_target (per class): {pick}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(summary["size"], summary["median_ci_width"], "o-", color="#2a9d8f")
    axes[0].plot(
        summary["size"], summary["p90_ci_width"], "o--", color="#2a9d8f", alpha=0.5
    )
    axes[0].axvline(pick, color="black", linestyle=":", alpha=0.7)
    axes[0].set_xlabel("per-class sample size")
    axes[0].set_ylabel("DoR CI width")
    axes[0].set_title("CI width vs n")

    axes[1].plot(summary["size"], summary["median_dor_std"], "o-", color="#e76f51")
    axes[1].plot(
        summary["size"], summary["p90_dor_std"], "o--", color="#e76f51", alpha=0.5
    )
    axes[1].axhline(0.02, color="black", linestyle=":", alpha=0.7)
    axes[1].axvline(pick, color="black", linestyle=":", alpha=0.7)
    axes[1].set_xlabel("per-class sample size")
    axes[1].set_ylabel("cross-seed DoR std")
    axes[1].set_title("Stability vs n")

    axes[2].plot(summary["size"], summary["median_neff_min"], "o-", color="#264653")
    axes[2].set_xlabel("per-class sample size")
    axes[2].set_ylabel("median n_eff_min")
    axes[2].set_title(f"n_eff @ corr={args.corr_range_m:g} m")

    fig.tight_layout()
    fig.savefig(plot_dir / "neff_calibration.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_dir / 'neff_calibration_summary.csv'}")
    print(f"Wrote {plot_dir / 'neff_calibration.png'}")


if __name__ == "__main__":
    main()
