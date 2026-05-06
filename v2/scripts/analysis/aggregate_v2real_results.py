"""Aggregate v2-real exhaustive comparison runs into a single robust ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "v2" / "data"

SCENARIOS = [(mask, corr) for mask in ("on", "off") for corr in (150, 300, 500)]


def main() -> None:
    rows = []
    for mask, corr in SCENARIOS:
        d = DATA / f"v2real_mask_{mask}_corr{corr}_exhaustive"
        csv = d / "sampling_strategy_comparison.csv"
        if not csv.exists():
            print(f"missing: {csv}")
            continue
        s = pd.read_csv(csv)
        s["mask"] = mask
        s["corr_range_m"] = corr
        rows.append(s)

    if not rows:
        raise SystemExit("no input scenarios found")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA / "v2real_exhaustive_summary_all.csv", index=False)

    rank_frames = []
    for (mask, corr), g in out.groupby(["mask", "corr_range_m"]):
        g = g.sort_values(
            ["median_ci_width", "pooled_auc"], ascending=[True, False]
        ).reset_index(drop=True)
        g["rank"] = g.index + 1
        rank_frames.append(g[["strategy", "rank", "mask", "corr_range_m"]])
    ranks = pd.concat(rank_frames, ignore_index=True)
    rob = (
        ranks.groupby("strategy", as_index=False)["rank"]
        .mean()
        .sort_values("rank")
        .rename(columns={"rank": "mean_rank"})
    )
    rob.to_csv(DATA / "v2real_exhaustive_robust_rank.csv", index=False)

    top5 = (
        out.sort_values(
            ["mask", "corr_range_m", "median_ci_width", "pooled_auc"],
            ascending=[True, True, True, False],
        )
        .groupby(["mask", "corr_range_m"], as_index=False)
        .head(5)
    )
    top5.to_csv(DATA / "v2real_exhaustive_top5_by_scenario.csv", index=False)

    print("\nRobust rank top 10 (lower = better; averaged across 6 scenarios):")
    print(rob.head(10).to_string(index=False))

    print("\nTop strategy per scenario:")
    print(
        top5.groupby(["mask", "corr_range_m"])
        .head(1)[
            [
                "mask",
                "corr_range_m",
                "strategy",
                "median_ci_width",
                "median_neff_min",
                "min_neff30_coverage",
                "pooled_auc",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
