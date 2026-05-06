"""Compare v1 and v2 per-site DoR outputs.

Writes:
  - wide join table (v1/v2 side by side)
  - metric deltas summary
  - category transition matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def category(df: pd.DataFrame, med_col: str, lo_col: str, hi_col: str) -> pd.Series:
    out = pd.Series("no_data", index=df.index)
    m = df[med_col].notna()
    out[m & (df[lo_col] > 0.5)] = "recovering"
    out[m & (df[hi_col] < 0.5)] = "degraded"
    out[m & (df[lo_col] <= 0.5) & (df[hi_col] >= 0.5)] = "indistinguishable"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True, help="v1 test_site_dor.csv")
    parser.add_argument("--v2", required=True, help="v2 test_site_dor_v2.csv")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = pd.read_csv(args.v1)
    v2 = pd.read_csv(args.v2)

    for df in (v1, v2):
        df["parent_id"] = df["parent_id"].astype(str)

    keep = [
        "parent_id",
        "parent_label",
        "applicable",
        "dor_median",
        "dor_median_ci_low",
        "dor_median_ci_high",
        "n_eff_min",
    ]
    v1 = v1[[c for c in keep if c in v1.columns]].copy()
    v2 = v2[[c for c in keep if c in v2.columns]].copy()

    v1 = v1.rename(columns={c: f"{c}_v1" for c in v1.columns if c != "parent_id"})
    v2 = v2.rename(columns={c: f"{c}_v2" for c in v2.columns if c != "parent_id"})

    m = v1.merge(v2, on="parent_id", how="inner")
    m["dor_delta_v2_minus_v1"] = m["dor_median_v2"] - m["dor_median_v1"]
    if "n_eff_min_v1" in m.columns and "n_eff_min_v2" in m.columns:
        m["n_eff_min_delta_v2_minus_v1"] = m["n_eff_min_v2"] - m["n_eff_min_v1"]

    m["category_v1"] = category(
        m,
        "dor_median_v1",
        "dor_median_ci_low_v1",
        "dor_median_ci_high_v1",
    )
    m["category_v2"] = category(
        m,
        "dor_median_v2",
        "dor_median_ci_low_v2",
        "dor_median_ci_high_v2",
    )
    m["category_changed"] = m["category_v1"] != m["category_v2"]

    wide_csv = out_dir / "v1_vs_v2_site_scores.csv"
    m.sort_values("parent_id").to_csv(wide_csv, index=False)

    trans = (
        m.groupby(["category_v1", "category_v2"], as_index=False)
        .size()
        .rename(columns={"size": "n_sites"})
        .sort_values(["category_v1", "category_v2"])
    )
    trans_csv = out_dir / "v1_vs_v2_category_transition.csv"
    trans.to_csv(trans_csv, index=False)

    summary = {
        "n_sites_joined": int(len(m)),
        "n_category_changed": int(m["category_changed"].sum()),
        "median_abs_dor_delta": float(m["dor_delta_v2_minus_v1"].abs().median()),
        "mean_dor_delta": float(m["dor_delta_v2_minus_v1"].mean()),
        "p90_abs_dor_delta": float(m["dor_delta_v2_minus_v1"].abs().quantile(0.9)),
        "mean_dor_v1": float(m["dor_median_v1"].mean()),
        "mean_dor_v2": float(m["dor_median_v2"].mean()),
    }
    if "n_eff_min_delta_v2_minus_v1" in m.columns:
        summary["mean_neff_delta_v2_minus_v1"] = float(
            m["n_eff_min_delta_v2_minus_v1"].mean()
        )
    summary_df = pd.DataFrame([summary])
    summary_csv = out_dir / "v1_vs_v2_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote {wide_csv}")
    print(f"Wrote {trans_csv}")
    print(f"Wrote {summary_csv}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nCategory transitions:")
    print(trans.to_string(index=False))


if __name__ == "__main__":
    main()
