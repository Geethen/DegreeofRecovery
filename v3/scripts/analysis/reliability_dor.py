"""Reliability decomposition for dor_median vs dor_knn.

Reuses per-probe scores from scorer_compare_site_scores.csv (same probes
across all variants — pure scorer comparison). Produces:

  v3/data/reliability_dor.csv     equal-frequency bin table (10 bins per scorer)
  v3/data/reliability_summary.csv ECE / MCE / Brier decomposition per scorer

Reliability bin row: (scorer, bin, n, mean_score, frac_good, gap)
where gap = mean_score - frac_good (positive = overconfident "good").

This is a calibration view, complementary to Brier. Brier already shows
dor_knn dominates (0.126 vs 0.158 in scorer compare); ECE/MCE clarify
*where* in the [0,1] interval each scorer is well or badly calibrated.

Usage:
  python v3/scripts/analysis/reliability_dor.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Map scorer columns in scorer_compare_site_scores.csv -> human label
SCORERS = {
    "med_cos_all": "dor_median",
    "mean_cos_k5": "dor_knn",
}


def equal_frequency_bins(s: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Bin scores into n_bins equal-frequency groups; return mean_score, frac_good per bin."""
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    order = np.argsort(s)
    s, y = s[order], y[order]
    n = len(s)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        s_bin = s[lo:hi]
        y_bin = y[lo:hi]
        rows.append({
            "bin":         b,
            "n":           int(hi - lo),
            "score_lo":    float(s_bin.min()),
            "score_hi":    float(s_bin.max()),
            "mean_score":  float(s_bin.mean()),
            "frac_good":   float(y_bin.mean()),
            "gap":         float(s_bin.mean() - y_bin.mean()),
        })
    return pd.DataFrame(rows)


def ece_mce(bins: pd.DataFrame) -> tuple[float, float]:
    """Expected and Maximum Calibration Error from a binned table."""
    n_total = bins["n"].sum()
    weights = bins["n"] / n_total
    abs_gap = bins["gap"].abs()
    return float((weights * abs_gap).sum()), float(abs_gap.max())


def brier(s: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(s)
    return float(np.mean((s[m] - y[m]) ** 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--site-scores",
        default="v3/data/scorer_compare_site_scores.csv",
        help="Per-probe scores CSV from compare_scorers.py.",
    )
    p.add_argument("--out-dir", default="v3/data")
    p.add_argument("--n-bins", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.site_scores)
    y = (df["probe_state"] == "good").to_numpy().astype(int)
    print(f"Probes: {len(df):,}  good={int(y.sum()):,}  bad={int((1-y).sum()):,}")

    bin_rows = []
    summary_rows = []
    for col, name in SCORERS.items():
        s = df[col].to_numpy()
        bins = equal_frequency_bins(s, y, args.n_bins)
        bins.insert(0, "scorer", name)
        bin_rows.append(bins)
        ece, mce = ece_mce(bins)
        summary_rows.append({
            "scorer":   name,
            "n":        int(np.isfinite(s).sum()),
            "brier":    brier(s, y.astype(float)),
            "ece":      ece,
            "mce":      mce,
            "n_bins":   args.n_bins,
        })

    bins_out = pd.concat(bin_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)

    bins_path = out_dir / "reliability_dor.csv"
    summary_path = out_dir / "reliability_summary.csv"
    bins_out.to_csv(bins_path, index=False)
    summary_out.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print(f"RELIABILITY DECOMPOSITION  ({args.n_bins} equal-frequency bins)")
    print("=" * 70)
    print("  gap > 0 : score over-promises good (overconfident on recovering)")
    print("  gap < 0 : score under-promises good (underconfident on recovering)\n")
    print(summary_out.to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    ))

    for _, row in summary_out.iterrows():
        scorer = row["scorer"]
        sub = bins_out[bins_out["scorer"] == scorer]
        print(f"\n  {scorer}  (ECE={row['ece']:.4f}  MCE={row['mce']:.4f})")
        print(f"  {'bin':>3}  {'n':>5}  {'score':>6}  {'mean_s':>7}  "
              f"{'frac_g':>7}  {'gap':>7}")
        for _, b in sub.iterrows():
            print(f"  {int(b['bin']):>3}  {int(b['n']):>5}  "
                  f"[{b['score_lo']:.2f},{b['score_hi']:.2f}]  "
                  f"{b['mean_score']:>7.4f}  {b['frac_good']:>7.4f}  "
                  f"{b['gap']:>+7.4f}")

    print(f"\nWrote: {bins_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
