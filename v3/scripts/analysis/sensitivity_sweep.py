"""Hyperparameter sensitivity sweep for v3 steps 2-4.

Grids over:
  deadband_halfwidth in {0.02, 0.05, 0.08, 0.12}
  delta              in {0.00, 0.02, 0.03, 0.05}
  knn_k              in {3, 5, 10}

For each combination, re-runs classification on the already-computed
within_parent_site_scores.csv (score_median, score_knn, t_med, t_knn columns
are fixed — only the decision boundaries change). This means the sweep takes
seconds rather than minutes.

Output:
  v3/data/sensitivity_sweep.csv  — one row per (hw, delta, k, probe_state)
  Console heatmaps of degraded_pct and error_rate for good/bad probes.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def classify_batch(
    score: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    t_lo: float, t_hi: float, delta: float,
) -> np.ndarray:
    out = np.full(len(score), "indistinguishable", dtype=object)
    finite = np.isfinite(score) & np.isfinite(lo) & np.isfinite(hi)
    out[~finite] = "no_data"
    out[finite & (lo > t_hi) & ((score - t_hi) >= delta)] = "recovering"
    out[finite & (hi < t_lo) & ((t_lo - score) >= delta)] = "degraded"
    return out


def sweep_one(
    df: pd.DataFrame,
    hw: float,
    delta: float,
    k: int,
) -> list[dict]:
    """Re-classify all rows under (hw, delta) and return summary rows.

    knn_k only affects which t_knn was calibrated to, but since we re-use the
    pre-computed t_knn from the original run (k=5), the k parameter here only
    changes the step4 score column used. Because the site_scores file stores
    only one knn score (computed at k=5), we note k in the output for
    reference but cannot re-score at different k without re-running bootstrap.
    See note below.
    """
    score_med = df["score_median"].to_numpy()
    score_knn = df["score_knn"].to_numpy()
    t_med = df["t_med"].to_numpy()
    t_knn = df["t_knn"].to_numpy()

    # CI bounds are not stored in site_scores; approximate via score ± 0
    # (i.e., use point-estimate-only classify: lo=hi=score).
    # This is equivalent to the non-bootstrap classify from test_steps_1_to_4.py
    # for a fair sensitivity comparison of the boundary parameters.
    lo_med = score_med
    hi_med = score_med
    lo_knn = score_knn
    hi_knn = score_knn

    cat_s2 = classify_batch(score_med, lo_med, hi_med, t_med - hw, t_med + hw, 0.0)
    cat_s3 = classify_batch(score_med, lo_med, hi_med, t_med - hw, t_med + hw, delta)
    cat_s4 = classify_batch(score_knn, lo_knn, hi_knn, t_knn - hw, t_knn + hw, delta)

    rows = []
    for probe in ("good", "bad"):
        mask = df["probe_state"].to_numpy() == probe
        for step, cats in (("step2_deadband", cat_s2),
                            ("step3_deadband_effect", cat_s3),
                            ("step4_knn", cat_s4)):
            c = cats[mask]
            n = mask.sum()
            if probe == "good":
                error_rate = float((c == "degraded").sum() / n)
                true_rate  = float((c == "recovering").sum() / n)
            else:
                error_rate = float((c == "recovering").sum() / n)
                true_rate  = float((c == "degraded").sum() / n)
            rows.append({
                "deadband_halfwidth": hw,
                "delta": delta,
                "knn_k": k,
                "probe_state": probe,
                "step": step,
                "n": int(n),
                "true_rate": true_rate,
                "error_rate": error_rate,
                "abstain_rate": float((c == "indistinguishable").sum() / n),
            })
    return rows


def print_heatmap(
    sweep_df: pd.DataFrame,
    step: str,
    probe: str,
    metric: str,
    hw_vals: list,
    delta_vals: list,
) -> None:
    sub = sweep_df[(sweep_df["step"] == step) & (sweep_df["probe_state"] == probe)
                   & (sweep_df["knn_k"] == sweep_df["knn_k"].iloc[0])]
    pivot = sub.pivot(index="deadband_halfwidth", columns="delta", values=metric)
    probe_tag = "good (false-deg)" if probe == "good" else "bad  (false-rec)"
    print(f"\n  {step} | probe={probe_tag} | {metric}")
    header = f"  {'hw \\ delta':<10}" + "".join(f"  d={d:.2f}" for d in delta_vals)
    print(header)
    for hw in hw_vals:
        if hw not in pivot.index:
            continue
        row_str = f"  hw={hw:.2f}      " + "".join(
            f"  {pivot.at[hw, d]:>6.1%}" if d in pivot.columns else "      --"
            for d in delta_vals
        )
        print(row_str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--site-scores",
        default="v3/data/within_parent_site_scores.csv",
        help="Path to within_parent_site_scores.csv from validate_steps_within_parent.py",
    )
    parser.add_argument("--out-dir", default="v3/data")
    args = parser.parse_args()

    df = pd.read_csv(args.site_scores)
    print(f"Loaded {len(df):,} probe rows from {args.site_scores}")

    # Note on k sweep: the stored score_knn was computed at k=5 (default).
    # Re-scoring at k=3 or k=10 would require re-running the bootstrap.
    # We sweep k in the output for labelling but scores are fixed at k=5.
    hw_vals   = [0.02, 0.05, 0.08, 0.12]
    delta_vals = [0.00, 0.02, 0.03, 0.05]
    k_vals    = [5]   # only k=5 scores are available in stored site_scores

    print(f"Grid: {len(hw_vals)} hw x {len(delta_vals)} delta x {len(k_vals)} k "
          f"= {len(hw_vals)*len(delta_vals)*len(k_vals)} combinations", flush=True)

    all_rows: list[dict] = []
    for hw, delta, k in itertools.product(hw_vals, delta_vals, k_vals):
        all_rows.extend(sweep_one(df, hw, delta, k))

    sweep_df = pd.DataFrame(all_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitivity_sweep.csv"
    sweep_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

    # ------------------------------------------------------------------
    # Console heatmaps
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SENSITIVITY SWEEP  (point-estimate classify, k=5 scores)")
    print("=" * 70)
    print("error% = fraction of probes hard-misclassified (not abstained)")

    for step in ("step2_deadband", "step3_deadband_effect", "step4_knn"):
        for probe in ("good", "bad"):
            print_heatmap(sweep_df, step, probe, "error_rate",  hw_vals, delta_vals)
            print_heatmap(sweep_df, step, probe, "abstain_rate", hw_vals, delta_vals)

    # Compact comparison: baseline (hw=0.05, delta=0.03) vs extremes
    print("\n" + "=" * 70)
    print("KEY TRADE-OFF: error% vs abstain% at default (hw=0.05, d=0.03)")
    print("=" * 70)
    default = sweep_df[
        (sweep_df["deadband_halfwidth"] == 0.05) &
        (sweep_df["delta"] == 0.03) &
        (sweep_df["knn_k"] == 5)
    ]
    for probe in ("good", "bad"):
        p_tag = "good (false-deg)" if probe == "good" else "bad  (false-rec)"
        print(f"\n  probe={p_tag}")
        print(f"  {'step':<30} {'error%':>7}  {'abstain%':>9}")
        for step in ("step2_deadband", "step3_deadband_effect", "step4_knn"):
            r = default[(default["probe_state"] == probe) & (default["step"] == step)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {step:<30} {r['error_rate']:>7.1%}  {r['abstain_rate']:>9.1%}")


if __name__ == "__main__":
    main()
