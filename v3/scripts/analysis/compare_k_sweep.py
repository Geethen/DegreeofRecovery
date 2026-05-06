"""Sweep k for mean_cos and mean_eucl; sweep sample size for med_cos_all (dor_median).

Strategies tested (random_N family — consistent method, varying n per class):
  random_40, random_60, random_100, random_150, random_200

For each strategy:
  - med_cos_all  : median + cosine + all refs  (dor_median equivalent)
  - mean_cos_k   : mean  + cosine + k nearest  (k swept over K_VALUES)
  - mean_eucl_k  : mean  + Euclidean + k nearest

kNN variants always use random_100 refs (the baseline strategy) so k is the
only thing varying. dor_median uses each strategy so sample size is the only
thing varying.

Outputs:
  v3/data/k_sweep_summary.csv
  v3/plots/k_sweep.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS = 1e-12

K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75]
STRATEGIES = ["random_40", "random_60", "random_100", "random_150", "random_200"]
STRATEGY_N = {"random_40": 40, "random_60": 60, "random_100": 100,
              "random_150": 150, "random_200": 200}
KNN_STRATEGY = "random_100"  # kNN variants always use this


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_mean_k(D_g: np.ndarray, D_b: np.ndarray, k: int) -> np.ndarray:
    if D_g.shape[1] < k or D_b.shape[1] < k:
        return np.full(D_g.shape[0], np.nan)
    Dg_k = np.partition(D_g, k - 1, axis=1)[:, :k]
    Db_k = np.partition(D_b, k - 1, axis=1)[:, :k]
    return np.mean(Db_k, axis=1) / (np.mean(Dg_k, axis=1) + np.mean(Db_k, axis=1) + EPS)


def score_median_all(D_g: np.ndarray, D_b: np.ndarray) -> np.ndarray:
    m_g = np.median(D_g, axis=1)
    m_b = np.median(D_b, axis=1)
    return m_b / (m_g + m_b + EPS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def roc_auc(scores: np.ndarray, y_good: np.ndarray) -> float:
    m = np.isfinite(scores)
    s, y = scores[m], y_good[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    s_s, y_s = s[order], y[order]
    n = len(s_s)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_s[j + 1] == s_s[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    n_pos = int(y_s.sum())
    n_neg = n - n_pos
    return float((ranks[y_s == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def youden_threshold(scores: np.ndarray, y_good: np.ndarray) -> float:
    m = np.isfinite(scores)
    s, y = scores[m], y_good[m]
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200))):
        pred = s >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        j = tp / (tp + fn + EPS) - fp / (fp + tn + EPS)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def compute_metrics(scores: np.ndarray, y_good: np.ndarray) -> dict:
    m = np.isfinite(scores)
    brier = float(np.mean((scores[m] - y_good[m].astype(float)) ** 2))
    auc = roc_auc(scores, y_good)
    t = youden_threshold(scores, y_good)
    pred = (scores >= t).astype(int)
    good_m = (y_good == 1) & m
    bad_m  = (y_good == 0) & m
    false_deg = float(1 - pred[good_m].mean()) if good_m.any() else float("nan")
    false_rec = float(pred[bad_m].mean())       if bad_m.any()  else float("nan")
    return dict(auc=auc, brier=brier, youden_t=t,
                false_deg=false_deg, false_rec=false_rec,
                bal_err=(false_deg + false_rec) / 2)


# ---------------------------------------------------------------------------
# Fold helpers
# ---------------------------------------------------------------------------

def make_folds(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    f = np.arange(n) % k
    rng.shuffle(f)
    return f


def load_strategy(parquet: str, strategy: str) -> pd.DataFrame:
    con = duckdb.connect()
    cols_sql = ", ".join(["parent_id", "parent_label", "ref_state"] + EMBED_COLS)
    df = con.execute(
        f"SELECT {cols_sql} FROM read_parquet(?) WHERE strategy = ?",
        [parquet, strategy],
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])].dropna(
        subset=["parent_id", "ref_state"] + EMBED_COLS
    ).reset_index(drop=True)
    return df


def build_parent_cache(df: pd.DataFrame, rng: np.random.Generator,
                       n_folds: int, dist: str = "cosine"):
    """Returns list of (good_idx, bad_idx, D, fa) per parent."""
    parents = []
    for _, sub in df.groupby("parent_id", sort=False):
        emb  = sub[EMBED_COLS].to_numpy(dtype=float)
        lbl  = sub["ref_state"].to_numpy()
        is_g = lbl == "good"
        is_b = lbl == "bad"
        if is_g.sum() < 2 or is_b.sum() < 2:
            continue
        good_idx = np.where(is_g)[0]
        bad_idx  = np.where(is_b)[0]
        D = cdist(emb, emb, metric=dist)
        fa = np.full(len(emb), -1, dtype=np.int8)
        fa[good_idx] = make_folds(len(good_idx), n_folds, rng)
        fa[bad_idx]  = make_folds(len(bad_idx),  n_folds, rng)
        parents.append((good_idx, bad_idx, D, fa))
    return parents


def run_probe_loop(parents, n_folds: int, score_fn, y_good_buf: list) -> np.ndarray:
    """Run scored probe loop; appends probe labels to y_good_buf if it's empty."""
    scores = []
    fill_labels = len(y_good_buf) == 0
    for good_idx, bad_idx, D, fa in parents:
        for f in range(n_folds):
            train_g = good_idx[fa[good_idx] != f]
            train_b = bad_idx[fa[bad_idx]   != f]
            for state, probe_idx in (("good", good_idx[fa[good_idx] == f]),
                                     ("bad",  bad_idx[fa[bad_idx]   == f])):
                if len(probe_idx) == 0:
                    continue
                Dg = D[probe_idx[:, None], train_g]
                Db = D[probe_idx[:, None], train_b]
                scores.append(score_fn(Dg, Db))
                if fill_labels:
                    y_good_buf.extend([1 if state == "good" else 0] * len(probe_idx))
    return np.concatenate(scores) if scores else np.array([])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(knn_df: pd.DataFrame, median_df: pd.DataFrame, out_path: Path) -> None:
    """Plot kNN k-sweep and dor_median n-sweep on a shared x-axis (k / n).

    Both series are plotted as lines. The x-axis represents k for kNN variants
    and n (refs per class) for dor_median. Tick positions are the union of all
    k and n values; labels show the value with a suffix indicating which axis
    it belongs to (k= for kNN-only points, n= for median-only points, plain
    number for shared values like 100).
    """
    metrics_cfg = [
        ("auc",     "ROC-AUC (↑ better)",            True),
        ("brier",   "Brier score (↓ better)",         False),
        ("bal_err", "Balanced error rate (↓ better)", False),
    ]

    k_vals = sorted(knn_df["k"].unique())                       # [1,2,3,5,7,10,15,20,30,50,75]
    n_vals = sorted(median_df["n_refs"].unique())               # [40,60,100,150,200]
    all_x  = sorted(set(k_vals) | set(n_vals))                  # union for tick positions

    cos_col  = "#1f77b4"
    eucl_col = "#ff7f0e"
    med_col  = "#2ca02c"

    def x_label(v):
        in_k = v in k_vals
        in_n = v in n_vals
        if in_k and in_n:
            return str(v)
        if in_k:
            return f"k={v}"
        return f"n={v}"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, (metric, ylabel, _) in zip(axes, metrics_cfg):
        # kNN lines
        for var, col, lab in [("mean_cos",  cos_col,  "mean cosine  (x = k)"),
                               ("mean_eucl", eucl_col, "mean Euclidean  (x = k)")]:
            sub = knn_df[knn_df["variant"] == var].sort_values("k")
            ax.plot(sub["k"], sub[metric], color=col, marker="o",
                    lw=2, ms=5, label=lab)

        # dor_median line over n
        med_sorted = median_df.sort_values("n_refs")
        ax.plot(med_sorted["n_refs"], med_sorted[metric], color=med_col,
                marker="s", lw=2, ms=5, ls="--", label="dor_median  (x = n refs)")

        ax.set_xscale("log")
        ax.set_xlabel("k  (kNN variants)  /  n refs per class  (dor_median)")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(all_x)
        ax.set_xticklabels([x_label(v) for v in all_x], rotation=60, ha="right", fontsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "kNN scorer k-sweep vs dor_median n-sweep on a shared axis\n"
        "kNN uses random_100 refs throughout; dor_median varies n per class",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--refs",
        default="v2/data/v2real_mask_on_corr300_exhaustive/sampling_strategy_selected_points.parquet",
    )
    parser.add_argument("--out-dir",   default="v3/data")
    parser.add_argument("--plots-dir", default="v3/plots")
    parser.add_argument("--n-folds",   type=int, default=5)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    out_dir   = Path(args.out_dir);   out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Part 1: kNN k-sweep on random_100
    # ------------------------------------------------------------------
    print(f"\nLoading {KNN_STRATEGY} for kNN sweep...")
    df_knn = load_strategy(args.refs, KNN_STRATEGY)
    print(f"  rows: {len(df_knn):,}  parents: {df_knn['parent_id'].nunique()}")

    print("Building cosine + Euclidean caches for kNN sweep...", flush=True)
    parents_cos  = build_parent_cache(df_knn, np.random.default_rng(args.seed), args.n_folds, "cosine")
    parents_eucl = build_parent_cache(df_knn, np.random.default_rng(args.seed), args.n_folds, "euclidean")
    print(f"  Parents: {len(parents_cos)}")

    knn_rows = []
    y_good_knn: list[int] = []

    print("Running kNN k-sweep...", flush=True)
    for k in K_VALUES:
        print(f"  k={k}", end="  ", flush=True)
        y_buf: list[int] = [] if not y_good_knn else y_good_knn
        fill = not bool(y_good_knn)

        s_cos  = run_probe_loop(parents_cos,  args.n_folds,
                                lambda Dg, Db, _k=k: score_mean_k(Dg, Db, _k),
                                y_buf if fill else [])
        s_eucl = run_probe_loop(parents_eucl, args.n_folds,
                                lambda Dg, Db, _k=k: score_mean_k(Dg, Db, _k),
                                [])

        if fill:
            y_good_knn = y_buf

        y_arr = np.array(y_good_knn, dtype=int)
        for var, s in [("mean_cos", s_cos), ("mean_eucl", s_eucl)]:
            m = compute_metrics(s, y_arr)
            knn_rows.append({"variant": var, "k": k, **m})

    print()
    knn_df = pd.DataFrame(knn_rows)

    # ------------------------------------------------------------------
    # Part 2: dor_median at each sample-size strategy
    # ------------------------------------------------------------------
    median_rows = []
    print("\nRunning dor_median across sample sizes...")
    for strat in STRATEGIES:
        n_refs = STRATEGY_N[strat]
        print(f"  {strat} (n={n_refs})...", flush=True)
        df_s = load_strategy(args.refs, strat)
        parents_s = build_parent_cache(df_s, np.random.default_rng(args.seed),
                                       args.n_folds, "cosine")
        y_buf: list[int] = []
        s = run_probe_loop(parents_s, args.n_folds, score_median_all, y_buf)
        y_arr = np.array(y_buf, dtype=int)
        m = compute_metrics(s, y_arr)
        median_rows.append({"strategy": strat, "n_refs": n_refs, **m})
        print(f"    probes={len(y_buf):,}  AUC={m['auc']:.4f}  brier={m['brier']:.4f}  bal_err={m['bal_err']:.4f}")

    median_df = pd.DataFrame(median_rows)

    # ------------------------------------------------------------------
    # Save + report
    # ------------------------------------------------------------------
    knn_df.to_csv(out_dir / "k_sweep_knn.csv", index=False)
    median_df.to_csv(out_dir / "k_sweep_median_sample_sizes.csv", index=False)

    print("\n" + "=" * 80)
    print("kNN k-sweep (random_100 refs)")
    print("=" * 80)
    print(f"  {'variant':<12} {'k':>4}  {'AUC':>6}  {'Brier':>6}  {'false_deg':>9}  {'false_rec':>9}  {'bal_err':>8}")
    for _, row in knn_df.iterrows():
        print(f"  {row['variant']:<12} {int(row['k']):>4}  {row['auc']:>6.4f}  {row['brier']:>6.4f}  "
              f"{row['false_deg']:>9.4f}  {row['false_rec']:>9.4f}  {row['bal_err']:>8.4f}")

    print("\n" + "=" * 80)
    print("dor_median at varying sample sizes")
    print("=" * 80)
    print(f"  {'strategy':<18} {'n':>4}  {'AUC':>6}  {'Brier':>6}  {'false_deg':>9}  {'false_rec':>9}  {'bal_err':>8}")
    for _, row in median_df.iterrows():
        print(f"  {row['strategy']:<18} {int(row['n_refs']):>4}  {row['auc']:>6.4f}  {row['brier']:>6.4f}  "
              f"{row['false_deg']:>9.4f}  {row['false_rec']:>9.4f}  {row['bal_err']:>8.4f}")

    make_plot(knn_df, median_df, plots_dir / "k_sweep.png")


if __name__ == "__main__":
    main()
