"""Compare DoR scorer variants on identical within-parent 5-fold splits.

Six scorer variants are evaluated as continuous DoR scores using the same
per-probe held-out splits (same n, same probes), so differences are purely
due to the scorer formula.

Variants
--------
  med_cos_all    median + cosine    + all refs   (= production dor_median)
  med_eucl_all   median + Euclidean + all refs
  med_cos_k5     median + cosine    + 5 nearest
  med_eucl_k5    median + Euclidean + 5 nearest
  mean_cos_k5    mean   + cosine    + 5 nearest  (= "fixed" score_knn)
  mean_eucl_k5   mean   + Euclidean + 5 nearest  (= current score_knn)

Score formula (all variants):
    s = agg(d_obs_to_bad_subset) / (agg(d_obs_to_good_subset) + agg(d_obs_to_bad_subset))

Higher score => closer to GOOD cloud (recovering). Probe label "good" expects
HIGH score, "bad" expects LOW score. (Score = median(d_bad) / (median(d_good) +
median(d_bad)): when obs is close to good refs, d_good is small so numerator
dominates → high score.)

Metrics
-------
  Continuous: Brier (target good=1, bad=0 — score is "recovering-ness").
              AUC with "good" as positive class (higher score => good).
  Calibrated classifier: Youden-J threshold per variant from same probe pool,
              report false-degraded (good probes called degraded) and
              false-recovering (bad probes called recovering).

Notes
-----
* No bootstrap CI / deadband / abstention — those are downstream steps that
  apply identically to any scorer; including them would obscure scorer-only
  differences. Pure scorer comparison.
* Threshold calibrated on the same probe scores it is evaluated on
  (in-sample). This slightly inflates accuracy uniformly across variants;
  the *relative* ordering is unaffected, which is what we care about.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS = 1e-12

VARIANTS = [
    ("med_cos_all",  "median", "cosine",    None),
    ("med_eucl_all", "median", "euclidean", None),
    ("med_cos_k5",   "median", "cosine",    5),
    ("med_eucl_k5",  "median", "euclidean", 5),
    ("mean_cos_k5",  "mean",   "cosine",    5),
    ("mean_eucl_k5", "mean",   "euclidean", 5),
]


def score_variant(D_g: np.ndarray, D_b: np.ndarray, agg: str, k: int | None) -> np.ndarray:
    """D_g, D_b: (n_probes, n_refs) precomputed distance matrices.
    Returns 1D score array length n_probes."""
    if k is not None:
        if D_g.shape[1] < k or D_b.shape[1] < k:
            return np.full(D_g.shape[0], np.nan)
        # k smallest along refs axis
        Dg_k = np.partition(D_g, k - 1, axis=1)[:, :k]
        Db_k = np.partition(D_b, k - 1, axis=1)[:, :k]
    else:
        Dg_k, Db_k = D_g, D_b

    if agg == "median":
        m_g = np.median(Dg_k, axis=1)
        m_b = np.median(Db_k, axis=1)
    elif agg == "mean":
        m_g = np.mean(Dg_k, axis=1)
        m_b = np.mean(Db_k, axis=1)
    else:
        raise ValueError(agg)
    return m_b / (m_g + m_b + EPS)


def youden_threshold(scores: np.ndarray, y_good: np.ndarray) -> float:
    """y_good=1 if probe is good (high score expected). Maximises TPR - FPR.
    Predicts good when score >= t."""
    m = np.isfinite(scores)
    s, y = scores[m], y_good[m]
    if len(np.unique(y)) < 2:
        return 0.5
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    best_t, best_j = 0.5, -1.0
    for t in cands:
        pred = s >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        j = tp / (tp + fn + EPS) - fp / (fp + tn + EPS)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def roc_auc(scores: np.ndarray, y_good: np.ndarray) -> float:
    """ROC-AUC with good as positive class (higher score => good). Tie-aware."""
    m = np.isfinite(scores)
    s, y = scores[m], y_good[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    # Average ranks for ties
    n = len(s_sorted)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed average
        ranks[i:j + 1] = avg_rank
        i = j + 1
    n_pos = int(y_sorted.sum())
    n_neg = n - n_pos
    sum_pos_ranks = ranks[y_sorted == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def make_folds(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    f = np.arange(n) % k
    rng.shuffle(f)
    return f


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--refs",
        default="v2/data/v2real_mask_on_corr300_exhaustive/sampling_strategy_selected_points.parquet",
    )
    p.add_argument("--strategy", default="random_100")
    p.add_argument("--out-dir", default="v3/data")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    cols_sql = ", ".join(["parent_id", "parent_label", "ref_state"] + EMBED_COLS)
    df = con.execute(
        f"SELECT {cols_sql} FROM read_parquet(?) WHERE strategy = ?",
        [args.refs, args.strategy],
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])].dropna(
        subset=["parent_id", "ref_state"] + EMBED_COLS
    ).reset_index(drop=True)
    print(f"Refs rows: {len(df):,}  parents: {df['parent_id'].nunique()}")

    rng = np.random.default_rng(args.seed)
    K = args.n_folds

    # Per-parent caches: precompute BOTH cosine and Euclidean full matrices once
    print("Precomputing distance matrices per parent...", flush=True)
    parents = []
    for pid, sub in df.groupby("parent_id", sort=False):
        emb = sub[EMBED_COLS].to_numpy(dtype=float)
        lbl = sub["ref_state"].to_numpy()
        is_g = lbl == "good"
        is_b = lbl == "bad"
        if is_g.sum() < args.knn_k + 1 or is_b.sum() < args.knn_k + 1:
            continue
        good_idx = np.where(is_g)[0]
        bad_idx = np.where(is_b)[0]
        Dcos = cdist(emb, emb, metric="cosine")
        Deuc = cdist(emb, emb, metric="euclidean")
        # Per-parent stratified folds for good and bad
        fa = np.full(len(emb), -1, dtype=np.int8)
        fa[good_idx] = make_folds(len(good_idx), K, rng)
        fa[bad_idx]  = make_folds(len(bad_idx),  K, rng)
        parents.append((pid, str(sub["parent_label"].iloc[0]),
                        good_idx, bad_idx, Dcos, Deuc, fa))
    print(f"Parents kept: {len(parents)}")

    # Probe loop: collect per-probe scores for each variant
    score_buf: dict[str, list[np.ndarray]] = {v[0]: [] for v in VARIANTS}
    pid_buf: list[str] = []
    label_buf: list[str] = []
    state_buf: list[str] = []  # "good" or "bad"
    fold_buf: list[int] = []

    for pid, label, good_idx, bad_idx, Dcos, Deuc, fa in parents:
        for f in range(K):
            train_g = good_idx[fa[good_idx] != f]
            train_b = bad_idx[fa[bad_idx]  != f]
            for state, probe_idx in (("good", good_idx[fa[good_idx] == f]),
                                     ("bad",  bad_idx[fa[bad_idx]  == f])):
                if len(probe_idx) == 0:
                    continue
                # Slice precomputed matrices: (n_probes, n_train)
                Dg_cos = Dcos[probe_idx[:, None], train_g]
                Db_cos = Dcos[probe_idx[:, None], train_b]
                Dg_euc = Deuc[probe_idx[:, None], train_g]
                Db_euc = Deuc[probe_idx[:, None], train_b]

                for name, agg, dist, k in VARIANTS:
                    Dg = Dg_cos if dist == "cosine" else Dg_euc
                    Db = Db_cos if dist == "cosine" else Db_euc
                    score_buf[name].append(score_variant(Dg, Db, agg, k))

                n_p = len(probe_idx)
                pid_buf.extend([pid] * n_p)
                label_buf.extend([label] * n_p)
                state_buf.extend([state] * n_p)
                fold_buf.extend([f] * n_p)

    site_df = pd.DataFrame({
        "parent_id":    pid_buf,
        "parent_label": label_buf,
        "probe_state":  state_buf,
        "fold":         fold_buf,
    })
    for name in score_buf:
        site_df[name] = np.concatenate(score_buf[name])

    n_total = len(site_df)
    n_good = (site_df["probe_state"] == "good").sum()
    n_bad  = (site_df["probe_state"] == "bad").sum()
    print(f"Probes: {n_total:,}  good={n_good:,}  bad={n_bad:,}")

    # Compute summary metrics per variant
    # Score interpretation: HIGH score => GOOD (recovering). Positive class = good.
    y_good = (site_df["probe_state"] == "good").to_numpy().astype(int)
    rows = []
    for name, agg, dist, k in VARIANTS:
        s = site_df[name].to_numpy()
        target = y_good.astype(float)  # good=1, bad=0
        m = np.isfinite(s)
        brier = float(np.mean((s[m] - target[m]) ** 2))
        auc = roc_auc(s, y_good)
        t = youden_threshold(s, y_good)
        pred_good = (s >= t).astype(int)  # 1 = predicted good (recovering)
        good_mask = (y_good == 1) & m
        bad_mask  = (y_good == 0) & m
        # false_deg: a good probe got predicted as not-good (degraded)
        false_deg = float(1 - pred_good[good_mask].mean()) if good_mask.any() else float("nan")
        # false_rec: a bad probe got predicted as good (recovering)
        false_rec = float(pred_good[bad_mask].mean()) if bad_mask.any() else float("nan")
        balanced_err = (false_deg + false_rec) / 2
        rows.append({
            "variant":      name,
            "aggregator":   agg,
            "distance":     dist,
            "subset":       "all" if k is None else f"k={k}",
            "n":            int(m.sum()),
            "auc":          auc,
            "brier":        brier,
            "youden_t":     t,
            "false_deg":    false_deg,
            "false_rec":    false_rec,
            "bal_err":      balanced_err,
        })
    summary = pd.DataFrame(rows)

    # Per-label breakdown
    per_label_rows = []
    for lbl in ("built_loss", "crop_loss"):
        sub = site_df[site_df["parent_label"] == lbl]
        y_g = (sub["probe_state"] == "good").to_numpy().astype(int)
        for name, agg, dist, k in VARIANTS:
            s = sub[name].to_numpy()
            m = np.isfinite(s)
            if m.sum() == 0:
                continue
            target = y_g.astype(float)
            brier = float(np.mean((s[m] - target[m]) ** 2))
            auc = roc_auc(s, y_g)
            t_global = summary.loc[summary["variant"] == name, "youden_t"].iloc[0]
            pred_good = (s >= t_global).astype(int)
            good_m = (y_g == 1) & m
            bad_m  = (y_g == 0) & m
            fd = float(1 - pred_good[good_m].mean()) if good_m.any() else float("nan")
            fr = float(pred_good[bad_m].mean()) if bad_m.any() else float("nan")
            per_label_rows.append({
                "variant": name, "label": lbl,
                "n": int(m.sum()), "auc": auc, "brier": brier,
                "false_deg": fd, "false_rec": fr,
                "bal_err": (fd + fr) / 2,
            })
    per_label = pd.DataFrame(per_label_rows)

    # Write outputs
    site_out    = out_dir / "scorer_compare_site_scores.csv"
    summary_out = out_dir / "scorer_compare_summary.csv"
    label_out   = out_dir / "scorer_compare_per_label.csv"
    site_df.to_csv(site_out, index=False)
    summary.to_csv(summary_out, index=False)
    per_label.to_csv(label_out, index=False)

    # Console report
    print("\n" + "=" * 84)
    print("SCORER COMPARISON  (within-parent 5-fold, same probes for all variants)")
    print("=" * 84)
    print(f"  All variants score the same {n_total:,} held-out probes "
          f"({n_good:,} good + {n_bad:,} bad).\n"
          f"  Lower Brier = better calibrated.  Higher AUC = better separation.\n"
          f"  false_deg  = good probe called degraded   (false alarm)\n"
          f"  false_rec  = bad  probe called recovering (miss)\n"
          f"  bal_err    = mean of the two (balanced error at Youden-J threshold)\n")

    cols = ["variant", "aggregator", "distance", "subset",
            "auc", "brier", "youden_t", "false_deg", "false_rec", "bal_err"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n  Per-label breakdown (using each variant's global Youden-J threshold)")
    print("  " + "-" * 80)
    print(per_label[["variant", "label", "auc", "brier", "false_deg", "false_rec", "bal_err"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nWrote: {site_out}")
    print(f"Wrote: {summary_out}")
    print(f"Wrote: {label_out}")


if __name__ == "__main__":
    main()
