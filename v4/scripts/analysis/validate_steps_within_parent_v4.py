"""v4 within-parent K-fold validation with per-stable-class threshold refit.

Mirrors v3/scripts/analysis/validate_steps_within_parent.py but:

  * Reads v4 stable refs (parent_label ∈ {stable_nature, stable_crop, stable_built}).
  * Calibrates `t_med`, `t_knn` separately per stable_class (Youden-J on the
    parent_label-restricted LOO score distribution).
  * Also reports the v3-pooled threshold as a transfer-check column.
  * Writes the per-class thresholds to JSON so `score_test_sites_v4.py` can
    pick them up at scoring time.

The probe semantics are unchanged:
  - probe = 'good' (held-out natural-state ref): error = wrongly called 'degraded'
    - probe = 'bad'  (held-out degraded-state ref): error = wrongly called 'regenerating'

For stable_built parents, the validator still computes error in this frame —
but at scoring time the test-site itself is expected to land near the bad
cloud (sanity check). The validator treats refs only; the sanity-check
interpretation belongs in the test-site scorer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from degree_of_recovery.core_batch import (
    CAT_DEGRADED,
    CAT_INDISTINGUISHABLE,
    CAT_NAMES,
    CAT_NO_DATA,
    bootstrap_ci_batch,
    classify_batch,
    cosine_dist_matrix,
    knn_mean_3d as _knn_mean_3d,
    median_3d as _median_3d,
)

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EPS = 1e-12

V3_KNN_T = 0.4859
V3_MED_T = 0.4728


# ---------------------------------------------------------------------------
# Threshold calibration (Youden-J on per-class LOO score distribution)
# ---------------------------------------------------------------------------

def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    m = np.isfinite(scores)
    y = (labels[m] == "good").astype(int)
    s = scores[m]
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


def loo_median_scores(emb: np.ndarray, lbl: np.ndarray) -> np.ndarray:
    is_g, is_b = lbl == "good", lbl == "bad"
    if is_g.sum() < 2 or is_b.sum() < 2:
        return np.full(len(emb), np.nan)
    D_g = cosine_dist_matrix(emb, emb[is_g])
    D_b = cosine_dist_matrix(emb, emb[is_b])
    D_g[np.where(is_g)[0], np.arange(is_g.sum())] = np.nan
    D_b[np.where(is_b)[0], np.arange(is_b.sum())] = np.nan
    m_g = np.nanmedian(D_g, axis=1) + EPS
    m_b = np.nanmedian(D_b, axis=1) + EPS
    return m_b / (m_g + m_b)


def loo_knn_scores(emb: np.ndarray, lbl: np.ndarray, k: int) -> np.ndarray:
    is_g, is_b = lbl == "good", lbl == "bad"
    if is_g.sum() < k + 1 or is_b.sum() < k + 1:
        return np.full(len(emb), np.nan)
    D_g = cosine_dist_matrix(emb, emb[is_g])
    D_b = cosine_dist_matrix(emb, emb[is_b])
    idx_g, idx_b = np.where(is_g)[0], np.where(is_b)[0]
    for i, gi in enumerate(idx_g):
        D_g[gi, i] = np.inf
    for i, bi in enumerate(idx_b):
        D_b[bi, i] = np.inf
    D_g = np.where(np.isinf(D_g), np.nan, D_g)
    D_b = np.where(np.isinf(D_b), np.nan, D_b)
    D_g_sorted = np.sort(D_g, axis=1)[:, :k]
    D_b_sorted = np.sort(D_b, axis=1)[:, :k]
    return np.nanmean(D_b_sorted, axis=1) / (np.nanmean(D_g_sorted, axis=1)
                                              + np.nanmean(D_b_sorted, axis=1) + EPS)


def calibrate_per_stable_class(df: pd.DataFrame, k: int
                               ) -> dict[str, tuple[float, float]]:
    """Returns {parent_label: (t_med, t_knn)} where parent_label is one of
    'stable_nature' / 'stable_crop' / 'stable_built'.
    """
    out: dict[str, tuple[float, float]] = {}
    for lbl, sub_df in df.groupby("parent_label", sort=False):
        all_ms, all_ml, all_ks, all_kl = [], [], [], []
        for _, sub in sub_df.groupby("parent_id", sort=False):
            x = sub[EMBED_COLS].to_numpy(dtype=float)
            lbl_arr = sub["ref_state"].to_numpy()
            ms = loo_median_scores(x, lbl_arr)
            m = np.isfinite(ms)
            if m.any():
                all_ms.append(ms[m]); all_ml.append(lbl_arr[m])
            ks = loo_knn_scores(x, lbl_arr, k)
            mk = np.isfinite(ks)
            if mk.any():
                all_ks.append(ks[mk]); all_kl.append(lbl_arr[mk])
        if not all_ms:
            out[str(lbl)] = (V3_MED_T, V3_KNN_T)
            continue
        t_m = calibrate_threshold(np.concatenate(all_ms), np.concatenate(all_ml))
        t_k = calibrate_threshold(np.concatenate(all_ks), np.concatenate(all_kl))
        out[str(lbl)] = (t_m, t_k)
    return out


# ---------------------------------------------------------------------------
# Per-parent cache (verbatim from v3)
# ---------------------------------------------------------------------------

class ParentCache:
    __slots__ = ("pid", "label", "emb", "is_g", "is_b",
                 "Dcos", "n", "good_idx", "bad_idx")

    def __init__(self, pid, label, emb, is_g, is_b):
        self.pid = pid
        self.label = label
        self.emb = emb
        self.is_g = is_g
        self.is_b = is_b
        self.n = len(emb)
        self.good_idx = np.where(is_g)[0]
        self.bad_idx = np.where(is_b)[0]
        self.Dcos = cosine_dist_matrix(emb, emb)


def build_parent_caches(df: pd.DataFrame) -> dict[str, ParentCache]:
    caches = {}
    for pid, sub in df.groupby("parent_id", sort=False):
        emb = sub[EMBED_COLS].to_numpy(dtype=float)
        lbl = sub["ref_state"].to_numpy()
        caches[str(pid)] = ParentCache(
            pid=str(pid),
            label=str(sub["parent_label"].iloc[0]),
            emb=emb,
            is_g=(lbl == "good"),
            is_b=(lbl == "bad"),
        )
    return caches


def make_folds(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    fold_ids = np.arange(n) % k
    rng.shuffle(fold_ids)
    return fold_ids


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(cats: np.ndarray, true_label: str) -> dict:
    n = max(len(cats), 1)
    if true_label == "good":
        return dict(
            true_rate=float((cats == "regenerating").sum() / n),
            error_rate=float((cats == "degraded").sum() / n),
            abstain_rate=float((cats == "indistinguishable").sum() / n),
            no_data_rate=float((cats == "no_data").sum() / n),
        )
    return dict(
        true_rate=float((cats == "degraded").sum() / n),
        error_rate=float((cats == "regenerating").sum() / n),
        abstain_rate=float((cats == "indistinguishable").sum() / n),
        no_data_rate=float((cats == "no_data").sum() / n),
    )


def brier(scores: np.ndarray, true_label: str) -> float:
    target = 1.0 if true_label == "good" else 0.0
    v = scores[np.isfinite(scores)]
    return float(np.mean((v - target) ** 2)) if len(v) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--refs",
        default="v4/data/v4_stable_refs_alphaearth.parquet",
    )
    parser.add_argument("--strategy", default="random_100")
    parser.add_argument("--out-dir", default="v4/data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadband-halfwidth", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument(
        "--calibration", choices=["global", "per_fold"], default="global",
        help=("global: calibrate per-class thresholds once via full LOO "
              "(matches v3 default — fast, mild leakage of prior). "
              "per_fold: recalibrate per class within each fold using "
              "training refs only (no leakage; ~K× slower)."),
    )
    args = parser.parse_args()

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
    print(f"Refs rows: {len(df):,}  parents: {df['parent_id'].nunique()}  "
          f"strategy: {args.strategy}")
    print(f"  parent_label split: "
          f"{df.groupby('parent_label')['parent_id'].nunique().to_dict()}")

    print("Building per-parent caches...", flush=True)
    caches = build_parent_caches(df)

    hw, delta, k_knn = args.deadband_halfwidth, args.delta, args.knn_k
    K = args.n_folds
    rng = np.random.default_rng(args.seed)

    # Stratified fold assignment per parent (needed before per_fold calibration).
    fold_assign: dict[str, np.ndarray] = {}
    for pid, c in caches.items():
        fa = np.full(c.n, -1, dtype=np.int8)
        fa[c.good_idx] = make_folds(len(c.good_idx), K, rng)
        fa[c.bad_idx] = make_folds(len(c.bad_idx), K, rng)
        fold_assign[pid] = fa

    # Threshold calibration: global (default, v3 parity) or per_fold (no leakage).
    per_class_thresholds: dict[str, tuple[float, float]] = {}
    fold_class_thresholds: dict[int, dict[str, tuple[float, float]]] = {}

    if args.calibration == "global":
        print("Calibrating thresholds per stable_class (Youden-J on full LOO)...",
              flush=True)
        per_class_thresholds = calibrate_per_stable_class(df, args.knn_k)
        for lbl, (tm, tk) in per_class_thresholds.items():
            print(f"  {lbl}: t_med = {tm:.4f}  t_knn = {tk:.4f}")
    else:
        print(f"Calibrating per-class thresholds per fold "
              f"({K} folds × {df['parent_label'].nunique()} classes)...",
              flush=True)
        for f in range(K):
            train_mask = pd.Series(True, index=df.index)
            train_rows = []
            for pid, c in caches.items():
                fa = fold_assign[pid]
                # Map cache row order back to the original df row indices.
                pid_rows = df.index[df["parent_id"] == pid]
                # Cache rows are in the same order as sub.iterrows on parent_id.
                # Build a boolean mask aligned to those rows.
                keep = (fa != f) & (fa != -1)
                train_rows.extend(pid_rows[keep].tolist())
            train_df = df.loc[train_rows].reset_index(drop=True)
            t_per_class = calibrate_per_stable_class(train_df, args.knn_k)
            fold_class_thresholds[f] = t_per_class
            for lbl, (tm, tk) in t_per_class.items():
                print(f"  fold {f}  {lbl}: t_med = {tm:.4f}  t_knn = {tk:.4f}")
        # Operational thresholds for the scorer = mean across folds (per class).
        all_classes = sorted({lbl for d in fold_class_thresholds.values() for lbl in d})
        for lbl in all_classes:
            tms = [fold_class_thresholds[f][lbl][0]
                   for f in range(K) if lbl in fold_class_thresholds[f]]
            tks = [fold_class_thresholds[f][lbl][1]
                   for f in range(K) if lbl in fold_class_thresholds[f]]
            per_class_thresholds[lbl] = (float(np.mean(tms)), float(np.mean(tks)))
        print("Per-class operational thresholds (mean across folds):")
        for lbl, (tm, tk) in per_class_thresholds.items():
            print(f"  {lbl}: t_med = {tm:.4f}  t_knn = {tk:.4f}")

    # Persist per-class thresholds for the scorer (both t_med and t_knn).
    thresholds_json = out_dir / "calibrated_thresholds_v4.json"
    payload = {
        lbl: {"t_med": float(tm), "t_knn": float(tk)}
        for lbl, (tm, tk) in per_class_thresholds.items()
    }
    with open(thresholds_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Wrote {thresholds_json}")

    print(f"Probing ({len(caches)} parents × {K} folds)...", flush=True)

    parent_id_buf, parent_label_buf, fold_buf = [], [], []
    probe_state_buf, probe_row_buf = [], []
    n_train_good_buf, n_train_bad_buf = [], []
    score_med_chunks, score_knn_chunks = [], []
    t_med_buf, t_knn_buf = [], []
    cat_base_chunks = []
    cat_s1_chunks, cat_s2_chunks, cat_s3_chunks, cat_s4_chunks = [], [], [], []
    cat_s4_v3t_chunks = []   # transfer check using v3 t_knn

    for pi, (pid, c) in enumerate(caches.items()):
        if pi % 20 == 0:
            print(f"  {pi}/{len(caches)}", flush=True)
        fa = fold_assign[pid]

        for f in range(K):
            if args.calibration == "per_fold":
                t_med, t_knn = fold_class_thresholds[f].get(
                    c.label, (V3_MED_T, V3_KNN_T))
            else:
                t_med, t_knn = per_class_thresholds.get(
                    c.label, (V3_MED_T, V3_KNN_T))
            train_g_idx = c.good_idx[fa[c.good_idx] != f]
            train_b_idx = c.bad_idx[fa[c.bad_idx] != f]
            if len(train_g_idx) == 0 or len(train_b_idx) == 0:
                continue

            for probe_state, probe_idx in (
                ("good", c.good_idx[fa[c.good_idx] == f]),
                ("bad",  c.bad_idx[fa[c.bad_idx] == f]),
            ):
                n_p = len(probe_idx)
                if n_p == 0:
                    continue

                Dg = c.Dcos[probe_idx[:, None], train_g_idx]
                Db = c.Dcos[probe_idx[:, None], train_b_idx]

                s_med, lo_med, hi_med = bootstrap_ci_batch(
                    Dg, Db, "median", args.n_boot, rng)
                s_knn, lo_knn, hi_knn = bootstrap_ci_batch(
                    Dg, Db, "knn", args.n_boot, rng, k=k_knn)

                cat_base = classify_batch(s_med, lo_med, hi_med, 0.5,        0.5,        0.0)
                cat_s1   = classify_batch(s_med, lo_med, hi_med, t_med,      t_med,      0.0)
                cat_s2   = classify_batch(s_med, lo_med, hi_med, t_med - hw, t_med + hw, 0.0)
                cat_s3   = classify_batch(s_med, lo_med, hi_med, t_med - hw, t_med + hw, delta)
                cat_s4   = classify_batch(s_knn, lo_knn, hi_knn, t_knn - hw, t_knn + hw, delta)
                cat_s4_v3t = classify_batch(
                    s_knn, lo_knn, hi_knn,
                    V3_KNN_T - hw, V3_KNN_T + hw, delta,
                )

                parent_id_buf.extend([pid] * n_p)
                parent_label_buf.extend([c.label] * n_p)
                fold_buf.extend([f] * n_p)
                probe_state_buf.extend([probe_state] * n_p)
                probe_row_buf.extend(probe_idx.tolist())
                n_train_good_buf.extend([len(train_g_idx)] * n_p)
                n_train_bad_buf.extend([len(train_b_idx)] * n_p)
                score_med_chunks.append(s_med)
                score_knn_chunks.append(s_knn)
                t_med_buf.extend([t_med] * n_p)
                t_knn_buf.extend([t_knn] * n_p)
                cat_base_chunks.append(cat_base)
                cat_s1_chunks.append(cat_s1)
                cat_s2_chunks.append(cat_s2)
                cat_s3_chunks.append(cat_s3)
                cat_s4_chunks.append(cat_s4)
                cat_s4_v3t_chunks.append(cat_s4_v3t)

    site_df = pd.DataFrame({
        "parent_id":     parent_id_buf,
        "parent_label":  parent_label_buf,
        "fold":          fold_buf,
        "probe_state":   probe_state_buf,
        "probe_row":     probe_row_buf,
        "n_train_good":  n_train_good_buf,
        "n_train_bad":   n_train_bad_buf,
        "score_median":  np.concatenate(score_med_chunks),
        "score_knn":     np.concatenate(score_knn_chunks),
        "t_med":         t_med_buf,
        "t_knn":         t_knn_buf,
        "category_baseline":              CAT_NAMES[np.concatenate(cat_base_chunks)],
        "category_step1_calibrated_t":    CAT_NAMES[np.concatenate(cat_s1_chunks)],
        "category_step2_deadband":        CAT_NAMES[np.concatenate(cat_s2_chunks)],
        "category_step3_deadband_effect": CAT_NAMES[np.concatenate(cat_s3_chunks)],
        "category_step4_knn":             CAT_NAMES[np.concatenate(cat_s4_chunks)],
        "category_step4_knn_v3t":         CAT_NAMES[np.concatenate(cat_s4_v3t_chunks)],
    })

    cat_cols = [c for c in site_df.columns if c.startswith("category_")]
    step_labels = [c.replace("category_", "") for c in cat_cols]
    score_col = {c: ("score_knn" if "knn" in c else "score_median") for c in cat_cols}

    summary_rows = []
    stable_labels = sorted(site_df["parent_label"].unique().tolist())
    for probe in ("good", "bad"):
        sub = site_df[site_df["probe_state"] == probe]
        for cat_col, step_lbl in zip(cat_cols, step_labels):
            for lbl in ["all"] + stable_labels:
                grp = sub if lbl == "all" else sub[sub["parent_label"] == lbl]
                if grp.empty:
                    continue
                m = metrics(grp[cat_col].to_numpy(), probe)
                summary_rows.append({
                    "probe_state": probe,
                    "step": step_lbl,
                    "parent_label": lbl,
                    "n": len(grp),
                    **m,
                    "brier": brier(grp[score_col[cat_col]].to_numpy(), probe),
                })
    summary_df = pd.DataFrame(summary_rows)

    site_out = out_dir / "within_parent_site_scores_v4.csv"
    summary_out = out_dir / "within_parent_summary_v4.csv"
    site_df.to_csv(site_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    step_order = [
        "baseline",
        "step1_calibrated_t",
        "step2_deadband",
        "step3_deadband_effect",
        "step4_knn",
        "step4_knn_v3t",
    ]
    print("\n" + "=" * 78)
    print(f"V4 WITHIN-PARENT {K}-FOLD VALIDATION  (per-class threshold refit)")
    print("=" * 78)
    for probe in ("good", "bad"):
        tag = ("GOOD probe (false-degraded)" if probe == "good"
               else "BAD probe  (false-regenerating)")
        print(f"\n{'-'*78}\n  {tag}\n{'-'*78}")
        print(f"  {'step':<33} {'n':>5}  {'true%':>6}  {'error%':>6}  "
              f"{'abstain%':>9}  {'brier':>6}")
        print(f"  {'-'*33}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*6}")
        sub = summary_df[(summary_df["probe_state"] == probe)
                         & (summary_df["parent_label"] == "all")]
        for step in step_order:
            r = sub[sub["step"] == step]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {step:<33} {int(r['n']):>5}  "
                  f"{r['true_rate']:>6.1%}  {r['error_rate']:>6.1%}  "
                  f"{r['abstain_rate']:>9.1%}  {r['brier']:>6.3f}")

    print(f"\n{'-'*78}\n  Per-stable_class error_rate (steps 1-4)\n{'-'*78}")
    print(f"  {'step':<33} {'class':<16} {'probe':>5}  {'error%':>6}")
    for step in step_order[1:]:
        for probe in ("good", "bad"):
            for lbl in stable_labels:
                r = summary_df[(summary_df["step"] == step)
                               & (summary_df["probe_state"] == probe)
                               & (summary_df["parent_label"] == lbl)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"  {step:<33} {lbl:<16} {probe:>5}  {r['error_rate']:>6.1%}")

    print(f"\nWrote: {site_out}")
    print(f"Wrote: {summary_out}")
    print(f"Wrote: {thresholds_json}")


if __name__ == "__main__":
    main()
