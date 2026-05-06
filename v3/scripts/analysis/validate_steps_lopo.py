"""Leave-one-parent-out (LOPO) validation of steps 1-4 decision rules.

Validation design
-----------------
For each parent p (158 total):
  x_obs_good = mean of p's own good-state embeddings  (known recovering)
  x_obs_bad  = mean of p's own bad-state embeddings   (known degraded)
  references = all OTHER parents' embeddings (157 parents × 200 pts each)
  Score each x_obs → category under each step's rule.

Ground truth:
  probe "good" → correct = NOT "degraded"   (false alarm = "degraded")
  probe "bad"  → correct = NOT "recovering" (miss = "recovering")

Thresholds calibrated once via full LOO across all 158 parents (held fixed).
Bootstrap vectorised: distances precomputed once per parent, resampling is
index-based and runs in microseconds.

Metrics:
  true_rate    fraction correctly labelled
  error_rate   fraction of hard mis-classifications
  abstain_rate fraction called "indistinguishable"
  brier        MSE(score, truth);  0=perfect, 0.25=random
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

EMBED_COLS = [f"A{i:02d}" for i in range(64)]


# ---------------------------------------------------------------------------
# Scoring — all vectorised over precomputed distance vectors
# ---------------------------------------------------------------------------

def cosine_dist_vec(x: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """1-D cosine distances from x to every row of pts."""
    nx = np.linalg.norm(x) + 1e-12
    npts = np.linalg.norm(pts, axis=1) + 1e-12
    return 1.0 - (pts @ x) / (npts * nx)


def score_median_from_dists(d_g: np.ndarray, d_b: np.ndarray) -> float:
    m_g = float(np.median(d_g))
    m_b = float(np.median(d_b))
    return m_b / (m_g + m_b + 1e-12)


def score_knn_from_dists(d_g: np.ndarray, d_b: np.ndarray, k: int) -> float:
    if len(d_g) < k or len(d_b) < k:
        return float("nan")
    mg = float(np.mean(np.partition(d_g, k - 1)[:k]))
    mb = float(np.mean(np.partition(d_b, k - 1)[:k]))
    return mb / (mg + mb + 1e-12)


def bootstrap_ci_from_dists(
    d_g: np.ndarray,
    d_b: np.ndarray,
    score_fn,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap CI by resampling precomputed distance vectors (no re-dot)."""
    point = score_fn(d_g, d_b)
    n_g, n_b = len(d_g), len(d_b)
    ig = rng.integers(0, n_g, size=(n_boot, n_g))
    ib = rng.integers(0, n_b, size=(n_boot, n_b))
    boots = np.array([score_fn(d_g[ig[i]], d_b[ib[i]]) for i in range(n_boot)])
    valid = boots[np.isfinite(boots)]
    if len(valid) < 10:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


# ---------------------------------------------------------------------------
# Threshold calibration (full LOO, called once)
# ---------------------------------------------------------------------------

def loo_median_scores(emb: np.ndarray, lbl: np.ndarray) -> np.ndarray:
    is_g, is_b = lbl == "good", lbl == "bad"
    if is_g.sum() < 2 or is_b.sum() < 2:
        return np.full(len(emb), np.nan)
    eps = 1e-12
    nx = np.linalg.norm(emb, axis=1, keepdims=True) + eps
    eg, eb = emb[is_g], emb[is_b]
    ng = np.linalg.norm(eg, axis=1, keepdims=True) + eps
    nb = np.linalg.norm(eb, axis=1, keepdims=True) + eps
    D_g = 1.0 - (emb @ eg.T) / (nx * ng.T)
    D_b = 1.0 - (emb @ eb.T) / (nx * nb.T)
    D_g[np.where(is_g)[0], np.arange(is_g.sum())] = np.nan
    D_b[np.where(is_b)[0], np.arange(is_b.sum())] = np.nan
    m_g = np.nanmedian(D_g, axis=1) + eps
    m_b = np.nanmedian(D_b, axis=1) + eps
    return m_b / (m_g + m_b)


def loo_knn_scores(emb: np.ndarray, lbl: np.ndarray, k: int) -> np.ndarray:
    is_g, is_b = lbl == "good", lbl == "bad"
    if is_g.sum() < k + 1 or is_b.sum() < k + 1:
        return np.full(len(emb), np.nan)
    dmat = np.linalg.norm(emb[:, None] - emb[None, :], axis=2)
    idx_g, idx_b = np.where(is_g)[0], np.where(is_b)[0]
    out = np.empty(len(emb))
    for i in range(len(emb)):
        dg = dmat[i, idx_g].copy()
        db = dmat[i, idx_b].copy()
        if is_g[i]:
            dg = np.delete(dg, np.where(idx_g == i)[0][0])
        if is_b[i]:
            db = np.delete(db, np.where(idx_b == i)[0][0])
        if len(dg) < k or len(db) < k:
            out[i] = np.nan
        else:
            out[i] = (np.mean(np.partition(db, k-1)[:k])
                      / (np.mean(np.partition(dg, k-1)[:k])
                         + np.mean(np.partition(db, k-1)[:k]) + 1e-12))
    return out


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
        j = tp / (tp + fn + 1e-12) - fp / (fp + tn + 1e-12)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def classify(score, lo, hi, t_lo, t_hi, delta) -> str:
    if not all(np.isfinite(v) for v in [score, lo, hi]):
        return "no_data"
    if lo > t_hi and (score - t_hi) >= delta:
        return "recovering"
    if hi < t_lo and (t_lo - score) >= delta:
        return "degraded"
    return "indistinguishable"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def metrics(cats: list[str], true_label: str) -> dict:
    c = np.array(cats)
    n = max(len(c), 1)
    if true_label == "good":
        return dict(true_rate=(c == "recovering").sum() / n,
                    error_rate=(c == "degraded").sum() / n,
                    abstain_rate=(c == "indistinguishable").sum() / n)
    return dict(true_rate=(c == "degraded").sum() / n,
                error_rate=(c == "recovering").sum() / n,
                abstain_rate=(c == "indistinguishable").sum() / n)


def brier(scores: np.ndarray, true_label: str) -> float:
    target = 1.0 if true_label == "good" else 0.0
    v = scores[np.isfinite(scores)]
    return float(np.mean((v - target) ** 2)) if len(v) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", default="v2/data/v2real_mask_on_corr300_exhaustive/sampling_strategy_selected_points.parquet")
    parser.add_argument("--strategy", default="random_100")
    parser.add_argument("--out-dir", default="v3/data")
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadband-halfwidth", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--knn-k", type=int, default=5)
    args = parser.parse_args()

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

    parents = df["parent_id"].unique()
    print(f"Parents: {len(parents)}, strategy: {args.strategy}")

    # ------------------------------------------------------------------
    # Calibrate thresholds once (full LOO)
    # ------------------------------------------------------------------
    print("Calibrating thresholds (full LOO)...", flush=True)
    all_ms, all_ml, all_ks, all_kl = [], [], [], []
    for _, sub in df.groupby("parent_id", sort=False):
        x = sub[EMBED_COLS].to_numpy(dtype=float)
        lbl = sub["ref_state"].to_numpy()
        ms = loo_median_scores(x, lbl)
        m = np.isfinite(ms)
        if m.any():
            all_ms.append(ms[m]); all_ml.append(lbl[m])
        ks = loo_knn_scores(x, lbl, args.knn_k)
        mk = np.isfinite(ks)
        if mk.any():
            all_ks.append(ks[mk]); all_kl.append(lbl[mk])

    t_med = calibrate_threshold(np.concatenate(all_ms), np.concatenate(all_ml))
    t_knn = calibrate_threshold(np.concatenate(all_ks), np.concatenate(all_kl))
    print(f"  t_med = {t_med:.4f}  t_knn = {t_knn:.4f}")

    hw, d, k = args.deadband_halfwidth, args.delta, args.knn_k

    # ------------------------------------------------------------------
    # LOPO scoring loop
    # ------------------------------------------------------------------
    print(f"LOPO scoring ({len(parents)} parents × 2 probes)...", flush=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    for i, pid in enumerate(parents):
        if i % 20 == 0:
            print(f"  {i}/{len(parents)}", flush=True)

        hold = df[df["parent_id"] == pid]
        rest = df[df["parent_id"] != pid]
        parent_label = str(hold["parent_label"].iloc[0])

        # Precompute reference point arrays once per parent
        rest_good = rest.loc[rest["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
        rest_bad  = rest.loc[rest["ref_state"] == "bad",  EMBED_COLS].to_numpy(dtype=float)

        for probe_state in ("good", "bad"):
            x_obs = hold.loc[hold["ref_state"] == probe_state, EMBED_COLS].to_numpy(dtype=float).mean(axis=0)

            # Precompute distances once — bootstrap resamples these vectors
            d_g_cos = cosine_dist_vec(x_obs, rest_good)
            d_b_cos = cosine_dist_vec(x_obs, rest_bad)
            d_g_euc = np.linalg.norm(rest_good - x_obs, axis=1)
            d_b_euc = np.linalg.norm(rest_bad  - x_obs, axis=1)

            def _med(dg, db): return score_median_from_dists(dg, db)
            def _knn(dg, db): return score_knn_from_dists(dg, db, k)

            s_med, lo_med, hi_med = bootstrap_ci_from_dists(d_g_cos, d_b_cos, _med, args.n_boot, rng)
            s_knn, lo_knn, hi_knn = bootstrap_ci_from_dists(d_g_euc, d_b_euc, _knn, args.n_boot, rng)

            rows.append({
                "parent_id":      pid,
                "parent_label":   parent_label,
                "probe_state":    probe_state,
                "score_median":   s_med,
                "score_knn":      s_knn,
                "t_med":          t_med,
                "t_knn":          t_knn,
                "category_baseline":              classify(s_med, lo_med, hi_med, 0.5,        0.5,        0.0),
                "category_step1_calibrated_t":    classify(s_med, lo_med, hi_med, t_med,      t_med,      0.0),
                "category_step2_deadband":        classify(s_med, lo_med, hi_med, t_med - hw, t_med + hw, 0.0),
                "category_step3_deadband_effect": classify(s_med, lo_med, hi_med, t_med - hw, t_med + hw, d),
                "category_step4_knn":             classify(s_knn, lo_knn, hi_knn, t_knn - hw, t_knn + hw, d),
            })

    site_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summarise metrics per step × probe × label
    # ------------------------------------------------------------------
    cat_cols = [c for c in site_df.columns if c.startswith("category_")]
    step_labels = [c.replace("category_", "") for c in cat_cols]
    score_col = {c: ("score_knn" if "knn" in c else "score_median") for c in cat_cols}

    summary_rows = []
    for probe in ("good", "bad"):
        sub = site_df[site_df["probe_state"] == probe]
        for cat_col, step_lbl in zip(cat_cols, step_labels):
            for lbl in ["all", "built_loss", "crop_loss"]:
                grp = sub if lbl == "all" else sub[sub["parent_label"] == lbl]
                if grp.empty:
                    continue
                m = metrics(grp[cat_col].tolist(), probe)
                summary_rows.append({
                    "probe_state":  probe,
                    "step":         step_lbl,
                    "parent_label": lbl,
                    "n":            len(grp),
                    **m,
                    "brier":        brier(grp[score_col[cat_col]].to_numpy(), probe),
                })

    summary_df = pd.DataFrame(summary_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    site_df.to_csv(out_dir / "lopo_site_scores.csv", index=False)
    summary_df.to_csv(out_dir / "lopo_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Console report  (steps 1 & 3 marked ◄, then 2 & 4)
    # ------------------------------------------------------------------
    step_order = [
        "baseline_t0.5",
        "step1_calibrated_t",      # test 1
        "step2_deadband",          # test 2
        "step3_deadband_effect",   # test 3 (adds effect-size gate to step2)
        "step4_knn",               # test 4
    ]

    print("\n" + "=" * 75)
    print("LOPO VALIDATION REPORT")
    print("=" * 75)
    print(
        "probe=good: x_obs known-GOOD  |  error = wrongly called 'degraded'\n"
        "probe=bad:  x_obs known-BAD   |  error = wrongly called 'recovering'\n"
        "Steps 1 & 3 use median scorer.  Steps 2 & 4 add deadband / kNN.\n"
        "brier: 0=perfect  0.25=random\n"
    )

    for probe in ("good", "bad"):
        tag = "GOOD probe (false-degraded)" if probe == "good" else "BAD probe  (false-recovering)"
        print(f"\n{'─'*75}")
        print(f"  {tag}")
        print(f"{'─'*75}")
        print(f"  {'step':<33} {'n':>4}  {'true%':>6}  {'error%':>6}  {'abstain%':>9}  {'brier':>6}")
        print(f"  {'─'*33}  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*6}")
        sub = summary_df[(summary_df["probe_state"] == probe) & (summary_df["parent_label"] == "all")]
        for step in step_order:
            r = sub[sub["step"] == step]
            if r.empty:
                continue
            r = r.iloc[0]
            tag2 = " ◄" if step in ("step1_calibrated_t", "step3_deadband_effect") else (
                   " ◄◄" if step in ("step2_deadband", "step4_knn") else "")
            print(
                f"  {step:<33} {int(r['n']):>4}  "
                f"{r['true_rate']:>6.1%}  {r['error_rate']:>6.1%}  "
                f"{r['abstain_rate']:>9.1%}  {r['brier']:>6.3f}{tag2}"
            )

    # Per-label breakdown for steps 1–4
    print(f"\n{'─'*75}")
    print("  Per-label breakdown (error_rate only)")
    print(f"  {'step':<33} {'label':<12} {'probe':>5}  {'error%':>6}")
    print(f"  {'─'*33}  {'─'*12}  {'─'*5}  {'─'*6}")
    for step in step_order[1:]:
        for probe in ("good", "bad"):
            for lbl in ("built_loss", "crop_loss"):
                r = summary_df[
                    (summary_df["step"] == step) &
                    (summary_df["probe_state"] == probe) &
                    (summary_df["parent_label"] == lbl)
                ]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"  {step:<33} {lbl:<12} {probe:>5}  {r['error_rate']:>6.1%}")

    print(f"\nWrote: {out_dir}/lopo_site_scores.csv")
    print(f"Wrote: {out_dir}/lopo_summary.csv")


if __name__ == "__main__":
    main()
