from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

EMBED_COLS = [f"A{i:02d}" for i in range(64)]


def load_refs(parquet_path: str, strategy: str | None) -> pd.DataFrame:
    con = duckdb.connect()
    cols = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [parquet_path]).df()
    all_cols = set(cols["column_name"].tolist())

    needed = ["parent_id", "parent_label", "ref_state"] + EMBED_COLS
    if "strategy" in all_cols:
        needed.append("strategy")

    for c in needed:
        if c not in all_cols:
            raise ValueError(f"Missing required column: {c}")

    select_sql = ", ".join(needed)
    query = f"SELECT {select_sql} FROM read_parquet(?)"
    params: list[object] = [parquet_path]
    if strategy and "strategy" in all_cols:
        query += " WHERE strategy = ?"
        params.append(strategy)

    df = con.execute(query, params).df()
    con.close()

    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])].copy()
    df = df.dropna(subset=EMBED_COLS + ["parent_id", "ref_state"]).reset_index(
        drop=True
    )
    return df


def load_test_site_embeddings(parquet_path: str) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    cols = ", ".join(["parent_id"] + EMBED_COLS)
    df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [parquet_path]).df()
    con.close()

    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=["parent_id"] + EMBED_COLS).reset_index(drop=True)

    out: dict[str, np.ndarray] = {}
    for pid, g in df.groupby("parent_id", sort=False):
        out[pid] = g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
    return out


def cosine_dists_to_set(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    nx = np.linalg.norm(x) + 1e-12
    np_pts = np.linalg.norm(points, axis=1) + 1e-12
    sims = (points @ x) / (np_pts * nx)
    return 1.0 - sims


def score_median_obs(x: np.ndarray, good: np.ndarray, bad: np.ndarray) -> float:
    d_g = cosine_dists_to_set(x, good)
    d_b = cosine_dists_to_set(x, bad)
    m_g = float(np.median(d_g))
    m_b = float(np.median(d_b))
    return float(m_b / (m_g + m_b + 1e-12))


def score_knn_obs(x: np.ndarray, good: np.ndarray, bad: np.ndarray, k: int) -> float:
    if len(good) < k or len(bad) < k:
        return float("nan")
    d_g = cosine_dists_to_set(x, good)
    d_b = cosine_dists_to_set(x, bad)
    m_g = float(np.mean(np.partition(d_g, k - 1)[:k]))
    m_b = float(np.mean(np.partition(d_b, k - 1)[:k]))
    return float(m_b / (m_g + m_b + 1e-12))


def bootstrap_ci(
    score_fn,
    x: np.ndarray,
    good: np.ndarray,
    bad: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    point = float(score_fn(x, good, bad))
    n_g = len(good)
    n_b = len(bad)
    if n_g == 0 or n_b == 0:
        return float("nan"), float("nan"), float("nan")

    ig = rng.integers(0, n_g, size=(n_boot, n_g))
    ib = rng.integers(0, n_b, size=(n_boot, n_b))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = score_fn(x, good[ig[i]], bad[ib[i]])

    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return point, lo, hi


def loo_median_scores(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    is_good = labels == "good"
    is_bad = labels == "bad"
    if is_good.sum() < 2 or is_bad.sum() < 2:
        return np.full(len(x), np.nan)

    x_good = x[is_good]
    x_bad = x[is_bad]
    eps = 1e-12
    nx = np.linalg.norm(x, axis=1) + eps
    ng = np.linalg.norm(x_good, axis=1) + eps
    nb = np.linalg.norm(x_bad, axis=1) + eps

    d_g = 1.0 - (x @ x_good.T) / np.outer(nx, ng)
    d_b = 1.0 - (x @ x_bad.T) / np.outer(nx, nb)

    good_rows = np.where(is_good)[0]
    bad_rows = np.where(is_bad)[0]
    d_g[good_rows, np.arange(len(good_rows))] = np.nan
    d_b[bad_rows, np.arange(len(bad_rows))] = np.nan

    m_g = np.nanmedian(d_g, axis=1) + eps
    m_b = np.nanmedian(d_b, axis=1) + eps
    return m_b / (m_g + m_b)


def loo_knn_scores(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    is_good = labels == "good"
    is_bad = labels == "bad"
    n_good = int(is_good.sum())
    n_bad = int(is_bad.sum())
    if n_good < k + 1 or n_bad < k + 1:
        return np.full(len(x), np.nan)

    nx = np.linalg.norm(x, axis=1) + 1e-12
    sims = (x @ x.T) / np.outer(nx, nx)
    dmat = 1.0 - sims
    out = np.empty(len(x), dtype=float)

    idx_good = np.where(is_good)[0]
    idx_bad = np.where(is_bad)[0]

    for i in range(len(x)):
        d_to_good = dmat[i, idx_good]
        d_to_bad = dmat[i, idx_bad]

        if is_good[i]:
            self_pos = int(np.where(idx_good == i)[0][0])
            d_to_good = np.delete(d_to_good, self_pos)
        if is_bad[i]:
            self_pos = int(np.where(idx_bad == i)[0][0])
            d_to_bad = np.delete(d_to_bad, self_pos)

        if len(d_to_good) < k or len(d_to_bad) < k:
            out[i] = np.nan
            continue

        m_g = float(np.mean(np.partition(d_to_good, k - 1)[:k]))
        m_b = float(np.mean(np.partition(d_to_bad, k - 1)[:k]))
        out[i] = m_b / (m_g + m_b + 1e-12)

    return out


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    m = np.isfinite(scores)
    y = (labels[m] == "good").astype(int)
    s = scores[m]
    if len(np.unique(y)) < 2:
        return 0.5

    thresholds = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    best_t = 0.5
    best_j = -1.0
    for t in thresholds:
        pred = s >= t
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def classify(
    score: float, ci_lo: float, ci_hi: float, t_low: float, t_high: float, delta: float
) -> str:
    if not np.isfinite(score) or not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
        return "no_data"
    if ci_lo > t_high and (score - t_high) >= delta:
        return "recovering"
    if ci_hi < t_low and (t_low - score) >= delta:
        return "degraded"
    return "indistinguishable"


def summarize(
    step: str, cats: pd.Series, t_low: float, t_high: float, delta: float
) -> dict[str, object]:
    counts = cats.value_counts(dropna=False).to_dict()
    total = int(cats.notna().sum())
    degraded = int(counts.get("degraded", 0))
    return {
        "step": step,
        "threshold_low": t_low,
        "threshold_high": t_high,
        "delta": delta,
        "recovering": int(counts.get("recovering", 0)),
        "indistinguishable": int(counts.get("indistinguishable", 0)),
        "degraded": degraded,
        "no_data": int(counts.get("no_data", 0)),
        "total": total,
        "degraded_pct": degraded / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v3 stepwise tests for 1-4 improvements."
    )
    parser.add_argument(
        "--refs",
        default="v2/data/v2real_mask_on_corr300_exhaustive/sampling_strategy_selected_points.parquet",
    )
    parser.add_argument(
        "--test-sites",
        default="v1/data/test_site_alphaearth_2024.parquet",
    )
    parser.add_argument("--strategy", default="random_100")
    parser.add_argument("--out-dir", default="v3/data")
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadband-halfwidth", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--knn-k", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_refs(args.refs, args.strategy)
    test_emb = load_test_site_embeddings(args.test_sites)

    print(f"Loaded refs rows: {len(refs):,}")
    print(f"Parents in refs: {refs['parent_id'].nunique()}")
    print(f"Parents with test embeddings: {len(test_emb)}")

    loo_med_scores: list[np.ndarray] = []
    loo_med_labels: list[np.ndarray] = []
    loo_knn_scores_all: list[np.ndarray] = []
    loo_knn_labels: list[np.ndarray] = []

    for _, sub in refs.groupby("parent_id", sort=False):
        x = sub[EMBED_COLS].to_numpy(dtype=float)
        labels = sub["ref_state"].to_numpy()

        med = loo_median_scores(x, labels)
        m = np.isfinite(med)
        if np.any(m):
            loo_med_scores.append(med[m])
            loo_med_labels.append(labels[m])

        knn = loo_knn_scores(x, labels, args.knn_k)
        mk = np.isfinite(knn)
        if np.any(mk):
            loo_knn_scores_all.append(knn[mk])
            loo_knn_labels.append(labels[mk])

    med_scores = np.concatenate(loo_med_scores) if loo_med_scores else np.array([0.5])
    med_labels = (
        np.concatenate(loo_med_labels) if loo_med_labels else np.array(["good"])
    )
    knn_scores = (
        np.concatenate(loo_knn_scores_all) if loo_knn_scores_all else np.array([0.5])
    )
    knn_labels = (
        np.concatenate(loo_knn_labels) if loo_knn_labels else np.array(["good"])
    )

    t_med = calibrate_threshold(med_scores, med_labels)
    t_knn = calibrate_threshold(knn_scores, knn_labels)

    print(f"Calibrated threshold (median_pairwise): {t_med:.4f}")
    print(f"Calibrated threshold (knn_margin): {t_knn:.4f}")

    rng = np.random.default_rng(args.seed)
    site_rows: list[dict[str, object]] = []

    for pid, sub in refs.groupby("parent_id", sort=False):
        if pid not in test_emb:
            continue

        good = sub.loc[sub["ref_state"] == "good", EMBED_COLS].to_numpy(dtype=float)
        bad = sub.loc[sub["ref_state"] == "bad", EMBED_COLS].to_numpy(dtype=float)
        if len(good) == 0 or len(bad) == 0:
            continue

        x_obs = test_emb[pid]
        parent_label = str(sub["parent_label"].iloc[0])

        s_med, lo_med, hi_med = bootstrap_ci(
            score_median_obs, x_obs, good, bad, args.n_boot, rng
        )

        def _knn_score(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
            return score_knn_obs(x, g, b, args.knn_k)

        s_knn, lo_knn, hi_knn = bootstrap_ci(
            _knn_score, x_obs, good, bad, args.n_boot, rng
        )

        row = {
            "parent_id": pid,
            "parent_label": parent_label,
            "score_median": s_med,
            "ci_lo_median": lo_med,
            "ci_hi_median": hi_med,
            "score_knn": s_knn,
            "ci_lo_knn": lo_knn,
            "ci_hi_knn": hi_knn,
        }

        row["category_baseline"] = classify(s_med, lo_med, hi_med, 0.5, 0.5, 0.0)
        row["category_step1_calibrated_t"] = classify(
            s_med, lo_med, hi_med, t_med, t_med, 0.0
        )

        t2_low = t_med - args.deadband_halfwidth
        t2_high = t_med + args.deadband_halfwidth
        row["category_step2_deadband"] = classify(
            s_med, lo_med, hi_med, t2_low, t2_high, 0.0
        )

        row["category_step3_deadband_effect"] = classify(
            s_med, lo_med, hi_med, t2_low, t2_high, args.delta
        )

        t4_low = t_knn - args.deadband_halfwidth
        t4_high = t_knn + args.deadband_halfwidth
        row["category_step4_knn"] = classify(
            s_knn, lo_knn, hi_knn, t4_low, t4_high, args.delta
        )

        site_rows.append(row)

    site_df = pd.DataFrame(site_rows).sort_values("parent_id").reset_index(drop=True)
    if site_df.empty:
        raise RuntimeError("No sites were scored. Check inputs and parent_id overlap.")

    summary_rows = [
        summarize("baseline_v2_t0.5", site_df["category_baseline"], 0.5, 0.5, 0.0),
        summarize(
            "step1_calibrated_threshold",
            site_df["category_step1_calibrated_t"],
            t_med,
            t_med,
            0.0,
        ),
        summarize(
            "step2_deadband",
            site_df["category_step2_deadband"],
            t_med - args.deadband_halfwidth,
            t_med + args.deadband_halfwidth,
            0.0,
        ),
        summarize(
            "step3_deadband_plus_effect_size",
            site_df["category_step3_deadband_effect"],
            t_med - args.deadband_halfwidth,
            t_med + args.deadband_halfwidth,
            args.delta,
        ),
        summarize(
            "step4_knn_primary",
            site_df["category_step4_knn"],
            t_knn - args.deadband_halfwidth,
            t_knn + args.deadband_halfwidth,
            args.delta,
        ),
    ]

    summary_df = pd.DataFrame(summary_rows)
    baseline_deg = int(
        summary_df.loc[summary_df["step"] == "baseline_v2_t0.5", "degraded"].iloc[0]
    )
    summary_df["degraded_change_vs_baseline"] = summary_df["degraded"] - baseline_deg

    site_out = out_dir / "step_1_to_4_site_scores.csv"
    summary_out = out_dir / "step_1_to_4_summary.csv"
    site_df.to_csv(site_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print("\nStep summary:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote: {site_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
