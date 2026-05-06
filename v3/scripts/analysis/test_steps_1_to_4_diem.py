"""v3 stepwise scoring using DIEM distances instead of cosine similarity.

DIEM (Dimension Insensitive Euclidean Metric) normalises Euclidean distances
by the expected distance distribution for the embedding dimensionality, making
the scores invariant to the number of bands.

Install the dependency once:
    pip install git+https://github.com/ftessari23/DIEM.git
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

EMBED_COLS = [f"A{i:02d}" for i in range(64)]


# ---------------------------------------------------------------------------
# DIEM helpers
# ---------------------------------------------------------------------------

def _diem_stats(n_dims: int, min_val: float, max_val: float):
    """Return (exp_center, vard) for the given embedding space.

    Wraps DIEM_Stat, which samples 100k random pairs to estimate the baseline
    Euclidean distance distribution for N-dimensional uniform data.
    """
    from diem_functions import DIEM_Stat  # type: ignore[import]

    exp_center, vard, *_ = DIEM_Stat(n_dims, max_val, min_val, fig_flag=0)
    return float(exp_center), float(vard)


def _diem_dist_one_to_many(x: np.ndarray, points: np.ndarray,
                            exp_center: float, vard: float,
                            min_val: float, max_val: float) -> np.ndarray:
    """DIEM distances from a single vector x (shape D,) to each row of points (N×D).

    DIEM expects matrices shaped (features, samples), i.e. column-per-sample.
    Returns an array of length N.
    """
    from diem_functions import getDIEM  # type: ignore[import]

    # x as a (D, 1) matrix; points as (D, N)
    mat1 = x.reshape(-1, 1)
    mat2 = points.T  # (D, N)

    diem_mat, _ = getDIEM(
        mat1, mat2,
        maxV=max_val, minV=min_val,
        exp_center=exp_center, vard=vard,
        Plot="off", Text="off",
    )
    # diem_mat shape is (1, N); return flat array
    return diem_mat.ravel().astype(float)


def _diem_dist_matrix(x: np.ndarray,
                       exp_center: float, vard: float,
                       min_val: float, max_val: float) -> np.ndarray:
    """Symmetric DIEM distance matrix for rows of x (shape N×D).

    Returns an (N, N) array.
    """
    from diem_functions import getDIEM  # type: ignore[import]

    mat = x.T  # (D, N)
    diem_mat, _ = getDIEM(
        mat, mat,
        maxV=max_val, minV=min_val,
        exp_center=exp_center, vard=vard,
        Plot="off", Text="off",
    )
    return np.array(diem_mat, dtype=float)


# ---------------------------------------------------------------------------
# Data loading (unchanged from test_steps_1_to_4.py)
# ---------------------------------------------------------------------------

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
    df = df.dropna(subset=EMBED_COLS + ["parent_id", "ref_state"]).reset_index(drop=True)
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


# ---------------------------------------------------------------------------
# Scoring (DIEM-based)
# ---------------------------------------------------------------------------

def score_median_obs(x: np.ndarray, good: np.ndarray, bad: np.ndarray,
                     exp_center: float, vard: float,
                     min_val: float, max_val: float) -> float:
    d_g = _diem_dist_one_to_many(x, good, exp_center, vard, min_val, max_val)
    d_b = _diem_dist_one_to_many(x, bad, exp_center, vard, min_val, max_val)
    m_g = float(np.median(d_g))
    m_b = float(np.median(d_b))
    return float(m_b / (m_g + m_b + 1e-12))


def score_knn_obs(x: np.ndarray, good: np.ndarray, bad: np.ndarray,
                  k: int,
                  exp_center: float, vard: float,
                  min_val: float, max_val: float) -> float:
    if len(good) < k or len(bad) < k:
        return float("nan")
    d_g = _diem_dist_one_to_many(x, good, exp_center, vard, min_val, max_val)
    d_b = _diem_dist_one_to_many(x, bad, exp_center, vard, min_val, max_val)
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


def loo_median_scores(x: np.ndarray, labels: np.ndarray,
                      exp_center: float, vard: float,
                      min_val: float, max_val: float) -> np.ndarray:
    """Leave-one-out median DIEM score for each reference point."""
    is_good = labels == "good"
    is_bad = labels == "bad"
    if is_good.sum() < 2 or is_bad.sum() < 2:
        return np.full(len(x), np.nan)

    # Full (N×N) DIEM distance matrix — avoids repeated getDIEM calls
    dmat = _diem_dist_matrix(x, exp_center, vard, min_val, max_val)

    good_rows = np.where(is_good)[0]
    bad_rows = np.where(is_bad)[0]

    d_g = dmat[:, good_rows].astype(float)
    d_b = dmat[:, bad_rows].astype(float)

    # Mask self-distances
    for local_i, global_i in enumerate(good_rows):
        d_g[global_i, local_i] = np.nan
    for local_i, global_i in enumerate(bad_rows):
        d_b[global_i, local_i] = np.nan

    eps = 1e-12
    m_g = np.nanmedian(d_g, axis=1) + eps
    m_b = np.nanmedian(d_b, axis=1) + eps
    return m_b / (m_g + m_b)


def loo_knn_scores(x: np.ndarray, labels: np.ndarray, k: int,
                   exp_center: float, vard: float,
                   min_val: float, max_val: float) -> np.ndarray:
    is_good = labels == "good"
    is_bad = labels == "bad"
    n_good = int(is_good.sum())
    n_bad = int(is_bad.sum())
    if n_good < k + 1 or n_bad < k + 1:
        return np.full(len(x), np.nan)

    dmat = _diem_dist_matrix(x, exp_center, vard, min_val, max_val)
    out = np.empty(len(x), dtype=float)

    idx_good = np.where(is_good)[0]
    idx_bad = np.where(is_bad)[0]

    for i in range(len(x)):
        d_to_good = dmat[i, idx_good].copy()
        d_to_bad = dmat[i, idx_bad].copy()

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


# ---------------------------------------------------------------------------
# Calibration and classification (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v3 stepwise tests using DIEM distances (replaces cosine similarity)."
    )
    parser.add_argument(
        "--refs",
        default="v2/data/v2real_mask_on_corr300_exhaustive/sampling_strategy_selected_points.parquet",
    )
    parser.add_argument(
        "--test-sites",
        default="degreeRecover/data/test_site_alphaearth_2024.parquet",
    )
    parser.add_argument("--strategy", default="random_100")
    parser.add_argument("--out-dir", default="v3/data")
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadband-halfwidth", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--knn-k", type=int, default=5)
    # DIEM data-range parameters; for AlphaEarth embeddings these are
    # typically 0-1 after normalisation but can be overridden.
    parser.add_argument("--emb-min", type=float, default=0.0,
                        help="Minimum value in embedding space (used by DIEM_Stat).")
    parser.add_argument("--emb-max", type=float, default=1.0,
                        help="Maximum value in embedding space (used by DIEM_Stat).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_refs(args.refs, args.strategy)
    test_emb = load_test_site_embeddings(args.test_sites)

    n_dims = len(EMBED_COLS)
    print(f"Loaded refs rows: {len(refs):,}")
    print(f"Parents in refs: {refs['parent_id'].nunique()}")
    print(f"Parents with test embeddings: {len(test_emb)}")
    print(f"Computing DIEM stats for N={n_dims}, range=[{args.emb_min}, {args.emb_max}] ...")

    exp_center, vard = _diem_stats(n_dims, args.emb_min, args.emb_max)
    print(f"  exp_center={exp_center:.6f}  vard={vard:.6f}")

    # Shared kwargs forwarded to every DIEM call
    diem_kw = dict(exp_center=exp_center, vard=vard,
                   min_val=args.emb_min, max_val=args.emb_max)

    # ------------------------------------------------------------------
    # LOO calibration pass
    # ------------------------------------------------------------------
    loo_med_scores: list[np.ndarray] = []
    loo_med_labels: list[np.ndarray] = []
    loo_knn_scores_all: list[np.ndarray] = []
    loo_knn_labels: list[np.ndarray] = []

    for _, sub in refs.groupby("parent_id", sort=False):
        x = sub[EMBED_COLS].to_numpy(dtype=float)
        labels = sub["ref_state"].to_numpy()

        med = loo_median_scores(x, labels, **diem_kw)
        m = np.isfinite(med)
        if np.any(m):
            loo_med_scores.append(med[m])
            loo_med_labels.append(labels[m])

        knn = loo_knn_scores(x, labels, args.knn_k, **diem_kw)
        mk = np.isfinite(knn)
        if np.any(mk):
            loo_knn_scores_all.append(knn[mk])
            loo_knn_labels.append(labels[mk])

    med_scores = np.concatenate(loo_med_scores) if loo_med_scores else np.array([0.5])
    med_labels = np.concatenate(loo_med_labels) if loo_med_labels else np.array(["good"])
    knn_scores = np.concatenate(loo_knn_scores_all) if loo_knn_scores_all else np.array([0.5])
    knn_labels = np.concatenate(loo_knn_labels) if loo_knn_labels else np.array(["good"])

    t_med = calibrate_threshold(med_scores, med_labels)
    t_knn = calibrate_threshold(knn_scores, knn_labels)

    print(f"Calibrated threshold (diem_median): {t_med:.4f}")
    print(f"Calibrated threshold (diem_knn):    {t_knn:.4f}")

    # ------------------------------------------------------------------
    # Per-site scoring
    # ------------------------------------------------------------------
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

        def _med(x, g, b):
            return score_median_obs(x, g, b, **diem_kw)

        def _knn(x, g, b):
            return score_knn_obs(x, g, b, args.knn_k, **diem_kw)

        s_med, lo_med, hi_med = bootstrap_ci(_med, x_obs, good, bad, args.n_boot, rng)
        s_knn, lo_knn, hi_knn = bootstrap_ci(_knn, x_obs, good, bad, args.n_boot, rng)

        row: dict[str, object] = {
            "parent_id": pid,
            "parent_label": parent_label,
            "score_diem_median": s_med,
            "ci_lo_diem_median": lo_med,
            "ci_hi_diem_median": hi_med,
            "score_diem_knn": s_knn,
            "ci_lo_diem_knn": lo_knn,
            "ci_hi_diem_knn": hi_knn,
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
            t_med, t_med, 0.0,
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

    site_out = out_dir / "step_1_to_4_diem_site_scores.csv"
    summary_out_path = out_dir / "step_1_to_4_diem_summary.csv"
    site_df.to_csv(site_out, index=False)
    summary_df.to_csv(summary_out_path, index=False)

    print("\nStep summary (DIEM):")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote: {site_out}")
    print(f"Wrote: {summary_out_path}")


if __name__ == "__main__":
    main()
