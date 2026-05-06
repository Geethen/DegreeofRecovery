"""Within-parent K-fold validation of steps 1-4 decision rules.

Validation design
-----------------
This is the structurally-correct counterpart to validate_steps_lopo.py.
The original framework (test_steps_1_to_4.py) scores each test site against
ONLY its own parent's good+bad references (200 pts). LOPO violated this by
scoring against all *other* parents' refs (~31k pts), inflating CIs and
forcing 100% abstention at steps 2-4. This script preserves the per-parent
reference design while still producing held-out probes.

For each parent p:
  Split p's good refs into K folds, same for bad.
  For each fold f:
    probe set    = p's fold-f good ∪ fold-f bad   (held-out individuals)
    reference set = p's other-fold good ∪ other-fold bad
  Score each probe individually against the retained refs of the same parent.

Ground truth (assumes near-perfect labels; some label noise possible):
  probe "good" → correct = NOT "degraded"   (false alarm = "degraded")
  probe "bad"  → correct = NOT "recovering" (miss = "recovering")

Thresholds:
  --calibration {global, per_fold}
    global   : calibrate t_med, t_knn once via full LOO across all parents
               (matches test_steps_1_to_4.py — fast, mild leakage of prior).
    per_fold : recalibrate per fold using *only* the current fold's training
               refs from all parents (no leakage; ~K× slower).
  Default: global.

Metric
------
Steps 1-4 all use cosine distance (unified). The prior Euclidean kNN (step 4)
has been replaced for coherence; the single Dcos matrix per parent is reused
for both median (steps 1-3) and kNN (step 4) scoring.

Speed design
------------
- Per-parent cosine distance matrix (200×200) built once with
  scipy.spatial.distance.cdist; fold scoring slices rows/cols by index.
- Bootstrap CIs fully vectorised in 3D: build (n_probes, n_boot, n_refs)
  resampled tensor and reduce along the refs axis directly — no Python
  bootstrap loop, no transpose/flatten round-trip.
- Quantiles via in-place sort + index lookup (no nanpercentile fallback).
- Categories stored as int8 codes during the loop; decoded to strings only
  at DataFrame construction.
- DataFrame built once at the end from column buffers (vs. 31k row-dicts).
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


# ---------------------------------------------------------------------------
# Vectorised scoring primitives
# ---------------------------------------------------------------------------

def cosine_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances |A| × |B| via scipy (no memory explosion)."""
    return cdist(A, B, metric="cosine")



# ---------------------------------------------------------------------------
# Vectorised bootstrap CI over a (n_probes, n_refs) distance matrix
# ---------------------------------------------------------------------------

def _median_3d(D_boot: np.ndarray) -> np.ndarray:
    """Median along last axis of a 3D (n_probes, n_boot, n_refs) array."""
    return np.median(D_boot, axis=2)


def _knn_mean_3d(D_boot: np.ndarray, k: int) -> np.ndarray:
    """Mean of k smallest along last axis of (n_probes, n_boot, n_refs)."""
    return np.mean(np.partition(D_boot, k - 1, axis=2)[..., :k], axis=2)


def bootstrap_ci_batch(
    D_g: np.ndarray,
    D_b: np.ndarray,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (point, lo, hi) each of shape (n_probes,).

    Fully vectorised: builds (n_probes, n_boot, n_refs) resampled matrices,
    reduces along the refs axis directly (no flatten/transpose), and uses
    sorted-quantile lookup (no nanpercentile fallback).
    """
    n_probes, n_g = D_g.shape
    n_b = D_b.shape[1]

    # Generate resample indices once
    ig = rng.integers(0, n_g, size=(n_boot, n_g))   # (n_boot, n_g)
    ib = rng.integers(0, n_b, size=(n_boot, n_b))   # (n_boot, n_b)

    # Build (n_probes, n_boot, n_refs) resampled distance arrays.
    # D_g[:, ig] gives shape (n_probes, n_boot, n_g) directly.
    Dg_boot = D_g[:, ig]
    Db_boot = D_b[:, ib]

    if metric == "median":
        mg_boot = _median_3d(Dg_boot)            # (n_probes, n_boot)
        mb_boot = _median_3d(Db_boot)
        point_g = np.median(D_g, axis=1)
        point_b = np.median(D_b, axis=1)
    elif metric == "knn":
        if n_g < k or n_b < k:
            nan = np.full(n_probes, np.nan)
            return nan, nan, nan
        mg_boot = _knn_mean_3d(Dg_boot, k)
        mb_boot = _knn_mean_3d(Db_boot, k)
        point_g = np.mean(np.partition(D_g, k - 1, axis=1)[:, :k], axis=1)
        point_b = np.mean(np.partition(D_b, k - 1, axis=1)[:, :k], axis=1)
    else:
        raise ValueError(metric)

    boots = mb_boot / (mg_boot + mb_boot + EPS)            # (n_probes, n_boot)
    point = point_b / (point_g + point_b + EPS)            # (n_probes,)

    # Fast quantile via sort + index lookup (no nanpercentile penalty)
    boots.sort(axis=1)                                      # in-place sort
    lo_idx = max(int(round(0.025 * (n_boot - 1))), 0)
    hi_idx = min(int(round(0.975 * (n_boot - 1))), n_boot - 1)
    lo = boots[:, lo_idx]
    hi = boots[:, hi_idx]
    return point, lo, hi


# ---------------------------------------------------------------------------
# Vectorised classify over arrays
# ---------------------------------------------------------------------------

# Category integer codes (decoded only at output time)
CAT_NO_DATA = 0
CAT_RECOVERING = 1
CAT_DEGRADED = 2
CAT_INDISTINGUISHABLE = 3
CAT_NAMES = np.array(
    ["no_data", "recovering", "degraded", "indistinguishable"], dtype=object
)


def classify_batch(
    score: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    t_lo: float, t_hi: float, delta: float,
) -> np.ndarray:
    """Returns int8 codes — string decoding deferred to output time."""
    out = np.full(len(score), CAT_NO_DATA, dtype=np.int8)
    finite = np.isfinite(score) & np.isfinite(lo) & np.isfinite(hi)
    rec = finite & (lo > t_hi) & ((score - t_hi) >= delta)
    deg = finite & (hi < t_lo) & ((t_lo - score) >= delta)
    out[finite] = CAT_INDISTINGUISHABLE
    out[rec] = CAT_RECOVERING
    out[deg] = CAT_DEGRADED
    return out


# ---------------------------------------------------------------------------
# Threshold calibration (Youden-J on LOO score distribution)
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
    """Per-row LOO median score within a single parent's refs."""
    is_g, is_b = lbl == "good", lbl == "bad"
    if is_g.sum() < 2 or is_b.sum() < 2:
        return np.full(len(emb), np.nan)
    D_g = cosine_dist_matrix(emb, emb[is_g])   # (n, n_good)
    D_b = cosine_dist_matrix(emb, emb[is_b])   # (n, n_bad)
    # Mask self-distance (leave-one-out)
    D_g[np.where(is_g)[0], np.arange(is_g.sum())] = np.nan
    D_b[np.where(is_b)[0], np.arange(is_b.sum())] = np.nan
    m_g = np.nanmedian(D_g, axis=1) + EPS
    m_b = np.nanmedian(D_b, axis=1) + EPS
    return m_b / (m_g + m_b)


def loo_knn_scores(emb: np.ndarray, lbl: np.ndarray, k: int) -> np.ndarray:
    """Per-row LOO kNN-margin score within a single parent's refs (cosine)."""
    is_g, is_b = lbl == "good", lbl == "bad"
    if is_g.sum() < k + 1 or is_b.sum() < k + 1:
        return np.full(len(emb), np.nan)
    D_g = cosine_dist_matrix(emb, emb[is_g])
    D_b = cosine_dist_matrix(emb, emb[is_b])
    idx_g, idx_b = np.where(is_g)[0], np.where(is_b)[0]
    # Mask self-distance
    for i, gi in enumerate(idx_g):
        D_g[gi, i] = np.inf
    for i, bi in enumerate(idx_b):
        D_b[bi, i] = np.inf
    D_g = np.where(np.isinf(D_g), np.nan, D_g)
    D_b = np.where(np.isinf(D_b), np.nan, D_b)
    D_g_sorted = np.sort(D_g, axis=1)[:, :k]
    D_b_sorted = np.sort(D_b, axis=1)[:, :k]
    mg = np.nanmean(D_g_sorted, axis=1)
    mb = np.nanmean(D_b_sorted, axis=1)
    return mb / (mg + mb + EPS)


def calibrate_global(df: pd.DataFrame, k: int) -> tuple[float, float]:
    """Full-LOO calibration across all parents (matches original framework)."""
    all_ms, all_ml, all_ks, all_kl = [], [], [], []
    for _, sub in df.groupby("parent_id", sort=False):
        x = sub[EMBED_COLS].to_numpy(dtype=float)
        lbl = sub["ref_state"].to_numpy()
        ms = loo_median_scores(x, lbl)
        m = np.isfinite(ms)
        if m.any():
            all_ms.append(ms[m]); all_ml.append(lbl[m])
        ks = loo_knn_scores(x, lbl, k)
        mk = np.isfinite(ks)
        if mk.any():
            all_ks.append(ks[mk]); all_kl.append(lbl[mk])
    t_med = calibrate_threshold(np.concatenate(all_ms), np.concatenate(all_ml))
    t_knn = calibrate_threshold(np.concatenate(all_ks), np.concatenate(all_kl))
    return t_med, t_knn


def calibrate_per_label(
    df: pd.DataFrame, k: int
) -> dict[str, tuple[float, float]]:
    """Separate Youden-J calibration for each parent_label (built_loss / crop_loss).

    Returns {label: (t_med, t_knn)}.  Falls back to pooled threshold if a
    label has fewer than 2 parents or only one ref_state present.
    """
    pooled_med, pooled_knn = calibrate_global(df, k)
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
            out[str(lbl)] = (pooled_med, pooled_knn)
            continue
        t_m = calibrate_threshold(np.concatenate(all_ms), np.concatenate(all_ml))
        t_k = calibrate_threshold(np.concatenate(all_ks), np.concatenate(all_kl))
        out[str(lbl)] = (t_m, t_k)
    return out


# ---------------------------------------------------------------------------
# Per-parent cache: distance matrices built once
# ---------------------------------------------------------------------------

class ParentCache:
    __slots__ = ("pid", "label", "emb", "is_g", "is_b",
                 "Dcos", "n", "good_idx", "bad_idx")

    def __init__(self, pid: str, label: str, emb: np.ndarray,
                 is_g: np.ndarray, is_b: np.ndarray):
        self.pid = pid
        self.label = label
        self.emb = emb
        self.is_g = is_g
        self.is_b = is_b
        self.n = len(emb)
        self.good_idx = np.where(is_g)[0]
        self.bad_idx = np.where(is_b)[0]
        self.Dcos = cosine_dist_matrix(emb, emb)   # (200, 200) — used for both median and kNN


def build_parent_caches(df: pd.DataFrame) -> dict[str, ParentCache]:
    caches: dict[str, ParentCache] = {}
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
            true_rate=float((cats == "recovering").sum() / n),
            error_rate=float((cats == "degraded").sum() / n),
            abstain_rate=float((cats == "indistinguishable").sum() / n),
            no_data_rate=float((cats == "no_data").sum() / n),
        )
    return dict(
        true_rate=float((cats == "degraded").sum() / n),
        error_rate=float((cats == "recovering").sum() / n),
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--refs",
        default="v2/data/v2real_mask_on_corr300_exhaustive/sampling_strategy_selected_points.parquet",
    )
    parser.add_argument("--strategy", default="random_100")
    parser.add_argument("--out-dir", default="v3/data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadband-halfwidth", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument(
        "--calibration", choices=["global", "per_fold"], default="global",
    )
    parser.add_argument(
        "--per-label-calibration", action="store_true", default=False,
        help="Calibrate t_med and t_knn separately for built_loss and crop_loss.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Build per-parent caches (distance matrices computed once)
    # ------------------------------------------------------------------
    print("Building per-parent caches...", flush=True)
    caches = build_parent_caches(df)

    # ------------------------------------------------------------------
    # Global threshold calibration (pooled or per-label)
    # ------------------------------------------------------------------
    per_label_thresholds: dict[str, tuple[float, float]] = {}
    if args.calibration == "global":
        print("Calibrating thresholds (full LOO)...", flush=True)
        t_med_global, t_knn_global = calibrate_global(df, args.knn_k)
        print(f"  t_med = {t_med_global:.4f}  t_knn = {t_knn_global:.4f}")
        if args.per_label_calibration:
            print("Calibrating per-label thresholds...", flush=True)
            per_label_thresholds = calibrate_per_label(df, args.knn_k)
            for lbl, (tm, tk) in per_label_thresholds.items():
                print(f"  {lbl}: t_med = {tm:.4f}  t_knn = {tk:.4f}")

    hw, delta, k_knn = args.deadband_halfwidth, args.delta, args.knn_k
    K = args.n_folds
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Assign stratified folds per parent
    # ------------------------------------------------------------------
    fold_assign: dict[str, np.ndarray] = {}
    for pid, c in caches.items():
        fa = np.full(c.n, -1, dtype=np.int8)
        fa[c.good_idx] = make_folds(len(c.good_idx), K, rng)
        fa[c.bad_idx] = make_folds(len(c.bad_idx), K, rng)
        fold_assign[pid] = fa

    # ------------------------------------------------------------------
    # Per-fold calibration (optional)
    # ------------------------------------------------------------------
    fold_thresholds: dict[int, tuple[float, float]] = {}
    if args.calibration == "per_fold":
        print(f"Calibrating thresholds per fold ({K} folds)...", flush=True)
        for f in range(K):
            ms_chunks, ml_chunks, ks_chunks, kl_chunks = [], [], [], []
            for pid, c in caches.items():
                fa = fold_assign[pid]
                train_mask = (fa != f) & (fa != -1)
                emb_t = c.emb[train_mask]
                lbl_t = np.where(c.is_g[train_mask], "good", "bad")
                ms = loo_median_scores(emb_t, lbl_t)
                m = np.isfinite(ms)
                if m.any():
                    ms_chunks.append(ms[m]); ml_chunks.append(lbl_t[m])
                ks = loo_knn_scores(emb_t, lbl_t, k_knn)
                mk = np.isfinite(ks)
                if mk.any():
                    ks_chunks.append(ks[mk]); kl_chunks.append(lbl_t[mk])
            t_m = calibrate_threshold(np.concatenate(ms_chunks), np.concatenate(ml_chunks))
            t_k = calibrate_threshold(np.concatenate(ks_chunks), np.concatenate(kl_chunks))
            fold_thresholds[f] = (t_m, t_k)
            print(f"  fold {f}: t_med = {t_m:.4f}  t_knn = {t_k:.4f}")

    # ------------------------------------------------------------------
    # Probe loop — batched per (parent × fold × probe_state)
    # Buffers per column avoid Python dict overhead at 31k rows.
    # ------------------------------------------------------------------
    print(f"Probing ({len(caches)} parents × {K} folds)...", flush=True)

    # Preallocate column buffers; sized roughly to total probes per metric.
    parent_id_buf: list[str] = []
    parent_label_buf: list[str] = []
    fold_buf: list[int] = []
    probe_state_buf: list[str] = []
    probe_row_buf: list[int] = []
    n_train_good_buf: list[int] = []
    n_train_bad_buf: list[int] = []
    score_med_chunks: list[np.ndarray] = []
    score_knn_chunks: list[np.ndarray] = []
    t_med_buf: list[float] = []
    t_knn_buf: list[float] = []
    cat_base_chunks: list[np.ndarray] = []
    cat_s1_chunks: list[np.ndarray] = []
    cat_s2_chunks: list[np.ndarray] = []
    cat_s3_chunks: list[np.ndarray] = []
    cat_s4_chunks: list[np.ndarray] = []

    for pi, (pid, c) in enumerate(caches.items()):
        if pi % 20 == 0:
            print(f"  {pi}/{len(caches)}", flush=True)
        fa = fold_assign[pid]

        for f in range(K):
            if args.calibration == "global":
                _base_med, _base_knn = t_med_global, t_knn_global
            else:
                _base_med, _base_knn = fold_thresholds[f]
            # Per-label override: swap in label-specific thresholds if requested.
            if args.per_label_calibration and c.label in per_label_thresholds:
                t_med, t_knn = per_label_thresholds[c.label]
            else:
                t_med, t_knn = _base_med, _base_knn

            train_g_idx = c.good_idx[fa[c.good_idx] != f]
            train_b_idx = c.bad_idx[fa[c.bad_idx] != f]

            if len(train_g_idx) == 0 or len(train_b_idx) == 0:
                continue

            for probe_state, probe_idx in (("good", c.good_idx[fa[c.good_idx] == f]),
                                            ("bad",  c.bad_idx[fa[c.bad_idx]  == f])):
                n_p = len(probe_idx)
                if n_p == 0:
                    continue

                # Slice precomputed cosine distance matrix: (n_probes, n_train_refs)
                Dg = c.Dcos[probe_idx[:, None], train_g_idx]
                Db = c.Dcos[probe_idx[:, None], train_b_idx]

                s_med, lo_med, hi_med = bootstrap_ci_batch(
                    Dg, Db, "median", args.n_boot, rng
                )
                s_knn, lo_knn, hi_knn = bootstrap_ci_batch(
                    Dg, Db, "knn", args.n_boot, rng, k=k_knn
                )

                cat_base = classify_batch(s_med, lo_med, hi_med, 0.5,        0.5,        0.0)
                cat_s1   = classify_batch(s_med, lo_med, hi_med, t_med,      t_med,      0.0)
                cat_s2   = classify_batch(s_med, lo_med, hi_med, t_med - hw, t_med + hw, 0.0)
                cat_s3   = classify_batch(s_med, lo_med, hi_med, t_med - hw, t_med + hw, delta)
                cat_s4   = classify_batch(s_knn, lo_knn, hi_knn, t_knn - hw, t_knn + hw, delta)

                # Append batch to column buffers
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

    # Single DataFrame construction at the end
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
    })

    # ------------------------------------------------------------------
    # Summarise per step × probe_state × parent_label
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

    suffix = "_per_label_cal" if args.per_label_calibration else ""
    site_out = out_dir / f"within_parent_site_scores{suffix}.csv"
    summary_out = out_dir / f"within_parent_summary{suffix}.csv"
    site_df.to_csv(site_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    step_order = [
        "baseline",
        "step1_calibrated_t",
        "step2_deadband",
        "step3_deadband_effect",
        "step4_knn",
    ]
    print("\n" + "=" * 78)
    print(f"WITHIN-PARENT {K}-FOLD VALIDATION  (calibration={args.calibration})")
    print("=" * 78)
    print(
        "probe=good: held-out good ref  |  error = wrongly called 'degraded'\n"
        "probe=bad:  held-out bad ref   |  error = wrongly called 'recovering'\n"
        "Note: labels assumed mostly correct; some label noise expected.\n"
        "brier: 0=perfect  0.25=random\n"
    )
    for probe in ("good", "bad"):
        tag = "GOOD probe (false-degraded)" if probe == "good" else "BAD probe  (false-recovering)"
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

    print(f"\n{'-'*78}\n  Per-label error_rate (steps 1-4)\n{'-'*78}")
    print(f"  {'step':<33} {'label':<12} {'probe':>5}  {'error%':>6}")
    for step in step_order[1:]:
        for probe in ("good", "bad"):
            for lbl in ("built_loss", "crop_loss"):
                r = summary_df[(summary_df["step"] == step)
                               & (summary_df["probe_state"] == probe)
                               & (summary_df["parent_label"] == lbl)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"  {step:<33} {lbl:<12} {probe:>5}  {r['error_rate']:>6.1%}")

    print(f"\nWrote: {site_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
