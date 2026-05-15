"""Vectorised 3-D bootstrap primitives for batched scorers.

Ported verbatim from v3/scripts/analysis/validate_steps_within_parent.py
and v4/scripts/analysis/validate_steps_within_parent_v4.py (byte-identical).

These operate on (n_probes, n_refs) distance matrices and use sorted-quantile
lookup for the CI rather than `nanpercentile`. The result is faster than
N independent calls to `core.bootstrap_ci` but is not bit-for-bit equal —
the CI percentiles use rounded indices into a sorted array, where
`np.nanpercentile` does linear interpolation. Use `core.bootstrap_ci` when
matching legacy outputs, `bootstrap_ci_batch` for new bulk work.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

EPS = 1e-12

# Category integer codes (decoded only at output time).
CAT_NO_DATA = 0
CAT_RECOVERING = 1
CAT_DEGRADED = 2
CAT_INDISTINGUISHABLE = 3
CAT_NAMES = np.array(
    ["no_data", "recovering", "degraded", "indistinguishable"], dtype=object
)


def cosine_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances |A| x |B| via scipy."""
    return cdist(A, B, metric="cosine")


def median_3d(D_boot: np.ndarray) -> np.ndarray:
    """Median along the last axis of a 3-D (n_probes, n_boot, n_refs) array."""
    return np.median(D_boot, axis=2)


def knn_mean_3d(D_boot: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k smallest along the last axis of (n_probes, n_boot, n_refs)."""
    return np.mean(np.partition(D_boot, k - 1, axis=2)[..., :k], axis=2)


def bootstrap_ci_batch(
    D_g: np.ndarray,
    D_b: np.ndarray,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised bootstrap CI over n_probes simultaneously.

    Returns (point, lo, hi), each shape (n_probes,). `metric` is "median"
    or "knn"; the kNN branch returns NaN for every probe if either pool
    has fewer than k references.
    """
    n_probes, n_g = D_g.shape
    n_b = D_b.shape[1]

    ig = rng.integers(0, n_g, size=(n_boot, n_g))
    ib = rng.integers(0, n_b, size=(n_boot, n_b))

    Dg_boot = D_g[:, ig]
    Db_boot = D_b[:, ib]

    if metric == "median":
        mg_boot = median_3d(Dg_boot)
        mb_boot = median_3d(Db_boot)
        point_g = np.median(D_g, axis=1)
        point_b = np.median(D_b, axis=1)
    elif metric == "knn":
        if n_g < k or n_b < k:
            nan = np.full(n_probes, np.nan)
            return nan, nan, nan
        mg_boot = knn_mean_3d(Dg_boot, k)
        mb_boot = knn_mean_3d(Db_boot, k)
        point_g = np.mean(np.partition(D_g, k - 1, axis=1)[:, :k], axis=1)
        point_b = np.mean(np.partition(D_b, k - 1, axis=1)[:, :k], axis=1)
    else:
        raise ValueError(metric)

    boots = mb_boot / (mg_boot + mb_boot + EPS)
    point = point_b / (point_g + point_b + EPS)

    boots.sort(axis=1)
    lo_idx = max(int(round(0.025 * (n_boot - 1))), 0)
    hi_idx = min(int(round(0.975 * (n_boot - 1))), n_boot - 1)
    return point, boots[:, lo_idx], boots[:, hi_idx]


def classify_batch(
    score: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    t_lo: float,
    t_hi: float,
    delta: float,
) -> np.ndarray:
    """Vectorised CI classifier with deadband + effect-size gate.

    Returns int8 codes (CAT_*); call `CAT_NAMES[out]` to decode.
    """
    out = np.full(len(score), CAT_NO_DATA, dtype=np.int8)
    finite = np.isfinite(score) & np.isfinite(lo) & np.isfinite(hi)
    rec = finite & (lo > t_hi) & ((score - t_hi) >= delta)
    deg = finite & (hi < t_lo) & ((t_lo - score) >= delta)
    out[finite] = CAT_INDISTINGUISHABLE
    out[rec] = CAT_RECOVERING
    out[deg] = CAT_DEGRADED
    return out
