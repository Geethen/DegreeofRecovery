"""Compare reference sampling strategies with spatial autocorrelation-aware diagnostics.

This script runs controlled, like-for-like experiments on the existing
reference embedding parquet and reports how strategy choice affects:
  - effective sample size (n_eff) under spatial autocorrelation,
  - DoR uncertainty (bootstrap CI width),
  - pooled good-vs-bad discriminability (ROC AUC),
  - applicability/coverage across parents.

It is designed to inform v2 sampling policy before modifying EE sampling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class Strategy:
    name: str
    method: str
    max_raw: int
    thin_m: float


DEFAULT_RANDOM_CAPS = [40, 60, 100, 150, 200]
DEFAULT_FSCS_CAPS = [60, 100, 150, 200]
DEFAULT_DIVERSE_CAPS = [60, 100, 150]
DEFAULT_THIN_CAPS = [60, 100, 150]
DEFAULT_THIN_DIST_M = [100, 150, 250, 300]


def build_strategies(exhaustive: bool) -> list[Strategy]:
    if not exhaustive:
        return [
            Strategy("random_100", "random", 100, 0.0),
            Strategy("random_60", "random", 60, 0.0),
            Strategy("feature_diverse_60", "feature_diverse", 60, 0.0),
            Strategy("fscs_100", "fscs", 100, 0.0),
            Strategy("spatial_thin_150_cap60", "spatial_thin", 60, 150.0),
        ]

    out: list[Strategy] = []
    for cap in DEFAULT_RANDOM_CAPS:
        out.append(Strategy(f"random_{cap}", "random", cap, 0.0))
    for cap in DEFAULT_FSCS_CAPS:
        out.append(Strategy(f"fscs_{cap}", "fscs", cap, 0.0))
    for cap in DEFAULT_DIVERSE_CAPS:
        out.append(Strategy(f"feature_diverse_{cap}", "feature_diverse", cap, 0.0))
    for cap in DEFAULT_THIN_CAPS:
        for d in DEFAULT_THIN_DIST_M:
            out.append(
                Strategy(f"spatial_thin_{d}_cap{cap}", "spatial_thin", cap, float(d))
            )
    for cap in [100, 150]:
        for d in [150, 300]:
            out.append(Strategy(f"hybrid_{d}_diverse_{cap}", "hybrid", cap, float(d)))
    return out


def load_refs(parquet_path: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute("SELECT * FROM read_parquet(?)", [parquet_path]).df()
    con.close()
    missing = [c for c in EMBED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing embedding columns: {missing[:5]}")
    df = df[df["ref_state"].isin(["good", "bad"])].copy()
    df = df.dropna(subset=EMBED_COLS + ["geo", "parent_id", "ref_state"]).reset_index(
        drop=True
    )

    # DuckDB materializes EE geometry as dict-like objects in this parquet.
    coords = df["geo"].map(
        lambda g: g.get("coordinates") if isinstance(g, dict) else None
    )
    df["lon"] = coords.map(lambda c: float(c[0]) if c else np.nan)
    df["lat"] = coords.map(lambda c: float(c[1]) if c else np.nan)
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    df["parent_id"] = df["parent_id"].astype(str)
    return df


def load_test_embeddings(parquet_path: str) -> dict[str, np.ndarray]:
    """Load real test-site embeddings as a dict keyed by parent_id.

    If a parent_id has multiple rows (rare), the median across rows is used.
    """
    con = duckdb.connect()
    df = con.execute("SELECT * FROM read_parquet(?)", [parquet_path]).df()
    con.close()
    missing = [c for c in EMBED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Test-site parquet missing embedding columns: {missing[:5]}")
    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=EMBED_COLS + ["parent_id"]).reset_index(drop=True)
    out: dict[str, np.ndarray] = {}
    for pid, g in df.groupby("parent_id", sort=False):
        out[pid] = g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
    return out


def haversine_matrix(lat_lon: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_lon[:, 0])
    lon = np.radians(lat_lon[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return (
        2.0
        * EARTH_RADIUS_M
        * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))
    )


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    sim = (x @ x.T) / (n @ n.T)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def compute_neff(dist_m: np.ndarray, corr_range_m: float) -> float:
    if len(dist_m) == 0:
        return 0.0
    if corr_range_m <= 0:
        return float(dist_m.shape[0])
    w = np.exp(-((dist_m / corr_range_m) ** 2))
    denom = float(np.sum(w))
    n = float(dist_m.shape[0])
    return float((n * n) / max(denom, 1e-12))


def random_select(n: int, max_raw: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(n)
    if n <= max_raw:
        return idx
    return np.sort(rng.choice(idx, size=max_raw, replace=False))


def farthest_point_select(
    emb: np.ndarray, max_raw: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(emb)
    if n <= max_raw:
        return np.arange(n)
    chosen = [int(rng.integers(0, n))]
    dmat = cosine_distance_matrix(emb)
    min_d = dmat[chosen[0]].copy()
    for _ in range(max_raw - 1):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, dmat[nxt])
    return np.sort(np.array(chosen, dtype=int))


def fscs_select(
    emb: np.ndarray,
    max_raw: int,
    rng: np.random.Generator,
    n_bands: int = 64,
    max_iter: int = 20,
) -> np.ndarray:
    """FSCS-style selection via k-means centroids in feature space.

    Uses the first `n_bands` embedding bands (default = all 64 AlphaEarth bands),
    runs a Lloyd loop, then keeps the closest point to each centroid.
    """
    n = len(emb)
    if n <= max_raw:
        return np.arange(n)

    x = emb[:, : min(n_bands, emb.shape[1])]
    k = min(max_raw, n)
    init = rng.choice(n, size=k, replace=False)
    centroids = x[init].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if np.any(m):
                centroids[j] = x[m].mean(axis=0)

    selected = []
    for j in range(k):
        m_idx = np.where(labels == j)[0]
        if len(m_idx) == 0:
            continue
        local = x[m_idx]
        c = centroids[j]
        best = m_idx[int(np.argmin(np.sum((local - c) ** 2, axis=1)))]
        selected.append(int(best))

    if len(selected) < max_raw:
        pool = np.array([i for i in range(n) if i not in set(selected)], dtype=int)
        if len(pool):
            fill_n = min(max_raw - len(selected), len(pool))
            fill = rng.choice(pool, size=fill_n, replace=False)
            selected.extend([int(i) for i in fill])

    return np.sort(np.array(selected[:max_raw], dtype=int))


def spatial_thin_select(
    lat_lon: np.ndarray,
    max_raw: int,
    thin_m: float,
    rng: np.random.Generator,
    refill: bool = False,
) -> np.ndarray:
    n = len(lat_lon)
    if n == 0:
        return np.array([], dtype=int)
    order = np.arange(n)
    rng.shuffle(order)
    dist = haversine_matrix(lat_lon)
    kept = []
    for i in order:
        if len(kept) >= max_raw:
            break
        if not kept:
            kept.append(int(i))
            continue
        d = dist[i, np.array(kept, dtype=int)]
        if float(np.min(d)) >= thin_m:
            kept.append(int(i))
    if refill and len(kept) < max_raw:
        remaining = [int(i) for i in order if int(i) not in set(kept)]
        kept.extend(remaining[: max_raw - len(kept)])
    return np.sort(np.array(kept, dtype=int))


def select_indices(
    sub: pd.DataFrame, strategy: Strategy, rng: np.random.Generator
) -> np.ndarray:
    emb = sub[EMBED_COLS].to_numpy(dtype=float)
    lat_lon = sub[["lat", "lon"]].to_numpy(dtype=float)
    n = len(sub)
    if strategy.method == "random":
        return random_select(n, strategy.max_raw, rng)
    if strategy.method == "feature_diverse":
        return farthest_point_select(emb, strategy.max_raw, rng)
    if strategy.method == "fscs":
        return fscs_select(emb, strategy.max_raw, rng)
    if strategy.method == "spatial_thin":
        return spatial_thin_select(
            lat_lon, strategy.max_raw, strategy.thin_m, rng, refill=False
        )
    if strategy.method == "hybrid":
        thin_idx = spatial_thin_select(
            lat_lon,
            max(strategy.max_raw * 2, strategy.max_raw),
            strategy.thin_m,
            rng,
            refill=False,
        )
        if len(thin_idx) == 0:
            return thin_idx
        thinned = emb[thin_idx]
        diverse_local = farthest_point_select(thinned, strategy.max_raw, rng)
        return np.sort(thin_idx[diverse_local])
    raise ValueError(strategy.method)


def cosine_dists_to_set(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    nx = np.linalg.norm(x) + 1e-12
    np_pts = np.linalg.norm(points, axis=1) + 1e-12
    sims = (points @ x) / (np_pts * nx)
    return 1.0 - sims


def dor_median_with_ci(
    d_g: np.ndarray, d_b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    mg = float(np.median(d_g))
    mb = float(np.median(d_b))
    point = mb / (mg + mb + 1e-12)

    n_g, n_b = len(d_g), len(d_b)
    ig = rng.integers(0, n_g, size=(n_boot, n_g))
    ib = rng.integers(0, n_b, size=(n_boot, n_b))
    med_g = np.median(d_g[ig], axis=1)
    med_b = np.median(d_b[ib], axis=1)
    boot = med_b / (med_g + med_b + 1e-12)
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
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


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sorted_scores = y_score[order]
    starts = np.where(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])[0]
    ends = np.r_[starts[1:], len(sorted_scores)]
    for s, e in zip(starts, ends, strict=True):
        if e - s > 1:
            avg_rank = (s + 1 + e) / 2.0
            ranks[order[s:e]] = avg_rank

    sum_ranks_pos = float(np.sum(ranks[pos]))
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def run_experiment(
    df: pd.DataFrame,
    strategy: Strategy,
    n_boot: int,
    seed: int,
    corr_range_m: float,
    test_emb: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    selected_rows = []
    auc_scores = []
    auc_labels = []
    fallback_count = 0

    for pid, parent in df.groupby("parent_id", sort=False):
        g = parent[parent["ref_state"] == "good"]
        b = parent[parent["ref_state"] == "bad"]
        if len(g) == 0 or len(b) == 0:
            continue

        idx_g = select_indices(g, strategy, rng)
        idx_b = select_indices(b, strategy, rng)
        g_sel = g.iloc[idx_g]
        b_sel = b.iloc[idx_b]

        selected = pd.concat([g_sel, b_sel], axis=0, ignore_index=True)
        selected = selected.copy()
        selected["strategy"] = strategy.name
        selected_rows.append(selected)

        g_dist = haversine_matrix(g_sel[["lat", "lon"]].to_numpy())
        b_dist = haversine_matrix(b_sel[["lat", "lon"]].to_numpy())
        n_eff_good = compute_neff(g_dist, corr_range_m)
        n_eff_bad = compute_neff(b_dist, corr_range_m)

        if test_emb is not None and pid in test_emb:
            x_obs = test_emb[pid]
            x_obs_source = "test_site"
        else:
            x_obs = parent[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
            x_obs_source = "ref_mean"
            if test_emb is not None:
                fallback_count += 1
        d_g = cosine_dists_to_set(x_obs, g_sel[EMBED_COLS].to_numpy(dtype=float))
        d_b = cosine_dists_to_set(x_obs, b_sel[EMBED_COLS].to_numpy(dtype=float))
        dor, ci_lo, ci_hi = dor_median_with_ci(d_g, d_b, n_boot=n_boot, rng=rng)

        x = selected[EMBED_COLS].to_numpy(dtype=float)
        y = (selected["ref_state"].to_numpy() == "good").astype(int)
        s = loo_median_scores(x, selected["ref_state"].to_numpy())
        mask = ~np.isnan(s)
        if int(np.sum(mask)) > 10 and len(np.unique(y[mask])) == 2:
            auc_scores.append(s[mask])
            auc_labels.append(y[mask])

        rows.append(
            {
                "strategy": strategy.name,
                "parent_id": pid,
                "parent_label": parent["parent_label"].iloc[0],
                "x_obs_source": x_obs_source,
                "n_good": int(len(g_sel)),
                "n_bad": int(len(b_sel)),
                "n_eff_good": float(n_eff_good),
                "n_eff_bad": float(n_eff_bad),
                "n_eff_min": float(min(n_eff_good, n_eff_bad)),
                "dor": float(dor),
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
                "ci_width": float(ci_hi - ci_lo),
            }
        )

    per_parent = pd.DataFrame(rows)

    auc = float("nan")
    if auc_scores:
        pooled_scores = np.concatenate(auc_scores)
        pooled_labels = np.concatenate(auc_labels)
        auc = roc_auc(pooled_labels, pooled_scores)

    if per_parent.empty:
        summary = pd.Series(
            {
                "strategy": strategy.name,
                "n_parents": 0,
                "median_ci_width": np.nan,
                "p90_ci_width": np.nan,
                "median_neff_min": np.nan,
                "p25_neff_min": np.nan,
                "min_neff10_coverage": np.nan,
                "min_neff20_coverage": np.nan,
                "min_neff30_coverage": np.nan,
                "pooled_auc": auc,
                "x_obs_fallback_parents": int(fallback_count),
            }
        )
        return per_parent, summary, pd.DataFrame()

    neff_min = per_parent["n_eff_min"]
    summary_dict = {
        "strategy": strategy.name,
        "n_parents": int(len(per_parent)),
        "median_ci_width": float(per_parent["ci_width"].median()),
        "p90_ci_width": float(per_parent["ci_width"].quantile(0.9)),
        "median_neff_min": float(neff_min.median()),
        "p25_neff_min": float(neff_min.quantile(0.25)),
        "min_neff10_coverage": float((neff_min >= 10).mean()),
        "min_neff20_coverage": float((neff_min >= 20).mean()),
        "min_neff30_coverage": float((neff_min >= 30).mean()),
        "pooled_auc": auc,
        "x_obs_fallback_parents": int(fallback_count),
    }
    # Per-label breakdowns (built_loss vs crop_loss vs others)
    for lbl, gdf in per_parent.groupby("parent_label"):
        suffix = str(lbl)
        summary_dict[f"median_ci_width__{suffix}"] = float(gdf["ci_width"].median())
        summary_dict[f"median_neff_min__{suffix}"] = float(gdf["n_eff_min"].median())
        summary_dict[f"median_dor__{suffix}"] = float(gdf["dor"].median())
        summary_dict[f"n_parents__{suffix}"] = int(len(gdf))
    summary = pd.Series(summary_dict)
    selected_out = pd.concat(selected_rows, ignore_index=True)
    return per_parent, summary, selected_out


def plot_summary(summary: pd.DataFrame, out_path: str) -> None:
    order = summary.sort_values("median_ci_width")["strategy"].tolist()
    s = summary.set_index("strategy").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(s["strategy"], s["median_ci_width"], color="#2a9d8f")
    axes[0].set_title("Median CI width (lower is better)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(s["strategy"], s["median_neff_min"], color="#264653")
    axes[1].set_title("Median min effective n")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(s["strategy"], s["min_neff30_coverage"], color="#e76f51")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Coverage with min n_eff >= 30")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="degreeRecover/data/recover_reference_samples_alphaearth.parquet",
        help="Path to reference embedding parquet.",
    )
    parser.add_argument(
        "--out-dir",
        default="v2/data",
        help="Output directory for summaries.",
    )
    parser.add_argument(
        "--plot-dir",
        default="v2/plots",
        help="Output directory for plots.",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--corr-range-m",
        type=float,
        default=300.0,
        help="Correlation range (m) used in n_eff approximation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run an expanded strategy matrix (random, FSCS, diverse, and multiple min-distance criteria).",
    )
    parser.add_argument(
        "--test-embeddings",
        default="degreeRecover/data/test_site_alphaearth_2024.parquet",
        help="Test-site embedding parquet for x_obs (operationally meaningful DoR).",
    )
    parser.add_argument(
        "--use-ref-mean",
        action="store_true",
        help="Use reference mean as x_obs instead of test-site embedding (legacy v1 behaviour).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_refs(args.input)

    test_emb: dict[str, np.ndarray] | None = None
    if not args.use_ref_mean:
        test_path = Path(args.test_embeddings)
        if test_path.exists():
            test_emb = load_test_embeddings(str(test_path))
            ref_pids = set(df["parent_id"].unique())
            test_pids = set(test_emb.keys())
            print(
                f"Loaded {len(test_emb)} test-site embeddings; "
                f"overlap with refs: {len(ref_pids & test_pids)} / {len(ref_pids)}"
            )
        else:
            print(
                f"WARNING: test-site embeddings not found at {test_path}; "
                "falling back to ref-mean x_obs."
            )

    strategies = build_strategies(exhaustive=args.exhaustive)

    all_parent = []
    all_summary = []
    all_selected = []
    for st in strategies:
        per_parent, summary, selected = run_experiment(
            df,
            strategy=st,
            n_boot=args.n_boot,
            seed=args.seed,
            corr_range_m=args.corr_range_m,
            test_emb=test_emb,
        )
        all_parent.append(per_parent)
        all_summary.append(summary)
        all_selected.append(selected)

    parent_out = pd.concat(all_parent, ignore_index=True)
    summary_out = pd.DataFrame(all_summary).sort_values("median_ci_width")
    selected_out = pd.concat(all_selected, ignore_index=True)

    parent_csv = out_dir / "sampling_strategy_by_parent.csv"
    summary_csv = out_dir / "sampling_strategy_comparison.csv"
    selected_csv = out_dir / "sampling_strategy_selected_points.csv"
    selected_parquet = out_dir / "sampling_strategy_selected_points.parquet"
    meta_json = out_dir / "sampling_strategy_run_meta.json"
    summary_png = plot_dir / "sampling_strategy_comparison.png"

    parent_out.to_csv(parent_csv, index=False)
    summary_out.to_csv(summary_csv, index=False)
    selected_out.to_csv(selected_csv, index=False)
    con = duckdb.connect()
    con.register("selected_df", selected_out)
    con.execute(
        "COPY selected_df TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(selected_parquet)],
    )
    con.close()
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input": args.input,
                "n_boot": args.n_boot,
                "corr_range_m": args.corr_range_m,
                "seed": args.seed,
                "x_obs_source": ("test_site" if test_emb is not None else "ref_mean"),
                "test_embeddings": (
                    args.test_embeddings if test_emb is not None else None
                ),
                "strategies": [s.__dict__ for s in strategies],
            },
            f,
            indent=2,
        )
    plot_summary(summary_out, summary_png)

    print("Wrote:")
    print(f"  {summary_csv}")
    print(f"  {parent_csv}")
    print(f"  {selected_csv}")
    print(f"  {selected_parquet}")
    print(f"  {summary_png}")
    print("\nTop strategies by median CI width:")
    print(
        summary_out[
            [
                "strategy",
                "median_ci_width",
                "median_neff_min",
                "min_neff30_coverage",
                "pooled_auc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
