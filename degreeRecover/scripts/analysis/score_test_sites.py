"""Compute Degree-of-Recovery (DoR) for each test site (parent point).

The extracted parquet contains good + bad reference embeddings around each
parent but not the parent itself. This script:

  1. Loads the reference embeddings parquet.
  2. Loads parent-point AlphaEarth embeddings from a cached parquet
     produced by `scripts/extraction/extract_test_site_embeddings.py`
     (run that first; it samples GEE once and writes a parquet so this
     script does not re-hit GEE on every run).
  3. Computes a per-point median-based DoR (primary score) with a
     bootstrap CI:

        d_g_i = cosine_dist(obs, good_i)   for each good ref i
        d_b_j = cosine_dist(obs, bad_j)    for each bad  ref j
        DoR   = median(d_b_j) / (median(d_g_i) + median(d_b_j))

     The references are bootstrapped (resampled with replacement, per
     class) to give a 95 % CI on DoR itself. Sites whose CI lies entirely
     above 0.5 are confidently recovering; sites whose CI straddles 0.5
     are statistically indistinguishable from the reference midpoint.

     Why median pairwise instead of centroid ratio: the reference clouds
     are often multimodal/heterogeneous (the good cloud especially),
     and centroids of multimodal clouds are unreliable summaries.
     Median pairwise distances are robust to that without assuming a
     unimodal Gaussian. Tradeoff: it discards the geometric structure
     of embedding space (no axis projection / direction), so we keep
     `dor_cosine` and the recovery-axis plot as diagnostics — they show
     *where* on the good→bad axis a site sits, useful to spot sites
     that fall off-axis entirely vs. legitimately between the clouds.

     stable_stable parents have no bad refs (the parent label asserts
     the site has not degraded). They are skipped — their rows have
     NaN scores and applicable=False.

Outputs:
  data/test_site_dor.csv
  plots/test_site_dor_with_ci.png      (primary)
  plots/test_site_recovery_axis.png    (diagnostic)

Usage:
  # one-time per (year, parent_asset):
  python scripts/extraction/extract_test_site_embeddings.py --year 2024
  # then:
  python scripts/analysis/score_test_sites.py
  python scripts/analysis/score_test_sites.py --year 2024
"""

import argparse
import json
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths / constants ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DEFAULT_INPUT = os.path.join(DATA_DIR, "recover_reference_samples_alphaearth.parquet")

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
YEAR = 2024
EARTH_RADIUS_M = 6371000.0

PRIMARY_SCORE = "dor_median"
DIAGNOSTIC_SCORES = ["dor_normalised", "dor_percentile", "dor_cosine"]
ALL_SCORES = [PRIMARY_SCORE] + DIAGNOSTIC_SCORES

N_BOOT_DEFAULT = 2000
SEED_DEFAULT = 42


def default_test_site_parquet(year):
    return os.path.join(DATA_DIR, f"test_site_alphaearth_{year}.parquet")


PARITY_MIN_FRACTION = 0.9


# ── Test-site embedding cache ───────────────────────────────────────
def load_parent_embeddings(
    parquet_path, refs_parent_ids, min_fraction=PARITY_MIN_FRACTION
):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Test-site embeddings parquet not found: {parquet_path}\n"
            f"Run: python scripts/extraction/extract_test_site_embeddings.py"
        )
    con = duckdb.connect()
    parents = con.execute(f"SELECT * FROM '{parquet_path}'").df()
    con.close()
    parents["parent_id"] = parents["parent_id"].astype(str)

    missing = [c for c in EMBED_COLS if c not in parents.columns]
    if missing:
        raise RuntimeError(f"{parquet_path} missing AlphaEarth bands: {missing[:3]}...")

    have = set(parents["parent_id"])
    want = set(map(str, refs_parent_ids))
    if not want:
        return parents
    absent = want - have
    fraction = (len(want) - len(absent)) / len(want)
    if absent and fraction < min_fraction:
        sample = sorted(absent)[:5]
        raise RuntimeError(
            f"parent_id parity check failed: only "
            f"{fraction:.1%} of refs parents are present in "
            f"{os.path.basename(parquet_path)} "
            f"({len(absent)}/{len(want)} missing, e.g. {sample}). "
            f"Re-extract with --overwrite to refresh the cache."
        )
    if absent:
        print(
            f"[warn] {len(absent)}/{len(want)} parent(s) "
            f"({1 - fraction:.1%}) in refs missing from "
            f"{os.path.basename(parquet_path)} — sites will be skipped. "
            f"Re-extract with --overwrite if needed."
        )
    return parents


def load_refs(parquet_path, strategy=None):
    con = duckdb.connect()
    refs = con.execute(f"SELECT * FROM '{parquet_path}'").df()
    con.close()
    refs["parent_id"] = refs["parent_id"].astype(str)

    if strategy is not None and "strategy" in refs.columns:
        refs = refs[refs["strategy"] == strategy].copy()

    refs = refs[refs["ref_state"].isin(["good", "bad"])].copy()
    refs = refs.dropna(subset=EMBED_COLS + ["geo", "parent_id", "ref_state"])

    coords = refs["geo"].map(
        lambda g: g.get("coordinates") if isinstance(g, dict) else None
    )
    refs["lon"] = coords.map(lambda c: float(c[0]) if c else np.nan)
    refs["lat"] = coords.map(lambda c: float(c[1]) if c else np.nan)
    refs = refs.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    return refs


def load_corr_range_map(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Correlation-range file not found: {path}")
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "per_parent" in payload:
            payload = payload["per_parent"]
        return {str(k): float(v) for k, v in payload.items()}
    df = pd.read_csv(path)
    if not {"parent_id", "corr_range_m"}.issubset(df.columns):
        raise RuntimeError(f"{path} must include columns: parent_id,corr_range_m")
    return {
        str(r.parent_id): float(r.corr_range_m)
        for r in df[["parent_id", "corr_range_m"]].itertuples(index=False)
    }


# ── Distance / scoring primitives ───────────────────────────────────
def cosine_dist(a, b, eps=1e-12):
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)


def cosine_dists_to_set(x, points, eps=1e-12):
    """Cosine distance from x to every row in points (vectorised)."""
    nx = np.linalg.norm(x) + eps
    np_pts = np.linalg.norm(points, axis=1) + eps
    sims = (points @ x) / (np_pts * nx)
    return 1.0 - sims


def haversine_matrix(lat_lon):
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


def compute_neff(dist_m, corr_range_m):
    if len(dist_m) == 0:
        return 0.0
    if corr_range_m <= 0:
        return float(dist_m.shape[0])
    w = np.exp(-((dist_m / corr_range_m) ** 2))
    n = float(dist_m.shape[0])
    denom = float(np.sum(w))
    return float((n * n) / max(denom, 1e-12))


def _median_dor(d_g, d_b, eps=1e-12):
    m_g = np.median(d_g)
    m_b = np.median(d_b)
    return m_b / (m_g + m_b + eps)


def dor_median_with_ci(d_g, d_b, n_boot=N_BOOT_DEFAULT, rng=None):
    """Median-based DoR with bootstrap CI on the DoR itself.

    Resamples each class with replacement (preserving its size) n_boot
    times, recomputing DoR each time, and reports the 2.5/97.5
    percentile.
    """
    if rng is None:
        rng = np.random.default_rng(SEED_DEFAULT)
    point = float(_median_dor(d_g, d_b))

    n_g, n_b = len(d_g), len(d_b)
    ig = rng.integers(0, n_g, size=(n_boot, n_g))
    ib = rng.integers(0, n_b, size=(n_boot, n_b))
    med_g = np.median(d_g[ig], axis=1)
    med_b = np.median(d_b[ib], axis=1)
    boot = med_b / (med_g + med_b + 1e-12)
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return point, ci_low, ci_high


def dor_normalised(d_g_obs, good, bad, eps=1e-12):
    """Linear normalised score using medians:

        score = (d_bad - d_obs) / (d_bad - d_good)

    where:
        d_obs  = median distance from obs to good refs
        d_bad  = median pairwise distance bad→good (inter-cloud span)
        d_good = 0                                   (good→itself)

    Equivalent to 1 - d_obs / d_bad. score = 1 at the good cloud,
    0 when obs is as far from good as the bad cloud is, < 0 if obs
    sits even further from good than the bad cloud does.
    """
    nb = np.linalg.norm(bad, axis=1) + eps
    ng = np.linalg.norm(good, axis=1) + eps
    sims = (bad @ good.T) / np.outer(nb, ng)
    d_bad_good = (1.0 - sims).ravel()
    d_bad = float(np.median(d_bad_good))
    d_obs = float(np.median(d_g_obs))
    if d_bad < eps:
        return np.nan
    return (d_bad - d_obs) / d_bad


def dor_percentile(score_obs, good, bad, eps=1e-12):
    """Rank of the test site's median-DoR within the LOO median-DoR
    distribution computed over the parent's own reference points.

    Vectorised: build the three cosine-distance matrices once, mask the
    diagonal of the within-class matrices with NaN, and use `nanmedian`
    along the appropriate axis.
    """
    n_g, n_b = len(good), len(bad)
    if n_g < 2 or n_b < 2:
        return np.nan

    ng = np.linalg.norm(good, axis=1) + eps
    nb = np.linalg.norm(bad, axis=1) + eps
    Dgg = 1.0 - (good @ good.T) / np.outer(ng, ng)
    Dbb = 1.0 - (bad @ bad.T) / np.outer(nb, nb)
    Dgb = 1.0 - (good @ bad.T) / np.outer(ng, nb)

    np.fill_diagonal(Dgg, np.nan)
    np.fill_diagonal(Dbb, np.nan)

    # good refs as obs: drop self from within-good distances
    med_g_g = np.nanmedian(Dgg, axis=1)
    med_b_g = np.median(Dgb, axis=1)
    scores_good = med_b_g / (med_g_g + med_b_g + eps)

    # bad refs as obs: drop self from within-bad distances
    med_g_b = np.median(Dgb, axis=0)
    med_b_b = np.nanmedian(Dbb, axis=1)
    scores_bad = med_b_b / (med_g_b + med_b_b + eps)

    ref_scores = np.concatenate([scores_good, scores_bad])
    return float(np.mean(ref_scores <= score_obs))


def dor_cosine_centroid(x, good, bad):
    """Diagnostic: centroid-based cosine DoR.

    score = 1 - cosine_dist(x, c_good) / cosine_dist(c_bad, c_good)
    Unreliable when reference clouds are multimodal — keep as a
    diagnostic to compare against the median-based primary score and
    to anchor the recovery-axis plot.
    """
    cg = good.mean(axis=0)
    cb = bad.mean(axis=0)
    d_inter = cosine_dist(cg, cb)
    d_obs = cosine_dist(x, cg)
    if d_inter < 1e-12:
        return np.nan
    return 1.0 - d_obs / d_inter


# ── Per-site scoring ────────────────────────────────────────────────
def compute_dor(
    parents_df,
    refs_df,
    embed_cols=EMBED_COLS,
    n_boot=N_BOOT_DEFAULT,
    seed=SEED_DEFAULT,
    corr_range_m_default=300.0,
    corr_range_by_parent=None,
):
    """Per-site DoR. Sites without both good and bad parent-specific
    references (notably stable_stable parents) are skipped — their rows
    are reported with NaN scores and applicable=False."""
    rng = np.random.default_rng(seed)

    parents_df = parents_df.copy()
    parents_df["parent_id"] = parents_df["parent_id"].astype(str)
    refs_df = refs_df.copy()
    refs_df["parent_id"] = refs_df["parent_id"].astype(str)

    # Group refs once, then look up by parent_id — avoids the O(N_refs)
    # per-parent astype/filter the iterrows version did.
    refs_groups = dict(tuple(refs_df.groupby("parent_id", sort=False)))

    embed_idx = [parents_df.columns.get_loc(c) for c in embed_cols]
    pid_idx = parents_df.columns.get_loc("parent_id")
    rows = []
    for p in parents_df.itertuples(index=False, name=None):
        pid = str(p[pid_idx])
        sub = refs_groups.get(pid)
        if sub is not None:
            good = sub.loc[sub["ref_state"] == "good", embed_cols].to_numpy()
            bad = sub.loc[sub["ref_state"] == "bad", embed_cols].to_numpy()
            label = sub["parent_label"].iloc[0]
        else:
            good = np.empty((0, len(embed_cols)))
            bad = np.empty((0, len(embed_cols)))
            label = np.nan
        x = np.array([p[i] for i in embed_idx], dtype=float)
        corr_range_m = float(
            corr_range_by_parent.get(pid, corr_range_m_default)
            if corr_range_by_parent is not None
            else corr_range_m_default
        )

        out = {
            "parent_id": pid,
            "parent_label": label,
            "n_good": int(len(good)),
            "n_bad": int(len(bad)),
            "corr_range_m": corr_range_m,
            "n_eff_good": np.nan,
            "n_eff_bad": np.nan,
            "n_eff_min": np.nan,
            "cos_dist_good": np.nan,
            "cos_dist_bad": np.nan,
            "dor_median": np.nan,
            "dor_median_ci_low": np.nan,
            "dor_median_ci_high": np.nan,
            "dor_normalised": np.nan,
            "dor_percentile": np.nan,
            "dor_cosine": np.nan,
            "applicable": False,
        }

        if len(good) == 0 or len(bad) == 0:
            rows.append(out)
            continue

        d_g = cosine_dists_to_set(x, good)
        d_b = cosine_dists_to_set(x, bad)

        g_xy = sub.loc[sub["ref_state"] == "good", ["lat", "lon"]].to_numpy()
        b_xy = sub.loc[sub["ref_state"] == "bad", ["lat", "lon"]].to_numpy()
        n_eff_good = compute_neff(haversine_matrix(g_xy), corr_range_m)
        n_eff_bad = compute_neff(haversine_matrix(b_xy), corr_range_m)
        out["n_eff_good"] = float(n_eff_good)
        out["n_eff_bad"] = float(n_eff_bad)
        out["n_eff_min"] = float(min(n_eff_good, n_eff_bad))

        dor, ci_low, ci_high = dor_median_with_ci(d_g, d_b, n_boot, rng)
        out["dor_median"] = dor
        out["dor_median_ci_low"] = ci_low
        out["dor_median_ci_high"] = ci_high

        out["dor_normalised"] = float(dor_normalised(d_g, good, bad))
        out["dor_percentile"] = float(dor_percentile(dor, good, bad))

        cg = good.mean(axis=0)
        cb = bad.mean(axis=0)
        out["cos_dist_good"] = float(cosine_dist(x, cg))
        out["cos_dist_bad"] = float(cosine_dist(x, cb))
        out["dor_cosine"] = float(dor_cosine_centroid(x, good, bad))

        out["applicable"] = True
        rows.append(out)
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────
def plot_dor_with_ci(dor, out_path):
    """Primary plot: dor_median per site with bootstrap 95 % CI bars,
    grouped by parent_label."""
    sub = dor.dropna(subset=["dor_median"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values(["parent_label", "dor_median"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(sub)), 5))
    labels = sorted(sub["parent_label"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    color_for = {lbl: cmap(i % 10) for i, lbl in enumerate(labels)}

    for i, row in sub.iterrows():
        c = color_for.get(row["parent_label"], "k")
        ax.errorbar(
            i,
            row["dor_median"],
            yerr=[
                [row["dor_median"] - row["dor_median_ci_low"]],
                [row["dor_median_ci_high"] - row["dor_median"]],
            ],
            fmt="o",
            color=c,
            capsize=4,
            lw=1.5,
            markersize=7,
        )

    ax.axhline(0.5, color="grey", ls="--", lw=1, label="midpoint (DoR = 0.5)")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["parent_id"], rotation=90, fontsize=7)
    ax.set_ylabel("dor_median  (1 = natural, 0 = degraded)")
    ax.set_xlabel("Test site")
    ax.set_ylim(-0.05, 1.05)

    from matplotlib.lines import Line2D

    legend_els = [
        Line2D(
            [0], [0], marker="o", color=color_for[lbl], lw=0, markersize=8, label=lbl
        )
        for lbl in labels
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")
    ax.set_title(
        "Per-site DoR with bootstrap 95 % CI\n"
        "(CI fully above 0.5 → confidently recovering)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_recovery_axis(
    parents_df, refs_df, dor, out_path, max_panels=6, embed_cols=EMBED_COLS
):
    """Diagnostic: project references + parent onto the good→bad axis to
    show *where* on that axis a site sits (off-axis vs between clouds)."""
    scoreable = dor.dropna(subset=["dor_cosine"])["parent_id"].tolist()
    if not scoreable:
        return
    parents_df = parents_df.copy()
    parents_df["parent_id"] = parents_df["parent_id"].astype(str)
    refs_df = refs_df.copy()
    refs_df["parent_id"] = refs_df["parent_id"].astype(str)

    pids = scoreable[:max_panels]
    n = len(pids)
    cols = min(n, 2)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(6 * cols, 3.2 * rows_n), squeeze=False
    )

    for ax, pid in zip(axes.flat, pids):
        ref_sub = refs_df[refs_df["parent_id"] == pid]
        good = ref_sub[ref_sub["ref_state"] == "good"][embed_cols].to_numpy()
        bad = ref_sub[ref_sub["ref_state"] == "bad"][embed_cols].to_numpy()
        cg = good.mean(axis=0)
        cb = bad.mean(axis=0)
        axis_unit = (cb - cg) / (np.linalg.norm(cb - cg) + 1e-12)
        scale = float(np.dot(cb - cg, axis_unit))

        def proj(arr):
            return (arr - cg) @ axis_unit / (scale + 1e-12)

        parent_row = parents_df[parents_df["parent_id"] == pid].iloc[0]
        x_parent = parent_row[embed_cols].to_numpy(dtype=float)
        proj_parent = float(proj(x_parent[None, :])[0])

        ax.hist(proj(good), bins=25, alpha=0.55, color="#2ca02c", label="good")
        ax.hist(proj(bad), bins=25, alpha=0.55, color="#d62728", label="bad")
        ax.axvline(
            proj_parent, color="black", lw=2, label=f"parent  (axis={proj_parent:.2f})"
        )
        ax.axvline(0, color="#2ca02c", ls=":", lw=1)
        ax.axvline(1, color="#d62728", ls=":", lw=1)
        row = dor.loc[dor["parent_id"] == pid].iloc[0]
        med = row.get("dor_median", np.nan)
        cos = row.get("dor_cosine", np.nan)
        ax.set_title(
            f"{pid} ({row['parent_label']})  dor_median={med:.2f}  dor_cosine={cos:.2f}"
        )
        ax.set_xlabel("Embedding axis  (0 = good centroid, 1 = bad centroid)")
        ax.legend(fontsize=8)

    for ax in axes.flat[len(pids) :]:
        ax.axis("off")
    fig.suptitle("Test-site position on the good→bad embedding axis (diagnostic)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _purge_legacy_outputs(plots_dir, data_dir):
    """Remove output files from previous script versions that this script
    no longer produces. Current outputs are overwritten on write, so they
    do not need pre-emptive deletion. Called only after a successful run
    so a mid-run crash never wipes the prior valid outputs."""
    legacy = [
        os.path.join(data_dir, "test_site_dist_stats.csv"),
        os.path.join(plots_dir, "test_site_dor_by_label.png"),
        os.path.join(plots_dir, "test_site_distance_scatter.png"),
        os.path.join(plots_dir, "test_site_percentile_rank.png"),
        os.path.join(plots_dir, "test_site_dist_distributions.png"),
    ]
    for f in legacy:
        if os.path.exists(f):
            os.remove(f)


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--strategy",
        default=None,
        help="If input has a strategy column, keep only this strategy.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument(
        "--test_sites",
        default=None,
        help="Parquet of parent-point AlphaEarth embeddings "
        "(default: data/test_site_alphaearth_<year>.parquet). "
        "Generate with "
        "scripts/extraction/extract_test_site_embeddings.py",
    )
    parser.add_argument("--plots_dir", default=PLOTS_DIR)
    parser.add_argument(
        "--out_csv", default=os.path.join(DATA_DIR, "test_site_dor.csv")
    )
    parser.add_argument(
        "--plot_prefix",
        default="test_site",
        help="Prefix for plot filenames, e.g. test_site_v2",
    )
    parser.add_argument(
        "--corr_range_m",
        type=float,
        default=300.0,
        help="Fallback correlation range (m) for n_eff if no parent-specific map is provided.",
    )
    parser.add_argument(
        "--corr_range_by_parent",
        default=None,
        help="CSV/JSON mapping parent_id -> corr_range_m (from variogram estimation).",
    )
    parser.add_argument("--n_boot", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"Loading refs from {args.input}")
    refs = load_refs(args.input, strategy=args.strategy)
    if args.strategy and "strategy" in refs.columns:
        print(f"  Strategy filter: {args.strategy} -> {len(refs):,} rows")

    parent_ids = refs["parent_id"].unique().tolist()
    print(f"Parents in refs: {len(parent_ids)}")

    test_sites_path = args.test_sites or default_test_site_parquet(args.year)
    print(f"Loading parent embeddings from {test_sites_path}")
    parents = load_parent_embeddings(test_sites_path, parent_ids)
    print(f"  Got {len(parents)} parent embeddings")

    corr_map = load_corr_range_map(args.corr_range_by_parent)
    if corr_map:
        overlap = len(set(parent_ids).intersection(corr_map.keys()))
        print(
            f"Loaded parent-specific corr ranges for {len(corr_map)} parents "
            f"(overlap with refs: {overlap}/{len(parent_ids)})"
        )

    dor = compute_dor(
        parents,
        refs,
        n_boot=args.n_boot,
        seed=args.seed,
        corr_range_m_default=args.corr_range_m,
        corr_range_by_parent=corr_map,
    )

    print_cols = [
        "parent_id",
        "parent_label",
        "n_good",
        "n_bad",
        "applicable",
        "corr_range_m",
        "n_eff_good",
        "n_eff_bad",
        "n_eff_min",
        "cos_dist_good",
        "cos_dist_bad",
        "dor_median",
        "dor_median_ci_low",
        "dor_median_ci_high",
        "dor_normalised",
        "dor_percentile",
        "dor_cosine",
    ]
    print("\nDoR per test site (primary = dor_median with 95 % CI):")
    print(dor[print_cols].round(4).to_string(index=False))

    n_applicable = int(dor["applicable"].sum())
    n_skipped = len(dor) - n_applicable
    print(
        f"\nApplicable sites: {n_applicable}/{len(dor)}  "
        f"(skipped (no good or no bad refs): {n_skipped})"
    )

    confident = dor["dor_median_ci_low"] > 0.5
    indistinguishable = (dor["dor_median_ci_low"] <= 0.5) & (
        dor["dor_median_ci_high"] >= 0.5
    )
    confident_low = dor["dor_median_ci_high"] < 0.5
    print(f"  CI entirely above 0.5  (confident recovery):    {int(confident.sum())}")
    print(
        f"  CI straddles 0.5       (indistinguishable):     "
        f"{int(indistinguishable.sum())}"
    )
    print(
        f"  CI entirely below 0.5  (confident degradation): {int(confident_low.sum())}"
    )

    print("\nEffective sample size diagnostics:")
    print(
        dor[dor["applicable"]]["n_eff_min"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(2)
        .to_string()
    )

    print("\nMean DoR by parent_label (applicable sites only):")
    print(
        dor[dor["applicable"]]
        .groupby("parent_label")[ALL_SCORES]
        .agg(["mean", "std", "count"])
        .round(3)
    )

    out_csv = args.out_csv
    dor.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")

    for path, fn, extra in [
        (
            os.path.join(args.plots_dir, f"{args.plot_prefix}_dor_with_ci.png"),
            plot_dor_with_ci,
            (dor,),
        ),
        (
            os.path.join(args.plots_dir, f"{args.plot_prefix}_recovery_axis.png"),
            plot_recovery_axis,
            (parents, refs, dor),
        ),
    ]:
        fn(*extra, path)
        print(f"  wrote {path}")

    _purge_legacy_outputs(args.plots_dir, DATA_DIR)


if __name__ == "__main__":
    main()
