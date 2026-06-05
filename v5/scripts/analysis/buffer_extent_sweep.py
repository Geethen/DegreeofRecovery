"""v5 joint inner/outer buffer sweep for DoR.

Companion to v5/scripts/analysis/dor_stability_sweep.py.

dor_stability_sweep.py varies only the *inner* exclusion radius (the minimum
distance between a ref and the parent). This script also varies the *outer*
ceiling — the maximum dist_m a ref can sit at — to answer the buffer-extent
side of the autocorrelation question: how far is "too far" before refs stop
being representative of the parent's ecoregion context?

For each (inner, outer) pair we keep refs with `inner <= dist_m < outer`,
recompute m_g, m_b and DoR per parent, and aggregate by class. The grid is
sweeping the v5 sampler's design space: inner ∈ {0..4 km}, outer ∈ {1..8 km}.

Outputs:
  v5/data/buffer_extent_per_site.csv                  — per-site DoR (+ bootstrap lower/upper 95% CI) at each (inner, outer)
  v5/data/buffer_extent_summary.csv                   — per-class median DoR / m_g / m_b at each (inner, outer)
  v5/data/buffer_extent_ghm_corr.csv                  — per-class Spearman ρ(DoR_point/lo/hi, parent GHM) at each (inner, outer)
  v5/plots/buffer_extent_dor_heatmap.png/.pdf         — per-class heatmap of median DoR vs (inner, outer)
  v5/plots/buffer_extent_count_heatmap.png/.pdf       — per-class heatmap of paired-site count vs (inner, outer)
  v5/plots/buffer_extent_ghm_corr_heatmap.png/.pdf    — per-class heatmap of Spearman ρ(DoR, GHM) vs (inner, outer)
  v5/plots/buffer_extent_ghm_corr_lo_heatmap.png/.pdf — per-class heatmap of Spearman ρ(DoR_lower_CI, GHM) [conservative]
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
V5_DATA  = BASE_DIR / "v5" / "data"
V5_PLOTS = BASE_DIR / "v5" / "plots"

STABLE_REFS_PATH = V5_DATA / "v5_stable_refs_alphaearth.parquet"
TEST_SITES_STAB  = BASE_DIR / "v4" / "data" / "test_site_alphaearth_2024_v4.parquet"

# Candidate (loss-site) refs + parent embeddings — added by the v5 candidate
# extraction (see retry_candidate_extraction.sh). Optional: the sweep includes
# built_loss/crop_loss only when BOTH files are present and --candidate is passed.
CANDIDATE_REFS_PATH = V5_DATA / "v5_candidate_refs_alphaearth.parquet"
TEST_SITES_CAND     = V5_DATA / "test_site_alphaearth_2024_candidate.parquet"

STRATEGY    = "random_100"
EMBED_COLS  = [f"A{i:02d}" for i in range(64)]
EPS         = 1e-12

INNER_THRESHOLDS = [0, 500, 1000, 2000, 3000, 4000]
OUTER_THRESHOLDS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

KNN_K       = 5
MIN_REFS    = 5

# Bootstrap settings for per-site DoR confidence intervals
N_BOOT      = 500
CI_LO_Q     = 0.025
CI_HI_Q     = 0.975
BOOT_SEED   = 42

PALETTE = {
    "stable_nature": "#009E73",
    "stable_crop":   "#0072B2",
    "stable_built":  "#E69F00",
    "built_loss":    "#D55E00",
    "crop_loss":     "#CC79A7",
}


def cosine_dists(x: np.ndarray, pts: np.ndarray) -> np.ndarray:
    nx  = np.linalg.norm(x) + EPS
    nps = np.linalg.norm(pts, axis=1) + EPS
    return 1.0 - (pts @ x) / (nps * nx)


def knn_components(d_g: np.ndarray, d_b: np.ndarray, k: int = KNN_K
                   ) -> tuple[float, float, float]:
    kk = min(k, len(d_g), len(d_b))
    if kk < 1:
        return (float("nan"), float("nan"), float("nan"))
    mg = float(np.mean(np.partition(d_g, kk - 1)[:kk]))
    mb = float(np.mean(np.partition(d_b, kk - 1)[:kk]))
    dor = mb / (mg + mb + EPS)
    return mg, mb, dor


def load_refs(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state", "dist_m", "ghm_aa"] + EMBED_COLS)
    df = con.execute(
        f"SELECT {cols} FROM read_parquet(?) WHERE strategy = ?",
        [str(path), STRATEGY],
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])]
    df = df.dropna(subset=["parent_id", "ref_state", "dist_m"] + EMBED_COLS).reset_index(drop=True)
    return df


def per_parent_ghm(refs: pd.DataFrame) -> dict[str, float]:
    """Median ghm_aa per parent (one scalar per parent). ghm_aa is sampled per
    ref point at 90 m; the per-parent median is a robust summary of the
    parent's human-modification context."""
    return (
        refs.dropna(subset=["ghm_aa"])
        .groupby("parent_id")["ghm_aa"].median()
        .to_dict()
    )


def load_test_emb(path: Path) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    cols = ", ".join(["parent_id"] + EMBED_COLS)
    df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [str(path)]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df.dropna(subset=["parent_id"] + EMBED_COLS).reset_index(drop=True)
    return {
        pid: g[EMBED_COLS].to_numpy(dtype=float).mean(axis=0)
        for pid, g in df.groupby("parent_id", sort=False)
    }


def _boot_dor_ci(d_g: np.ndarray, d_b: np.ndarray, rng: np.random.Generator,
                 n_boot: int = N_BOOT, k: int = KNN_K
                 ) -> tuple[float, float]:
    """Return (lower, upper) percentiles of the DoR bootstrap distribution.
    Resamples good/bad cosine-distance arrays with replacement, recomputes
    DoR = m_b / (m_g + m_b + EPS) per replicate."""
    if len(d_g) < k or len(d_b) < k:
        return (float("nan"), float("nan"))
    # Pre-allocate the resample index matrices once
    bg = rng.integers(0, len(d_g), size=(n_boot, len(d_g)))
    bb = rng.integers(0, len(d_b), size=(n_boot, len(d_b)))
    dgs = d_g[bg]
    dbs = d_b[bb]
    # k-NN means per replicate (vectorised partition)
    mg = np.mean(np.partition(dgs, k - 1, axis=1)[:, :k], axis=1)
    mb = np.mean(np.partition(dbs, k - 1, axis=1)[:, :k], axis=1)
    dor = mb / (mg + mb + EPS)
    return float(np.quantile(dor, CI_LO_Q)), float(np.quantile(dor, CI_HI_Q))


def score_grid(refs: pd.DataFrame, test_emb: dict[str, np.ndarray],
               inners: list[int], outers: list[int],
               with_ci: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED) if with_ci else None
    rows = []
    for pid, grp in refs.groupby("parent_id", sort=False):
        if pid not in test_emb:
            continue
        x = test_emb[pid]
        lbl = str(grp["parent_label"].iloc[0])
        dist = grp["dist_m"].to_numpy()
        state = grp["ref_state"].to_numpy()
        emb = grp[EMBED_COLS].to_numpy(dtype=float)
        cdist = cosine_dists(x, emb)

        for inr in inners:
            for outr in outers:
                if outr <= inr:
                    continue
                keep = (dist >= inr) & (dist < outr)
                dg = cdist[keep & (state == "good")]
                db = cdist[keep & (state == "bad")]
                row = {
                    "parent_id": pid, "parent_label": lbl,
                    "inner_m": int(inr), "outer_m": int(outr),
                    "n_good": int(len(dg)), "n_bad": int(len(db)),
                }
                if len(dg) < MIN_REFS or len(db) < MIN_REFS:
                    row.update({"m_g": float("nan"), "m_b": float("nan"),
                                "dor": float("nan")})
                    if with_ci:
                        row.update({"dor_lo": float("nan"), "dor_hi": float("nan")})
                    rows.append(row)
                    continue
                mg, mb, dor = knn_components(dg, db)
                row.update({"m_g": mg, "m_b": mb, "dor": dor})
                if with_ci:
                    lo, hi = _boot_dor_ci(dg, db, rng)
                    row.update({"dor_lo": lo, "dor_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)


def ghm_correlations(per_site: pd.DataFrame, ghm: dict[str, float]) -> pd.DataFrame:
    """Per (class, inner, outer): Spearman corr of DoR (point estimate AND
    bootstrap lower/upper CI) vs per-parent median ghm_aa. The lower-CI
    correlation is the conservative estimate — it answers 'even in the
    worst-plausible-case for each parent, does DoR still rank against GHM?'"""
    from scipy.stats import pearsonr, spearmanr
    per_site = per_site.copy()
    per_site["ghm_aa_parent"] = per_site["parent_id"].map(ghm)
    has_ci = "dor_lo" in per_site.columns
    rows = []
    for (lbl, inr, outr), g in per_site.groupby(
        ["parent_label", "inner_m", "outer_m"], sort=True
    ):
        v = g.dropna(subset=["dor", "ghm_aa_parent"])
        if len(v) < 10:
            d = {
                "parent_label": lbl, "inner_m": int(inr), "outer_m": int(outr),
                "n": int(len(v)),
                "spearman_dor_ghm": float("nan"), "spearman_p": float("nan"),
                "pearson_dor_ghm":  float("nan"), "pearson_p":  float("nan"),
            }
            if has_ci:
                d.update({"spearman_dor_lo_ghm": float("nan"),
                          "spearman_dor_lo_p": float("nan"),
                          "spearman_dor_hi_ghm": float("nan"),
                          "spearman_dor_hi_p": float("nan")})
            rows.append(d)
            continue
        rs, ps = spearmanr(v["dor"], v["ghm_aa_parent"])
        rp, pp = pearsonr(v["dor"], v["ghm_aa_parent"])
        d = {
            "parent_label": lbl, "inner_m": int(inr), "outer_m": int(outr),
            "n": int(len(v)),
            "spearman_dor_ghm": float(rs), "spearman_p": float(ps),
            "pearson_dor_ghm":  float(rp), "pearson_p":  float(pp),
        }
        if has_ci:
            vlo = v.dropna(subset=["dor_lo"])
            vhi = v.dropna(subset=["dor_hi"])
            if len(vlo) >= 10:
                rlo, plo = spearmanr(vlo["dor_lo"], vlo["ghm_aa_parent"])
            else:
                rlo, plo = float("nan"), float("nan")
            if len(vhi) >= 10:
                rhi, phi = spearmanr(vhi["dor_hi"], vhi["ghm_aa_parent"])
            else:
                rhi, phi = float("nan"), float("nan")
            d.update({
                "spearman_dor_lo_ghm": float(rlo), "spearman_dor_lo_p": float(plo),
                "spearman_dor_hi_ghm": float(rhi), "spearman_dor_hi_p": float(phi),
            })
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["parent_label", "inner_m", "outer_m"]).reset_index(drop=True)


def summarise(per_site: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lbl, inr, outr), g in per_site.groupby(
        ["parent_label", "inner_m", "outer_m"], sort=True
    ):
        valid = g.dropna(subset=["dor"])
        rows.append({
            "parent_label": lbl,
            "inner_m": int(inr),
            "outer_m": int(outr),
            "n_sites_valid": int(len(valid)),
            "n_sites_total": int(len(g)),
            "median_dor": float(valid["dor"].median()) if len(valid) else float("nan"),
            "median_m_g": float(valid["m_g"].median()) if len(valid) else float("nan"),
            "median_m_b": float(valid["m_b"].median()) if len(valid) else float("nan"),
            "median_n_good": float(valid["n_good"].median()) if len(valid) else float("nan"),
            "median_n_bad":  float(valid["n_bad"].median())  if len(valid) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["parent_label", "inner_m", "outer_m"]).reset_index(drop=True)


def plot_heatmaps(summary: pd.DataFrame, value: str, title: str,
                  out_png: Path, out_pdf: Path, fmt: str = "{:.3f}") -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42,
    })
    labels = sorted(summary["parent_label"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.0),
                             sharey=True)
    if len(labels) == 1:
        axes = [axes]
    # shared colour range across panels for visual comparability
    vmin = summary[value].min()
    vmax = summary[value].max()
    for ax, lbl in zip(axes, labels):
        sub = summary[summary["parent_label"] == lbl]
        pivot = sub.pivot(index="inner_m", columns="outer_m", values=value)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c)/1000:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(r)/1000:g}" for r in pivot.index])
        ax.set_xlabel("Outer ceiling (km)")
        if ax is axes[0]:
            ax.set_ylabel("Inner exclusion (km)")
        ax.set_title(lbl.replace("stable_", "").title())
        # annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if (v - vmin) / (vmax - vmin + EPS) < 0.55 else "black",
                        fontsize=6.5)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", action="store_true",
                    help="also include v5 candidate (built_loss/crop_loss) refs + "
                         "their parent embeddings, if both parquets are present")
    args = ap.parse_args()

    V5_DATA.mkdir(parents=True, exist_ok=True)
    V5_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading stable refs ...")
    refs = load_refs(STABLE_REFS_PATH)
    print(f"  {len(refs):,} rows, {refs['parent_id'].nunique()} parents")

    print("Loading test-site embeddings ...")
    test_emb = load_test_emb(TEST_SITES_STAB)
    print(f"  {len(test_emb)} test-site embeddings")

    if args.candidate:
        if CANDIDATE_REFS_PATH.exists() and TEST_SITES_CAND.exists():
            print("Loading candidate (loss-site) refs + parent embeddings ...")
            cand = load_refs(CANDIDATE_REFS_PATH)
            refs = pd.concat([refs, cand], ignore_index=True)
            test_emb.update(load_test_emb(TEST_SITES_CAND))
            print(f"  +{len(cand):,} candidate ref rows, "
                  f"{cand['parent_id'].nunique()} loss parents; "
                  f"{len(test_emb)} total test-site embeddings")
        else:
            miss = [str(p) for p in (CANDIDATE_REFS_PATH, TEST_SITES_CAND) if not p.exists()]
            print(f"  [skip] --candidate requested but missing: {miss}")

    print(f"\nScoring grid: {len(INNER_THRESHOLDS)} inner × {len(OUTER_THRESHOLDS)} outer ...")
    print(f"  Bootstrap CI: B={N_BOOT}, {int((CI_HI_Q-CI_LO_Q)*100)}% CI")
    per_site = score_grid(refs, test_emb, INNER_THRESHOLDS, OUTER_THRESHOLDS,
                          with_ci=True)
    per_site_out = V5_DATA / "buffer_extent_per_site.csv"
    per_site.to_csv(per_site_out, index=False)
    print(f"  Wrote {per_site_out}  ({len(per_site):,} rows)")

    summary = summarise(per_site)
    summary_out = V5_DATA / "buffer_extent_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"  Wrote {summary_out}")

    print("\nComputing per-parent median GHM ...")
    ghm = per_parent_ghm(refs)
    print(f"  {len(ghm)} parents with ghm_aa")

    corr = ghm_correlations(per_site, ghm)
    # Join paired-site count from summary so we can require ≥90% retention
    corr = corr.merge(
        summary[["parent_label", "inner_m", "outer_m", "n_sites_valid"]],
        on=["parent_label", "inner_m", "outer_m"],
        how="left",
    )
    # Per-class desirability rank: prefer most-negative ρ; tie-break on count
    corr["rank_in_class"] = (
        corr.groupby("parent_label")
        .apply(lambda g: g.assign(_r=(
            g["spearman_dor_ghm"].rank(ascending=True, method="min") * 10000
            - g["n_sites_valid"].fillna(0).rank(ascending=False, method="min")
        ))["_r"].rank(method="min").astype(int))
        .reset_index(level=0, drop=True)
    )
    corr_out = V5_DATA / "buffer_extent_ghm_corr.csv"
    corr.to_csv(corr_out, index=False)
    print(f"  Wrote {corr_out}")

    print("\nPer-class Spearman ρ(DoR, GHM) at (inner_m, outer_m):")
    for lbl in sorted(corr["parent_label"].unique()):
        sub = corr[corr["parent_label"] == lbl]
        piv = sub.pivot(index="inner_m", columns="outer_m", values="spearman_dor_ghm")
        print(f"\n  {lbl}")
        print(piv.round(3).to_string())

    if "spearman_dor_lo_ghm" in corr.columns:
        print("\nPer-class Spearman ρ(DoR_lower_CI, GHM) at (inner_m, outer_m) "
              "[conservative]:")
        for lbl in sorted(corr["parent_label"].unique()):
            sub = corr[corr["parent_label"] == lbl]
            piv = sub.pivot(index="inner_m", columns="outer_m",
                            values="spearman_dor_lo_ghm")
            print(f"\n  {lbl}")
            print(piv.round(3).to_string())

    # Conservative ρ column: dor_lo if available, else fall back to point estimate
    rho_col = "spearman_dor_lo_ghm" if "spearman_dor_lo_ghm" in corr.columns else "spearman_dor_ghm"
    rho_label = "ρ_lo" if rho_col == "spearman_dor_lo_ghm" else "ρ"

    # ---- Per-class winners under "strong-negative ρ_lo, ≥90% paired sites" --
    print(f"\nPer-class winners (most-negative {rho_label} at ≥90% paired-site retention):")
    winners = {}
    for lbl in sorted(corr["parent_label"].unique()):
        sub = corr[corr["parent_label"] == lbl].dropna(subset=[rho_col])
        max_n = sub["n_sites_valid"].max()
        elig = sub[sub["n_sites_valid"] >= 0.90 * max_n]
        if elig.empty:
            print(f"  {lbl:<16}: no cell meets the 90% retention floor")
            continue
        row = elig.sort_values(rho_col).iloc[0]
        winners[lbl] = (int(row["inner_m"]), int(row["outer_m"]))
        print(
            f"  {lbl:<16}: inner={int(row['inner_m']):>4}m, outer={int(row['outer_m']):>4}m  "
            f"{rho_label}={row[rho_col]:+.3f}  "
            f"(point ρ={row['spearman_dor_ghm']:+.3f})  n={int(row['n_sites_valid'])}"
        )

    print(f"\nTop-5 cells per class (lowest {rho_label} first):")
    for lbl in sorted(corr["parent_label"].unique()):
        sub = (
            corr[corr["parent_label"] == lbl]
            .dropna(subset=[rho_col])
            .sort_values(rho_col)
            .head(5)
        )
        print(f"\n  {lbl}")
        for _, r in sub.iterrows():
            print(
                f"    inner={int(r['inner_m']):>4}m, outer={int(r['outer_m']):>4}m  "
                f"{rho_label}={r[rho_col]:+.3f}  point ρ={r['spearman_dor_ghm']:+.3f}  "
                f"p={r['spearman_p']:.2g}  "
                f"n={int(r['n_sites_valid']) if pd.notna(r['n_sites_valid']) else 0}"
            )

    # ---- Single-config compromise: minimise mean rank across classes ---------
    # Rebuild rank using the conservative ρ
    corr["rank_in_class_lo"] = (
        corr.groupby("parent_label")[rho_col]
        .rank(ascending=True, method="min")
    )
    print(f"\nSingle-config compromise (averaged rank-by-{rho_label} across classes):")
    avg_rank = (
        corr.dropna(subset=[rho_col])
        .groupby(["inner_m", "outer_m"])["rank_in_class_lo"]
        .mean()
        .sort_values()
    )
    print(f"  Top 5 (inner, outer) configs by average rank-by-{rho_label}:")
    for (inr, outr), ar in avg_rank.head(5).items():
        cells = corr[(corr["inner_m"] == inr) & (corr["outer_m"] == outr)].dropna(subset=[rho_col])
        per_class = {
            r["parent_label"]: r[rho_col] for _, r in cells.iterrows()
        }
        rho_str = "  ".join(f"{k.replace('stable_',''):<6}={v:+.3f}" for k, v in sorted(per_class.items()))
        print(f"    inner={int(inr):>4}m, outer={int(outr):>4}m  avg_rank={ar:5.1f}   {rho_str}")

    print("\nPer-class median DoR at (inner_m, outer_m):")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[summary["parent_label"] == lbl]
        pivot = sub.pivot(index="inner_m", columns="outer_m", values="median_dor")
        print(f"\n  {lbl}")
        print(pivot.round(4).to_string())

    print("\nWriting heatmaps ...")
    plot_heatmaps(summary, "median_dor",
                  "Median DoR by (inner exclusion, outer ceiling)",
                  V5_PLOTS / "buffer_extent_dor_heatmap.png",
                  V5_PLOTS / "buffer_extent_dor_heatmap.pdf",
                  fmt="{:.3f}")
    print(f"  Wrote {V5_PLOTS / 'buffer_extent_dor_heatmap.png'}")

    plot_heatmaps(summary, "n_sites_valid",
                  "Paired-site count (DoR computable) by (inner, outer)",
                  V5_PLOTS / "buffer_extent_count_heatmap.png",
                  V5_PLOTS / "buffer_extent_count_heatmap.pdf",
                  fmt="{:.0f}")
    print(f"  Wrote {V5_PLOTS / 'buffer_extent_count_heatmap.png'}")

    plot_heatmaps(corr, "spearman_dor_ghm",
                  "Spearman ρ(DoR, parent GHM) by (inner, outer)",
                  V5_PLOTS / "buffer_extent_ghm_corr_heatmap.png",
                  V5_PLOTS / "buffer_extent_ghm_corr_heatmap.pdf",
                  fmt="{:+.2f}")
    print(f"  Wrote {V5_PLOTS / 'buffer_extent_ghm_corr_heatmap.png'}")

    if "spearman_dor_lo_ghm" in corr.columns:
        plot_heatmaps(corr, "spearman_dor_lo_ghm",
                      "Spearman ρ(DoR_lower_CI, parent GHM) [conservative]",
                      V5_PLOTS / "buffer_extent_ghm_corr_lo_heatmap.png",
                      V5_PLOTS / "buffer_extent_ghm_corr_lo_heatmap.pdf",
                      fmt="{:+.2f}")
        print(f"  Wrote {V5_PLOTS / 'buffer_extent_ghm_corr_lo_heatmap.png'}")


if __name__ == "__main__":
    main()
