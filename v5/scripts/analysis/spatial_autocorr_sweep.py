"""v5 within-pool spatial-autocorrelation sweep across (inner, outer) buffers.

Goal axis: "reduce spatial autocorrelation" -- we want the reference pool used to
score a site to behave like independent samples, not spatially-clustered
pseudoreplicates. The empirical variogram of the AlphaEarth reference embeddings
(measured within each parent) shows cosine similarity between two same-parent refs
decaying steeply with geographic separation up to ~2-3 km, then plateauing:

    band(m)     mean cosine-sim (nature good)
    0-100       0.94
    250-500     0.87
    1000-2000   0.82
    3000-4000   0.80   <- correlation range largely reached
    6000-8000   0.78

So refs closer than ~2-3 km are meaningfully correlated; a larger inner exclusion
drops the most-correlated near pairs and lowers the pool's mean autocorrelation.

This sweep measures, per parent and per (inner, outer) buffer, the MEAN pairwise
embedding cosine similarity among the retained refs (good and bad pooled, and also
per state) -- a direct, model-free index of within-pool spatial autocorrelation.
Lower is better (more independent). It is aggregated to a per-class mean so it can
enter the buffer-desirability hypercube as the spatial-independence axis, replacing
the earlier GHM-correlation proxy (GHM correlation is legitimate signal, not a
nuisance to minimise; within-pool autocorrelation IS a nuisance).

Works for any parent_label present in the input parquet(s): stable_{nature,crop,
built} from the v5 stable refs, and built_loss / crop_loss from the v5 candidate
refs once extracted. Pass --candidate to include the candidate parquet.

Outputs:
  v5/data/spatial_autocorr_per_site.csv  -- per-(parent, inner, outer) mean pair-sim (all/good/bad) + n
  v5/data/spatial_autocorr_summary.csv   -- per-(class, inner, outer) mean within-pool autocorrelation
  v5/plots/spatial_autocorr_heatmap.png  -- per-class heatmap of mean autocorrelation vs (inner, outer)
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

STABLE_REFS_PATH    = V5_DATA / "v5_stable_refs_alphaearth.parquet"
CANDIDATE_REFS_PATH = V5_DATA / "v5_candidate_refs_alphaearth.parquet"

STRATEGY    = "random_100"
EMBED_COLS  = [f"A{i:02d}" for i in range(64)]
EPS         = 1e-12

INNER_THRESHOLDS = [0, 500, 1000, 2000, 3000, 4000]
OUTER_THRESHOLDS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

MIN_REFS    = 5
MAX_PAIRS   = 4000     # cap pairs sampled per (parent, state, cell) for speed
RNG_SEED    = 42

PALETTE = {
    "stable_nature": "#009E73",
    "stable_crop":   "#0072B2",
    "stable_built":  "#E69F00",
    "built_loss":    "#D55E00",
    "crop_loss":     "#CC79A7",
}


def load_refs(path: Path, strategy: str | None) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state", "dist_m"] + EMBED_COLS)
    if strategy:
        df = con.execute(
            f"SELECT {cols} FROM read_parquet(?) WHERE strategy = ?",
            [str(path), strategy],
        ).df()
    else:
        df = con.execute(f"SELECT {cols} FROM read_parquet(?)", [str(path)]).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])]
    df = df.dropna(subset=["parent_id", "ref_state", "dist_m"] + EMBED_COLS).reset_index(drop=True)
    return df


def _mean_pair_sim(Xn: np.ndarray, rng: np.random.Generator,
                   max_pairs: int = MAX_PAIRS) -> float:
    """Mean cosine similarity over pairs of unit-normed rows. Exhaustive when the
    number of pairs is small; random-sampled (with replacement, self-pairs dropped)
    when large. Returns NaN if < 2 rows."""
    n = Xn.shape[0]
    if n < 2:
        return float("nan")
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        S = Xn @ Xn.T
        iu = np.triu_indices(n, k=1)
        return float(np.mean(S[iu]))
    ii = rng.integers(0, n, max_pairs)
    jj = rng.integers(0, n, max_pairs)
    ok = ii != jj
    ii, jj = ii[ok], jj[ok]
    sims = np.einsum("ij,ij->i", Xn[ii], Xn[jj])
    return float(np.mean(sims))


def run_sweep(refs: pd.DataFrame, inners: list[int], outers: list[int]
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    # pre-extract per-parent unit-normed embeddings + dist + state
    parents = {}
    for (pid, lbl), grp in refs.groupby(["parent_id", "parent_label"], sort=False):
        X = grp[EMBED_COLS].to_numpy(dtype=float)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        parents[pid] = (lbl, X, grp["ref_state"].to_numpy(), grp["dist_m"].to_numpy())

    rows = []
    n_par = len(parents)
    for k, (pid, (lbl, X, state, dist)) in enumerate(parents.items(), 1):
        if k % 200 == 0 or k == n_par:
            print(f"    autocorr parent {k}/{n_par}")
        good = state == "good"
        bad = state == "bad"
        for inr in inners:
            for outr in outers:
                if outr <= inr:
                    continue
                keep = (dist >= inr) & (dist < outr)
                ig = keep & good
                ib = keep & bad
                ng, nb = int(ig.sum()), int(ib.sum())
                row = {"parent_id": pid, "parent_label": lbl,
                       "inner_m": int(inr), "outer_m": int(outr),
                       "n_good": ng, "n_bad": nb}
                if ng + nb < MIN_REFS:
                    row.update({"sim_all": float("nan"), "sim_good": float("nan"),
                                "sim_bad": float("nan")})
                    rows.append(row)
                    continue
                row["sim_good"] = _mean_pair_sim(X[ig], rng) if ng >= 2 else float("nan")
                row["sim_bad"]  = _mean_pair_sim(X[ib], rng) if nb >= 2 else float("nan")
                row["sim_all"]  = _mean_pair_sim(X[keep], rng)
                rows.append(row)

    per_site = pd.DataFrame(rows)
    summary = (
        per_site.groupby(["parent_label", "inner_m", "outer_m"])
        .agg(n_sites=("parent_id", "nunique"),
             mean_sim_all=("sim_all", "mean"),
             median_sim_all=("sim_all", "median"),
             mean_sim_good=("sim_good", "mean"),
             mean_sim_bad=("sim_bad", "mean"))
        .reset_index()
        .sort_values(["parent_label", "inner_m", "outer_m"])
        .reset_index(drop=True)
    )
    return per_site, summary


def plot_heatmaps(summary: pd.DataFrame, value: str, title: str, out_png: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    labels = sorted(summary["parent_label"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.0),
                             sharey=True)
    if len(labels) == 1:
        axes = [axes]
    vmin = float(np.nanmin(summary[value]))
    vmax = float(np.nanmax(summary[value]))
    for ax, lbl in zip(axes, labels):
        sub = summary[summary["parent_label"] == lbl]
        pivot = sub.pivot(index="inner_m", columns="outer_m", values=value)
        cmap = plt.get_cmap("magma_r").copy()
        cmap.set_bad("#e8e8e8")
        im = ax.imshow(np.ma.masked_invalid(pivot.values), origin="lower",
                       aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c)/1000:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(r)/1000:g}" for r in pivot.index])
        ax.set_xlabel("Outer ceiling (km)")
        if ax is axes[0]:
            ax.set_ylabel("Inner exclusion (km)")
        ax.set_title(lbl.replace("stable_", "").replace("_", "-").title())
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if (v - vmin) / (vmax - vmin + EPS) > 0.55 else "black",
                        fontsize=6.2)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", action="store_true",
                    help="also include v5 candidate (built_loss/crop_loss) refs if present")
    args = ap.parse_args()

    V5_DATA.mkdir(parents=True, exist_ok=True)
    V5_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading stable refs ...")
    refs = load_refs(STABLE_REFS_PATH, strategy=STRATEGY)
    print(f"  {len(refs):,} rows, {refs['parent_id'].nunique()} parents")

    if args.candidate and CANDIDATE_REFS_PATH.exists():
        print("Loading candidate (loss-site) refs ...")
        cand = load_refs(CANDIDATE_REFS_PATH, strategy=None)
        print(f"  {len(cand):,} rows, {cand['parent_id'].nunique()} parents")
        refs = pd.concat([refs, cand], ignore_index=True)
    elif args.candidate:
        print(f"  [skip] candidate parquet not found at {CANDIDATE_REFS_PATH}")

    print(f"\nSpatial-autocorrelation sweep: {len(INNER_THRESHOLDS)} inner x "
          f"{len(OUTER_THRESHOLDS)} outer ...")
    per_site, summary = run_sweep(refs, INNER_THRESHOLDS, OUTER_THRESHOLDS)

    per_site_out = V5_DATA / "spatial_autocorr_per_site.csv"
    per_site.to_csv(per_site_out, index=False)
    print(f"  Wrote {per_site_out}  ({len(per_site):,} rows)")

    summary_out = V5_DATA / "spatial_autocorr_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"  Wrote {summary_out}")

    print("\nPer-class mean within-pool autocorrelation (sim_all) at (inner_m, outer_m):")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[summary["parent_label"] == lbl]
        piv = sub.pivot(index="inner_m", columns="outer_m", values="mean_sim_all")
        print(f"\n  {lbl}")
        print(piv.round(4).to_string())

    print("\nWriting heatmap ...")
    plot_heatmaps(summary, "mean_sim_all",
                  "Within-pool reference autocorrelation (mean pairwise cosine sim) "
                  "by (inner, outer)\nlower = more spatially independent",
                  V5_PLOTS / "spatial_autocorr_heatmap.png")
    print(f"  Wrote {V5_PLOTS / 'spatial_autocorr_heatmap.png'}")


if __name__ == "__main__":
    main()
