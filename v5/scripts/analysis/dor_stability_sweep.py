"""v5 DoR-stability sweep across minimum exclusion radii.

Question (refining v5/scripts/analysis/min_distance_exclusion_sweep.py):
  100 m preserves sample sizes, but does it stabilise the DoR score?
  The v4 distance-stratified analysis (v4/scripts/analysis/distance_stratified_dor.py)
  showed median built-loss DoR rising by +0.029 from 0 -> 2 km, driven by m_b
  growing faster than m_g — the signature of bad-pool pseudoreplication. That
  analysis used coarse steps (0, 500, 1000, 2000, 3000 m). It cannot answer
  where the curve plateaus between 0 and 500 m.

  This script re-scores every site at a finer grid (0, 25, 50, 75, 100, 150,
  200, 300, 500, 1000, 2000 m), keeps the per-site pairing, decomposes into
  m_g / m_b, and reports the smallest threshold beyond which the median delta
  DoR no longer moves materially.

Tolerance for "stable": per-class |median(DoR_thr - DoR_thr_next_step)| < 0.005
  AND the cumulative shift relative to that step is < 0.005. (0.005 is well
  below the per-site bootstrap SE the v4 scorer reports.)

Outputs:
  v5/data/dor_stability_per_site.csv          -- per-site DoR/m_g/m_b at each threshold
  v5/data/dor_stability_summary.csv           -- per-class median DoR, m_g, m_b vs threshold
  v5/data/dor_stability_deltas.csv            -- per-class step-deltas and cumulative shifts from the largest threshold
  v5/plots/dor_stability_curves.png/.pdf      -- median DoR / m_g / m_b vs threshold (three panels)
  v5/plots/dor_stability_built_loss.png/.pdf  -- per-site DoR trajectories for built-loss
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
V5_DATA  = BASE_DIR / "v5" / "data"
V5_PLOTS = BASE_DIR / "v5" / "plots"

STABLE_REFS_PATH = BASE_DIR / "v5" / "data" / "v5_stable_refs_alphaearth.parquet"
CAND_REFS_PATH   = BASE_DIR / "v2" / "data" / "recover_reference_samples_v2_mask_on_large_alphaearth.parquet"
TEST_SITES_CAND  = BASE_DIR / "v1" / "data" / "test_site_alphaearth_2024.parquet"
TEST_SITES_STAB  = BASE_DIR / "v4" / "data" / "test_site_alphaearth_2024_v4.parquet"

STRATEGY    = "random_100"
EMBED_COLS  = [f"A{i:02d}" for i in range(64)]
EPS         = 1e-12

THRESHOLDS  = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 3000, 4000]
KNN_K       = 5
MIN_REFS    = 5     # below this, the score is not defined at knn5

STABILITY_TOL = 0.005   # |delta median DoR| below this counts as plateaued

PALETTE = {
    "stable_nature": "#009E73",
    "stable_built":  "#E69F00",
    "built_loss":    "#D55E00",
}
LABEL_NAMES = {
    "stable_nature": "Stable near-natural",
    "stable_built":  "Stable built-up",
    "built_loss":    "Built-loss (candidate)",
}


# ---------------------------------------------------------------------------
# Scoring helpers (inlined to keep v5 self-contained)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_refs_with_emb(path: Path, strategy: str | None) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Score at each threshold (per site, paired)
# ---------------------------------------------------------------------------

def score_all_thresholds(refs: pd.DataFrame, test_emb: dict[str, np.ndarray],
                        thresholds: list[int]) -> pd.DataFrame:
    """For each (parent_id, threshold): compute m_g, m_b, DoR.
    NaNs where either pool has < MIN_REFS surviving refs.
    """
    rows = []
    for pid, grp in refs.groupby("parent_id", sort=False):
        if pid not in test_emb:
            continue
        x = test_emb[pid]
        lbl = str(grp["parent_label"].iloc[0])
        dist = grp["dist_m"].to_numpy()
        state = grp["ref_state"].to_numpy()
        emb = grp[EMBED_COLS].to_numpy(dtype=float)
        # Pre-compute cosine distances once
        cdist = cosine_dists(x, emb)

        for thr in thresholds:
            keep = dist >= thr
            dg = cdist[keep & (state == "good")]
            db = cdist[keep & (state == "bad")]
            if len(dg) < MIN_REFS or len(db) < MIN_REFS:
                rows.append({
                    "parent_id": pid, "parent_label": lbl, "min_dist_m": int(thr),
                    "n_good": int(len(dg)), "n_bad": int(len(db)),
                    "m_g": float("nan"), "m_b": float("nan"), "dor": float("nan"),
                })
                continue
            mg, mb, dor = knn_components(dg, db)
            rows.append({
                "parent_id": pid, "parent_label": lbl, "min_dist_m": int(thr),
                "n_good": int(len(dg)), "n_bad": int(len(db)),
                "m_g": mg, "m_b": mb, "dor": dor,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary + plateau detection
# ---------------------------------------------------------------------------

def summarise(per_site: pd.DataFrame) -> pd.DataFrame:
    """Per-(class, threshold) median DoR, m_g, m_b on the *paired* subset of
    sites that have a defined DoR at every threshold (so deltas are honest).
    """
    rows = []
    for lbl, grp in per_site.groupby("parent_label", sort=True):
        # Identify sites that survive every threshold
        pivot_dor = grp.pivot(index="parent_id", columns="min_dist_m", values="dor")
        paired = pivot_dor.dropna()
        paired_ids = set(paired.index.tolist())
        sub = grp[grp["parent_id"].isin(paired_ids)]
        for thr, g in sub.groupby("min_dist_m", sort=True):
            rows.append({
                "parent_label":  lbl,
                "min_dist_m":    int(thr),
                "n_sites_paired": int(len(paired_ids)),
                "median_dor":    float(g["dor"].median()),
                "p25_dor":       float(g["dor"].quantile(0.25)),
                "p75_dor":       float(g["dor"].quantile(0.75)),
                "median_m_g":    float(g["m_g"].median()),
                "median_m_b":    float(g["m_b"].median()),
                "median_n_good": float(g["n_good"].median()),
                "median_n_bad":  float(g["n_bad"].median()),
            })
    return pd.DataFrame(rows).sort_values(["parent_label", "min_dist_m"]).reset_index(drop=True)


def compute_deltas(summary: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """Per-(class, threshold): step-delta (this - previous step) and
    residual-to-largest (this - last threshold). Both based on paired medians.
    """
    rows = []
    last_thr = max(thresholds)
    for lbl, grp in summary.groupby("parent_label", sort=True):
        s = grp.sort_values("min_dist_m").reset_index(drop=True)
        end_dor = float(s.loc[s["min_dist_m"] == last_thr, "median_dor"].iloc[0])
        for i, r in s.iterrows():
            thr = int(r["min_dist_m"])
            step = float(r["median_dor"] - s["median_dor"].iloc[i - 1]) if i > 0 else 0.0
            residual = float(r["median_dor"] - end_dor)
            rows.append({
                "parent_label":  lbl,
                "min_dist_m":    thr,
                "median_dor":    float(r["median_dor"]),
                "step_delta":    step,
                "residual_to_max": residual,
            })
    return pd.DataFrame(rows)


def plateau_threshold(deltas: pd.DataFrame, tol: float) -> dict[str, int | None]:
    """For each class, return the smallest threshold T such that:
      - every step from T onward has |step_delta| < tol,
      - |residual_to_max| at T is < tol.
    Returns None if no such T exists in the grid.
    """
    out: dict[str, int | None] = {}
    for lbl, grp in deltas.groupby("parent_label", sort=True):
        g = grp.sort_values("min_dist_m").reset_index(drop=True)
        chosen = None
        for i in range(len(g)):
            tail = g.iloc[i:]
            steps_ok = (tail["step_delta"].abs() < tol).all()
            res_ok = abs(g["residual_to_max"].iloc[i]) < tol
            if steps_ok and res_ok:
                chosen = int(g["min_dist_m"].iloc[i])
                break
        out[lbl] = chosen
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(summary: pd.DataFrame, deltas: pd.DataFrame,
                plateaus: dict[str, int | None],
                out_png: Path, out_pdf: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })
    labels = sorted(summary["parent_label"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    fig.subplots_adjust(wspace=0.3)

    # Panel A: median DoR vs threshold (per class)
    axA = axes[0]
    for lbl in labels:
        sub = summary[summary["parent_label"] == lbl].sort_values("min_dist_m")
        color = PALETTE.get(lbl, "#333333")
        axA.plot(sub["min_dist_m"], sub["median_dor"], marker="o", markersize=3.5,
                 color=color, label=LABEL_NAMES.get(lbl, lbl))
        axA.fill_between(sub["min_dist_m"], sub["p25_dor"], sub["p75_dor"],
                         color=color, alpha=0.10, linewidth=0)
        thr = plateaus.get(lbl)
        if thr is not None:
            axA.axvline(thr, color=color, linewidth=0.6, linestyle=":", alpha=0.8)
    axA.set_xlabel("Minimum exclusion radius (m)")
    axA.set_ylabel("Median DoR (knn5)")
    axA.set_title("(a) DoR vs exclusion radius")
    axA.legend(fontsize=5.5, framealpha=0.85)

    # Panel B: step delta in median DoR
    axB = axes[1]
    for lbl in labels:
        sub = deltas[deltas["parent_label"] == lbl].sort_values("min_dist_m")
        axB.plot(sub["min_dist_m"], sub["step_delta"], marker="o", markersize=3.5,
                 color=PALETTE.get(lbl, "#333333"),
                 label=LABEL_NAMES.get(lbl, lbl))
    axB.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    axB.axhline(STABILITY_TOL, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
    axB.axhline(-STABILITY_TOL, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
    axB.set_xlabel("Minimum exclusion radius (m)")
    axB.set_ylabel("Step delta in median DoR")
    axB.set_title(f"(b) Step delta (|delta| < {STABILITY_TOL} = stable)")
    axB.legend(fontsize=5.5, framealpha=0.85)

    # Panel C: m_g / m_b decomposition (built-loss only — that's where the
    # asymmetric drift happens)
    axC = axes[2]
    for lbl in labels:
        sub = summary[summary["parent_label"] == lbl].sort_values("min_dist_m")
        color = PALETTE.get(lbl, "#333333")
        # m_b solid, m_g dashed
        axC.plot(sub["min_dist_m"], sub["median_m_b"], marker="o", markersize=3.0,
                 color=color, linestyle="-",
                 label=f"{LABEL_NAMES.get(lbl, lbl)}: m_b")
        axC.plot(sub["min_dist_m"], sub["median_m_g"], marker="s", markersize=3.0,
                 color=color, linestyle="--", alpha=0.7,
                 label=f"{LABEL_NAMES.get(lbl, lbl)}: m_g")
    axC.set_xlabel("Minimum exclusion radius (m)")
    axC.set_ylabel("Median cosine distance (knn5 mean)")
    axC.set_title("(c) m_g (dashed) / m_b (solid) decomposition")
    axC.legend(fontsize=4.8, framealpha=0.85, ncol=1)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_built_loss_trajectories(per_site: pd.DataFrame,
                                 out_png: Path, out_pdf: Path) -> None:
    """Per-site DoR trajectories for built-loss only, with overall median."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "lines.linewidth": 1.0, "pdf.fonttype": 42,
    })

    sub = per_site[per_site["parent_label"] == "built_loss"].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="parent_id", columns="min_dist_m", values="dor")
    pivot = pivot.dropna()
    if pivot.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    thrs = sorted([int(t) for t in pivot.columns])
    for pid, row in pivot.iterrows():
        ax.plot(thrs, [row[t] for t in thrs],
                color=PALETTE["built_loss"], alpha=0.15, linewidth=0.6)
    median_curve = [pivot[t].median() for t in thrs]
    ax.plot(thrs, median_curve, color="black", linewidth=1.8, marker="o",
            markersize=4, label="Median")
    ax.axhline(0.5, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Minimum exclusion radius (m)")
    ax.set_ylabel("DoR (knn5)")
    ax.set_title(f"Built-loss DoR vs exclusion radius (n={len(pivot)})")
    ax.legend(fontsize=6, framealpha=0.85)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    V5_DATA.mkdir(parents=True, exist_ok=True)
    V5_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading stable refs (with embeddings) ...")
    refs_stable = load_refs_with_emb(STABLE_REFS_PATH, strategy=STRATEGY)
    print(f"  {len(refs_stable):,} rows, {refs_stable['parent_id'].nunique()} parents")

    has_cand = CAND_REFS_PATH.exists() and TEST_SITES_CAND.exists()
    if has_cand:
        print("Loading candidate refs (with embeddings) ...")
        refs_cand = load_refs_with_emb(CAND_REFS_PATH, strategy=None)
        print(f"  {len(refs_cand):,} rows, {refs_cand['parent_id'].nunique()} parents")
    else:
        print(f"  [skip] candidate refs/test sites missing — running stable-only sweep")

    print("Loading test-site embeddings ...")
    test_emb_stab = load_test_emb(TEST_SITES_STAB)
    print(f"  stable: {len(test_emb_stab)}")
    if has_cand:
        test_emb_cand = load_test_emb(TEST_SITES_CAND)
        print(f"  candidate: {len(test_emb_cand)}")

    print("\nScoring stable sites at each threshold ...")
    per_site_stab = score_all_thresholds(refs_stable, test_emb_stab, THRESHOLDS)
    print(f"  {per_site_stab['parent_id'].nunique()} sites x {len(THRESHOLDS)} thresholds")

    if has_cand:
        print("Scoring candidate sites at each threshold ...")
        per_site_cand = score_all_thresholds(refs_cand, test_emb_cand, THRESHOLDS)
        print(f"  {per_site_cand['parent_id'].nunique()} sites x {len(THRESHOLDS)} thresholds")
        per_site = pd.concat([per_site_stab, per_site_cand], ignore_index=True)
    else:
        per_site = per_site_stab
    per_site_out = V5_DATA / "dor_stability_per_site.csv"
    per_site.to_csv(per_site_out, index=False)
    print(f"  Wrote {per_site_out}")

    summary = summarise(per_site)
    summary_out = V5_DATA / "dor_stability_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"  Wrote {summary_out}")

    deltas = compute_deltas(summary, THRESHOLDS)
    deltas_out = V5_DATA / "dor_stability_deltas.csv"
    deltas.to_csv(deltas_out, index=False)
    print(f"  Wrote {deltas_out}")

    plateaus = plateau_threshold(deltas, STABILITY_TOL)

    print("\nPer-class median DoR by threshold (paired sites only):")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[summary["parent_label"] == lbl]
        n = int(sub["n_sites_paired"].iloc[0])
        print(f"\n  {lbl}  (n_paired = {n})")
        for _, r in sub.iterrows():
            d = deltas[(deltas["parent_label"] == lbl)
                       & (deltas["min_dist_m"] == r["min_dist_m"])].iloc[0]
            print(f"    {int(r['min_dist_m']):>5} m: "
                  f"med DoR = {r['median_dor']:.4f}  "
                  f"(m_g={r['median_m_g']:.4f}, m_b={r['median_m_b']:.4f})  "
                  f"step delta = {d['step_delta']:+.4f}  "
                  f"residual = {d['residual_to_max']:+.4f}")

    print(f"\nSmallest threshold beyond which |step delta| < {STABILITY_TOL} "
          f"and |residual_to_max| < {STABILITY_TOL}:")
    for lbl, thr in plateaus.items():
        if thr is None:
            print(f"  {lbl:<16}: never plateaus within the tested grid")
        else:
            print(f"  {lbl:<16}: plateau at {thr} m")

    print("\nWriting figures ...")
    plot_curves(summary, deltas, plateaus,
                V5_PLOTS / "dor_stability_curves.png",
                V5_PLOTS / "dor_stability_curves.pdf")
    print(f"  Wrote {V5_PLOTS / 'dor_stability_curves.png'}")

    plot_built_loss_trajectories(per_site,
                                 V5_PLOTS / "dor_stability_built_loss.png",
                                 V5_PLOTS / "dor_stability_built_loss.pdf")
    print(f"  Wrote {V5_PLOTS / 'dor_stability_built_loss.png'}")


if __name__ == "__main__":
    main()
