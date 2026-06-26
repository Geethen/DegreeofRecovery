"""v5 good/bad reference-pool separability sweep across (inner, outer) buffers.

The companion sweeps quantify three of the five buffer goals already:
  - min_distance_exclusion_sweep.py : contamination + sample-size survival
  - dor_stability_sweep.py          : DoR plateau vs inner exclusion
  - buffer_extent_sweep.py          : DoR/GHM decorrelation + bootstrap CI width

The remaining goal -- "separability between the good and bad reference pools" --
was never scored as a function of the buffer. v4 calibrated a decision threshold
by leave-one-out (LOO) scoring of every reference pixel (v4_methods.md, Stage 5)
and picking the cut-off that maximised Youden's J. This script reuses that exact
LOO construction but, per the v5 decision, reports **F1 and MCC** (Matthews
correlation coefficient) at the MCC-optimal threshold instead of Youden's J --
MCC is the more honest single-number summary of 2x2 separability under class
imbalance, and F1 is reported alongside for interpretability.

Definition (pooled across all sites, per class):
  For each parent, build the ref-to-ref cosine-distance matrix once. For each
  (inner, outer) buffer keep refs with `inner <= dist_m < outer`. For every
  surviving ref r:
      m_g(r) = mean of the k=5 smallest cosine distances from r to *other good*
               refs in the buffer (r itself excluded);
      m_b(r) = mean of the k=5 smallest cosine distances from r to *bad* refs;
      s(r)   = m_b / (m_g + m_b)      [the same DoR functional form]
    A known-good ref is a positive (it *should* score high / regenerating); a
  known-bad ref is a negative. Pool s(r) and the true labels across all parents
  of the class, sweep the threshold over the pooled score distribution, and pick
  the threshold maximising MCC. Report F1 and MCC at that threshold, plus the
  threshold-free ROC AUC for reference.

This makes separability a direct function of the buffer: too tight a pool (large
inner, large lower bound) starves the kNN and the score gets noisy; too wide an
outer ceiling pulls in refs from outside the parent's ecoregion context and
blurs the good/bad geometry. The optimum balances both.

Outputs:
  v5/data/separability_per_site.csv   -- per-(parent, inner, outer) LOO scores summary (n, median good/bad score)
  v5/data/separability_summary.csv    -- per-(class, inner, outer) pooled F1, MCC, ROC AUC, opt threshold, n_pos/n_neg
  v5/plots/separability_mcc_heatmap.png  -- per-class heatmap of pooled MCC vs (inner, outer)
  v5/plots/separability_f1_heatmap.png   -- per-class heatmap of pooled F1  vs (inner, outer)
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

# Same design grid as buffer_extent_sweep.py so the metrics line up cell-for-cell
INNER_THRESHOLDS = [0, 500, 1000, 2000, 3000, 4000]
OUTER_THRESHOLDS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

KNN_K    = 5
MIN_REFS = 5          # need >= k+1 good and >= k bad for a defined LOO good-score

PALETTE = {
    "stable_nature": "#009E73",
    "stable_crop":   "#0072B2",
    "stable_built":  "#E69F00",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_refs(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(["parent_id", "parent_label", "ref_state", "dist_m"] + EMBED_COLS)
    df = con.execute(
        f"SELECT {cols} FROM read_parquet(?) WHERE strategy = ?",
        [str(path), STRATEGY],
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str)
    df = df[df["ref_state"].isin(["good", "bad"])]
    df = df.dropna(subset=["parent_id", "ref_state", "dist_m"] + EMBED_COLS).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# LOO scoring
# ---------------------------------------------------------------------------

def _knn_mean_rows(D: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k smallest finite values in each row of D.
    Self-distances must already be set to +inf on the diagonal where relevant."""
    if D.shape[1] == 0:
        return np.full(D.shape[0], np.nan)
    kk = min(k, D.shape[1])
    part = np.partition(D, kk - 1, axis=1)[:, :kk]
    return np.mean(part, axis=1)


def parent_loo_scores(emb: np.ndarray, state: np.ndarray, dist: np.ndarray,
                      inner: float, outer: float, k: int = KNN_K
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) for one parent at one (inner, outer) buffer.

    labels: 1 for known-good (positive), 0 for known-bad (negative).
    scores: s = m_b / (m_g + m_b), the LOO DoR functional value of each ref.
    A known-good ref's m_g excludes itself; a known-bad ref's m_b excludes itself.
    Empty arrays if the buffer leaves < MIN_REFS in either pool.
    """
    keep = (dist >= inner) & (dist < outer)
    gi = np.where(keep & (state == "good"))[0]
    bi = np.where(keep & (state == "bad"))[0]
    if len(gi) < MIN_REFS or len(bi) < MIN_REFS:
        return np.empty(0), np.empty(0)

    Xg = emb[gi]
    Xb = emb[bi]
    Xg = Xg / (np.linalg.norm(Xg, axis=1, keepdims=True) + EPS)
    Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + EPS)

    # cosine distance = 1 - cos similarity
    Dgg = 1.0 - Xg @ Xg.T
    np.fill_diagonal(Dgg, np.inf)            # exclude self for good-to-good
    Dbb = 1.0 - Xb @ Xb.T
    np.fill_diagonal(Dbb, np.inf)            # exclude self for bad-to-bad
    Dgb = 1.0 - Xg @ Xb.T                    # good rows -> bad cols
    Dbg = Dgb.T                              # bad rows  -> good cols

    mg_g = _knn_mean_rows(Dgg, k)            # good ref: dist to other good
    mb_g = _knn_mean_rows(Dgb, k)            # good ref: dist to bad
    mg_b = _knn_mean_rows(Dbg, k)            # bad ref:  dist to good
    mb_b = _knn_mean_rows(Dbb, k)            # bad ref:  dist to other bad

    s_good = mb_g / (mg_g + mb_g + EPS)      # should be high
    s_bad  = mb_b / (mg_b + mb_b + EPS)      # should be low

    scores = np.concatenate([s_good, s_bad])
    labels = np.concatenate([np.ones(len(s_good)), np.zeros(len(s_bad))])
    ok = np.isfinite(scores)
    return scores[ok], labels[ok]


# ---------------------------------------------------------------------------
# Pooled threshold metrics (F1, MCC, ROC AUC)
# ---------------------------------------------------------------------------

def _mcc_from_counts(tp, tn, fp, fn) -> np.ndarray:
    # Cast to float64 first: the fourfold product under the sqrt overflows int64
    # once the pooled count reaches ~10^5 (nature pools ~4x10^5 refs).
    tp = tp.astype(np.float64); tn = tn.astype(np.float64)
    fp = fp.astype(np.float64); fn = fn.astype(np.float64)
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0)
    return out


def _f1_from_counts(tp, fp, fn) -> np.ndarray:
    tp = tp.astype(np.float64); fp = fp.astype(np.float64); fn = fn.astype(np.float64)
    denom = (2.0 * tp + fp + fn)
    return np.where(denom > 0, 2.0 * tp / np.where(denom == 0, 1.0, denom), 0.0)


def pooled_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Sweep all candidate thresholds (midpoints of sorted unique scores), pick
    the one maximising MCC. Return MCC, F1 (at that threshold), the threshold,
    plus threshold-free ROC AUC and base rates. Rule: predict positive (good)
    when score > threshold."""
    n = len(scores)
    n_pos = int(labels.sum())
    n_neg = int(n - n_pos)
    base = {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "mcc": float("nan"), "f1": float("nan"),
        "opt_threshold": float("nan"), "roc_auc": float("nan"),
        "good_score_median": float("nan"), "bad_score_median": float("nan"),
    }
    if n_pos < 5 or n_neg < 5:
        return base

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order].astype(np.int64)

    pos_total = y_sorted.sum()
    neg_total = len(y_sorted) - pos_total

    # cumulative counts of pos/neg at-or-below each sorted position
    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)

    # Candidate thresholds between distinct scores: everything strictly above
    # index i is predicted positive. Evaluate at each i (predict + for >score[i]).
    # TP = pos above i, FP = neg above i, FN = pos at-or-below, TN = neg at-or-below
    tp = pos_total - cum_pos
    fp = neg_total - cum_neg
    fn = cum_pos
    tn = cum_neg

    mcc = _mcc_from_counts(tp, tn, fp, fn)
    f1  = _f1_from_counts(tp, fp, fn)

    # Only consider split points at score-value boundaries (avoid ties)
    boundary = np.ones(len(s_sorted), dtype=bool)
    boundary[:-1] = s_sorted[1:] != s_sorted[:-1]

    mcc_masked = np.where(boundary, mcc, -np.inf)
    best = int(np.argmax(mcc_masked))
    thr = float(s_sorted[best])

    # ROC AUC via Mann-Whitney U (rank-based), robust + exact for ties
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    auc = float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) /
                (n_pos * n_neg + EPS))

    base.update({
        "mcc": float(mcc[best]),
        "f1": float(f1[best]),
        "opt_threshold": thr,
        "roc_auc": auc,
        "good_score_median": float(np.median(scores[labels == 1])),
        "bad_score_median":  float(np.median(scores[labels == 0])),
    })
    return base


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(refs: pd.DataFrame, inners: list[int], outers: list[int]
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pre-extract per-parent arrays once
    parents = {}
    for (pid, lbl), grp in refs.groupby(["parent_id", "parent_label"], sort=False):
        parents[pid] = (
            lbl,
            grp[EMBED_COLS].to_numpy(dtype=float),
            grp["ref_state"].to_numpy(),
            grp["dist_m"].to_numpy(),
        )

    per_site_rows = []
    # pooled[(lbl, inr, outr)] -> (list_scores, list_labels)
    pooled: dict[tuple, tuple[list, list]] = {}

    n_par = len(parents)
    for j, (pid, (lbl, emb, state, dist)) in enumerate(parents.items(), 1):
        if j % 100 == 0 or j == n_par:
            print(f"    LOO scoring parent {j}/{n_par}")
        for inr in inners:
            for outr in outers:
                if outr <= inr:
                    continue
                sc, lab = parent_loo_scores(emb, state, dist, inr, outr)
                key = (lbl, int(inr), int(outr))
                if key not in pooled:
                    pooled[key] = ([], [])
                if len(sc):
                    pooled[key][0].append(sc)
                    pooled[key][1].append(lab)
                    per_site_rows.append({
                        "parent_id": pid, "parent_label": lbl,
                        "inner_m": int(inr), "outer_m": int(outr),
                        "n_good": int((lab == 1).sum()), "n_bad": int((lab == 0).sum()),
                        "good_score_median": float(np.median(sc[lab == 1])) if (lab == 1).any() else float("nan"),
                        "bad_score_median":  float(np.median(sc[lab == 0])) if (lab == 0).any() else float("nan"),
                    })
                else:
                    per_site_rows.append({
                        "parent_id": pid, "parent_label": lbl,
                        "inner_m": int(inr), "outer_m": int(outr),
                        "n_good": 0, "n_bad": 0,
                        "good_score_median": float("nan"),
                        "bad_score_median": float("nan"),
                    })

    summary_rows = []
    for (lbl, inr, outr), (scs, labs) in sorted(pooled.items()):
        if scs:
            scores = np.concatenate(scs)
            labels = np.concatenate(labs)
        else:
            scores = np.empty(0)
            labels = np.empty(0)
        m = pooled_metrics(scores, labels)
        m.update({"parent_label": lbl, "inner_m": inr, "outer_m": outr,
                  "n_parents": int(len(scs))})
        summary_rows.append(m)

    per_site = pd.DataFrame(per_site_rows)
    summary = pd.DataFrame(summary_rows)
    cols = ["parent_label", "inner_m", "outer_m", "n_parents",
            "n", "n_pos", "n_neg", "mcc", "f1", "roc_auc", "opt_threshold",
            "good_score_median", "bad_score_median"]
    summary = summary[cols].sort_values(["parent_label", "inner_m", "outer_m"]).reset_index(drop=True)
    return per_site, summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmaps(summary: pd.DataFrame, value: str, title: str,
                  out_png: Path, fmt: str = "{:.3f}") -> None:
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
        for i in range(pivot.shape[0]):
            for jj in range(pivot.shape[1]):
                v = pivot.values[i, jj]
                if np.isnan(v):
                    continue
                ax.text(jj, i, fmt.format(v), ha="center", va="center",
                        color="white" if (v - vmin) / (vmax - vmin + EPS) < 0.55 else "black",
                        fontsize=6.5)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", action="store_true",
                    help="also include v5 candidate (built_loss/crop_loss) refs if present")
    args = ap.parse_args()

    V5_DATA.mkdir(parents=True, exist_ok=True)
    V5_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading stable refs (with embeddings) ...")
    refs = load_refs(STABLE_REFS_PATH)
    print(f"  {len(refs):,} rows, {refs['parent_id'].nunique()} parents")

    if args.candidate and CANDIDATE_REFS_PATH.exists():
        print("Loading candidate (loss-site) refs ...")
        cand = load_refs(CANDIDATE_REFS_PATH)
        refs = pd.concat([refs, cand], ignore_index=True)
        print(f"  +{len(cand):,} rows, {cand['parent_id'].nunique()} loss parents")
    elif args.candidate:
        print(f"  [skip] candidate parquet not found at {CANDIDATE_REFS_PATH}")

    print(f"\nLOO separability sweep: {len(INNER_THRESHOLDS)} inner x "
          f"{len(OUTER_THRESHOLDS)} outer, pooled per class ...")
    per_site, summary = run_sweep(refs, INNER_THRESHOLDS, OUTER_THRESHOLDS)

    per_site_out = V5_DATA / "separability_per_site.csv"
    per_site.to_csv(per_site_out, index=False)
    print(f"  Wrote {per_site_out}  ({len(per_site):,} rows)")

    summary_out = V5_DATA / "separability_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"  Wrote {summary_out}")

    print("\nPer-class pooled MCC at (inner_m, outer_m):")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[summary["parent_label"] == lbl]
        piv = sub.pivot(index="inner_m", columns="outer_m", values="mcc")
        print(f"\n  {lbl}")
        print(piv.round(3).to_string())

    print("\nPer-class pooled F1 at (inner_m, outer_m):")
    for lbl in sorted(summary["parent_label"].unique()):
        sub = summary[summary["parent_label"] == lbl]
        piv = sub.pivot(index="inner_m", columns="outer_m", values="f1")
        print(f"\n  {lbl}")
        print(piv.round(3).to_string())

    print("\nWriting heatmaps ...")
    plot_heatmaps(summary, "mcc",
                  "Pooled good/bad separability MCC by (inner, outer)",
                  V5_PLOTS / "separability_mcc_heatmap.png", fmt="{:.3f}")
    print(f"  Wrote {V5_PLOTS / 'separability_mcc_heatmap.png'}")
    plot_heatmaps(summary, "f1",
                  "Pooled good/bad separability F1 by (inner, outer)",
                  V5_PLOTS / "separability_f1_heatmap.png", fmt="{:.3f}")
    print(f"  Wrote {V5_PLOTS / 'separability_f1_heatmap.png'}")


if __name__ == "__main__":
    main()
