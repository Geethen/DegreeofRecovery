"""Compare degree-of-recovery scoring methods on extracted reference embeddings.

Input
-----
data/recover_reference_samples_alphaearth.parquet — 64 AlphaEarth bands
(A00..A63) plus `ref_state` ('good'/'bad'), `parent_id`, `parent_label`.

Per-parent we have:
  - 'good' = natural-state references (non-built / non-crop / non-water
    WorldCover pixels within a 1 km buffer of the parent)
  - 'bad'  = degraded-state references (built or crop pixels, depending
    on the parent's label; absent for `stable_stable` parents)

A degree-of-recovery (DoR) score should approach 1 for embeddings near the
natural state and 0 for embeddings near the degraded state. We use the
extracted reference points as a labelled validation set: a good scoring
method gives 'good' points high scores and 'bad' points low scores,
quantified by ROC AUC.

Methods compared (all per-parent):
  1. Euclidean centroid ratio (leave-one-out)
  2. Cosine centroid ratio (leave-one-out)
  3. 1-NN class margin
  4. Logistic regression probability (5-fold CV when possible)

Outputs (under plots/):
  recovery_score_hists.png    — score histograms by method & ref_state
  recovery_roc.png            — ROC curves, all methods overlaid
  recovery_per_parent_auc.png — per-parent AUC by method (if >1 parent)
  recovery_pca.png            — PCA scatter for up to 4 parents

Usage
-----
  python scripts/analysis/recovery_score.py
  python scripts/analysis/recovery_score.py --input <path.parquet>
"""

import argparse
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DEFAULT_INPUT = os.path.join(
    DATA_DIR, "recover_reference_samples_alphaearth.parquet")

EMBED_COLS = [f"A{i:02d}" for i in range(64)]


# ── Data loading ────────────────────────────────────────────────────
def load(parquet_path):
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{parquet_path}'").df()
    con.close()

    missing = [c for c in EMBED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing embedding columns: {missing[:5]}...")

    n_before = len(df)
    df = df.dropna(subset=EMBED_COLS).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} rows with NaN bands "
              f"(water/edge pixels)")
    return df


def summarise(df):
    print(f"Rows: {len(df):,}")
    print("\nCounts by parent_label × ref_state:")
    print(df.groupby(['parent_label', 'ref_state']).size()
          .unstack(fill_value=0))
    n_parents = df['parent_id'].nunique()
    print(f"\nUnique parents: {n_parents}")
    print("Parents per label:")
    print(df.groupby('parent_label')['parent_id'].nunique())


# ── Score methods ───────────────────────────────────────────────────
def _safe_norm(v, axis=None, eps=1e-12):
    return np.linalg.norm(v, axis=axis) + eps


def score_centroid_loo(X, labels, metric='euclidean'):
    """Leave-one-out centroid distance ratio (vectorised).

    For each point, recompute its own class centroid excluding itself,
    then score = d(point, bad_centroid) / (d(point, good_centroid) +
    d(point, bad_centroid)). Higher = closer to natural.
    """
    is_good = labels == 'good'
    is_bad = labels == 'bad'
    n_good, n_bad = is_good.sum(), is_bad.sum()
    if n_good < 2 or n_bad < 2:
        return np.full(len(X), np.nan)

    good_sum = X[is_good].sum(axis=0)
    bad_sum = X[is_bad].sum(axis=0)
    c_good_full = good_sum / n_good
    c_bad_full = bad_sum / n_bad

    # Per-row centroid: LOO for the row's own class, full for the other.
    c_g = np.broadcast_to(c_good_full, X.shape).copy()
    c_b = np.broadcast_to(c_bad_full, X.shape).copy()
    c_g[is_good] = (good_sum - X[is_good]) / (n_good - 1)
    c_b[is_bad] = (bad_sum - X[is_bad]) / (n_bad - 1)

    eps = 1e-12
    if metric == 'euclidean':
        d_g = np.linalg.norm(X - c_g, axis=1) + eps
        d_b = np.linalg.norm(X - c_b, axis=1) + eps
    elif metric == 'cosine':
        nx = np.linalg.norm(X, axis=1) + eps
        ng = np.linalg.norm(c_g, axis=1) + eps
        nb = np.linalg.norm(c_b, axis=1) + eps
        d_g = 1.0 - np.einsum('ij,ij->i', X, c_g) / (nx * ng)
        d_b = 1.0 - np.einsum('ij,ij->i', X, c_b) / (nx * nb)
    else:
        raise ValueError(metric)
    return d_b / (d_g + d_b)


def score_median_pairwise(X, labels):
    """Per-point median pairwise distance ratio (LOO).

    For each point, compute cosine distances to every same- and other-class
    reference (excluding self), then:
        score = median(d_bad) / (median(d_good) + median(d_bad))

    Robust to multimodal/heterogeneous reference clouds where centroid
    summaries are unreliable. This is the primary score used for
    test-site DoR (see score_test_sites.py); included here so it can be
    benchmarked against the other methods on the labelled references.
    """
    is_good = labels == 'good'
    is_bad = labels == 'bad'
    n_good, n_bad = is_good.sum(), is_bad.sum()
    if n_good < 2 or n_bad < 2:
        return np.full(len(X), np.nan)

    X_good = X[is_good]
    X_bad = X[is_bad]
    eps = 1e-12
    nx = np.linalg.norm(X, axis=1) + eps
    ng = np.linalg.norm(X_good, axis=1) + eps
    nb = np.linalg.norm(X_bad, axis=1) + eps

    D_g = 1.0 - (X @ X_good.T) / np.outer(nx, ng)   # (N, n_good)
    D_b = 1.0 - (X @ X_bad.T) / np.outer(nx, nb)    # (N, n_bad)

    # Mask self-distance (the 0 entry) for rows that belong to each class.
    good_rows = np.where(is_good)[0]
    bad_rows = np.where(is_bad)[0]
    D_g[good_rows, np.arange(n_good)] = np.nan
    D_b[bad_rows, np.arange(n_bad)] = np.nan

    m_g = np.nanmedian(D_g, axis=1) + eps
    m_b = np.nanmedian(D_b, axis=1) + eps
    return m_b / (m_g + m_b)


def score_knn_margin(X, labels, k=5):
    """k-NN class margin (LOO).

    For each point, find its k nearest neighbours within each class
    (excluding itself), score = d_bad_mean / (d_good_mean + d_bad_mean).
    """
    is_good = labels == 'good'
    is_bad = labels == 'bad'
    if is_good.sum() < k + 1 or is_bad.sum() < k + 1:
        return np.full(len(X), np.nan)

    X_good = X[is_good]
    X_bad = X[is_bad]
    nn_good = NearestNeighbors(n_neighbors=k + 1).fit(X_good)
    nn_bad = NearestNeighbors(n_neighbors=k + 1).fit(X_bad)

    dg, _ = nn_good.kneighbors(X, n_neighbors=k + 1)  # (N, k+1)
    db, _ = nn_bad.kneighbors(X, n_neighbors=k + 1)

    # Drop the nearest neighbour (self, distance 0) for in-class rows;
    # keep the first k for out-of-class rows.
    d_g = np.where(is_good[:, None], dg[:, 1:], dg[:, :k])
    d_b = np.where(is_bad[:, None], db[:, 1:], db[:, :k])

    m_g = d_g.mean(axis=1) + 1e-12
    m_b = d_b.mean(axis=1) + 1e-12
    return m_b / (m_g + m_b)


def score_logreg(X, labels, n_splits=5, random_state=42):
    """Logistic regression P(good) under stratified k-fold CV.

    Falls back to in-sample fit if either class has < n_splits samples.
    """
    y = (labels == 'good').astype(int)
    if y.sum() < 2 or (1 - y).sum() < 2:
        return np.full(len(X), np.nan)

    scores = np.full(len(X), np.nan)
    n_splits = min(n_splits, y.sum(), (1 - y).sum())

    lr_kwargs = dict(solver='liblinear', max_iter=200, tol=1e-3, C=1.0)
    if n_splits < 2:
        clf = LogisticRegression(**lr_kwargs).fit(X, y)
        return clf.predict_proba(X)[:, 1]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(**lr_kwargs).fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return scores


METHODS = {
    'median_pairwise': score_median_pairwise,
    'euclidean':       lambda X, l: score_centroid_loo(X, l, metric='euclidean'),
    'cosine':          lambda X, l: score_centroid_loo(X, l, metric='cosine'),
    'knn_margin':      lambda X, l: score_knn_margin(X, l, k=5),
    'logreg':          score_logreg,
}


# ── Per-parent scoring ──────────────────────────────────────────────
def score_all(df):
    """Compute every method's score per parent. Returns df with new cols."""
    out = df.copy()
    for name in METHODS:
        out[f"score_{name}"] = np.nan

    for _, sub in df.groupby('parent_id'):
        idx = sub.index.to_numpy()
        X = sub[EMBED_COLS].to_numpy()
        labels = sub['ref_state'].to_numpy()

        for name, fn in METHODS.items():
            out.loc[idx, f"score_{name}"] = fn(X, labels)

    return out


# ── AUC summaries ───────────────────────────────────────────────────
def overall_auc(scored, method):
    col = f"score_{method}"
    sub = scored.dropna(subset=[col])
    sub = sub[sub['ref_state'].isin(['good', 'bad'])]
    if sub['ref_state'].nunique() < 2:
        return np.nan
    y = (sub['ref_state'] == 'good').astype(int)
    return roc_auc_score(y, sub[col])


def per_parent_auc(scored, method):
    col = f"score_{method}"
    rows = []
    for pid, sub in scored.groupby('parent_id'):
        sub = sub.dropna(subset=[col])
        if sub['ref_state'].nunique() < 2:
            continue
        y = (sub['ref_state'] == 'good').astype(int)
        rows.append({
            'parent_id': pid,
            'parent_label': sub['parent_label'].iloc[0],
            'auc': roc_auc_score(y, sub[col]),
            'n': len(sub),
        })
    return pd.DataFrame(rows)


# ── Plots ───────────────────────────────────────────────────────────
def plot_score_histograms(scored, out_path):
    methods = list(METHODS)
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(4 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        col = f"score_{m}"
        for state, color in [('bad', '#d62728'), ('good', '#2ca02c')]:
            vals = scored.loc[scored['ref_state'] == state, col].dropna()
            if len(vals):
                ax.hist(vals, bins=30, alpha=0.6, label=state, color=color)
        ax.set_title(f"{m}\nAUC = {overall_auc(scored, m):.3f}")
        ax.set_xlabel("DoR score (1 = natural)")
        ax.legend()
    axes[0].set_ylabel("Count")
    fig.suptitle("Recovery score by method (in-sample LOO)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(scored, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for m in METHODS:
        col = f"score_{m}"
        sub = scored.dropna(subset=[col])
        sub = sub[sub['ref_state'].isin(['good', 'bad'])]
        if sub['ref_state'].nunique() < 2:
            continue
        y = (sub['ref_state'] == 'good').astype(int)
        fpr, tpr, _ = roc_curve(y, sub[col])
        a = roc_auc_score(y, sub[col])
        ax.plot(fpr, tpr, label=f"{m} (AUC={a:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: separating good vs bad references")
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_parent_auc(scored, out_path):
    rows = []
    for m in METHODS:
        df_aucs = per_parent_auc(scored, m)
        df_aucs['method'] = m
        rows.append(df_aucs)
    df_aucs = pd.concat(rows, ignore_index=True)
    if df_aucs.empty:
        print("  [skip] per-parent AUC: no parent has both good and bad")
        return df_aucs

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = df_aucs['method'].unique()
    pos = np.arange(len(methods))
    box = [df_aucs.loc[df_aucs['method'] == m, 'auc'].to_numpy()
           for m in methods]
    ax.boxplot(box, positions=pos, widths=0.5, showmeans=True)
    for i, m in enumerate(methods):
        ys = df_aucs.loc[df_aucs['method'] == m, 'auc']
        ax.scatter(np.full(len(ys), i) + np.random.uniform(-0.1, 0.1, len(ys)),
                   ys, alpha=0.4, s=20)
    ax.set_xticks(pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Per-parent ROC AUC")
    ax.set_title("Per-parent separability by method")
    ax.axhline(0.5, color='grey', ls='--', lw=1)
    ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return df_aucs


def plot_pca(df, out_path, max_parents=4):
    parents = df.groupby('parent_id').filter(
        lambda g: g['ref_state'].nunique() == 2
    )['parent_id'].unique()[:max_parents]
    if len(parents) == 0:
        print("  [skip] PCA: no parent with both good and bad")
        return

    pca_global = PCA(n_components=2,
                     svd_solver='randomized',
                     random_state=42).fit(df[EMBED_COLS].to_numpy())

    n = len(parents)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             squeeze=False)
    for ax, pid in zip(axes.flat, parents):
        sub = df[df['parent_id'] == pid]
        Z = pca_global.transform(sub[EMBED_COLS].to_numpy())
        for state, color in [('bad', '#d62728'), ('good', '#2ca02c')]:
            mask = (sub['ref_state'] == state).to_numpy()
            ax.scatter(Z[mask, 0], Z[mask, 1], s=14, alpha=0.7,
                       c=color, label=state)
        ax.set_title(f"parent {pid} ({sub['parent_label'].iloc[0]})")
        ax.set_xlabel(f"PC1 ({pca_global.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca_global.explained_variance_ratio_[1]:.0%})")
        ax.legend()
    for ax in axes.flat[len(parents):]:
        ax.axis('off')
    fig.suptitle("PCA of AlphaEarth embeddings (global PCs, per-parent panels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input parquet path")
    parser.add_argument("--plots_dir", default=PLOTS_DIR,
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"Loading {args.input}")
    df = load(args.input)
    summarise(df)

    print("\nScoring (this may take a moment for large inputs)...")
    scored = score_all(df)

    print("\nOverall AUC (good vs bad, all parents pooled):")
    for m in METHODS:
        a = overall_auc(scored, m)
        print(f"  {m:>11s} : {a:.3f}" if not np.isnan(a)
              else f"  {m:>11s} : n/a")

    hist_path = os.path.join(args.plots_dir, "recovery_score_hists.png")
    roc_path = os.path.join(args.plots_dir, "recovery_roc.png")
    auc_path = os.path.join(args.plots_dir, "recovery_per_parent_auc.png")
    pca_path = os.path.join(args.plots_dir, "recovery_pca.png")

    plot_score_histograms(scored, hist_path)
    print(f"  wrote {hist_path}")
    plot_roc(scored, roc_path)
    print(f"  wrote {roc_path}")
    auc_df = plot_per_parent_auc(scored, auc_path)
    if not auc_df.empty:
        print(f"  wrote {auc_path}")
        print("\nPer-parent AUC summary (mean ± std):")
        print(auc_df.groupby('method')['auc']
              .agg(['mean', 'std', 'count']).round(3))
    plot_pca(df, pca_path)
    print(f"  wrote {pca_path}")


if __name__ == "__main__":
    main()
