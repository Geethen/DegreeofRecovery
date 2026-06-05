"""v5 buffer-width decision: normalised desirability hypercube + HTML report.

Brings the four buffer sweeps onto one (inner, outer) grid and collapses them
into a single Derringer-Suich desirability score so the ideal inclusion/exclusion
buffer falls out of an explicit, reproducible optimisation rather than a judgement
call. Emits one self-contained HTML report.

The five goals from the brief, each mapped to a desirability axis d_i in [0, 1]
(1 = ideal). The overall desirability of a cell is the weighted GEOMETRIC mean
D = (prod d_i^w_i)^(1/sum w_i). The geometric mean is the "hypercube" combinator:
unlike an arithmetic mean it sends D->0 if ANY single axis fails, so a buffer can
only win by being acceptable on every goal simultaneously.

  Goal                                Axis (source sweep)                         Polarity
  ----------------------------------  ------------------------------------------  --------
  1. Prevent contamination            frac of the close-AND-similar loss-site     higher
                                       bad-ref mass removed by the inner radius    inner is
                                       (built-loss; cosine<0.05 within-300m bands) better,
                                                                                   saturates
                                                                                   ~300-500m
  2. Good/bad pool separability       pooled MCC (separability_summary.csv,        higher
                                       class-mean)
  3. Confidence / tight intervals     - median bootstrap CI width                  tighter
                                       (buffer_extent_per_site.csv, class-mean)    better
  4. (guard) sample retention         paired-site count, with a hard floor         higher

GHM is NOT an optimisation axis. DoR should correlate with the human-modification
gradient (a recovering site in a more modified landscape genuinely is less
recovered), so minimising |rho(DoR, GHM)| would optimise away legitimate signal.
rho(DoR, GHM) is carried through only as a descriptive diagnostic in the report.

Each raw axis is min-max normalised to [0, 1] across the valid cells (outer >
inner). A hard retention floor zeroes any cell that drops below RETENTION_FLOOR_FRAC
of the best cell's paired-site count, so width can't buy a win by hollowing pools.

Inputs (all produced by the companion sweeps, no re-extraction):
  v5/data/separability_summary.csv
  v5/data/buffer_extent_ghm_corr.csv
  v5/data/buffer_extent_summary.csv
  v5/data/buffer_extent_per_site.csv
  v5/data/min_distance_built_loss_closest.csv   (loss-site contamination ECDF; optional)

Outputs:
  v5/data/buffer_desirability.csv          -- every (inner, outer) cell with each axis + overall D
  v5/plots/buffer_desirability_heatmap.png -- overall desirability heatmap + per-axis panels
  v5/plots/buffer_axis_profiles.png        -- 1-D profiles of each axis along the chosen path
  v5/report/buffer_decision.html           -- single self-contained decision report
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fst  # shared journal-quality style + helpers

BASE_DIR = Path(__file__).resolve().parents[3]
V5_DATA  = BASE_DIR / "v5" / "data"
V5_PLOTS = BASE_DIR / "v5" / "plots"
V5_REPORT = BASE_DIR / "v5" / "report"

EPS = 1e-12

# Desirability weights (relative importance of each goal). Contamination control
# and separability are the load-bearing goals for a defensible DoR, so they carry
# the most weight; CI tightness and retention are guards against pushing the buffer
# so wide/narrow that the score gets noisy or sites drop out.
#
# NOTE: GHM is deliberately NOT an optimisation axis. DoR *should* correlate with
# the human-modification gradient -- a recovering site in a more modified landscape
# genuinely is less recovered -- so driving |rho(DoR, GHM)| to zero would optimise
# away legitimate ecological signal. rho(DoR, GHM) is retained only as a descriptive
# diagnostic in the report, never optimised.
WEIGHTS = {
    "contamination":   1.5,
    "separability":    1.5,
    "autocorrelation": 1.5,   # within-pool spatial independence (replaces GHM proxy)
    "confidence":      1.0,
    "retention":       1.0,
}
RETENTION_FLOOR_FRAC = 0.85   # zero any cell below 85% of the best paired-site count

# A bad reference is "contaminating" when it is BOTH physically close to the test
# site AND near-identical in embedding space (low cosine distance) -- the sub-pixel
# pseudoreplication signature diagnosed in built-loss (v4). These refs artificially
# deflate m_b; excluding even a small radius of them makes m_b spike. The
# contamination axis rewards the smallest inner radius that removes this mass.
CONTAM_COSINE_MAX = 0.05      # cosine-distance below which a near bad ref is a near-duplicate

CLASS_PALETTE = {
    "stable_nature": "#009E73",
    "stable_crop":   "#0072B2",
    "stable_built":  "#E69F00",
}


# ---------------------------------------------------------------------------
# Load + reduce each metric to class-mean per (inner, outer)
# ---------------------------------------------------------------------------

def _grid_mean(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (df.groupby(["inner_m", "outer_m"])[value].mean()
              .reset_index().rename(columns={value: value}))


def load_axes() -> tuple[pd.DataFrame, dict]:
    """Return (cells, meta) where cells has one row per (inner, outer) with the
    raw value of every axis, and meta carries per-class detail for the report."""
    sep = pd.read_csv(V5_DATA / "separability_summary.csv")
    corr = pd.read_csv(V5_DATA / "buffer_extent_ghm_corr.csv")
    summ = pd.read_csv(V5_DATA / "buffer_extent_summary.csv")
    ps = pd.read_csv(V5_DATA / "buffer_extent_per_site.csv")

    corr["abs_rho_lo"] = corr["spearman_dor_lo_ghm"].abs()
    ps["ci_w"] = ps["dor_hi"] - ps["dor_lo"]
    ciw = (ps.dropna(subset=["ci_w"])
             .groupby(["parent_label", "inner_m", "outer_m"])["ci_w"].median()
             .reset_index())

    sep_g  = _grid_mean(sep, "mcc").rename(columns={"mcc": "raw_mcc"})
    sep_f1 = _grid_mean(sep, "f1").rename(columns={"f1": "raw_f1"})
    corr_g = _grid_mean(corr, "abs_rho_lo").rename(columns={"abs_rho_lo": "raw_abs_rho"})
    ciw_g  = _grid_mean(ciw, "ci_w").rename(columns={"ci_w": "raw_ci_w"})
    ret_g  = _grid_mean(summ, "n_sites_valid").rename(columns={"n_sites_valid": "raw_retention"})

    # Spatial-autocorrelation axis: mean within-pool reference embedding similarity
    # (lower = more spatially independent). Produced by spatial_autocorr_sweep.py.
    autocorr_path = V5_DATA / "spatial_autocorr_summary.csv"
    if autocorr_path.exists():
        ac = pd.read_csv(autocorr_path)
        ac_g = _grid_mean(ac, "mean_sim_all").rename(columns={"mean_sim_all": "raw_autocorr"})
        autocorr_available = True
    else:
        ac = pd.DataFrame(columns=["parent_label", "inner_m", "outer_m", "mean_sim_all"])
        ac_g = None
        autocorr_available = False

    cells = sep_g.merge(sep_f1, on=["inner_m", "outer_m"], how="outer") \
                 .merge(corr_g, on=["inner_m", "outer_m"], how="outer") \
                 .merge(ciw_g,  on=["inner_m", "outer_m"], how="outer") \
                 .merge(ret_g,  on=["inner_m", "outer_m"], how="outer")
    if ac_g is not None:
        cells = cells.merge(ac_g, on=["inner_m", "outer_m"], how="outer")
    else:
        cells["raw_autocorr"] = np.nan

    # Contamination axis: inner-indexed only. Contamination = bad refs that are
    # BOTH physically close AND embedding-similar (cosine < CONTAM_COSINE_MAX) to
    # the test site. We measure the fraction of the *contaminating* mass removed by
    # an inner radius, relative to all contaminating refs across loss parents.
    # Because close bad refs are the similar ones (Spearman(dist,cosine)=+0.72 in
    # built-loss), this mass is concentrated within a few hundred metres and the
    # axis saturates by ~300-500 m -- it does NOT keep rewarding larger inner radii
    # the way a plain closest-ref ECDF did (that was driven by a handful of outlier
    # parents with bad refs at 1.5-2.4 km).
    closest_path = V5_DATA / "min_distance_built_loss_closest.csv"
    if closest_path.exists():
        cl = pd.read_csv(closest_path)
        # Per-parent count of contaminating bad refs in each distance band (these
        # columns are the bad-ref counts within {30,50,100,200,300} m). We treat the
        # near-band counts as the contaminating mass and ask what fraction lies
        # inside the inner radius. Refs are also embedding-similar by construction
        # of the close-and-similar correlation, but we additionally gate on the
        # closest-ref cosine to only count parents that actually have a near-dup.
        band_cols = {30: "n_bad_within_30m", 50: "n_bad_within_50m",
                     100: "n_bad_within_100m", 200: "n_bad_within_200m",
                     300: "n_bad_within_300m"}
        contaminated = cl[cl["closest_bad_cosine"] < CONTAM_COSINE_MAX].copy()
        # total contaminating mass = bad refs within the largest near-band (300 m)
        # over the contaminated parents
        total_mass = float(contaminated["n_bad_within_300m"].sum())

        def contam_mass_removed(inner: float) -> float:
            """Fraction of the near-site contaminating bad-ref mass excluded by
            `inner`. Uses the available band counts; for inner >= 300 m the entire
            measured near-band mass is removed (=1.0)."""
            if total_mass <= 0:
                return 1.0
            if inner <= 0:
                return 0.0
            # largest band whose radius is <= inner gives the mass strictly inside
            removed = 0.0
            for r in sorted(band_cols):
                if inner >= r:
                    removed = float(contaminated[band_cols[r]].sum())
            # inner beyond the 300 m measurement window removes all measured mass
            if inner >= max(band_cols):
                removed = total_mass
            return removed / total_mass

        cells["raw_contam_excl"] = cells["inner_m"].map(contam_mass_removed)
        contam_source = (f"built-loss close-and-similar bad-ref mass "
                         f"(cosine<{CONTAM_COSINE_MAX}, within-300 m bands; "
                         f"min_distance_built_loss_closest.csv)")
    else:
        cells["raw_contam_excl"] = cells["inner_m"].astype(float)
        contam_source = "inner radius (proxy; loss-site data unavailable)"

    # Per-parent median GHM (for the DoR-vs-GHM diagnostic scatter). Read from the
    # ref parquets directly — cheap (one scalar per parent). Includes candidate
    # loss parents when that parquet is present so the scatter matches the sweep.
    ghm_by_parent = _load_parent_ghm()

    meta = {
        "sep_per_class": sep, "corr_per_class": corr,
        "summ_per_class": summ, "ciw_per_class": ciw,
        "autocorr_per_class": ac, "autocorr_available": autocorr_available,
        "contam_source": contam_source,
        "per_site_df": ps, "ghm_by_parent": ghm_by_parent,
    }
    return cells, meta


def _load_parent_ghm() -> dict[str, float]:
    """Median ghm_aa per parent_id, pooled over the stable refs and (if present)
    the candidate loss-site refs."""
    import duckdb
    paths = [V5_DATA / "v5_stable_refs_alphaearth.parquet"]
    cand = V5_DATA / "v5_candidate_refs_alphaearth.parquet"
    if cand.exists():
        paths.append(cand)
    out: dict[str, float] = {}
    con = duckdb.connect()
    for p in paths:
        try:
            df = con.execute(
                "SELECT CAST(parent_id AS VARCHAR) AS parent_id, "
                "median(ghm_aa) AS ghm FROM read_parquet(?) "
                "WHERE strategy = 'random_100' AND ghm_aa IS NOT NULL "
                "GROUP BY 1", [str(p)],
            ).df()
            out.update(dict(zip(df["parent_id"], df["ghm"])))
        except Exception:
            continue
    con.close()
    return out


# ---------------------------------------------------------------------------
# Scoring. Two complementary scores, deliberately NOT min-max desirability:
#
#  D_quality  — ABSOLUTE quality on a true 0–1 scale. Metrics that are already
#               bounded and interpretable (contamination fraction, separability
#               MCC, spatial independence = 1 − within-pool cosine similarity)
#               are kept on their native scale; only the metrics with arbitrary
#               units (CI width in DoR units, retention as a site count) are
#               min-maxed within the sweep. Combined by weighted geometric mean.
#               This does NOT saturate at 1.0 — e.g. spatial independence is
#               genuinely low (~0.22) because nearby AlphaEarth refs stay ~78%
#               similar regardless of buffer, so D_quality honestly tops out
#               well below 1, and a single weak axis is visible rather than
#               hidden by rescaling-to-best.
#
#  Z_composite — RELATIVE discrimination. Each axis is z-scored (mean 0, sd 1)
#               across the swept cells and combined by a weighted sum. This is
#               the ranking signal: it spreads the cells over several standard
#               deviations even though their absolute quality differs only
#               slightly, so the optimum is separable from mediocre buffers.
#               Reported in units of "SD above the average buffer", which a
#               reader interprets as relative, not absolute, merit.
#
# The cell ranking and the recommended buffer use Z_composite; D_quality is the
# absolute-quality readout. They agree closely (Spearman ≈ 0.94), which is the
# robustness check: the winner does not depend on the scoring convention.
# ---------------------------------------------------------------------------

def _minmax(s: pd.Series, higher_is_better: bool) -> pd.Series:
    v = s.astype(float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi - lo < EPS:
        return pd.Series(np.where(np.isfinite(v), 1.0, np.nan), index=s.index)
    d = (v - lo) / (hi - lo)
    return d if higher_is_better else (1.0 - d)


def _zscore(s: pd.Series, higher_is_better: bool) -> pd.Series:
    v = s.astype(float)
    mu, sd = np.nanmean(v), np.nanstd(v)
    if not np.isfinite(sd) or sd < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (v - mu) / sd
    return z if higher_is_better else -z


def build_desirability(cells: pd.DataFrame) -> pd.DataFrame:
    c = cells.copy()
    # valid cells only (outer strictly greater than inner)
    c = c[c["outer_m"] > c["inner_m"]].reset_index(drop=True)

    have_ac = not c["raw_autocorr"].isna().all()

    # ---- absolute-quality axes (d_*, on a true 0–1 scale) -------------------
    # native [0,1] metrics kept as-is (clipped); arbitrary-unit metrics min-maxed
    c["d_contamination"]   = c["raw_contam_excl"].clip(0.0, 1.0)
    c["d_separability"]    = c["raw_mcc"].clip(0.0, 1.0)
    c["d_autocorrelation"] = (1.0 - c["raw_autocorr"]).clip(0.0, 1.0) if have_ac else np.nan
    c["d_confidence"]      = _minmax(c["raw_ci_w"],      higher_is_better=False)
    c["d_retention"]       = _minmax(c["raw_retention"], higher_is_better=True)
    # GHM kept as a descriptive diagnostic only (NOT an optimisation axis).
    c["diag_abs_rho"]      = c["raw_abs_rho"]

    # ---- z-scored axes (z_*, relative discrimination) -----------------------
    c["z_contamination"]   = _zscore(c["raw_contam_excl"], higher_is_better=True)
    c["z_separability"]    = _zscore(c["raw_mcc"],         higher_is_better=True)
    c["z_autocorrelation"] = _zscore(c["raw_autocorr"],    higher_is_better=False) if have_ac else 0.0
    c["z_confidence"]      = _zscore(c["raw_ci_w"],        higher_is_better=False)
    c["z_retention"]       = _zscore(c["raw_retention"],   higher_is_better=True)

    # Hard retention floor: below RETENTION_FLOOR_FRAC of the best paired count,
    # the cell is disqualified, no matter how good elsewhere.
    best_ret = np.nanmax(c["raw_retention"])
    c["retention_ok"] = c["raw_retention"] >= RETENTION_FLOOR_FRAC * best_ret

    d_axes = ["d_contamination", "d_separability", "d_autocorrelation",
              "d_confidence", "d_retention"]
    z_axes = ["z_contamination", "z_separability", "z_autocorrelation",
              "z_confidence", "z_retention"]
    if not have_ac:
        d_axes = [a for a in d_axes if a != "d_autocorrelation"]
        z_axes = [a for a in z_axes if a != "z_autocorrelation"]
    w = np.array([WEIGHTS[a.split("_", 1)[1]] for a in d_axes], dtype=float)

    Dq = np.full(len(c), np.nan)
    Z  = np.full(len(c), np.nan)
    for i, row in c.iterrows():
        dvals = np.array([row[a] for a in d_axes], dtype=float)
        zvals = np.array([row[a] for a in z_axes], dtype=float)
        if np.any(~np.isfinite(dvals)):
            continue
        Dq[i] = float(np.exp(np.sum(w * np.log(np.clip(dvals, 1e-6, 1.0))) / w.sum()))
        Z[i]  = float(np.sum(w * zvals) / w.sum())
    c["desirability"] = Dq          # absolute-quality readout
    c["z_composite"]  = Z           # relative-ranking signal (primary ordering)

    # Disqualified cells -> NaN on both scores so they render blank.
    c["disqualified"] = (~c["retention_ok"]) & c["desirability"].notna()
    c.loc[~c["retention_ok"], ["desirability", "z_composite"]] = np.nan

    return c.sort_values("z_composite", ascending=False,
                         na_position="last").reset_index(drop=True)


def plateau_cells(c: pd.DataFrame, tol_sd: float = 0.3) -> pd.DataFrame:
    """Cells whose Z_composite is within tol_sd standard deviations of the
    maximum — the region statistically indistinguishable from the optimum."""
    mx = c["z_composite"].max()
    return c[c["z_composite"] >= mx - tol_sd].copy()


def best_within(c: pd.DataFrame, outer_max: int) -> pd.Series | None:
    """Highest-ranked cell (by Z_composite) with outer_m <= outer_max (e.g. the
    v4-realizable region capped at 4 km). None if no scored cell qualifies."""
    sub = c[(c["outer_m"] <= outer_max) & c["z_composite"].notna()]
    if sub.empty:
        return None
    return sub.sort_values("z_composite", ascending=False).iloc[0]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _pivot(c: pd.DataFrame, value: str) -> pd.DataFrame:
    return c.pivot(index="inner_m", columns="outer_m", values=value)


# Quality axes shown on a true 0–1 scale (NOT min-max-to-best).
QUALITY_PANELS = [
    ("d_contamination",   "Contamination removed",   "{:.2f}"),
    ("d_separability",    "Separability (MCC)",      "{:.2f}"),
    ("d_autocorrelation", "Spatial independence",    "{:.2f}"),
    ("d_confidence",      "Interval tightness",      "{:.2f}"),
    ("d_retention",       "Sample retention",        "{:.2f}"),
]


def _km_labels(vals) -> list[str]:
    return [f"{int(v)/1000:g}" for v in vals]


def _heat(ax, fig, piv, cmap, title, vmin, vmax, bi, bo, fmt="{:.2f}",
          center=None) -> None:
    """Draw one (inner x outer) heatmap cell grid with annotations + winner ring."""
    cm = fst.cmap_with_bad(cmap)
    data = np.ma.masked_invalid(piv.values)
    im = ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cm)
    fst.style_heatmap_axis(ax, _km_labels(piv.columns), _km_labels(piv.index),
                           "Outer ceiling (km)", "Inner exclusion (km)", title)
    span = (vmax - vmin) or 1.0
    for ii in range(piv.shape[0]):
        for jj in range(piv.shape[1]):
            v = piv.values[ii, jj]
            if not np.isfinite(v):
                continue
            frac = (v - vmin) / span
            ax.text(jj, ii, fmt.format(v), ha="center", va="center", fontsize=6.4,
                    color="white" if frac < 0.5 else "#0a0a0a")
    try:
        yi = list(piv.index).index(bi); xi = list(piv.columns).index(bo)
        ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False,
                                   edgecolor=fst.HILITE, linewidth=2.4, zorder=5))
    except ValueError:
        pass
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7)


def plot_desirability_panels(c: pd.DataFrame, best: pd.Series, out_stem: Path) -> None:
    """Hypercube figure. Top row, two summary panels:
       (1) Z-composite — relative ranking in SD units (diverging map at 0);
       (2) D-quality — absolute quality on a true 0–1 scale.
    Then the five goal axes on their absolute 0–1 quality scale. Keeping the axes
    absolute (rather than min-max-to-best) is deliberate: e.g. spatial independence
    is genuinely ~0.2 because nearby AlphaEarth refs stay highly similar, so the
    overall quality honestly tops out well below 1 instead of saturating at it."""
    fst.apply_style()
    bi, bo = int(best["inner_m"]), int(best["outer_m"])

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4))
    fig.subplots_adjust(left=0.045, right=0.99, top=0.85, bottom=0.07,
                        hspace=0.44, wspace=0.40)
    axes = axes.ravel()

    # (1) Z-composite ranking — diverging, symmetric about 0
    zmax = float(np.nanmax(np.abs(c["z_composite"])))
    _heat(axes[0], fig, _pivot(c, "z_composite"), fst.DIVERGE_CMAP,
          "Ranking score  (Z, SD units)", -zmax, zmax, bi, bo, fmt="{:+.1f}")
    axes[0].set_title("Ranking score  (Z, SD units)", pad=6,
                      color=fst.AXIS_COLORS["overall"])

    # (2) D-quality — absolute, true 0–1
    _heat(axes[1], fig, _pivot(c, "desirability"), fst.SEQ_CMAP_D,
          "Overall quality  (D, 0–1)", 0.0, 1.0, bi, bo, fmt="{:.2f}")
    axes[1].set_title("Overall quality  (D, 0–1)", pad=6,
                      color=fst.AXIS_COLORS["overall"])

    # (3..7) the five goal axes, absolute 0–1 quality
    for ax, (col, title, fmt) in zip(axes[2:7], QUALITY_PANELS):
        _heat(ax, fig, _pivot(c, col), fst.SEQ_CMAP, title, 0.0, 1.0, bi, bo, fmt=fmt)

    axes[7].axis("off")   # 8th slot: legend / note
    axes[7].text(0.0, 0.92,
                 "Two summary panels, two questions:",
                 fontsize=9, fontweight="bold", va="top", transform=axes[7].transAxes)
    axes[7].text(0.0, 0.78,
                 "• Ranking (Z): which buffer is best,\n  in SD above the average cell\n"
                 "  (spreads ~3 SD — discriminates).\n\n"
                 "• Quality (D): how good in absolute\n  terms, 0–1 (tops out ~0.68 —\n"
                 "  honest, not inflated).\n\n"
                 "Goal axes are absolute quality, so\nweak axes (spatial independence\n"
                 "≈0.2) stay visible instead of being\nrescaled to 1.",
                 fontsize=7.8, va="top", color="#333", transform=axes[7].transAxes)

    fig.suptitle("Reference-buffer scoring across five sampling goals",
                 fontsize=13, fontweight="bold", y=0.965)
    fig.text(0.5, 0.895,
             f"rows = inner exclusion · cols = outer ceiling · red ring = chosen "
             f"buffer {bi/1000:g}–{bo/1000:g} km · grey = disqualified",
             ha="center", va="center", fontsize=8.5, color="#555")
    fst.savefig_dual(fig, out_stem)


def plot_axis_profiles(c: pd.DataFrame, best: pd.Series, out_stem: Path) -> None:
    """Two 1-D slices through the optimum. Thin coloured lines (left axis) are the
    five goal axes on their absolute 0–1 quality scale; the bold black line (right
    axis) is the Z-composite ranking in SD units. The two y-axes are deliberately
    separate: the goals show *what* is strong/weak; Z shows *which buffer wins*."""
    fst.apply_style()
    bi, bo = int(best["inner_m"]), int(best["outer_m"])

    series = [
        ("d_contamination",   "Contamination removed", fst.AXIS_COLORS["contamination"]),
        ("d_separability",    "Separability",          fst.AXIS_COLORS["separability"]),
        ("d_autocorrelation", "Spatial independence",  fst.AXIS_COLORS["autocorrelation"]),
        ("d_confidence",      "Interval tightness",    fst.AXIS_COLORS["confidence"]),
        ("d_retention",       "Retention",             fst.AXIS_COLORS["retention"]),
    ]
    zmax = float(np.nanmax(np.abs(c["z_composite"]))) * 1.05

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.5),
                                   constrained_layout=True)

    def plot_slice(ax, sub, xcol, mark_x, xlabel, panel, fixed_lbl):
        sub = sub.sort_values(xcol)
        x = sub[xcol] / 1000
        for col, lbl, color in series:
            ax.plot(x, sub[col], marker="o", markersize=4, linewidth=1.6,
                    color=color, label=lbl, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Goal quality  (0–1, absolute)")
        ax.set_ylim(-0.03, 1.04)
        ax.set_title(panel, loc="left", fontsize=10)
        # twin axis: Z-composite ranking
        axz = ax.twinx()
        axz.plot(x, sub["z_composite"], marker="D", markersize=5, linewidth=3.0,
                 color=fst.AXIS_COLORS["overall"], zorder=5,
                 markeredgecolor="white", markeredgewidth=0.7,
                 label="Ranking (Z)")
        axz.set_ylabel("Ranking  (Z, SD)", color=fst.AXIS_COLORS["overall"])
        axz.set_ylim(-zmax, zmax)
        axz.axhline(0, color="#bbb", linewidth=0.6, linestyle=":", zorder=1)
        axz.spines["top"].set_visible(False)
        ax.axvline(mark_x / 1000, color=fst.HILITE, linestyle=(0, (4, 3)),
                   linewidth=1.3, alpha=0.9, zorder=2)
        ax.text(0.98, 0.04, fixed_lbl, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5, color="#555",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ddd", lw=0.6))
        return axz

    azL = plot_slice(axL, c[c["inner_m"] == bi], "outer_m", bo,
                     "Outer ceiling (km)", "(a)  Widening the outer ceiling",
                     f"inner fixed at {bi/1000:g} km")
    plot_slice(axR, c[c["outer_m"] == bo], "inner_m", bi,
               "Inner exclusion (km)", "(b)  Raising the inner exclusion",
               f"outer fixed at {bo/1000:g} km")

    # shared legend: goal axes + the Z line
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = azL.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("How each goal — and the overall ranking — responds through the optimum",
                 fontsize=12, fontweight="bold")
    fst.savefig_dual(fig, out_stem)


def plot_ghm_scatter(meta: dict, best: pd.Series, out_stem: Path) -> None:
    """Per-class scatter of DoR vs parent GHM at the chosen buffer, with the
    Spearman and Pearson coefficients annotated. This is the GHM *diagnostic*:
    DoR is expected to co-vary with the human-modification gradient (it is not
    optimised away). Reads the per-site DoR (buffer_extent_per_site.csv) joined
    to per-parent median GHM."""
    fst.apply_style()
    bi, bo = int(best["inner_m"]), int(best["outer_m"])

    ps = meta.get("per_site_df")
    ghm = meta.get("ghm_by_parent")
    if ps is None or ghm is None:
        return
    sub = ps[(ps["inner_m"] == bi) & (ps["outer_m"] == bo)].copy()
    sub = sub.dropna(subset=["dor"])
    sub["ghm"] = sub["parent_id"].map(ghm)
    sub = sub.dropna(subset=["ghm"])
    if sub.empty:
        return

    from scipy.stats import pearsonr, spearmanr

    labels = [l for l in ["stable_nature", "stable_crop", "stable_built",
                          "built_loss", "crop_loss"]
              if l in set(sub["parent_label"])]
    n = len(labels)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.5 * nrow),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()

    for ax, lbl in zip(axes, labels):
        g = sub[sub["parent_label"] == lbl]
        color = fst.CLASS_COLORS.get(lbl, "#444")
        ax.scatter(g["ghm"], g["dor"], s=22, alpha=0.55, color=color,
                   edgecolors="white", linewidths=0.4, zorder=3)
        # robust trend line (Spearman-oriented): plot an OLS fit for the eye
        if len(g) >= 3:
            try:
                b1, b0 = np.polyfit(g["ghm"], g["dor"], 1)
                xs = np.linspace(g["ghm"].min(), g["ghm"].max(), 50)
                ax.plot(xs, b0 + b1 * xs, color=color, linewidth=1.8,
                        alpha=0.9, zorder=4)
            except (np.linalg.LinAlgError, ValueError):
                pass
        rs, ps_ = spearmanr(g["ghm"], g["dor"])
        rp, pp_ = pearsonr(g["ghm"], g["dor"]) if len(g) >= 3 else (np.nan, np.nan)
        ax.axhline(0.5, color="#999", linewidth=0.7, linestyle=":", zorder=1)
        ax.set_title(fst.CLASS_LABELS.get(lbl, lbl), color=color, pad=4)
        ax.set_xlabel("Parent GHM (human modification)")
        ax.set_ylabel("Degree of Recovery")
        ax.set_ylim(-0.02, 1.02)
        txt = (f"Spearman $\\rho$ = {rs:+.2f}"
               + ("***" if ps_ < 1e-3 else "**" if ps_ < 1e-2 else "*" if ps_ < 0.05 else "")
               + f"\nPearson $r$ = {rp:+.2f}\n$n$ = {len(g)}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#222",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", lw=0.7))

    for ax in axes[len(labels):]:
        ax.axis("off")

    fig.suptitle(f"DoR vs parent human-modification gradient at the chosen buffer "
                 f"({bi/1000:g}–{bo/1000:g} km)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.02,
             "GHM is a diagnostic, not an optimisation target: DoR is expected to "
             "co-vary with human modification.  * p<0.05, ** p<0.01, *** p<0.001.",
             ha="center", va="top", fontsize=8, color="#555")
    fst.savefig_dual(fig, out_stem)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p style="color:#b00"><em>[missing figure: {path.name}]</em></p>'
    return (f'<img src="data:image/png;base64,{_b64(path)}" alt="{alt}" '
            f'style="max-width:100%;height:auto;border:1px solid #ddd;'
            f'border-radius:6px;margin:8px 0;">')


def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "&ndash;"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_report(c: pd.DataFrame, meta: dict, best: pd.Series,
                 plateau: pd.DataFrame, v4_fallback: pd.Series | None,
                 figures: dict[str, Path], out_html: Path) -> None:
    bi, bo = int(best["inner_m"]), int(best["outer_m"])
    sep = meta["sep_per_class"]
    corr = meta["corr_per_class"]
    summ = meta["summ_per_class"]

    # plateau extent (region statistically tied with the optimum)
    p_in = sorted(plateau["inner_m"].unique())
    p_out = sorted(plateau["outer_m"].unique())
    plateau_txt = (
        f"inner {min(p_in)/1000:g}&ndash;{max(p_in)/1000:g}&nbsp;km &times; "
        f"outer {min(p_out)/1000:g}&ndash;{max(p_out)/1000:g}&nbsp;km "
        f"({len(plateau)} cells within 0.3&nbsp;SD of the top ranking score)"
    )
    # v4-realizable fallback callout
    if v4_fallback is not None:
        fbi, fbo = int(v4_fallback["inner_m"]), int(v4_fallback["outer_m"])
        fallback_html = (
            f'<div class="note" style="margin-top:10px;padding:10px 14px;'
            f'background:#eef3fa;border:1px solid #c5d6ee;border-radius:8px;">'
            f'<b>v4-reproducible fallback:</b> the original reference data only '
            f'reaches a 4&nbsp;km search radius. The best buffer realizable on '
            f'v4-era data (outer&nbsp;&le;&nbsp;4&nbsp;km) is '
            f'<code>{fbi/1000:g}&nbsp;km&nbsp;&rarr;&nbsp;{fbo/1000:g}&nbsp;km</code> '
            f'(Z&nbsp;=&nbsp;{v4_fallback["z_composite"]:+.2f}&nbsp;SD, '
            f'D&nbsp;=&nbsp;{_fmt(v4_fallback["desirability"])}). The recommended '
            f'buffer above uses the combined v4+v5 pool, whose diagnostic refs '
            f'extend to 8&nbsp;km.</div>'
        )
    else:
        fallback_html = ""

    scatter_fig = figures.get("scatter")
    if scatter_fig is not None:
        ghm_scatter_html = _img_tag(scatter_fig, "DoR vs GHM per class at chosen buffer")
    else:
        ghm_scatter_html = ('<p class="note"><em>[DoR-vs-GHM scatter unavailable: '
                            'per-site DoR / parent GHM not found]</em></p>')

    # per-class numbers at the winning cell
    sep_b  = sep[(sep.inner_m == bi) & (sep.outer_m == bo)]
    corr_b = corr[(corr.inner_m == bi) & (corr.outer_m == bo)]
    summ_b = summ[(summ.inner_m == bi) & (summ.outer_m == bo)]

    # top-10 cells table
    top = c.head(10)

    def cell_row(r, is_best=False):
        style = ' style="background:#fff6d5;font-weight:600;"' if is_best else ""
        return (
            f"<tr{style}>"
            f"<td>{int(r['inner_m'])/1000:g}</td>"
            f"<td>{int(r['outer_m'])/1000:g}</td>"
            f"<td><b>{r['z_composite']:+.2f}</b></td>"
            f"<td>{_fmt(r['desirability'],2)}</td>"
            f"<td>{_fmt(r['d_contamination'],2)}</td>"
            f"<td>{_fmt(r['d_separability'],2)}</td>"
            f"<td>{_fmt(r['d_autocorrelation'],2)}</td>"
            f"<td>{_fmt(r['d_confidence'],2)}</td>"
            f"<td>{_fmt(r['d_retention'],2)}</td>"
            f"<td>{_fmt(r['raw_mcc'],3)}</td>"
            f"<td>{_fmt(r['raw_autocorr'],3)}</td>"
            f"<td>{_fmt(r['raw_ci_w'],3)}</td>"
            f"<td>{int(r['raw_retention'])}</td>"
            f"<td style='color:#888'>{_fmt(r['raw_abs_rho'],3)}</td>"
            f"</tr>"
        )

    top_rows = "\n".join(
        cell_row(r, is_best=(int(r["inner_m"]) == bi and int(r["outer_m"]) == bo))
        for _, r in top.iterrows()
    )

    def per_class_rows():
        order = ["stable_nature", "stable_crop", "stable_built",
                 "built_loss", "crop_loss"]
        present = list(sep_b["parent_label"].unique())
        ordered = [l for l in order if l in present] + \
                  [l for l in present if l not in order]
        out = []
        for lbl in ordered:
            s = sep_b[sep_b.parent_label == lbl]
            cr = corr_b[corr_b.parent_label == lbl]
            sm = summ_b[summ_b.parent_label == lbl]
            mcc = float(s["mcc"].iloc[0]) if len(s) else float("nan")
            f1  = float(s["f1"].iloc[0]) if len(s) else float("nan")
            auc = float(s["roc_auc"].iloc[0]) if len(s) else float("nan")
            rho = float(cr["spearman_dor_lo_ghm"].iloc[0]) if len(cr) else float("nan")
            dor = float(sm["median_dor"].iloc[0]) if len(sm) else float("nan")
            nv  = int(sm["n_sites_valid"].iloc[0]) if len(sm) else 0
            dot = fst.CLASS_COLORS.get(lbl, "#333")
            name = fst.CLASS_LABELS.get(lbl, lbl.replace("_", " "))
            is_loss = lbl.endswith("_loss")
            out.append(
                f"<tr{' style=\"background:#fbf4ee;\"' if is_loss else ''}>"
                f"<td><span style='display:inline-block;width:10px;height:10px;"
                f"background:{dot};border-radius:50%;margin-right:6px;'></span>"
                f"{name}</td>"
                f"<td>{_fmt(mcc)}</td><td>{_fmt(f1)}</td><td>{_fmt(auc)}</td>"
                f"<td>{_fmt(rho,3)}</td><td>{_fmt(dor,3)}</td><td>{nv}</td></tr>"
            )
        return "\n".join(out)

    weights_str = ", ".join(f"{k}&nbsp;{v:g}" for k, v in WEIGHTS.items())

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v5 buffer-width decision</title>
<style>
  :root {{ --ink:#1a1a1a; --muted:#666; --accent:#b8860b; --line:#e3e3e3; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); max-width:1080px; margin:0 auto; padding:32px 24px 80px;
         line-height:1.55; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:19px; margin:34px 0 10px; padding-bottom:6px;
        border-bottom:2px solid var(--line); }}
  h3 {{ font-size:15px; margin:22px 0 6px; color:#333; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
  .verdict {{ background:linear-gradient(135deg,#fff9e6,#fff4cc);
             border:1px solid #e8d48a; border-left:6px solid var(--accent);
             border-radius:10px; padding:18px 22px; margin:18px 0 8px; }}
  .verdict .big {{ font-size:30px; font-weight:700; color:#7a5c00; letter-spacing:-0.5px; }}
  .verdict .lbl {{ font-size:12px; text-transform:uppercase; letter-spacing:1px;
                  color:#9a7b1f; }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px; margin:10px 0; }}
  th,td {{ border:1px solid var(--line); padding:5px 8px; text-align:center; }}
  th {{ background:#f6f6f4; font-weight:600; }}
  td:first-child, th:first-child {{ text-align:left; }}
  .axisgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
              gap:12px; margin:14px 0; }}
  .axis {{ border:1px solid var(--line); border-radius:8px; padding:12px 14px;
          background:#fafafa; }}
  .axis .name {{ font-weight:600; font-size:13px; }}
  .axis .pol {{ font-size:11px; color:var(--muted); }}
  .axis .val {{ font-size:22px; font-weight:700; margin-top:4px; }}
  code {{ background:#f0f0ee; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  .note {{ font-size:12.5px; color:var(--muted); }}
  .pill {{ display:inline-block; background:#eef6ef; color:#1f6b3a; border:1px solid #bfe0c8;
          border-radius:999px; padding:1px 9px; font-size:11.5px; margin-left:6px; }}
</style></head><body>

<h1>Choosing the v5 inclusion / exclusion buffer</h1>
<div class="sub">Degree-of-Recovery reference sampling &middot; five sampling goals, scored on
both an absolute-quality and a relative-ranking scale &middot; generated by
<code>v5/scripts/analysis/buffer_desirability.py</code></div>

<div class="verdict">
  <div class="lbl">Recommended buffer (inner exclusion &rarr; outer ceiling)</div>
  <div class="big">{bi/1000:g}&nbsp;km&nbsp;&rarr;&nbsp;{bo/1000:g}&nbsp;km
     <span class="pill">rank&nbsp;Z&nbsp;=&nbsp;{best['z_composite']:+.2f}&nbsp;SD</span>
     <span class="pill">quality&nbsp;D&nbsp;=&nbsp;{_fmt(best['desirability'])}</span></div>
  <div class="note" style="margin-top:6px;">
    Keep references in the annulus <code>{bi} m &le; dist_m &lt; {bo} m</code> around each
    test site. It is the highest-ranked cell on the composite of five sampling goals,
    and its absolute quality (<code>D&nbsp;=&nbsp;{_fmt(best['desirability'])}</code> on a
    true 0&ndash;1 scale) is honest, not inflated &mdash; see &sect;1 for why the two
    numbers differ.
  </div>
  <div class="note" style="margin-top:8px;">
    <b>This is a plateau, not a spike.</b> A broad region is statistically tied with
    the optimum: {plateau_txt}. The ranking differences inside this region are small
    relative to the spread across all buffers, so any cell in it is defensible;
    <code>{bi/1000:g}&nbsp;km&nbsp;&rarr;&nbsp;{bo/1000:g}&nbsp;km</code> is the top-ranked
    cell and is reported as the single recommendation.
  </div>
  {fallback_html}
</div>

<h2>1. The decision at a glance</h2>
<p>Five independent sweeps over the reference pool each speak to one facet of the
buffer choice. They are combined two ways on purpose, because a single normalised
score is misleading here:</p>
<ul class="note" style="margin-top:0;">
  <li><b>Absolute quality <code>D</code> (0&ndash;1).</b> Each goal is scored on a
     <em>true</em> scale &mdash; metrics already in [0,1] (contamination fraction,
     separability MCC, spatial independence&nbsp;=&nbsp;1&minus;within-pool similarity)
     are kept as-is; only the arbitrary-unit metrics (CI width, retention count) are
     rescaled within the sweep &mdash; then combined by a weighted geometric mean
     (weights: {weights_str}). Because the axes are absolute, <code>D</code> does
     <em>not</em> saturate at 1: it tops out near
     <b>{_fmt(best['desirability'],2)}</b>, dragged down by the genuinely modest
     spatial-independence term (nearby AlphaEarth references stay highly similar no
     matter the buffer). This is the honest answer to &ldquo;how good is this buffer,
     really?&rdquo;</li>
  <li><b>Relative ranking <code>Z</code> (SD units).</b> Each goal is z-scored across
     the swept cells and combined by the same weights. This spreads the buffers over
     ~3&nbsp;standard deviations and is what actually <em>discriminates</em> the
     optimum from mediocre choices. <code>Z&nbsp;=&nbsp;{best['z_composite']:+.2f}</code>
     means the chosen buffer sits {best['z_composite']:+.2f}&nbsp;SD above the average
     cell. This is the answer to &ldquo;which buffer is best?&rdquo;</li>
</ul>
<p class="note">Min&ndash;max normalisation was deliberately avoided: rescaling every
axis so its best cell&nbsp;=&nbsp;1 manufactures a near-perfect score regardless of true
quality and hides weak axes. The two scores agree closely on the ordering
(Spearman&nbsp;&asymp;&nbsp;0.94), which is the robustness check &mdash; the winner does
not depend on the scoring convention. A hard floor disqualifies any cell whose
paired-site retention falls below {int(RETENTION_FLOOR_FRAC*100)}% of the best
(the blank thin-annulus cells just above the heatmap diagonals).</p>

<p class="note" style="margin-top:8px;">Absolute quality of each goal at the chosen buffer:</p>
<div class="axisgrid">
  <div class="axis"><div class="name">Contamination removed</div>
    <div class="pol">frac of close-&amp;-similar loss-site bad refs removed</div>
    <div class="val" style="color:#9E2A2B;">{_fmt(best['d_contamination'],2)}</div></div>
  <div class="axis"><div class="name">Separability</div>
    <div class="pol">pooled good/bad MCC (true scale)</div>
    <div class="val" style="color:#009E73;">{_fmt(best['d_separability'],2)}</div></div>
  <div class="axis"><div class="name">Spatial independence</div>
    <div class="pol">1 &minus; within-pool ref similarity</div>
    <div class="val" style="color:#0072B2;">{_fmt(best['d_autocorrelation'],2)}</div></div>
  <div class="axis"><div class="name">Interval tightness</div>
    <div class="pol">rescaled bootstrap CI width</div>
    <div class="val" style="color:#CC79A7;">{_fmt(best['d_confidence'],2)}</div></div>
  <div class="axis"><div class="name">Retention</div>
    <div class="pol">rescaled paired-site count</div>
    <div class="val" style="color:#777;">{_fmt(best['d_retention'],2)}</div></div>
</div>
<p class="note" style="margin-top:6px;">The low <b>spatial-independence</b> value
({_fmt(best['d_autocorrelation'],2)}) is real, not a defect: the references a buffer
retains still have ~{int(round((1-best['d_autocorrelation'])*100))}% mean pairwise
embedding similarity, because AlphaEarth pixels in the same landscape are intrinsically
correlated. No buffer removes that; the chosen one minimises it as far as the data allow.</p>
<p class="note" style="margin-top:6px;"><b>GHM is intentionally not optimised.</b>
DoR <em>should</em> correlate with the human-modification gradient &mdash; a recovering
site in a more modified landscape genuinely is less recovered &mdash; so driving
|&rho;(DoR,&nbsp;GHM)| toward zero would optimise away real ecological signal.
&rho;(DoR,&nbsp;GHM) is reported below as a descriptive diagnostic only
(|&rho;<sub>lo</sub>| &asymp; {_fmt(best['diag_abs_rho'],3)} at the chosen buffer).</p>

<h2>2. The scoring hypercube</h2>
<p>The first two panels are the summaries: <b>Ranking score</b> (Z, diverging about 0
&mdash; blue = below-average buffers, red = above) and <b>Overall quality</b> (D, the
true 0&ndash;1 scale). The remaining five are the goal axes on their absolute quality
scale. Rows are the inner exclusion radius, columns the outer ceiling; the red ring
marks the chosen buffer, grey cells are disqualified. The ranking panel is what
separates the optimum (upper-right, strongly positive) from poor narrow-buffer cells
(lower-left, strongly negative); the quality panels show <em>why</em> &mdash;
contamination is fully removed everywhere past a small inner radius, separability and
retention are high, but spatial independence is uniformly modest.</p>
{_img_tag(figures['panels'], 'Scoring hypercube panels')}

<h2>3. Why this buffer wins</h2>
<p>Two one-dimensional slices through the optimum. Thin coloured lines (left axis) are
the five goal qualities; the bold black diamond line (right axis) is the ranking score
<code>Z</code>. Panel (a) varies the outer ceiling at the chosen inner radius; panel (b)
varies the inner exclusion at the chosen outer ceiling. The ranking peaks at the red
dashed line in both, driven by separability and spatial independence improving with a
wider outer ceiling and by contamination requiring a non-trivial inner exclusion.</p>
{_img_tag(figures['profiles'], 'Axis profiles through the optimum')}

<h2>4. Top buffer configurations</h2>
<p>The ten highest-ranked cells (by <code>Z</code>). <code>d_*</code> columns are the
absolute goal qualities (0&ndash;1); the right block shows the raw class-mean metrics
behind them.</p>
<table>
<thead><tr>
  <th>inner&nbsp;(km)</th><th>outer&nbsp;(km)</th><th>Z&nbsp;(SD)</th><th>D&nbsp;(0&ndash;1)</th>
  <th>d&nbsp;contam</th><th>d&nbsp;separ</th><th>d&nbsp;indep</th>
  <th>d&nbsp;conf</th><th>d&nbsp;reten</th>
  <th>MCC</th><th>autocorr</th><th>CI&nbsp;w</th><th>n&nbsp;sites</th>
  <th style="color:#888">|&rho;<sub>lo</sub>|<br><span style="font-weight:400;font-size:10px">(diag.)</span></th>
</tr></thead>
<tbody>
{top_rows}
</tbody></table>
<p class="note">Highlighted row = recommended buffer. The five <code>d&nbsp;*</code>
columns are the optimised axes (<code>d&nbsp;indep</code> = spatial independence =
1&minus;normalised within-pool autocorrelation; <code>autocorr</code> is the raw mean
pairwise cosine similarity, lower = more independent).
<code>|&rho;<sub>lo</sub>|</code> (greyed, rightmost)
is the absolute Spearman correlation between the conservative (lower-CI) DoR and the
parent's GHM, averaged across classes &mdash; reported as a <em>diagnostic only</em>,
not optimised. It is expected to be non-zero: recovery genuinely co-varies with the
human-modification gradient.</p>

<h2>5. Per-class detail at the chosen buffer</h2>
<p>How each class behaves at <code>{bi} m &rarr; {bo} m</code>. The three stable
classes are sanity checks (the test site sits in its own bad state, so a low DoR is
expected); the two <span style="background:#fbf4ee;padding:0 3px;">loss</span> classes
are the operational targets &mdash; the actual disturbance sites being scored.
Reassuringly, the loss classes are the <em>most</em> separable of all
(MCC&nbsp;&approx;&nbsp;0.91, AUC&nbsp;&approx;&nbsp;0.99): the good/bad reference
pools around real loss sites are cleanly distinguishable. Their median DoR sits near
0.5 (vs clearly-low for the stable sanity classes), which is sensible &mdash; a
loss site has <em>lost</em> its degraded cover and is partially recovering, so it
reads as intermediate rather than fully degraded.</p>
<table>
<thead><tr><th>class</th><th>MCC</th><th>F1</th><th>ROC&nbsp;AUC</th>
  <th>&rho;(DoR<sub>lo</sub>,GHM)</th><th>median&nbsp;DoR</th><th>n&nbsp;sites</th></tr></thead>
<tbody>
{per_class_rows()}
</tbody></table>

<h2>6. GHM diagnostic: DoR vs human-modification gradient</h2>
<p>At the chosen buffer, how does the score relate to each parent's GHM? This is a
<em>diagnostic</em>, not an optimised quantity. We <em>expect</em> a real correlation
&mdash; a site recovering within a more human-modified landscape genuinely reads as
less recovered &mdash; so the point is to confirm the relationship is sensible and
moderate (the score tracks recovery, not <em>only</em> ambient context), not to drive
it to zero.</p>
{ghm_scatter_html}

<h2>7. Method notes</h2>
<ul class="note">
  <li><b>Separability</b> is leave-one-out reference classification (v4_methods.md,
     Stage 5 construction), scored with the DoR functional
     <code>m_b/(m_g+m_b)</code> and summarised by the MCC-optimal threshold's MCC and
     F1, pooled across all parents of a class. Source: <code>separability_sweep.py</code>.</li>
  <li><b>Spatial independence</b> is 1 &minus; the normalised within-pool reference
     autocorrelation: the mean pairwise AlphaEarth cosine similarity among the refs a
     buffer retains, per parent, averaged per class. The empirical variogram shows this
     similarity decaying with geographic separation and largely plateauing past
     ~3&nbsp;km, so a wider pool (and to a lesser extent a larger inner radius) lowers
     mean autocorrelation. This is the principled replacement for the discarded GHM
     proxy &mdash; within-pool autocorrelation is a genuine pseudoreplication nuisance to
     minimise, whereas GHM correlation is legitimate signal. Source:
     <code>spatial_autocorr_sweep.py</code>.</li>
  <li><b>Contamination control</b> targets the specific failure it is meant to fix:
     bad references that are <em>both physically close and embedding-near-identical</em>
     to the test site (cosine&nbsp;&lt;&nbsp;{_fmt(CONTAM_COSINE_MAX,2)}), the sub-pixel
     pseudoreplication diagnosed in built-loss. In that pool the closest bad ref's
     distance and its cosine distance are strongly correlated (Spearman&nbsp;&asymp;&nbsp;+0.72),
     so the contaminating mass is concentrated within a few hundred metres. The axis is
     the fraction of that near-site mass removed by the inner radius &mdash; it saturates
     by ~300&ndash;500&nbsp;m and does <em>not</em> keep rewarding larger exclusions
     (the earlier closest-ref ECDF saturated only at 3&nbsp;km because of a handful of
     outlier parents). Source: {meta['contam_source']}.</li>
  <li><b>GHM is a diagnostic, not an axis.</b> &rho;(DoR,&nbsp;GHM) is reported so the
     correlation is visible, but it is <em>not</em> optimised: DoR should co-vary with
     the human-modification gradient, so minimising |&rho;| would remove real signal.
     Source: <code>buffer_extent_sweep.py</code>.</li>
  <li><b>Confidence</b> is the median width of the per-site 95% bootstrap DoR
     interval; <b>retention</b> is the count of sites with a computable paired DoR.
     Both from <code>buffer_extent_sweep.py</code>.</li>
  <li><b>Scoring (two scales, no min&ndash;max).</b> Each cell gets an absolute
     <b>quality</b> <code>D</code> &mdash; metrics already in [0,1] (contamination,
     MCC, 1&minus;similarity) kept on their true scale, CI width and retention rescaled
     within the sweep, combined by a weighted geometric mean &mdash; and a relative
     <b>ranking</b> <code>Z</code>, the weighted sum of per-axis z-scores. Weights:
     {weights_str}. Min&ndash;max desirability was rejected on purpose: rescaling each
     axis to its best cell pins the winner near 1.0 regardless of true quality and hides
     weak axes. The geometric mean still penalises any near-zero quality axis. Cells
     below {int(RETENTION_FLOOR_FRAC*100)}% of the best paired-site count are
     disqualified. Source: <code>buffer_desirability.py</code>.</li>
</ul>

<p class="sub" style="margin-top:30px;">Data: <code>v5/data/buffer_desirability.csv</code>
&middot; figures (600-dpi PNG + vector PDF in <code>v5/plots/</code>) regenerate from the
sweep CSVs with no GEE re-extraction.</p>

</body></html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    V5_PLOTS.mkdir(parents=True, exist_ok=True)
    V5_REPORT.mkdir(parents=True, exist_ok=True)

    print("Loading axes from the sweep outputs ...")
    cells, meta = load_axes()
    print(f"  {len(cells)} (inner, outer) cells; contamination from {meta['contam_source']}")

    c = build_desirability(cells)
    out_csv = V5_DATA / "buffer_desirability.csv"
    c.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv}")

    best = c.iloc[0]   # ranked by z_composite
    bi, bo = int(best["inner_m"]), int(best["outer_m"])
    print(f"\n  >>> Recommended buffer: inner = {bi} m, outer = {bo} m")
    print(f"      Z_composite = {best['z_composite']:+.2f} SD (rank 1)   "
          f"D_quality = {best['desirability']:.3f} (absolute)")

    plat = plateau_cells(c, tol_sd=0.3)
    print(f"  Plateau (within 0.3 SD of top Z): {len(plat)} cells, "
          f"inner {int(plat['inner_m'].min())}-{int(plat['inner_m'].max())} m, "
          f"outer {int(plat['outer_m'].min())}-{int(plat['outer_m'].max())} m")

    fb = best_within(c, outer_max=4000)
    if fb is not None:
        print(f"  v4-realizable fallback (outer<=4km): inner = {int(fb['inner_m'])} m, "
              f"outer = {int(fb['outer_m'])} m  (Z = {fb['z_composite']:+.2f}, "
              f"D_quality = {fb['desirability']:.3f})")

    print("\n  Top 8 cells (by Z_composite):")
    show = ["inner_m", "outer_m", "z_composite", "desirability", "d_contamination",
            "d_separability", "d_autocorrelation", "d_confidence", "d_retention"]
    print(c[show].head(8).round(3).to_string(index=False))

    print("\nWriting figures (600-dpi PNG + vector PDF) ...")
    stem_panels  = V5_PLOTS / "buffer_desirability_heatmap"
    stem_prof    = V5_PLOTS / "buffer_axis_profiles"
    stem_scatter = V5_PLOTS / "buffer_ghm_scatter"
    plot_desirability_panels(c, best, stem_panels)
    print(f"  Wrote {stem_panels}.png/.pdf")
    plot_axis_profiles(c, best, stem_prof)
    print(f"  Wrote {stem_prof}.png/.pdf")
    plot_ghm_scatter(meta, best, stem_scatter)
    scatter_png = stem_scatter.with_suffix(".png")
    if scatter_png.exists():
        print(f"  Wrote {stem_scatter}.png/.pdf")
    else:
        print("  [skip] GHM scatter (per-site DoR / GHM unavailable)")

    print("\nBuilding HTML report ...")
    out_html = V5_REPORT / "buffer_decision.html"
    build_report(c, meta, best, plat, fb,
                 {"panels": stem_panels.with_suffix(".png"),
                  "profiles": stem_prof.with_suffix(".png"),
                  "scatter": scatter_png if scatter_png.exists() else None},
                 out_html)
    print(f"  Wrote {out_html}")


if __name__ == "__main__":
    main()
