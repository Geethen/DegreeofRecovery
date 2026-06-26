"""Self-contained HTML report: DoR-vs-GHM scatter plots for three scoring schemes.

Plots each test site's human-modification context (mean GHM all-threats `AA`,
2022, of its selected reference pool) against three Degree-of-Regeneration scores:

  1. v4 DoR          — kNN-5 cosine `dor_knn` from v4/data/test_site_dor_v4.csv
  2. v5 DoR          — kNN-5 cosine `dor` at the operational buffer (inner 3 km,
                       outer 8 km) from v5/data/buffer_extent_per_site.csv
  3. Ecoregion DoR   — `pct_dor` percentile-of-ecoregion score from
                       v5/data/test_site_ecoregion_percentile.csv (rescaled 0-1)

GHM is a *diagnostic*, not an optimisation target: DoR is expected to co-vary with
the human-modification gradient (a site regenerating in a more modified landscape
genuinely reads as less regenerated). The per-class Spearman/Pearson correlations are
reported per panel to characterise — not correct — that relationship, mirroring the
v5 methods' GHM treatment.

Output: v5/report/ghm_score_scatter.html (single file, Plotly via CDN).

Usage:
  python3 v5/scripts/analysis/build_ghm_score_scatter.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(BASE, "v5", "data")
V4_DATA = os.path.join(BASE, "v4", "data")
OUT = os.path.join(BASE, "v5", "report", "ghm_score_scatter.html")

V5_INNER_M, V5_OUTER_M = 3000, 8000  # operational buffer

CLASS_COLORS = {
    "stable_nature": "#2e7d32",
    "stable_crop":   "#f9a825",
    "stable_built":  "#6d4c41",
    "built_loss":    "#c62828",
    "crop_loss":     "#ad1457",
}
CLASS_ORDER = ["built_loss", "crop_loss", "stable_nature",
               "stable_crop", "stable_built"]


def load() -> pd.DataFrame:
    eco = pd.read_csv(os.path.join(DATA, "test_site_ecoregion_percentile.csv"))
    eco = eco[["parent_id", "parent_label", "pct_dor"]].copy()
    eco["eco_dor"] = eco["pct_dor"] / 100.0  # percentile -> 0-1

    v5 = pd.read_csv(os.path.join(DATA, "buffer_extent_per_site.csv"))
    v5 = v5[(v5.inner_m == V5_INNER_M) & (v5.outer_m == V5_OUTER_M)]
    v5 = v5[["parent_id", "dor"]].rename(columns={"dor": "dor_v5"})

    v4 = pd.read_csv(os.path.join(V4_DATA, "test_site_dor_v4.csv"))
    v4 = v4[["parent_id", "dor_knn"]].rename(columns={"dor_knn": "dor_v4"})

    # Per-parent GHM = mean ghm_aa over the SELECTED (operational) reference pool —
    # the human-modification context of the neighbourhood the score is built on.
    refs = []
    for f in ("v5_candidate_refs_alphaearth.parquet",
              "v5_stable_refs_alphaearth.parquet"):
        refs.append(pd.read_parquet(
            os.path.join(DATA, f), columns=["parent_id", "ghm_aa", "selection"]))
    refs = pd.concat(refs, ignore_index=True)
    ghm = (refs[refs.selection == "selected"]
           .groupby("parent_id").ghm_aa.mean()
           .rename("ghm").reset_index())

    m = (eco.merge(ghm, on="parent_id", how="left")
            .merge(v5, on="parent_id", how="left")
            .merge(v4, on="parent_id", how="left"))
    return m


def corr(x: pd.Series, y: pd.Series):
    """Return (rho, p, n) or (None, None, n) when too few pairs."""
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(sub) < 5:
        return None, None, len(sub)
    rs, ps = stats.spearmanr(sub.x, sub.y)
    return rs, ps, len(sub)


def cell(rho, p):
    """Coloured table cell HTML for a correlation: blue=+, red=-, * if p<.05."""
    if rho is None:
        return '<td class="v dim">—</td>'
    sig = "*" if (p is not None and p < 0.05) else ""
    shade = min(abs(rho), 0.9)
    bg = (f"rgba(33,102,172,{0.10 + 0.55 * shade:.2f})" if rho >= 0
          else f"rgba(178,24,43,{0.10 + 0.55 * shade:.2f})")
    return (f'<td class="v" style="background:{bg}">'
            f'{rho:+.2f}<span class="star">{sig}</span></td>')


METRICS = [
    ("dor_v4",  "v4 DoR (kNN-5)",
     "v4 kNN-5 cosine DoR — fixed-buffer references."),
    ("dor_v5",  "v5 DoR (kNN-5, 3–8 km)",
     "v5 kNN-5 cosine DoR at the operational buffer (inner 3 km, outer 8 km)."),
    ("eco_dor", "Ecoregion DoR (percentile)",
     "Site similarity expressed as a percentile within its RESOLVE ecoregion "
     "reference cloud (rescaled to 0–1)."),
]
METRIC_LABEL = {"dor_v4": "v4 DoR", "dor_v5": "v5 DoR",
                "eco_dor": "Ecoregion DoR"}


def build_panels(df: pd.DataFrame):
    """Return (panels, ghm_corr) — ghm_corr is {row -> {metric_col -> (rho,p,n)}}
    with row ∈ {"all class", <class>...}."""
    panels = []
    ghm_corr = {"all class": {}}
    for cls in CLASS_ORDER:
        ghm_corr[cls] = {}

    for i, (col, title, blurb) in enumerate(METRICS):
        sub = df[["parent_id", "parent_label", "ghm", col]].dropna(
            subset=["ghm", col])
        ghm_corr["all class"][col] = corr(sub.ghm, sub[col])
        for cls in CLASS_ORDER:
            cs = sub[sub.parent_label == cls]
            ghm_corr[cls][col] = corr(cs.ghm, cs[col])

        traces = []
        for cls in CLASS_ORDER:
            cs = sub[sub.parent_label == cls]
            if not len(cs):
                continue
            traces.append({
                "type": "scatter", "mode": "markers", "name": cls,
                "legendgroup": cls, "showlegend": i == 0,
                "x": cs.ghm.round(4).tolist(),
                "y": cs[col].round(4).tolist(),
                "text": cs.parent_id.tolist(),
                "marker": {"size": 5, "opacity": 0.55,
                           "color": CLASS_COLORS[cls],
                           "line": {"width": 0}},
                "hovertemplate": (f"<b>{cls}</b><br>GHM=%{{x:.3f}}<br>"
                                  f"{title}=%{{y:.3f}}<br>%{{text}}<extra></extra>"),
            })
        # LOWESS-free linear fit line across all classes for visual trend
        fit = sub.dropna(subset=["ghm", col])
        if len(fit) >= 5:
            b, a = np.polyfit(fit.ghm, fit[col], 1)
            xs = np.linspace(fit.ghm.min(), fit.ghm.max(), 50)
            traces.append({
                "type": "scatter", "mode": "lines", "name": "trend",
                "showlegend": False, "hoverinfo": "skip",
                "x": xs.round(4).tolist(), "y": (a + b * xs).round(4).tolist(),
                "line": {"color": "#111", "width": 1.6, "dash": "dash"},
            })

        layout = {
            "title": {"text": title, "font": {"size": 14}, "x": 0.02},
            "xaxis": {"title": "GHM (all-threats AA, 2022)", "range": [0, 1],
                      "gridcolor": "#eee", "zeroline": False},
            "yaxis": {"title": "DoR score",
                      "range": [-0.02, 1.02], "gridcolor": "#eee",
                      "zeroline": False},
            "margin": {"l": 55, "r": 12, "t": 34, "b": 44},
            "plot_bgcolor": "#fff", "paper_bgcolor": "#fff",
            "legend": {"orientation": "h", "y": -0.2, "font": {"size": 11}},
            "font": {"family": "Helvetica Neue,Arial,sans-serif", "size": 11.5},
        }
        panels.append({
            "id": f"plot{i}", "title": title, "blurb": blurb,
            "data": traces, "layout": layout,
        })
    return panels, ghm_corr


def score_agreement(df: pd.DataFrame):
    """Pairwise Spearman between the three scores (how much they agree)."""
    pairs = [("dor_v4", "dor_v5"), ("dor_v4", "eco_dor"), ("dor_v5", "eco_dor")]
    out = []
    for a, b in pairs:
        rho, p, n = corr(df[a], df[b])
        out.append((METRIC_LABEL[a], METRIC_LABEL[b], rho, p, n))
    return out


def render(panels, ghm_corr, agree, n_total) -> str:
    plot_divs = "\n".join(
        f'<div class="card"><div class="cap">{p["blurb"]}</div>'
        f'<div id="{p["id"]}" class="plot"></div></div>'
        for p in panels)

    # ── GHM-correlation matrix: rows = class, columns = the three scores ──
    head = "".join(f"<th>{lbl}</th>" for lbl in METRIC_LABEL.values())
    body_rows = []
    for row in ["all class"] + CLASS_ORDER:
        cells = "".join(cell(*ghm_corr[row][col][:2]) for col in METRIC_LABEL)
        # smallest n across the row's three scores, for the n column
        ns = [ghm_corr[row][col][2] for col in METRIC_LABEL]
        n_disp = f"{min(ns)}–{max(ns)}" if min(ns) != max(ns) else str(min(ns))
        rowcls = " class='allrow'" if row == "all class" else ""
        body_rows.append(
            f"<tr{rowcls}><td class='k'>{row}</td>{cells}"
            f"<td class='v dim'>{n_disp}</td></tr>")
    ghm_table = (
        "<table class='mx'><thead><tr><th>class</th>"
        f"{head}<th>n</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>")

    # ── Score-vs-score agreement table ──
    agree_rows = "".join(
        f"<tr><td class='k'>{a} vs {b}</td>{cell(rho, p)}"
        f"<td class='v dim'>{n}</td></tr>"
        for a, b, rho, p, n in agree)
    agree_table = (
        "<table class='mx'><thead><tr><th>score pair</th>"
        "<th>Spearman ρ</th><th>n</th></tr></thead>"
        f"<tbody>{agree_rows}</tbody></table>")

    spec = json.dumps([{"id": p["id"], "data": p["data"],
                        "layout": p["layout"]} for p in panels])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DoR vs GHM — v4 / v5 / ecoregion</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ --ink:#111; --muted:#555; --line:#ddd; }}
  body {{ margin:0; font-family:"Helvetica Neue",Arial,sans-serif; color:var(--ink);
         background:#fafafa; }}
  header {{ padding:16px 22px 12px; border-bottom:1px solid var(--line); background:#fff; }}
  header h1 {{ margin:0 0 4px; font-size:19px; }}
  header p {{ margin:0; font-size:12.8px; color:var(--muted); max-width:96ch; line-height:1.5; }}
  header p em {{ color:var(--ink); font-style:italic; }}
  main {{ padding:14px 18px 0; }}
  .row {{ display:flex; gap:14px; flex-wrap:wrap; }}
  .card {{ flex:1 1 380px; min-width:340px; background:#fff; border:1px solid var(--line);
          border-radius:9px; padding:10px 10px 4px; }}
  .card .cap {{ font-size:12px; color:var(--muted); line-height:1.45; padding:0 4px 4px; }}
  .plot {{ width:100%; height:380px; }}
  h2.sec {{ font-size:15px; margin:22px 22px 4px; }}
  p.subnote {{ font-size:12px; color:var(--muted); margin:0 22px 8px; max-width:96ch; }}
  .crow {{ display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start;
          padding:0 22px 8px; }}
  table.mx {{ border-collapse:collapse; font-size:12.5px; background:#fff;
             border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  table.mx th {{ background:#f3f3f3; color:var(--muted); font-weight:600;
                text-align:right; padding:7px 11px; font-size:11.5px;
                border-bottom:1px solid var(--line); }}
  table.mx th:first-child {{ text-align:left; }}
  table.mx td {{ padding:6px 11px; border-bottom:1px solid #eee;
                text-align:right; font-variant-numeric:tabular-nums; }}
  table.mx td.k {{ text-align:left; color:var(--ink); }}
  table.mx td.dim {{ color:var(--muted); }}
  table.mx tr.allrow td {{ font-weight:600; border-bottom:2px solid var(--line); }}
  table.mx .star {{ color:#111; font-weight:700; }}
  table.mx tr:last-child td {{ border-bottom:none; }}
  footer {{ font-size:11.5px; color:var(--muted); padding:10px 22px 18px;
           border-top:1px solid var(--line); background:#fff; margin-top:16px; }}
  code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:11.5px; }}
</style></head>
<body>
<header>
    <h1>Degree of Regeneration vs Human Modification</h1>
  <p>Each point is one test site. The x-axis is its <em>human-modification context</em>
     — the mean Global Human Modification all-threats value (GHM <code>AA</code>, 2022)
     over the site's <em>selected</em> reference pool. The three panels plot three DoR
     scoring schemes against that gradient: v4 (fixed buffer), v5 (operational
     3–8&nbsp;km buffer), and the ecoregion percentile score. GHM is reported as a
     <em>diagnostic</em>, never optimised: DoR is <em>expected</em> to fall as human
     modification rises, so a moderate negative slope is the correct, ecologically
     sensible behaviour — not a confound to remove.</p>
</header>
<main>
  <div class="row">{plot_divs}</div>
</main>
<h2 class="sec">DoR vs GHM — Spearman ρ by class</h2>
<p class="subnote">Each cell is the rank correlation between a score and GHM context
  for that class. <span style="color:#2166ac">Blue</span> = positive,
  <span style="color:#b2182b">red</span> = negative; deeper shade = stronger;
  <b>*</b> marks p&nbsp;&lt;&nbsp;0.05. Weak negative values are expected and correct.</p>
<div class="crow">{ghm_table}</div>
<h2 class="sec">Do the three scores agree? (Spearman ρ between schemes)</h2>
<p class="subnote">How closely the scoring schemes rank the same sites. v4 and v5 are
  near-substitutable; the ecoregion percentile is a looser, related signal.</p>
<div class="crow">{agree_table}</div>
<footer>
  Self-contained report — {n_total} test sites with a GHM context and at least one
  DoR score. Dashed line = OLS fit across all classes (visual trend only).
  GHM context = mean <code>ghm_aa</code> over selected references; ecoregion DoR
  rescaled from percentile to 0–1. Generated by
  <code>build_ghm_score_scatter.py</code>.
</footer>
<script>
const SPEC = {spec};
SPEC.forEach(p => Plotly.newPlot(p.id, p.data, p.layout,
  {{responsive:true, displaylogo:false,
    modeBarButtonsToRemove:['lasso2d','select2d']}}));
</script>
</body></html>"""


def main():
    df = load()
    panels, ghm_corr = build_panels(df)
    agree = score_agreement(df)
    n_total = df.dropna(subset=["ghm"]).loc[
        lambda d: d[["dor_v4", "dor_v5", "eco_dor"]].notna().any(axis=1)].shape[0]
    html = render(panels, ghm_corr, agree, n_total)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"[OK] wrote {OUT}")
    print("DoR vs GHM (all class):")
    for col, lbl in METRIC_LABEL.items():
        rho, p, n = ghm_corr["all class"][col]
        print(f"  {lbl}: rho={rho:+.2f} p={p:.2g} n={n}")
    print("Score agreement:")
    for a, b, rho, p, n in agree:
        print(f"  {a} vs {b}: rho={rho:+.2f} n={n}")


if __name__ == "__main__":
    main()
