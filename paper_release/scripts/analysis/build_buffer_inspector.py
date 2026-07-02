#!/usr/bin/env python3
"""Build a self-contained interactive site-inspection page for the v5 DoR scatters.

Companion to the *static* `buffer_ghm_scatter.{png,pdf}` (see `plot_ghm_scatter`
in buffer_desirability.py). The static figure confirms the per-class DoR-vs-GHM

distributions; this page lets you click any point and jump to the site on the
ground (Google Earth + Esri Wayback time-series imagery) to sanity-check the
score against what is actually there.

Inputs (all local, no Earth Engine, no DoR recomputation):
  - v5/data/buffer_extent_per_site.csv  — per-site DoR + 95% bootstrap CI at every
    swept (inner, outer); filtered here to the chosen 3 km -> 8 km cell.
  - v4/data/test_site_alphaearth_2024_v4.parquet         — parent lat/lon (stable)
  - v5/data/test_site_alphaearth_2024_candidate.parquet  — parent lat/lon (loss)
  - per-parent median GHM via _load_parent_ghm() (buffer_desirability.py)
  - v4/data/calibrated_thresholds_v4.json  — per-class DoR thresholds (stable only)

Output: v5/report/buffer_inspector.html  (open in any browser; Plotly via CDN).

This script consumes the analysis outputs; it does NOT modify buffer_desirability.py,
figstyle.py, the figures, or v5_methods.md.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

# Project layout: this file lives in v5/scripts/analysis/.
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # .../degree_of_recovery
V4_DATA = ROOT / "v4" / "data"
V5_DATA = ROOT / "v5" / "data"
V5_REPORT = ROOT / "v5" / "report"

# Reuse the analysis helpers + the figure palette so this view matches the report.
sys.path.insert(0, str(HERE.parent))
from buffer_desirability import _load_parent_ghm  # noqa: E402
import figstyle as fst                            # noqa: E402

# Chosen buffer (the cell whose sites are inspected). See v5-buffer-decision.
INNER_M, OUTER_M = 3000, 8000

CLASS_ORDER = ["stable_nature", "stable_crop", "stable_built",
               "built_loss", "crop_loss"]


def load_sites() -> pd.DataFrame:
    """Join per-site DoR + CI (at the chosen buffer) to parent lat/lon and GHM.

    Mirrors plot_ghm_scatter's filtering: drop sites without a computable DoR or
    without a GHM value, exactly as the static scatter does. Also left-joins the
    ecoregion percentile rank (see score_test_sites_by_ecoregion.py) so each site
    carries its DoR *and* its rank within its ecoregion's reference distribution."""
    ps = pd.read_csv(V5_DATA / "buffer_extent_per_site.csv")
    ps = ps[(ps.inner_m == INNER_M) & (ps.outer_m == OUTER_M)].copy()
    ps = ps.dropna(subset=["dor"])

    con = duckdb.connect()
    geo = con.execute(
        """
        SELECT CAST(parent_id AS VARCHAR) AS parent_id,
               geo.coordinates[1] AS lon, geo.coordinates[2] AS lat
        FROM read_parquet([?, ?])
        """,
        [str(V4_DATA / "test_site_alphaearth_2024_v4.parquet"),
         str(V5_DATA / "test_site_alphaearth_2024_candidate.parquet")],
    ).df()
    con.close()

    ghm = _load_parent_ghm()
    df = ps.merge(geo, on="parent_id", how="left")
    df["ghm"] = df["parent_id"].map(ghm)
    df = df.dropna(subset=["lat", "lon", "ghm"]).reset_index(drop=True)

    df = df.merge(load_percentiles(), on="parent_id", how="left")
    return df


def load_percentiles() -> pd.DataFrame:
    """Ecoregion percentile ranks per parent (test_site_ecoregion_percentile.csv).

    Returns parent_id + the percentile columns used by the panel. Absent file =>
    empty frame so the merge is a no-op and the inspector still builds."""
    p = V5_DATA / "test_site_ecoregion_percentile.csv"
    cols = ["parent_id", "eco_id", "pct_vs_all", "pct_vs_good",
            "pct_vs_bad", "pct_dor", "n_refs_good", "n_refs_bad"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    pct = pd.read_csv(p)
    pct["parent_id"] = pct["parent_id"].astype(str)
    keep = [c for c in cols if c in pct.columns]
    return pct[keep]


def load_thresholds() -> dict[str, float]:
    """Per-class DoR threshold for the categorical call. Only the three stable
    classes are calibrated in v4; loss classes fall through to 'n/a'."""
    p = V4_DATA / "calibrated_thresholds_v4.json"
    if p.exists():
        try:
            return {k: float(v) for k, v in json.loads(p.read_text()).items()}
        except (ValueError, json.JSONDecodeError):
            pass
    return {}


def build_payload(df: pd.DataFrame, thr: dict[str, float]) -> list[dict]:
    """One record per site for embedding in the page."""
    def _opt(v, nd=1):
        return None if pd.isna(v) else round(float(v), nd)

    def _optint(v):
        return None if pd.isna(v) else int(v)

    recs = []
    for r in df.itertuples(index=False):
        recs.append({
            "parent_id": str(r.parent_id),
            "parent_label": r.parent_label,
            "lat": round(float(r.lat), 6),
            "lon": round(float(r.lon), 6),
            "dor": round(float(r.dor), 4),
            "dor_lo": None if pd.isna(r.dor_lo) else round(float(r.dor_lo), 4),
            "dor_hi": None if pd.isna(r.dor_hi) else round(float(r.dor_hi), 4),
            "ghm": round(float(r.ghm), 4),
            "n_good": None if pd.isna(r.n_good) else int(r.n_good),
            "n_bad": None if pd.isna(r.n_bad) else int(r.n_bad),
            # ecoregion percentile rank (score_test_sites_by_ecoregion.py)
            "eco_id": _optint(getattr(r, "eco_id", float("nan"))),
            "pct_all": _opt(getattr(r, "pct_vs_all", float("nan"))),
            "pct_good": _opt(getattr(r, "pct_vs_good", float("nan"))),
            "pct_bad": _opt(getattr(r, "pct_vs_bad", float("nan"))),
            "pct_dor": _opt(getattr(r, "pct_dor", float("nan"))),
        })
    return recs


def _build_stamp() -> str:
    """A short provenance string for the footer: build date + git short-SHA
    (with a '-dirty' marker if the tree has uncommitted changes), so a saved
    copy of this page can be traced back to the run that produced it."""
    date = datetime.now().strftime("%Y-%m-%d")
    sha = ""
    try:
        sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if sha and dirty:
            sha += "-dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return f"built {date}" + (f" · {sha}" if sha else "")


def render_html(records: list[dict], thr: dict[str, float]) -> str:
    classes = [c for c in CLASS_ORDER
               if any(r["parent_label"] == c for r in records)]
    config = {
        "classOrder": classes,
        "classColors": {c: fst.CLASS_COLORS.get(c, "#444") for c in classes},
        "classLabels": {c: fst.CLASS_LABELS.get(c, c) for c in classes},
        "thresholds": thr,
        "innerKm": INNER_M / 1000,
        "outerKm": OUTER_M / 1000,
    }
    data_json = json.dumps(records, separators=(",", ":"))
    config_json = json.dumps(config)
    return _HTML_TEMPLATE.replace("/*__DATA__*/", data_json) \
                         .replace("/*__CONFIG__*/", config_json) \
                         .replace("__BUILD__", _build_stamp())


# --- The page. Plotly from CDN; subplots = one panel per class; click -> panel.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v5 DoR site inspector — DoR vs GHM at 3 km → 8 km</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --ink:#111; --muted:#555; --line:#ddd; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Helvetica Neue",Arial,sans-serif; color:var(--ink);
         background:#fafafa; }
  header { padding:14px 20px 8px; border-bottom:1px solid var(--line); background:#fff; }
  header h1 { margin:0 0 2px; font-size:18px; }
  header p { margin:0; font-size:12.5px; color:var(--muted); }
  header p.legend { margin-top:5px; font-size:12px; }
  header p.note { margin-top:4px; font-size:12px; max-width:74ch; }
  header p.note em { font-style:italic; color:var(--ink); }
  .lk { font-weight:700; margin:0 3px 0 12px; vertical-align:middle; }
  .lk.fit { color:#444; } .lk.one { color:#bbb; } .lk.half { color:#bbb; }
  .lk:first-child { margin-left:0; }
  .stamp { font-variant-numeric:tabular-nums; }
  .wrap { display:flex; gap:0; align-items:stretch; min-height:calc(100vh - 64px); }
  #plot { flex:1 1 auto; min-width:0; padding:8px 4px; }
  #panel { flex:0 0 330px; border-left:1px solid var(--line); background:#fff;
           padding:16px 18px; overflow-y:auto; }
  #panel h2 { margin:0 0 4px; font-size:15px; }
  #panel .hint { color:var(--muted); font-size:12.5px; line-height:1.5; }
  .pid { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
         word-break:break-all; color:var(--muted); }
  .chip { display:inline-block; padding:2px 9px; border-radius:11px; color:#fff;
          font-size:12px; font-weight:600; margin:6px 0 10px; }
  table.kv { width:100%; border-collapse:collapse; font-size:13px; margin:2px 0 12px; }
  table.kv td { padding:4px 2px; border-bottom:1px solid #eee; vertical-align:top; }
  table.kv td.k { color:var(--muted); width:46%; }
  table.kv td.v { text-align:right; font-variant-numeric:tabular-nums; }
  .call { font-size:13px; padding:7px 10px; border-radius:6px; margin:2px 0 14px;
          border:1px solid var(--line); background:#f6f6f6; }
  .call b { font-weight:700; }
  .pcthead { font-size:12.5px; font-weight:700; color:var(--ink); margin:10px 0 2px;
             display:flex; justify-content:space-between; align-items:baseline; }
  .pcthead .eco { font-weight:400; font-size:11px; color:var(--muted);
                  font-family:ui-monospace,Menlo,monospace; }
  .pctnum { font-variant-numeric:tabular-nums; font-size:12.5px; }
  .pctbar { height:5px; background:#eee; border-radius:3px; overflow:hidden; margin-top:2px; }
  .pctfill { height:100%; border-radius:3px; }
  a.maplink { display:block; padding:9px 12px; margin:7px 0; border-radius:7px;
              text-decoration:none; font-size:13.5px; font-weight:600;
              border:1px solid var(--line); color:#fff; }
  a.ge { background:#1a73e8; } a.ge:hover { background:#1664cf; }
  a.wb { background:#0a7d4f; } a.wb:hover { background:#086a43; }
  a.es { background:#444; }    a.es:hover { background:#333; }
  .status { min-height:1.2em; margin:6px 0 0; font-size:12px; color:var(--muted); }
  .coords { font-size:12px; color:var(--muted); margin:8px 0 2px;
            font-variant-numeric:tabular-nums; }
  footer { font-size:11.5px; color:var(--muted); padding:6px 20px 14px;
           border-top:1px solid var(--line); background:#fff; }
  code { background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:11.5px; }
</style>
</head>
<body>
<header>
  <h1>Degree-of-Regeneration site inspector</h1>
  <p>DoR vs parent human-modification gradient (GHM) at the chosen reference
     buffer (<span id="bufstr"></span>). Click any point to inspect that site on
     the ground. Companion to <code>buffer_ghm_scatter.png</code>.</p>
  <p class="legend">
     <span class="lk fit">━</span> OLS fit (DoR on GHM)
     <span class="lk one">– –</span> 1:1 line (y = x)
     <span class="lk half">·····</span> per-class call threshold (<code>0.5*</code> = uncalibrated guide)
     &nbsp;·&nbsp; ρ = Spearman rank, R² = linear fit, n = sites
  </p>
  <p class="note">A near-flat fit is the expected, healthy result: DoR should
     <em>not</em> track GHM. A steep slope or high R² would flag that the score is
     being driven by the surrounding modification gradient rather than on-site
     recovery — worth inspecting on the ground.</p>
</header>
<div class="wrap">
  <div id="plot"></div>
  <aside id="panel">
    <h2>No site selected</h2>
    <p class="hint">Click a point in any panel to see its scores and to open the
       site in Google Earth and the Esri Wayback imagery time-series. The dotted
       line marks DoR&nbsp;=&nbsp;0.5.</p>
  </aside>
</div>
<footer id="foot"></footer>

<script>
const DATA   = /*__DATA__*/;
const CONFIG = /*__CONFIG__*/;

// ----- build per-class subplots -------------------------------------------
const classes = CONFIG.classOrder;
const ncol = Math.min(3, classes.length);
const nrow = Math.ceil(classes.length / ncol);

const traces = [];
const annotations = [];
const shapes = [];
const layoutAxes = {};
const byClass = {};
classes.forEach(c => byClass[c] = DATA.filter(d => d.parent_label === c));

function spearman(xs, ys) {
  const n = xs.length; if (n < 3) return NaN;
  const rank = a => {
    const idx = a.map((v,i)=>[v,i]).sort((p,q)=>p[0]-q[0]);
    const r = new Array(a.length);
    for (let i=0;i<idx.length;) {
      let j=i; while (j+1<idx.length && idx[j+1][0]===idx[i][0]) j++;
      const avg = (i+j)/2 + 1;
      for (let k=i;k<=j;k++) r[idx[k][1]] = avg;
      i = j+1;
    }
    return r;
  };
  const rx = rank(xs), ry = rank(ys);
  const mx = rx.reduce((a,b)=>a+b,0)/n, my = ry.reduce((a,b)=>a+b,0)/n;
  let num=0, dx=0, dy=0;
  for (let i=0;i<n;i++){ const a=rx[i]-mx, b=ry[i]-my; num+=a*b; dx+=a*a; dy+=b*b; }
  return num/Math.sqrt(dx*dy);
}

// Ordinary-least-squares fit of y on x, plus Pearson r. Returns the fitted line
// endpoints over the observed x-range so it can be drawn as a Plotly shape.
function linfit(xs, ys) {
  const n = xs.length;
  if (n < 3) return null;
  const mx = xs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n;
  let sxx=0, sxy=0, syy=0;
  for (let i=0;i<n;i++){ const a=xs[i]-mx, b=ys[i]-my; sxx+=a*a; sxy+=a*b; syy+=b*b; }
  if (sxx === 0 || syy === 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const r = sxy / Math.sqrt(sxx * syy);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  return { slope, intercept, r, r2: r*r,
           x0, y0: intercept + slope*x0,
           x1, y1: intercept + slope*x1 };
}

// Explicit per-panel domains (we lay the grid out by hand rather than using
// Plotly's `grid:` so the inter-panel gaps are generous and nothing squashes).
const COL_GAP = 0.07, ROW_GAP = 0.13;   // fraction of paper between panels
const cellW = (1 - COL_GAP * (ncol - 1)) / ncol;
const cellH = (1 - ROW_GAP * (nrow - 1)) / nrow;
const axisOf = {};   // parent_label -> {x:'x1', y:'y1'}

classes.forEach((c, i) => {
  const g = byClass[c];
  const color = CONFIG.classColors[c];
  const xId = 'x' + (i+1), yId = 'y' + (i+1);
  axisOf[c] = { x: xId, y: yId };
  traces.push({
    type: 'scatter', mode: 'markers',
    x: g.map(d=>d.ghm), y: g.map(d=>d.dor),
    // customdata[0] = parent_id (click lookup); [1] = class label (hover)
    customdata: g.map(d=>[d.parent_id, CONFIG.classLabels[c]]),
    marker: { size: 7, color: color, opacity: 0.6,
              line: { color:'white', width:0.5 } },
    xaxis: xId, yaxis: yId, name: CONFIG.classLabels[c], showlegend: false,
    hovertemplate: '<b>%{customdata[1]}</b><br>'
                 + 'DoR %{y:.3f} · GHM %{x:.3f}<br>'
                 + '<span style="font-size:10px">%{customdata[0]} · click to inspect</span>'
                 + '<extra></extra>',
  });

  const col = i % ncol, row = Math.floor(i / ncol);
  const x0 = col * (cellW + COL_GAP);
  const x1 = x0 + cellW;
  const y1 = 1 - row * (cellH + ROW_GAP);   // top of this panel (paper coords)
  const y0 = y1 - cellH;

  // Per-class call threshold (the boundary the side-panel "Call" actually uses).
  // Calibrated classes get their v4 threshold; uncalibrated (loss) classes fall
  // back to a 0.5 guide, drawn lighter and labelled so the difference is clear.
  const thr = CONFIG.thresholds[c];
  const tVal = (thr === undefined ? 0.5 : thr);
  const tCal = thr !== undefined;
  shapes.push({ type:'line', xref:xId+' domain', yref:yId,
                x0:0, x1:1, y0:tVal, y1:tVal,
                line:{ color: tCal ? '#999' : '#ccc', width:1, dash:'dot' } });
  annotations.push({
    xref:xId+' domain', yref:yId, x:0.98, y:tVal,
    text: tCal ? tVal.toFixed(2) : '0.5*',
    showarrow:false, xanchor:'right', yanchor:'bottom',
    font:{ size:9, color: tCal ? '#999' : '#bbb' },
  });
  // 1:1 identity line (y = x). DoR and GHM share the [0,1] scale, so the
  // diagonal is a visual reference for "DoR tracks the modification gradient";
  // it is NOT an expected relationship, just a fixed guide.
  shapes.push({ type:'line', xref:xId, yref:yId, layer:'below',
                x0:0, x1:1, y0:0, y1:1,
                line:{ color:'#ccc', width:1, dash:'dash' } });
  // OLS fit of DoR on GHM, drawn across the observed GHM range in the class hue.
  const fit = linfit(g.map(d=>d.ghm), g.map(d=>d.dor));
  if (fit) {
    shapes.push({ type:'line', xref:xId, yref:yId,
                  x0:fit.x0, y0:fit.y0, x1:fit.x1, y1:fit.y1,
                  line:{ color:color, width:1.6 } });
  }
  // class title above the panel
  annotations.push({
    xref:'paper', yref:'paper', x:(x0+x1)/2, y:Math.min(y1+0.025, 1.0),
    text:'<b>'+CONFIG.classLabels[c]+'</b>', showarrow:false,
    font:{ size:13, color:color }, xanchor:'center', yanchor:'bottom',
  });
  // Spearman ρ (rank), R² (linear fit) + n, inside the panel
  const rs = spearman(g.map(d=>d.ghm), g.map(d=>d.dor));
  const rsStr = isNaN(rs) ? '–' : (rs>=0?'+':'') + rs.toFixed(2);
  const r2Str = fit ? fit.r2.toFixed(2) : '–';
  annotations.push({
    xref:xId+' domain', yref:yId+' domain', x:0.04, y:0.96,
    text:'ρ = '+rsStr+'<br>R² = '+r2Str+'<br>n = '+g.length,
    showarrow:false, align:'left', xanchor:'left', yanchor:'top',
    font:{ size:11, color:'#222' },
    bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ddd', borderwidth:1,
    borderpad:3,
  });

  // bind the axes to their hand-computed domains
  const lastRow = row === nrow - 1 || (i + ncol >= classes.length);
  layoutAxes['xaxis'+(i+1)] = {
    domain:[x0, x1], anchor:yId, range:[-0.02, 1.02],
    title:{ text: lastRow ? 'Parent GHM' : '', font:{size:11} },
    zeroline:false, showline:true, linecolor:'#333', ticks:'outside',
  };
  layoutAxes['yaxis'+(i+1)] = {
    domain:[y0, y1], anchor:xId, range:[-0.02, 1.02],
    title:{ text: col === 0 ? 'DoR' : '', font:{size:11} },
    zeroline:false, showline:true, linecolor:'#333', ticks:'outside',
  };
});

const layout = Object.assign({
  margin:{ l:55, r:14, t:34, b:46 },
  height: 380 * nrow,
  shapes, annotations,
  hovermode:'closest', plot_bgcolor:'white', paper_bgcolor:'white',
  font:{ family:'Helvetica Neue, Arial, sans-serif', size:11, color:'#111' },
}, layoutAxes);

Plotly.newPlot('plot', traces, layout,
               { responsive:true, displaylogo:false,
                 modeBarButtonsToRemove:['lasso2d','select2d'] });

// ----- selection highlight -------------------------------------------------
let hiTrace = null;
function highlight(d) {
  const ax = axisOf[d.parent_label];
  if (hiTrace === null) {
    Plotly.addTraces('plot', {
      type:'scatter', mode:'markers',
      x:[d.ghm], y:[d.dor],
      xaxis: ax.x, yaxis: ax.y,
      marker:{ size:16, color:'rgba(0,0,0,0)', symbol:'circle-open',
               line:{ color:'#D81B60', width:3 } },
      showlegend:false, hoverinfo:'skip',
    }).then(gd => { hiTrace = gd.data.length - 1; });
  } else {
    Plotly.restyle('plot', { x:[[d.ghm]], y:[[d.dor]],
                             xaxis:ax.x, yaxis:ax.y }, [hiTrace]);
  }
}

// ----- click -> side panel -------------------------------------------------
const byId = {};
DATA.forEach(d => byId[d.parent_id] = d);

function fmt(x, nd) { return (x===null||x===undefined||isNaN(x)) ? '–' : Number(x).toFixed(nd); }

function call(d) {
  const t = CONFIG.thresholds[d.parent_label];
  if (t === undefined) return '<span style="color:#888">not calibrated for this class</span>';
  const lo = d.dor_lo, hi = d.dor_hi;
  if (lo !== null && lo > t)  return '<b style="color:#009E73">regenerating</b> (CI entirely above '+t.toFixed(2)+')';
  if (hi !== null && hi < t)  return '<b style="color:#D55E00">degraded</b> (CI entirely below '+t.toFixed(2)+')';
  return '<b style="color:#888">indistinguishable</b> (CI spans '+t.toFixed(2)+')';
}

// Ecoregion percentile block: where this site's 2024 embedding ranks within its
// ecoregion's reference distribution (score_test_sites_by_ecoregion.py). Higher
// pct_good = more like the ecoregion's natural refs; higher pct_bad = more like
// its degraded refs; pct_dor blends the two into a single 0-100 regeneration rank.
function pctBar(label, v, color) {
  if (v === null || v === undefined || isNaN(v)) {
    return '<tr><td class="k">'+label+'</td><td class="v">–</td></tr>';
  }
  const w = Math.max(0, Math.min(100, v));
  return '<tr><td class="k">'+label+'</td>'
       + '<td class="v" style="padding-bottom:2px">'
       + '<div class="pctnum">'+v.toFixed(1)+'</div>'
       + '<div class="pctbar"><div class="pctfill" style="width:'+w+'%;background:'+color+'"></div></div>'
       + '</td></tr>';
}

function pctBlock(d) {
  if (d.pct_all === null && d.pct_dor === null) return '';
  const eco = (d.eco_id === null || d.eco_id === undefined) ? '?' : d.eco_id;
  return '<div class="pcthead">Ecoregion percentile <span class="eco">eco '+eco+'</span></div>'
       + '<table class="kv">'
       + pctBar('vs good refs', d.pct_good, '#009E73')
       + pctBar('vs bad refs',  d.pct_bad,  '#D55E00')
       + pctBar('vs all refs',  d.pct_all,  '#666')
       + pctBar('recovery rank (pct_dor)', d.pct_dor, '#1a73e8')
       + '</table>';
}

function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  ta.style.top = '0';
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    document.execCommand('copy');
  } finally {
    document.body.removeChild(ta);
  }
  return Promise.resolve();
}

function wireClipboardBackup() {
  const status = document.getElementById('link-status');
  if (!status) return;
  const links = document.querySelectorAll('#panel .maplink.ge, #panel .maplink.wb');
  links.forEach(link => {
    link.addEventListener('click', () => {
      copyTextToClipboard(link.href)
        .then(() => {
          status.textContent = 'Link copied to clipboard. Paste it if the new tab does not open.';
        })
        .catch(() => {
          status.textContent = 'Could not copy automatically. Use the link address from the browser if it does not open.';
        });
    });
  });
}

function showSite(d) {
  highlight(d);
  const color = CONFIG.classColors[d.parent_label];
  const label = CONFIG.classLabels[d.parent_label];
  const ge = 'https://earth.google.com/web/search/'+d.lat.toFixed(6)+','+d.lon.toFixed(6)+'/';
  const e = 0.005;
  const wb = 'https://livingatlas.arcgis.com/wayback/?ext='
           + (d.lon-e).toFixed(6)+','+(d.lat-e).toFixed(6)+','
           + (d.lon+e).toFixed(6)+','+(d.lat+e).toFixed(6);
  const es = 'https://www.arcgis.com/apps/mapviewer/index.html?center='
           + d.lon+','+d.lat+'&level=17&basemapUrl='
           + encodeURIComponent('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer');
  const ciStr = (d.dor_lo!==null && d.dor_hi!==null)
      ? '['+fmt(d.dor_lo,3)+', '+fmt(d.dor_hi,3)+']' : '–';
  document.getElementById('panel').innerHTML =
    '<h2>'+fmt(d.dor,3)+' <span style="font-size:13px;color:#888;font-weight:400">DoR</span></h2>'
    + '<span class="chip" style="background:'+color+'">'+label+'</span>'
    + '<div class="pid">'+d.parent_id+'</div>'
    + '<table class="kv">'
    + '<tr><td class="k">DoR (point)</td><td class="v">'+fmt(d.dor,3)+'</td></tr>'
    + '<tr><td class="k">95% CI</td><td class="v">'+ciStr+'</td></tr>'
    + '<tr><td class="k">GHM</td><td class="v">'+fmt(d.ghm,3)+'</td></tr>'
    + '<tr><td class="k">paired refs (good / bad)</td><td class="v">'+fmt(d.n_good,0)+' / '+fmt(d.n_bad,0)+'</td></tr>'
    + '</table>'
    + pctBlock(d)
    + '<div class="call"><b>Call:</b> '+call(d)+'</div>'
    + '<a class="maplink ge" id="ge-link" target="_blank" rel="noopener">Open in Google Earth ↗</a>'
    + '<a class="maplink wb" id="wb-link" target="_blank" rel="noopener">Esri Wayback (imagery time-series) ↗</a>'
    + '<a class="maplink es" id="es-link" target="_blank" rel="noopener">Esri World Imagery (current) ↗</a>'
    + '<div class="status" id="link-status">Tip: Ctrl-click to force a new tab. Google Earth opens with a red pin at the site; the link is also copied to your clipboard as a backup.</div>'
    + '<div class="coords">lat '+d.lat+', lon '+d.lon+'</div>';
  document.getElementById('ge-link').href = ge;
  document.getElementById('wb-link').href = wb;
  document.getElementById('es-link').href = es;
  wireClipboardBackup();
}

document.getElementById('plot').on('plotly_click', ev => {
  const pt = ev.points[0];
  if (!pt || pt.customdata === undefined) return;
  const pid = Array.isArray(pt.customdata) ? pt.customdata[0] : pt.customdata;
  const d = byId[pid];
  if (d) showSite(d);
});

// ----- header + footer text -----------------------------------------------
document.getElementById('bufstr').textContent =
  CONFIG.innerKm + ' km → ' + CONFIG.outerKm + ' km';
document.getElementById('foot').innerHTML =
  DATA.length + ' sites across ' + classes.length + ' classes at the chosen buffer · '
  + 'GHM is a diagnostic, not an optimisation target · '
  + 'scores are read from <code>buffer_extent_per_site.csv</code> (no recomputation) · '
  + '<span class="stamp">__BUILD__</span>';
</script>
</body>
</html>
"""


def main() -> None:
    df = load_sites()
    thr = load_thresholds()
    records = build_payload(df, thr)
    html = render_html(records, thr)
    V5_REPORT.mkdir(parents=True, exist_ok=True)
    out = V5_REPORT / "buffer_inspector.html"
    out.write_text(html, encoding="utf-8")
    n_by_class = df["parent_label"].value_counts().reindex(CLASS_ORDER).dropna()
    print(f"wrote {out}  ({len(records)} sites)")
    for cls, n in n_by_class.items():
        print(f"  {cls:<14} {int(n):>4}")
    if thr:
        print("thresholds:", {k: round(v, 3) for k, v in thr.items()})


if __name__ == "__main__":
    main()
