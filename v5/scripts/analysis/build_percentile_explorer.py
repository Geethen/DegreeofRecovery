#!/usr/bin/env python3
"""Build a self-contained interactive report for the ecoregion percentile ranks.

Companion to the DoR site inspector (build_buffer_inspector.py). Where the
inspector ranks each site's DoR vs the parent's GHM at the chosen buffer, this
report explores the *ecoregion percentile* produced by
score_test_sites_by_ecoregion.py: how each parent test site's 2024 embedding
ranks within the cosine-similarity distribution of its ecoregion's reference
points (the C_eco methodology from RECOVER/score_by_bioregion.py).

Three things to explore:
  * Regeneration quadrant: pct_vs_good (x) vs pct_vs_bad (y), one panel per class.
    Top-left = looks like good refs, not bad = regenerated. Click a point to open
    the site in Google Earth / Esri Wayback (same UX as the DoR inspector).
  * Per-class percentile distributions (good / bad / regeneration-rank), as overlaid
    histograms — the population-level signal.
  * A sortable per-class summary table (median percentiles, n).

Inputs (all local, no Earth Engine):
  v5/data/test_site_ecoregion_percentile.csv             — the percentiles
  v4/data/test_site_alphaearth_2024_v4.parquet           — stable parent lat/lon
  v5/data/test_site_alphaearth_2024_candidate.parquet    — loss   parent lat/lon

Output: v5/report/percentile_explorer.html
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve()
# This builder lives in myprojects/recover; point at the project tree explicitly.
ROOT = Path("/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery")
V4_DATA = ROOT / "v4" / "data"
V5_DATA = ROOT / "v5" / "data"
V5_REPORT = ROOT / "v5" / "report"
ANALYSIS = ROOT / "v5" / "scripts" / "analysis"

sys.path.insert(0, str(ANALYSIS))
import figstyle as fst  # noqa: E402

CLASS_ORDER = ["stable_nature", "stable_crop", "stable_built",
               "built_loss", "crop_loss"]


def load_data() -> pd.DataFrame:
    pct = pd.read_csv(V5_DATA / "test_site_ecoregion_percentile.csv")
    pct["parent_id"] = pct["parent_id"].astype(str)
    pct = pct.dropna(subset=["pct_vs_all"]).reset_index(drop=True)

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
    geo["parent_id"] = geo["parent_id"].astype(str)
    df = pct.merge(geo, on="parent_id", how="left")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return df


def build_payload(df: pd.DataFrame) -> list[dict]:
    def _o(v, nd=1):
        return None if pd.isna(v) else round(float(v), nd)

    recs = []
    for r in df.itertuples(index=False):
        recs.append({
            "parent_id": str(r.parent_id),
            "parent_label": r.parent_label,
            "eco_id": None if pd.isna(r.eco_id) else int(r.eco_id),
            "lat": round(float(r.lat), 6),
            "lon": round(float(r.lon), 6),
            "pct_all": _o(r.pct_vs_all),
            "pct_good": _o(r.pct_vs_good),
            "pct_bad": _o(r.pct_vs_bad),
            "pct_dor": _o(r.pct_dor),
            "n_good": None if pd.isna(r.n_refs_good) else int(r.n_refs_good),
            "n_bad": None if pd.isna(r.n_refs_bad) else int(r.n_refs_bad),
        })
    return recs


def _build_stamp() -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    sha = ""
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        if sha and dirty:
            sha += "-dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return f"built {date}" + (f" · {sha}" if sha else "")


def render_html(records: list[dict]) -> str:
    classes = [c for c in CLASS_ORDER if any(r["parent_label"] == c for r in records)]
    config = {
        "classOrder": classes,
        "classColors": {c: fst.CLASS_COLORS.get(c, "#444") for c in classes},
        "classLabels": {c: fst.CLASS_LABELS.get(c, c) for c in classes},
    }
    return (_HTML_TEMPLATE
            .replace("/*__DATA__*/", json.dumps(records, separators=(",", ":")))
            .replace("/*__CONFIG__*/", json.dumps(config))
            .replace("__BUILD__", _build_stamp()))


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v5 ecoregion percentile explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --ink:#111; --muted:#555; --line:#ddd; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Helvetica Neue",Arial,sans-serif; color:var(--ink); background:#fafafa; }
  header { padding:14px 20px 8px; border-bottom:1px solid var(--line); background:#fff; }
  header h1 { margin:0 0 2px; font-size:18px; }
  header p { margin:0; font-size:12.5px; color:var(--muted); max-width:80ch; }
  header p.note { margin-top:5px; }
  header p.note em { color:var(--ink); font-style:italic; }
  .tabs { display:flex; gap:4px; padding:8px 20px 0; background:#fff; border-bottom:1px solid var(--line); }
  .tab { padding:7px 14px; font-size:13px; font-weight:600; cursor:pointer; border:1px solid var(--line);
         border-bottom:none; border-radius:7px 7px 0 0; background:#f3f3f3; color:var(--muted); }
  .tab.active { background:#fff; color:var(--ink); position:relative; top:1px; }
  .view { display:none; }
  .view.active { display:block; }
  .wrap { display:flex; gap:0; align-items:stretch; min-height:calc(100vh - 150px); }
  #plot, #histplot { flex:1 1 auto; min-width:0; padding:8px 4px; }
  #panel { flex:0 0 330px; border-left:1px solid var(--line); background:#fff; padding:16px 18px; overflow-y:auto; }
  #panel h2 { margin:0 0 4px; font-size:15px; }
  #panel .hint { color:var(--muted); font-size:12.5px; line-height:1.5; }
  .pid { font-family:ui-monospace,Menlo,monospace; font-size:12.5px; word-break:break-all; color:var(--muted); }
  .chip { display:inline-block; padding:2px 9px; border-radius:11px; color:#fff; font-size:12px; font-weight:600; margin:6px 0 10px; }
  table.kv { width:100%; border-collapse:collapse; font-size:13px; margin:2px 0 12px; }
  table.kv td { padding:4px 2px; border-bottom:1px solid #eee; vertical-align:top; }
  table.kv td.k { color:var(--muted); width:50%; }
  table.kv td.v { text-align:right; font-variant-numeric:tabular-nums; }
  .pctnum { font-variant-numeric:tabular-nums; font-size:12.5px; }
  .pctbar { height:5px; background:#eee; border-radius:3px; overflow:hidden; margin-top:2px; }
  .pctfill { height:100%; border-radius:3px; }
  a.maplink { display:block; padding:9px 12px; margin:7px 0; border-radius:7px; text-decoration:none;
              font-size:13.5px; font-weight:600; border:1px solid var(--line); color:#fff; }
  a.ge { background:#1a73e8; } a.ge:hover { background:#1664cf; }
  a.wb { background:#0a7d4f; } a.wb:hover { background:#086a43; }
  a.es { background:#444; }    a.es:hover { background:#333; }
  .coords { font-size:12px; color:var(--muted); margin:8px 0 2px; font-variant-numeric:tabular-nums; }
  .status { min-height:1.2em; margin:6px 0 0; font-size:12px; color:var(--muted); }
  table.summary { border-collapse:collapse; font-size:13px; margin:14px 20px; }
  table.summary th, table.summary td { padding:6px 12px; border-bottom:1px solid var(--line); text-align:right;
                                       font-variant-numeric:tabular-nums; }
  table.summary th { background:#f3f3f3; text-align:right; cursor:pointer; user-select:none; }
  table.summary td.lbl, table.summary th.lbl { text-align:left; }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }
  footer { font-size:11.5px; color:var(--muted); padding:6px 20px 14px; border-top:1px solid var(--line); background:#fff; }
  code { background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:11.5px; }
</style>
</head>
<body>
<header>
  <h1>Ecoregion percentile explorer</h1>
  <p>Each parent test site's 2024 embedding, ranked within the cosine-similarity
     distribution of its <b>ecoregion's reference points</b> (C_eco method from
     <code>score_by_bioregion.py</code>). 0 = outlier vs the reference cloud,
     100 = as reference-like as the most reference-like points.</p>
  <p class="note">Regeneration reading: a regenerated site sits <em>high on vs-good</em>
     and <em>low on vs-bad</em> (top-left of the quadrant). <code>pct_dor</code>
    blends the two into a single 0–100 regeneration rank.</p>
</header>
<div class="tabs">
  <div class="tab active" data-view="quadrant">Recovery quadrant</div>
  <div class="tab" data-view="dist">Distributions</div>
  <div class="tab" data-view="table">Class summary</div>
</div>

<div class="view active" id="view-quadrant">
  <div class="wrap">
    <div id="plot"></div>
    <aside id="panel">
      <h2>No site selected</h2>
      <p class="hint">Click a point to see its ecoregion percentiles and open the
         site in Google Earth and the Esri Wayback imagery time-series.</p>
    </aside>
  </div>
</div>
<div class="view" id="view-dist"><div id="histplot"></div></div>
<div class="view" id="view-table"><div id="summary"></div></div>

<footer id="foot"></footer>

<script>
const DATA   = /*__DATA__*/;
const CONFIG = /*__CONFIG__*/;
const classes = CONFIG.classOrder;
const byClass = {};
classes.forEach(c => byClass[c] = DATA.filter(d => d.parent_label === c));
const byId = {};
DATA.forEach(d => byId[d.parent_id] = d);
function fmt(x, nd){ return (x===null||x===undefined||isNaN(x))?'–':Number(x).toFixed(nd); }

// ---------- tabs -----------------------------------------------------------
let histDrawn = false, tableDrawn = false;
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  const v = t.dataset.view;
  document.getElementById('view-'+v).classList.add('active');
  if (v === 'dist' && !histDrawn) { drawHist(); histDrawn = true; }
  if (v === 'table' && !tableDrawn) { drawTable(); tableDrawn = true; }
  if (v === 'quadrant') Plotly.Plots.resize('plot');
  if (v === 'dist') Plotly.Plots.resize('histplot');
}));

// ---------- quadrant scatter (per-class panels) ----------------------------
const ncol = Math.min(3, classes.length);
const nrow = Math.ceil(classes.length / ncol);
const COL_GAP = 0.07, ROW_GAP = 0.13;
const cellW = (1 - COL_GAP*(ncol-1))/ncol;
const cellH = (1 - ROW_GAP*(nrow-1))/nrow;
const axisOf = {};
const traces = [], shapes = [], annotations = [], layoutAxes = {};

classes.forEach((c, i) => {
  const g = byClass[c], color = CONFIG.classColors[c];
  const xId = 'x'+(i+1), yId = 'y'+(i+1);
  axisOf[c] = { x:xId, y:yId };
  traces.push({
    type:'scatter', mode:'markers',
    x: g.map(d=>d.pct_good), y: g.map(d=>d.pct_bad),
    customdata: g.map(d=>[d.parent_id, CONFIG.classLabels[c]]),
    marker:{ size:7, color:color, opacity:0.6, line:{color:'white', width:0.5} },
    xaxis:xId, yaxis:yId, showlegend:false,
    hovertemplate:'<b>%{customdata[1]}</b><br>vs good %{x:.0f} · vs bad %{y:.0f}<br>'
                + '<span style="font-size:10px">%{customdata[0]} · click to inspect</span><extra></extra>',
  });
  const col = i % ncol, row = Math.floor(i/ncol);
  const x0 = col*(cellW+COL_GAP), x1 = x0+cellW;
  const y1 = 1 - row*(cellH+ROW_GAP), y0 = y1-cellH;
  // 1:1 diagonal: above it = more bad-like than good-like
  shapes.push({ type:'line', xref:xId, yref:yId, layer:'below', x0:0,x1:100,y0:0,y1:100,
                line:{color:'#ddd', width:1, dash:'dash'} });
  // median crosshairs at 50
  shapes.push({ type:'line', xref:xId, yref:yId, layer:'below', x0:50,x1:50,y0:0,y1:100,
                line:{color:'#eee', width:1} });
  shapes.push({ type:'line', xref:xId, yref:yId, layer:'below', x0:0,x1:100,y0:50,y1:50,
                line:{color:'#eee', width:1} });
  annotations.push({ xref:'paper', yref:'paper', x:(x0+x1)/2, y:Math.min(y1+0.025,1.0),
                     text:'<b>'+CONFIG.classLabels[c]+'</b>', showarrow:false,
                     font:{size:13, color:color}, xanchor:'center', yanchor:'bottom' });
  // median pct_dor label
  const dors = g.map(d=>d.pct_dor).filter(v=>v!==null&&!isNaN(v)).sort((a,b)=>a-b);
  const med = dors.length ? dors[Math.floor(dors.length/2)] : NaN;
  annotations.push({ xref:xId+' domain', yref:yId+' domain', x:0.04, y:0.96,
    text:'median pct_dor = '+(isNaN(med)?'–':med.toFixed(0))+'<br>n = '+g.length,
    showarrow:false, align:'left', xanchor:'left', yanchor:'top', font:{size:11, color:'#222'},
    bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ddd', borderwidth:1, borderpad:3 });
  const lastRow = (i + ncol >= classes.length);
  layoutAxes['xaxis'+(i+1)] = { domain:[x0,x1], anchor:yId, range:[-2,102],
    title:{text: lastRow ? 'percentile vs GOOD refs' : '', font:{size:11}},
    zeroline:false, showline:true, linecolor:'#333', ticks:'outside' };
  layoutAxes['yaxis'+(i+1)] = { domain:[y0,y1], anchor:xId, range:[-2,102],
    title:{text: col===0 ? 'percentile vs BAD refs' : '', font:{size:11}},
    zeroline:false, showline:true, linecolor:'#333', ticks:'outside' };
});
const layout = Object.assign({
  margin:{l:55,r:14,t:34,b:46}, height:380*nrow, shapes, annotations,
  hovermode:'closest', plot_bgcolor:'white', paper_bgcolor:'white',
  font:{family:'Helvetica Neue, Arial, sans-serif', size:11, color:'#111'},
}, layoutAxes);
Plotly.newPlot('plot', traces, layout,
  { responsive:true, displaylogo:false, modeBarButtonsToRemove:['lasso2d','select2d'] });

// ---------- selection highlight + side panel -------------------------------
let hiTrace = null;
function highlight(d) {
  const ax = axisOf[d.parent_label];
  if (hiTrace === null) {
    Plotly.addTraces('plot', { type:'scatter', mode:'markers',
      x:[d.pct_good], y:[d.pct_bad], xaxis:ax.x, yaxis:ax.y,
      marker:{ size:16, color:'rgba(0,0,0,0)', symbol:'circle-open', line:{color:'#D81B60', width:3} },
      showlegend:false, hoverinfo:'skip' }).then(gd => { hiTrace = gd.data.length-1; });
  } else {
    Plotly.restyle('plot', { x:[[d.pct_good]], y:[[d.pct_bad]], xaxis:ax.x, yaxis:ax.y }, [hiTrace]);
  }
}
function bar(label, v, color) {
  if (v===null||v===undefined||isNaN(v)) return '<tr><td class="k">'+label+'</td><td class="v">–</td></tr>';
  const w = Math.max(0, Math.min(100, v));
  return '<tr><td class="k">'+label+'</td><td class="v" style="padding-bottom:2px">'
       + '<div class="pctnum">'+v.toFixed(1)+'</div>'
       + '<div class="pctbar"><div class="pctfill" style="width:'+w+'%;background:'+color+'"></div></div></td></tr>';
}
function copyTextToClipboard(text){
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const ta=document.createElement('textarea'); ta.value=text; ta.setAttribute('readonly','');
  ta.style.position='fixed'; ta.style.left='-9999px'; document.body.appendChild(ta); ta.select();
  try{document.execCommand('copy');}finally{document.body.removeChild(ta);} return Promise.resolve();
}
function showSite(d) {
  highlight(d);
  const color = CONFIG.classColors[d.parent_label], label = CONFIG.classLabels[d.parent_label];
  const ge = 'https://earth.google.com/web/search/'+d.lat.toFixed(6)+','+d.lon.toFixed(6)+'/';
  const e = 0.005;
  const wb = 'https://livingatlas.arcgis.com/wayback/?ext='
           + (d.lon-e).toFixed(6)+','+(d.lat-e).toFixed(6)+','+(d.lon+e).toFixed(6)+','+(d.lat+e).toFixed(6);
  const es = 'https://www.arcgis.com/apps/mapviewer/index.html?center='+d.lon+','+d.lat+'&level=17&basemapUrl='
           + encodeURIComponent('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer');
  document.getElementById('panel').innerHTML =
      '<h2>'+fmt(d.pct_dor,1)+' <span style="font-size:13px;color:#888;font-weight:400">recovery rank</span></h2>'
    + '<span class="chip" style="background:'+color+'">'+label+'</span>'
    + '<div class="pid">'+d.parent_id+' · eco '+(d.eco_id===null?'?':d.eco_id)+'</div>'
    + '<table class="kv">'
    + bar('vs good refs', d.pct_good, '#009E73')
    + bar('vs bad refs',  d.pct_bad,  '#D55E00')
    + bar('vs all refs',  d.pct_all,  '#666')
    + bar('recovery rank (pct_dor)', d.pct_dor, '#1a73e8')
    + '<tr><td class="k">refs (good / bad)</td><td class="v">'+fmt(d.n_good,0)+' / '+fmt(d.n_bad,0)+'</td></tr>'
    + '</table>'
    + '<a class="maplink ge" id="ge-link" target="_blank" rel="noopener">Open in Google Earth ↗</a>'
    + '<a class="maplink wb" id="wb-link" target="_blank" rel="noopener">Esri Wayback (imagery time-series) ↗</a>'
    + '<a class="maplink es" id="es-link" target="_blank" rel="noopener">Esri World Imagery (current) ↗</a>'
    + '<div class="status" id="link-status">Tip: Ctrl-click to force a new tab. The link is copied to your clipboard as a backup.</div>'
    + '<div class="coords">lat '+d.lat+', lon '+d.lon+'</div>';
  document.getElementById('ge-link').href = ge;
  document.getElementById('wb-link').href = wb;
  document.getElementById('es-link').href = es;
  const status = document.getElementById('link-status');
  document.querySelectorAll('#panel .maplink.ge, #panel .maplink.wb').forEach(link =>
    link.addEventListener('click', () => copyTextToClipboard(link.href)
      .then(()=>{status.textContent='Link copied to clipboard. Paste it if the new tab does not open.';})
      .catch(()=>{status.textContent='Could not copy automatically.';})));
}
document.getElementById('plot').on('plotly_click', ev => {
  const pt = ev.points[0]; if (!pt || pt.customdata===undefined) return;
  const pid = Array.isArray(pt.customdata) ? pt.customdata[0] : pt.customdata;
  const d = byId[pid]; if (d) showSite(d);
});

// ---------- distributions (overlaid histograms per metric) -----------------
function drawHist() {
  const metrics = [['pct_good','percentile vs GOOD refs'],
                   ['pct_bad','percentile vs BAD refs'],
                   ['pct_dor','recovery rank (pct_dor)']];
  const traces = [], layoutAxes = {}, annotations = [];
  metrics.forEach((m, mi) => {
    const xId='x'+(mi+1), yId='y'+(mi+1);
    classes.forEach(c => {
      const vals = byClass[c].map(d=>d[m[0]]).filter(v=>v!==null&&!isNaN(v));
      traces.push({ type:'histogram', x:vals, xaxis:xId, yaxis:yId,
        name:CONFIG.classLabels[c], legendgroup:c, showlegend: mi===0,
        marker:{color:CONFIG.classColors[c]}, opacity:0.55,
        xbins:{start:0,end:100,size:5}, histnorm:'probability density' });
    });
    const y1 = 1 - mi*(1/metrics.length) , y0 = y1 - (1/metrics.length) + 0.06;
    layoutAxes['xaxis'+(mi+1)] = { domain:[0,1], anchor:yId, range:[0,100],
      title:{text:m[1], font:{size:11}}, showline:true, linecolor:'#333', ticks:'outside' };
    layoutAxes['yaxis'+(mi+1)] = { domain:[y0,y1], anchor:xId, title:{text:'density', font:{size:10}},
      showline:true, linecolor:'#333', ticks:'outside' };
    annotations.push({ xref:'paper', yref:'paper', x:0, y:Math.min(y1+0.02,1.0),
      text:'<b>'+m[1]+'</b>', showarrow:false, font:{size:12}, xanchor:'left', yanchor:'bottom' });
  });
  const layout = Object.assign({ barmode:'overlay', height:760,
    margin:{l:55,r:14,t:30,b:40}, annotations,
    legend:{orientation:'h', y:1.06, x:0}, plot_bgcolor:'white', paper_bgcolor:'white',
    font:{family:'Helvetica Neue, Arial, sans-serif', size:11, color:'#111'} }, layoutAxes);
  Plotly.newPlot('histplot', traces, layout, { responsive:true, displaylogo:false });
}

// ---------- class summary table (sortable) ---------------------------------
function median(a){ const s=a.filter(v=>v!==null&&!isNaN(v)).sort((x,y)=>x-y);
  return s.length? s[Math.floor(s.length/2)] : NaN; }
let summaryRows = classes.map(c => {
  const g = byClass[c];
  return { label:CONFIG.classLabels[c], color:CONFIG.classColors[c], n:g.length,
    good:median(g.map(d=>d.pct_good)), bad:median(g.map(d=>d.pct_bad)),
    all:median(g.map(d=>d.pct_all)), dor:median(g.map(d=>d.pct_dor)) };
});
let sortKey='dor', sortDir=-1;
function drawTable() {
  summaryRows.sort((a,b)=> (a[sortKey]<b[sortKey]?-1:a[sortKey]>b[sortKey]?1:0)*sortDir);
  const th = (k,t,cls)=>'<th class="'+(cls||'')+'" data-k="'+k+'">'+t+(sortKey===k?(sortDir<0?' ▾':' ▴'):'')+'</th>';
  let html = '<table class="summary"><thead><tr>'
    + th('label','Class','lbl') + th('n','n') + th('good','median vs good')
    + th('bad','median vs bad') + th('all','median vs all') + th('dor','median pct_dor')
    + '</tr></thead><tbody>';
  summaryRows.forEach(r => {
    html += '<tr><td class="lbl"><span class="swatch" style="background:'+r.color+'"></span>'+r.label+'</td>'
      + '<td>'+r.n+'</td><td>'+fmt(r.good,1)+'</td><td>'+fmt(r.bad,1)+'</td>'
      + '<td>'+fmt(r.all,1)+'</td><td>'+fmt(r.dor,1)+'</td></tr>';
  });
  html += '</tbody></table>';
  const div = document.getElementById('summary');
  div.innerHTML = html;
  div.querySelectorAll('th').forEach(h => h.addEventListener('click', () => {
    const k = h.dataset.k; if (k===sortKey) sortDir*=-1; else { sortKey=k; sortDir=(k==='label')?1:-1; }
    drawTable();
  }));
}

// ---------- footer ---------------------------------------------------------
document.getElementById('foot').innerHTML =
  DATA.length + ' scored test sites across ' + classes.length + ' classes · '
  + 'percentiles read from <code>test_site_ecoregion_percentile.csv</code> (C_eco method) · '
  + '<span class="stamp">__BUILD__</span>';
</script>
</body>
</html>
"""


def main() -> None:
    df = load_data()
    records = build_payload(df)
    html = render_html(records)
    V5_REPORT.mkdir(parents=True, exist_ok=True)
    out = V5_REPORT / "percentile_explorer.html"
    out.write_text(html, encoding="utf-8")
    n_by_class = df["parent_label"].value_counts().reindex(CLASS_ORDER).dropna()
    print(f"wrote {out}  ({len(records)} sites)")
    for cls, n in n_by_class.items():
        print(f"  {cls:<14} {int(n):>4}")


if __name__ == "__main__":
    main()
