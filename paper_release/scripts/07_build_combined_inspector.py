"""Step 7 — interactive site inspector for the combined test-site DoR results.

Rebuilds the buffer inspector (scripts/analysis/build_buffer_inspector.py)
against the LATEST combined outputs of this pipeline:

  outputs/data/test_site_scores_combined.csv   (buffer + ecoregion DoR, step 5)
  outputs/data/target_groups.parquet           (lon/lat per PLOTID, step 0)

The HTML template, scatter layout, click-to-Google-Earth/Wayback panel, and the
ecoregion-percentile block are reused UNCHANGED by importing them from
build_buffer_inspector.py — only the data feeding it changes:

  * points are the three transition groups (stable_natural, artificial_reversion,
    stable_artificial), one scatter panel per group, mapped onto the template's
    per-class colour/label/threshold slots;
  * the x-axis is parent GHM (all-threats AA, 2022, 90 m). GHM comes from the v5
    ref parquets where a parent_id exists (967 sites) and from the GEE extraction
    cached in ghm_281_no_parent.csv for the 281 no-parent sites (step: build_spatial);
  * DoR is dor_knn with its 95% bootstrap CI; the side panel's ecoregion block is
    fed from pct_vs_good/bad/all + pct_dor.

Output: outputs/report/combined_inspector.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CACHED, OUT_DATA, OUT_REPORT, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

# Reuse the whole page (template + payload builder + provenance stamp).
from build_buffer_inspector import (  # noqa: E402
    build_payload,
    render_html,
)
from buffer_desirability import _load_parent_ghm  # noqa: E402

DATA = OUT_DATA
COMBINED = DATA / "test_site_scores_combined.csv"
TARGETS = DATA / "target_groups.parquet"
GHM_281 = CACHED / "ghm_281_no_parent.csv"
OUT = OUT_REPORT / "combined_inspector.html"

# Map the three transition groups onto the template's class slots. Colours/labels
# reuse the v5 figstyle palette; thresholds follow step-3 GROUP_THRESHOLD
# (reversion is called against the built-up threshold, as in 03_score_buffer_dor).
GROUP_COLOR = {
    "stable_natural": "#009E73",        # green  (= stable_nature)
    "artificial_reversion": "#D55E00",  # orange-red (= built_loss hue)
    "stable_artificial": "#E69F00",     # amber  (= stable_built)
}
GROUP_LABEL = {
    "stable_natural": "Stable natural",
    "artificial_reversion": "Artificial reversion",
    "stable_artificial": "Stable artificial",
}
GROUP_ORDER = ["stable_natural", "artificial_reversion", "stable_artificial"]
GROUP_THRESHOLD_KEY = {
    "stable_natural": "stable_nature",
    "artificial_reversion": "stable_built",
    "stable_artificial": "stable_built",
}

import json  # noqa: E402

THRESH_JSON = CACHED / "v4_calibrated_thresholds.json"


def load_ghm() -> dict[str, float]:
    """PLOTID -> GHM. Parent-keyed refs for the 967 with a parent_id; the cached
    GEE extraction for the 281 no-parent sites."""
    tgt = pd.read_parquet(TARGETS)[["PLOTID", "parent_id"]]
    tgt["PLOTID"] = tgt["PLOTID"].astype(str)
    tgt["parent_id"] = tgt["parent_id"].astype("string")
    ref_ghm = _load_parent_ghm()
    ghm = {r.PLOTID: ref_ghm.get(r.parent_id) for r in tgt.itertuples()}
    if GHM_281.exists():
        g281 = pd.read_csv(GHM_281, dtype={"PLOTID": str})
        for r in g281.itertuples():
            if ghm.get(r.PLOTID) is None and pd.notna(r.ghm_aa):
                ghm[r.PLOTID] = float(r.ghm_aa)
    return {k: v for k, v in ghm.items() if v is not None}


def load_sites() -> pd.DataFrame:
    """Combined DoR + lon/lat + GHM, shaped to what build_payload/render expect:
    columns parent_id, parent_label, lat, lon, dor, dor_lo, dor_hi, ghm, n_good,
    n_bad, eco_id, pct_vs_good/bad/all, pct_dor. One row per scored+located site."""
    df = pd.read_csv(COMBINED, dtype={"PLOTID": str, "parent_id": str})
    tgt = pd.read_parquet(TARGETS)[["PLOTID", "longitude", "latitude"]]
    tgt["PLOTID"] = tgt["PLOTID"].astype(str)
    df = df.merge(tgt, on="PLOTID", how="left")
    df["ghm"] = df["PLOTID"].map(load_ghm())

    # inspector shows the buffer scatter -> require a computable DoR, coords, GHM.
    df = df.dropna(subset=["dor_knn", "latitude", "longitude", "ghm"]).copy()

    out = pd.DataFrame({
        # key each point by PLOTID so byId lookups are unique (parent_id repeats
        # for the no-parent set); parent_label drives the panel/colour.
        "parent_id": df["PLOTID"],
        "parent_label": df["group"],
        "lat": df["latitude"].astype(float),
        "lon": df["longitude"].astype(float),
        "dor": df["dor_knn"].astype(float),
        "dor_lo": df["dor_knn_ci_low"].astype(float),
        "dor_hi": df["dor_knn_ci_high"].astype(float),
        "ghm": df["ghm"].astype(float),
        "n_good": df["n_good"],
        "n_bad": df["n_bad"],
        "eco_id": df["eco_id"],
        "pct_vs_good": df["pct_vs_good"],
        "pct_vs_bad": df["pct_vs_bad"],
        "pct_vs_all": df["pct_vs_all"],
        "pct_dor": df["pct_dor"],
    })
    return out


def main() -> None:
    # Patch the template's class metadata to our three groups. render_html reads
    # CONFIG.classColors/Labels via fst.CLASS_COLORS/LABELS and filters CLASS_ORDER,
    # so we inject group-level entries into those module globals before rendering.
    import build_buffer_inspector as bbi
    import figstyle as fst

    fst.CLASS_COLORS.update(GROUP_COLOR)
    fst.CLASS_LABELS.update(GROUP_LABEL)
    bbi.CLASS_ORDER = GROUP_ORDER

    thr_raw = json.loads(THRESH_JSON.read_text())
    # threshold per GROUP label (what render/call look up by parent_label)
    thr = {g: float(thr_raw[GROUP_THRESHOLD_KEY[g]]) for g in GROUP_ORDER}

    df = load_sites()
    records = build_payload(df, thr)
    html = render_html(records, thr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    print(f"wrote {OUT}  ({len(records)} sites)")
    for g in GROUP_ORDER:
        n = sum(r["parent_label"] == g for r in records)
        print(f"  {g:<22} {n:>4}")
    print("thresholds:", {k: round(v, 3) for k, v in thr.items()})


if __name__ == "__main__":
    main()
