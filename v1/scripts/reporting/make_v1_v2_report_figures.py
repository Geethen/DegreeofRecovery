"""Create publication-quality figures for the v1/v2 recovery methodology report.

Outputs:
  - v1/report/plots/v1_v2_comparison_overview.png
  - v1/report/plots/v1_v2_examples_2018_vs_2024.png
  - v1/report/plots/v1_v2_examples_links.csv  (Google Earth URLs)

The example imagery uses Sentinel-2 true-colour composites from Earth Engine
for the same site in 2018 and 2024, so the visual change between dates can be
read alongside the v1 vs v2 score change.
"""

from __future__ import annotations

import io
from pathlib import Path

import duckdb
import ee
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[3]
REPORT_PLOTS = ROOT / "v1" / "report" / "plots"
V1_SUMMARY = ROOT / "v1" / "data" / "dor_summary_by_label_v1.csv"
V2_SUMMARY = ROOT / "v2" / "data" / "dor_summary_by_label_v2.csv"
V1_V2_SUMMARY = ROOT / "v2" / "data" / "v1_vs_v2" / "v1_vs_v2_summary.csv"
V1_V2_SITES = ROOT / "v2" / "data" / "v1_vs_v2" / "v1_vs_v2_site_scores.csv"
V1_V2_TRANSITIONS = (
    ROOT / "v2" / "data" / "v1_vs_v2" / "v1_vs_v2_category_transition.csv"
)
TEST_SITES = ROOT / "v1" / "data" / "test_site_alphaearth_2024.parquet"

EXAMPLES = [
    ("00000000000000000176", "indistinguishable \u2192 recovering"),
    ("00000000000000000442", "recovering \u2192 degraded"),
    ("00000000000000000291", "recovering \u2192 recovering"),
    ("00000000000000000632", "degraded \u2192 degraded"),
]

CATEGORY_ORDER = ["recovering", "indistinguishable", "degraded"]
CATEGORY_COLORS = {
    "recovering": "#2ca02c",
    "indistinguishable": "#dab600",
    "degraded": "#d62728",
}
V1_COLOR = "#4c78a8"
V2_COLOR = "#f58518"

# Thumbnail buffer (one side of square ROI in metres) and pixel size.
THUMB_RADIUS_M = 1500
THUMB_DIM = 640


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def apply_pub_style() -> None:
    """Clean, paper-ready matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def initialize_ee() -> None:
    try:
        ee.Initialize(project="ee-gsingh")
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project="ee-gsingh")


def load_coords() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        "SELECT parent_id, geo FROM read_parquet(?)", [str(TEST_SITES)]
    ).df()
    con.close()
    df["parent_id"] = df["parent_id"].astype(str).str.zfill(20)
    df["lon"] = df["geo"].apply(
        lambda g: g["coordinates"][0] if isinstance(g, dict) else np.nan
    )
    df["lat"] = df["geo"].apply(
        lambda g: g["coordinates"][1] if isinstance(g, dict) else np.nan
    )
    return df[["parent_id", "lon", "lat"]]


def google_earth_url(lon: float, lat: float) -> str:
    """Return a Google Earth Web search URL for the target point."""
    return f"https://earth.google.com/web/search/{lat:.6f},{lon:.6f}/"


def google_maps_pin_url(lon: float, lat: float) -> str:
    """Google Maps URL that always drops a pin and offers a satellite view."""
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def write_kml(rows: list[dict], path: Path) -> None:
    """Write a KML with one placemark per example so users can drag/drop
    into Google Earth and see a pin at each site."""
    placemarks = []
    for r in rows:
        placemarks.append(
            f"  <Placemark>\n"
            f"    <name>{r['parent_id'][-6:]}  {r['transition']}</name>\n"
            f"    <description><![CDATA["
            f"label: {r['parent_label']}<br/>"
            f"DoR v1 = {r['dor_v1']:.2f}, DoR v2 = {r['dor_v2']:.2f}"
            f"]]></description>\n"
            f"    <Point><coordinates>{r['lon']:.6f},{r['lat']:.6f},0</coordinates></Point>\n"
            f"  </Placemark>"
        )
    body = "\n".join(placemarks)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "<Document>\n"
        "  <name>RECOVER v1 vs v2 examples</name>\n"
        f"{body}\n"
        "</Document>\n"
        "</kml>\n"
    )
    path.write_text(kml, encoding="utf-8")


# ---------------------------------------------------------------------------
# Overview figure
# ---------------------------------------------------------------------------
def make_overview_figure() -> None:
    v1 = pd.read_csv(V1_SUMMARY).set_index("par_label")
    v2 = pd.read_csv(V2_SUMMARY).set_index("par_label")
    trans = pd.read_csv(V1_V2_TRANSITIONS)
    summ = pd.read_csv(V1_V2_SUMMARY).iloc[0]
    sites = pd.read_csv(V1_V2_SITES)

    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.30, 1.0],
        hspace=0.45,
        wspace=0.32,
    )

    # ---- Panel A: grouped bars per label, v1 (light) vs v2 (solid) ----
    ax = fig.add_subplot(gs[0, 0])
    labels = ["built_loss", "crop_loss"]
    bar_w = 0.12
    in_pair_gap = 0.02
    cat_gap = 0.08
    pair_w = 2 * bar_w + in_pair_gap
    group_w = len(CATEGORY_ORDER) * pair_w + (len(CATEGORY_ORDER) - 1) * cat_gap
    x_centres = np.arange(len(labels))
    max_count = 0
    for li, lbl in enumerate(labels):
        group_left = x_centres[li] - group_w / 2
        for ci, cat in enumerate(CATEGORY_ORDER):
            pair_left = group_left + ci * (pair_w + cat_gap)
            x_v1 = pair_left + bar_w / 2
            x_v2 = x_v1 + bar_w
            v1_val = int(v1.loc[lbl, cat])
            v2_val = int(v2.loc[lbl, cat])
            max_count = max(max_count, v1_val, v2_val)
            ax.bar(
                x_v1,
                v1_val,
                width=bar_w,
                color=CATEGORY_COLORS[cat],
                alpha=0.45,
                edgecolor="black",
                linewidth=0.4,
            )
            ax.bar(
                x_v2,
                v2_val,
                width=bar_w,
                color=CATEGORY_COLORS[cat],
                alpha=1.0,
                edgecolor="black",
                linewidth=0.4,
            )
    ax.set_xticks(x_centres)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sites (n)")
    ax.set_title("A  Site categories by disturbance label")
    ax.set_ylim(0, max_count * 1.18)
    cat_handles = [
        mpatches.Patch(
            facecolor=CATEGORY_COLORS[c], edgecolor="black", linewidth=0.4, label=c
        )
        for c in CATEGORY_ORDER
    ]
    ver_handles = [
        mpatches.Patch(
            facecolor="#888888",
            alpha=0.45,
            edgecolor="black",
            linewidth=0.4,
            label="v1",
        ),
        mpatches.Patch(
            facecolor="#888888", alpha=1.0, edgecolor="black", linewidth=0.4, label="v2"
        ),
    ]
    # Keep legends inside Panel A (bottom-left) to avoid overlap with Panel B.
    leg1 = ax.legend(
        handles=cat_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.23),
        title="Category",
        title_fontsize=9,
        borderaxespad=0.2,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=ver_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        title="Version",
        title_fontsize=9,
        borderaxespad=0.2,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )

    # ---- Panel B: transition matrix ----
    ax = fig.add_subplot(gs[0, 1])
    cats = CATEGORY_ORDER
    mat = np.zeros((3, 3), dtype=int)
    for _, row in trans.iterrows():
        if row["category_v1"] in cats and row["category_v2"] in cats:
            i = cats.index(row["category_v1"])
            j = cats.index(row["category_v2"])
            mat[i, j] = int(row["n_sites"])
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0)
    threshold = mat.max() * 0.6
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(mat[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if mat[i, j] > threshold else "#333333",
                fontweight="bold",
            )
    for k in range(3):
        ax.add_patch(
            Rectangle(
                (k - 0.5, k - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#444",
                linewidth=1.3,
                linestyle="--",
            )
        )
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_yticklabels(cats)
    ax.set_xlabel("v2 category")
    ax.set_ylabel("v1 category")
    ax.set_title("B  Category transitions (v1 \u2192 v2)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)

    # ---- Panel C: per-site DoR distributions ----
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 23)
    ax.hist(
        sites["dor_median_v1"].dropna(),
        bins=bins,
        alpha=0.55,
        color=V1_COLOR,
        edgecolor="black",
        linewidth=0.3,
        label="v1",
    )
    ax.hist(
        sites["dor_median_v2"].dropna(),
        bins=bins,
        alpha=0.55,
        color=V2_COLOR,
        edgecolor="black",
        linewidth=0.3,
        label="v2",
    )
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.51, ymax * 0.95, "midpoint", fontsize=8, color="#333", va="top")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Site DoR (1 = natural, 0 = degraded)")
    ax.set_ylabel("Sites (n)")
    ax.set_title("C  Distribution of site DoR")
    ax.legend(loc="upper left")

    # ---- Panel D: paired v1 vs v2 scatter ----
    ax = fig.add_subplot(gs[1, 1])
    sub = sites.dropna(subset=["dor_median_v1", "dor_median_v2"]).copy()
    sub["changed"] = sub["category_changed"]
    no_change = sub[~sub["changed"]]
    changed = sub[sub["changed"]]
    ax.scatter(
        no_change["dor_median_v1"],
        no_change["dor_median_v2"],
        s=18,
        c="#999999",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        label=f"unchanged (n={len(no_change)})",
    )
    ax.scatter(
        changed["dor_median_v1"],
        changed["dor_median_v2"],
        s=24,
        c="#d62728",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label=f"category changed (n={len(changed)})",
    )
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="#bbb", linewidth=0.6)
    ax.axhline(0.5, color="#bbb", linewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("DoR v1")
    ax.set_ylabel("DoR v2")
    ax.set_title("D  Per-site v1 vs v2 DoR")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")

    foot = (
        f"n = {int(summ['n_sites_joined'])} sites    "
        f"category changes = {int(summ['n_category_changed'])}    "
        f"median |\u0394 DoR| = {summ['median_abs_dor_delta']:.3f}    "
        f"P90 |\u0394 DoR| = {summ['p90_abs_dor_delta']:.3f}    "
        f"mean DoR: v1 = {summ['mean_dor_v1']:.3f}, v2 = {summ['mean_dor_v2']:.3f}"
    )
    fig.suptitle(
        "v1 vs v2 Degree-of-Recovery comparison", fontsize=14, fontweight="bold", y=1.00
    )
    fig.text(0.5, -0.01, foot, ha="center", va="bottom", fontsize=9, color="#333333")

    REPORT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_PLOTS / "v1_v2_comparison_overview.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sentinel-2 example imagery
# ---------------------------------------------------------------------------
def fetch_s2_thumb(
    lon: float,
    lat: float,
    date_start: str,
    date_end: str,
    radius_m: int = THUMB_RADIUS_M,
    dim: int = THUMB_DIM,
) -> np.ndarray:
    """Median S2-SR true-colour composite over the date window."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m).bounds()
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median()
        .select(["B4", "B3", "B2"])
    )
    url = image.getThumbURL(
        {
            "region": region,
            "dimensions": dim,
            "format": "png",
            "min": 200,
            "max": 3000,
            "gamma": 1.2,
        }
    )
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return plt.imread(io.BytesIO(resp.content), format="png")


def _draw_panel(ax, img, year_label, scale_total_m=THUMB_RADIUS_M * 2):
    ax.imshow(img)
    h, w = img.shape[:2]
    # Centre marker (50 m on the ground)
    px_per_m = w / scale_total_m
    ring_r_m = 50
    ring = Circle(
        (w / 2, h / 2),
        radius=ring_r_m * px_per_m,
        fill=False,
        edgecolor="#ffd400",
        linewidth=1.8,
    )
    ax.add_patch(ring)
    ax.add_patch(Circle((w / 2, h / 2), radius=2.5, color="#ffd400"))
    # Scale bar (500 m)
    bar_m = 500
    bar_px = bar_m * px_per_m
    pad = w * 0.04
    y_bar = h - pad
    ax.add_patch(
        Rectangle(
            (pad, y_bar - 5),
            bar_px,
            5,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
        )
    )
    ax.text(
        pad + bar_px / 2,
        y_bar - 8,
        f"{bar_m} m",
        color="white",
        fontsize=8,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    # Year tag
    ax.text(
        0.02,
        0.96,
        year_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, pad=2.5, edgecolor="none"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#333")
        spine.set_linewidth(0.8)


def make_examples_figure() -> None:
    initialize_ee()
    sites = pd.read_csv(V1_V2_SITES)
    coords = load_coords()
    sites["parent_id"] = sites["parent_id"].astype(str).str.zfill(20)
    sites = sites.merge(coords, on="parent_id", how="left")

    n = len(EXAMPLES)
    panel_in = 5.0  # per-panel width and height in inches (square thumbs)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(2 * panel_in + 0.4, panel_in * n + 0.6),
        gridspec_kw={"wspace": 0.04, "hspace": 0.18},
    )

    link_rows: list[dict] = []
    for r, (pid, transition) in enumerate(EXAMPLES):
        row = sites.loc[sites["parent_id"] == pid].iloc[0]
        lon, lat = float(row["lon"]), float(row["lat"])

        img_2018 = fetch_s2_thumb(lon, lat, "2018-01-01", "2018-12-31")
        img_2024 = fetch_s2_thumb(lon, lat, "2024-01-01", "2024-12-31")

        ax_l = axes[r, 0]
        ax_r = axes[r, 1]
        _draw_panel(ax_l, img_2018, "2018")
        _draw_panel(ax_r, img_2024, "2024")

        # Single row title spanning both panels (avoids the right-column
        # overlap from before).
        ax_pos_l = ax_l.get_position()
        ax_pos_r = ax_r.get_position()
        x_centre = (ax_pos_l.x0 + ax_pos_r.x1) / 2
        y_top = ax_pos_l.y1 + 0.012
        title_left = (
            f"{transition}   \u2022   {row['parent_label_v1']}   "
            f"\u2022   DoR v1 = {row['dor_median_v1']:.2f}, "
            f"v2 = {row['dor_median_v2']:.2f}"
        )
        fig.text(
            x_centre,
            y_top,
            title_left,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        # Coordinates as an in-panel overlay on the right image (top-right).
        ax_r.text(
            0.98,
            0.96,
            f"{lat:.4f}\u00b0, {lon:.4f}\u00b0",
            transform=ax_r.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=2.5, edgecolor="none"),
        )

        link_rows.append(
            {
                "parent_id": pid,
                "transition": transition,
                "parent_label": row["parent_label_v1"],
                "lat": lat,
                "lon": lon,
                "dor_v1": float(row["dor_median_v1"]),
                "dor_v2": float(row["dor_median_v2"]),
                "google_earth_url": google_earth_url(lon, lat),
                "google_maps_pin_url": google_maps_pin_url(lon, lat),
            }
        )

    fig.suptitle(
        "Representative sites — Sentinel-2 true colour, 2018 vs 2024",
        fontsize=13,
        fontweight="bold",
        y=0.998,
    )
    fig.text(
        0.5,
        0.003,
        "Yellow ring marks the parent location (50 m radius). Composites are "
        "median S2-SR with cloud-pixel < 30%. Imagery is visual context only; "
        "scoring uses the AlphaEarth embedding.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444",
    )

    REPORT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_PLOTS / "v1_v2_examples_2018_vs_2024.png")
    plt.close(fig)

    pd.DataFrame(link_rows).to_csv(
        REPORT_PLOTS / "v1_v2_examples_links.csv", index=False
    )
    write_kml(link_rows, REPORT_PLOTS / "v1_v2_examples.kml")


if __name__ == "__main__":
    apply_pub_style()
    make_overview_figure()
    make_examples_figure()
