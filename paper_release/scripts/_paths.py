"""Shared path layout for the paper_release pipeline scripts.

Every script in this directory lives at paper_release/scripts/<name>.py and
resolves paths relative to the paper_release/ root, so the package works
unmodified wherever it is checked out (no hardcoded absolute paths).

Layout:
  data/raw/            input shapefile (samples_recover_w_ref_label.*)
  data/cached/          small (~100 MB) parquet/json caches shipped with the repo
  data/ecoregion_refs/  ref_samples_eco{id}_2024.parquet — a small sample ships
                         with the repo; the rest are downloaded (see
                         data/ecoregion_refs/DOWNLOAD.md) for full reproduction
  outputs/              everything the pipeline scripts generate
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "src"

DATA = ROOT / "data"
RAW = DATA / "raw"
CACHED = DATA / "cached"
ECO_REFS = DATA / "ecoregion_refs"

OUTPUTS = ROOT / "outputs"
OUT_DATA = OUTPUTS / "data"
OUT_FIGURES = OUTPUTS / "figures"
OUT_REPORT = OUTPUTS / "report"
OUT_LOGS = OUTPUTS / "logs"

SHP = RAW / "samples_recover_w_ref_label.shp"
