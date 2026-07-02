# Ecoregion reference points — full set

This directory ships a **sample of 5 ecoregions** (eco323, eco296, eco717, eco289,
eco61 — chosen because together they contain sites from all three transition
groups scored by the pipeline, and are among the smallest files) so the
reproduction scripts run out of the box without any download.

The paper's published result scores sites against **all 317** ecoregion
reference files (~193 MB total). To reproduce the full result, download the
remaining 312 files and place them in this directory (`data/ecoregion_refs/`).

## Where to get them

<!-- TODO: replace with the Zenodo/OSF record DOI once the dataset is archived.
     Until then, contact the corresponding author for the full ecoregion
     reference set (test_site_scoring/data/ref_samples_eco*_2024.parquet in the
     internal working repository). -->

## Verifying what you have

`MANIFEST.csv` lists every one of the 317 files with its expected size and
SHA-256 checksum (including the 5 already shipped). After downloading, verify:

```bash
cd data/ecoregion_refs
python - <<'PY'
import csv, hashlib
from pathlib import Path

missing, bad = [], []
with open("MANIFEST.csv") as f:
    for row in csv.DictReader(f):
        p = Path(row["filename"])
        if not p.exists():
            missing.append(p.name)
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != row["sha256"]:
            bad.append(p.name)

print(f"{317 - len(missing)}/317 files present, {len(bad)} checksum mismatches")
if missing[:5]:
    print("missing (first 5):", missing[:5])
if bad[:5]:
    print("checksum mismatch (first 5):", bad[:5])
PY
```

## Partial downloads are fine

The scoring scripts (`03_score_buffer_dor.py`, `04_score_ecoregion_dor.py`) degrade
gracefully: sites whose ecoregion has no reference file on disk are reported as
unscored rather than raising an error (buffer-DoR does not depend on this
directory at all — see the note below). Download only the ecoregions you need,
or all 317 for the exact published coverage (1,248 sites).

Note: **local-buffer DoR** (`03_score_buffer_dor.py`, the `dor_knn` score) does
**not** depend on this directory — it uses only the files in `data/cached/` and
scores ~894 sites regardless of what is present here. Only **ecoregion-percentile
DoR** (`04_score_ecoregion_dor.py`, `pct_dor`) is limited by ecoregion coverage.
