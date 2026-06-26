"""Percentile-rank each parent test site against its ecoregion's reference distribution.

Adapts the `percentileofscore` methodology from RECOVER/scripts/analysis/
score_by_bioregion.py (metric C_eco) to the v5 ecoregion DoR data, using
AlphaEarth 64-band embeddings.

For each ecoregion:
  * Build a baseline distribution of natural-reference -> all-reference mean
    cosine similarities (the "eco baseline", sampled like C_eco).
  * For each parent TEST SITE in that ecoregion, compute the 2024 test
    embedding's mean cosine similarity to the ecoregion's reference points,
    then percentileofscore(baseline, value) -> percentile rank in [0, 100].

A high percentile => the test site looks like (is as similar to the references
as) the most reference-like points in its ecoregion; a low percentile => it is
an outlier relative to the ecoregion's reference cloud.

Two reference pools are scored separately so the result is interpretable:
  * pct_vs_good : similarity to the ecoregion's GOOD (natural) refs
  * pct_vs_bad  : similarity to the ecoregion's BAD (non-natural) refs
  * pct_vs_all  : similarity to ALL refs in the ecoregion (C_eco analogue)

Inputs (all local, no Earth Engine):
  v1-ecoregion/data/ref_samples_eco{eco_id}.parquet  — ecoregion-wide FSCS refs
      (natural=1 -> good, natural=0 -> bad; one file per ecoregion)
  v5/data/v5_stable_refs_alphaearth.parquet   — parent_id -> eco_id + label map
  v5/data/v5_candidate_refs_alphaearth.parquet — parent_id -> eco_id + label map
  v4/data/test_site_alphaearth_2024_v4.parquet   (stable parent test sites)
  v5/data/test_site_alphaearth_2024_candidate.parquet (loss parent test sites)

Output:
  v5/data/test_site_ecoregion_percentile.csv

Note: only ecoregions with a ref_samples_eco{id}.parquet file will be scored.
Parents whose ecoregion has no file receive NaN percentiles.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path("/data/P-Prosjekter2/155020_recover/WP1/degree_of_recovery")
V1_ECO_DATA = BASE_DIR / "v1-ecoregion" / "data"
V4_DATA = BASE_DIR / "v4" / "data"
V5_DATA = BASE_DIR / "v5" / "data"

# v5 refs used only for parent_id -> eco_id + parent_label mapping
STABLE_REFS = V5_DATA / "v5_stable_refs_alphaearth.parquet"
CAND_REFS   = V5_DATA / "v5_candidate_refs_alphaearth.parquet"
TS_STABLE   = V4_DATA / "test_site_alphaearth_2024_v4.parquet"
TS_CAND     = V5_DATA / "test_site_alphaearth_2024_candidate.parquet"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
STRATEGY = "random_100"

# Baseline subsample sizes (mirrors score_by_bioregion C_eco: n_cosine_baseline=500)
N_BASELINE = 500
MIN_REFS = 10          # need at least this many refs in a pool to score
BOOT_SEED = 42

OUT_PATH = V5_DATA / "test_site_ecoregion_percentile.csv"


def load_parent_meta() -> pd.DataFrame:
    """parent_id -> eco_id + parent_label, sourced from v5 buffer refs."""
    con = duckdb.connect()
    frames = []
    for p in (STABLE_REFS, CAND_REFS):
        if not p.exists():
            continue
        df = con.execute(
            "SELECT DISTINCT parent_id, CAST(eco_id AS INT) AS eco_id, parent_label "
            "FROM read_parquet(?) WHERE strategy = ?",
            [str(p), STRATEGY],
        ).df()
        frames.append(df)
    con.close()
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("parent_id")
    meta["parent_id"] = meta["parent_id"].astype(str)
    return meta


def load_eco_refs(eco_id: int) -> pd.DataFrame | None:
    """Load FSCS ecoregion refs for one ecoregion; return None if file missing."""
    p = V1_ECO_DATA / f"ref_samples_eco{eco_id}.parquet"
    if not p.exists():
        return None
    con = duckdb.connect()
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)
    df = con.execute(
        f'SELECT "natural", {embed_sql} FROM read_parquet(?)', [str(p)]
    ).df()
    con.close()
    df = df.dropna(subset=EMBED_COLS).reset_index(drop=True)
    df["ref_state"] = df["natural"].map({1: "good", 0: "bad"})
    return df


def load_test_sites() -> pd.DataFrame:
    """One mean embedding per parent test site (2024)."""
    con = duckdb.connect()
    cols = ", ".join(["parent_id"] + EMBED_COLS)
    frames = []
    for p in (TS_STABLE, TS_CAND):
        if not p.exists():
            continue
        frames.append(con.execute(f"SELECT {cols} FROM read_parquet(?)", [str(p)]).df())
    con.close()
    ts = pd.concat(frames, ignore_index=True)
    ts["parent_id"] = ts["parent_id"].astype(str)
    ts = ts.dropna(subset=EMBED_COLS).reset_index(drop=True)
    # collapse to one mean embedding per parent
    return ts.groupby("parent_id", sort=False)[EMBED_COLS].mean().reset_index()


def eco_baseline(ref_embeds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Baseline distribution of (sampled ref) -> (all refs) mean cosine sim.

    Mirrors C_eco in score_by_bioregion.build_natural_baselines: sample up to
    N_BASELINE reference points, take each one's mean cosine similarity to the
    full reference pool. This is the null distribution a test site is ranked in.
    """
    n = len(ref_embeds)
    sample_n = min(N_BASELINE, n)
    idx = rng.choice(n, sample_n, replace=False)
    return cosine_similarity(ref_embeds[idx], ref_embeds).mean(axis=1)


def score() -> pd.DataFrame:
    meta = load_parent_meta()
    ts   = load_test_sites()

    p2e = meta.set_index("parent_id")["eco_id"].to_dict()
    p2l = meta.set_index("parent_id")["parent_label"].to_dict()

    # Discover available ecoregion FSCS ref files
    eco_files = {
        int(re.search(r"eco(\d+)", f.name).group(1)): f
        for f in V1_ECO_DATA.glob("ref_samples_eco*.parquet")
    }
    print(f"Found {len(eco_files)} ecoregion ref files in v1-ecoregion/data/")
    print(f"Loaded parent meta: {len(meta):,} parents, {meta['eco_id'].nunique()} ecos")
    print(f"Loaded {len(ts):,} parent test sites")

    # Load eco refs on demand; cache so each file is read once
    eco_cache: dict[int, pd.DataFrame | None] = {}

    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    for pid, trow in ts.set_index("parent_id")[EMBED_COLS].iterrows():
        eid = p2e.get(pid)
        rec = {"parent_id": pid, "eco_id": eid, "parent_label": p2l.get(pid)}

        if eid is None or eid < 0 or eid not in eco_files:
            rec.update({"n_refs_all": 0, "n_refs_good": 0, "n_refs_bad": 0,
                        "sim_all": np.nan, "pct_vs_all": np.nan,
                        "sim_good": np.nan, "pct_vs_good": np.nan,
                        "sim_bad": np.nan, "pct_vs_bad": np.nan,
                        "pct_dor": np.nan})
            rows.append(rec)
            continue

        if eid not in eco_cache:
            eco_cache[eid] = load_eco_refs(eid)
        g = eco_cache[eid]
        if g is None:
            rec.update({"n_refs_all": 0, "n_refs_good": 0, "n_refs_bad": 0,
                        "sim_all": np.nan, "pct_vs_all": np.nan,
                        "sim_good": np.nan, "pct_vs_good": np.nan,
                        "sim_bad": np.nan, "pct_vs_bad": np.nan,
                        "pct_dor": np.nan})
            rows.append(rec)
            continue

        x = trow.to_numpy(dtype=float).reshape(1, -1)

        for pool, mask in (("all", slice(None)),
                           ("good", (g["ref_state"] == "good").to_numpy()),
                           ("bad",  (g["ref_state"] == "bad").to_numpy())):
            sub = g.iloc[mask] if pool != "all" else g
            emb = sub[EMBED_COLS].to_numpy(dtype=float)
            rec[f"n_refs_{pool}"] = int(len(emb))
            if len(emb) < MIN_REFS:
                rec[f"sim_{pool}"] = np.nan
                rec[f"pct_vs_{pool}"] = np.nan
                continue
            # test-site mean cosine sim to this pool
            test_sim = float(cosine_similarity(x, emb).mean())
            base = eco_baseline(emb, rng)
            rec[f"sim_{pool}"] = test_sim
            rec[f"pct_vs_{pool}"] = float(percentileofscore(base, test_sim, kind="mean"))

        # A single directional percentile: high pct_vs_good AND low pct_vs_bad
        # => most regenerated. Mirrors the DoR idea in percentile space.
        sg, sb = rec.get("pct_vs_good"), rec.get("pct_vs_bad")
        if sg is not None and sb is not None and not (np.isnan(sg) or np.isnan(sb)):
            rec["pct_dor"] = float((sg + (100.0 - sb)) / 2.0)
        else:
            rec["pct_dor"] = np.nan
        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    df = score()
    n_scored = int(df["pct_vs_all"].notna().sum())
    n_unmapped = int(df["eco_id"].isna().sum() + (df["eco_id"].fillna(-1) < 0).sum())
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df):,} test sites, {n_scored:,} scored, "
          f"{n_unmapped} unmapped/no-eco)")
    print("\nPer-class summary (median percentile):")
    summ = (df.dropna(subset=["pct_vs_all"])
              .groupby("parent_label")
              .agg(n=("parent_id", "size"),
                   med_pct_all=("pct_vs_all", "median"),
                   med_pct_good=("pct_vs_good", "median"),
                   med_pct_bad=("pct_vs_bad", "median"),
                   med_pct_dor=("pct_dor", "median"))
              .round(1))
    print(summ.to_string())


if __name__ == "__main__":
    main()
