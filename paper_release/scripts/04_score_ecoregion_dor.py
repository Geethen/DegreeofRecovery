"""Step 4 — ecoregion-percentile DoR for the three target groups, at the 2024 vintage.

Fixes the year caveat: the original ecoregion scorer ranked 2024 test sites against
2022-vintage FSCS references. Here we read ONLY the 2024 ecoregion refs produced in
step 2 (ref_samples_eco{id}_2024.parquet) so test site and references share the 2024
vintage.

Methodology is reused unchanged from v5/scripts/analysis/score_test_sites_by_ecoregion.py
(percentileofscore against the C_eco mean-ref-to-all-refs baseline; pct_vs_good/bad/all
and the directional pct_dor = (pct_vs_good + (100 - pct_vs_bad)) / 2).

No silent 2022 fallback: if a site's ecoregion has a 2022 file but no 2024 file, we RAISE
(the whole point of this run is a consistent 2024 vintage). Sites whose ecoregion has no
ref file at all (never sampled) are reported unscored, as in the original script.

Parent (test-site) 2024 embeddings = union of v4 stable + v5 candidate + step-1 missing,
so all three groups are covered (incl. the 281 sites with no buffer refs).

Output: outputs/data/test_site_dor_ecoregion_2024.csv  (ref_vintage=2024).

NOTE: full reproduction of the paper's coverage requires the complete
ref_samples_eco{id}_2024.parquet set in data/ecoregion_refs/ (see
data/ecoregion_refs/DOWNLOAD.md). A small sample ships with the repo so this
script runs out of the box and scores the subset of sites in the sampled
ecoregions; sites in unsampled ecoregions are reported unscored (same
behaviour as the original run when an ecoregion was never sampled at all).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CACHED, DATA, ECO_REFS, OUT_DATA  # noqa: E402

TARGETS = OUT_DATA / "target_groups.parquet"
ECO_DIR_2024 = ECO_REFS                                  # ref_samples_eco{id}_2024.parquet
ECO_DIR_2022 = DATA / "ecoregion_refs_2022"               # ref_samples_eco{id}.parquet (legacy, download only)

TS_STABLE = CACHED / "v4_test_site_alphaearth_2024.parquet"
TS_CAND = CACHED / "v5_test_site_alphaearth_2024_candidate.parquet"
# step-1 output: prefer a freshly re-extracted copy in outputs/data/ (if step 1
# was re-run against Earth Engine), else fall back to the shipped cache.
TS_MISSING = (OUT_DATA / "test_site_alphaearth_2024_missing.parquet"
              if (OUT_DATA / "test_site_alphaearth_2024_missing.parquet").exists()
              else CACHED / "test_site_alphaearth_2024_missing.parquet")

OUT = OUT_DATA / "test_site_dor_ecoregion_2024.csv"

EMBED_COLS = [f"A{i:02d}" for i in range(64)]
N_BASELINE = 500
MIN_REFS = 10
SEED = 42


def have_2024() -> set[int]:
    return {int(re.search(r"eco(\d+)_2024", f.name).group(1))
            for f in ECO_DIR_2024.glob("ref_samples_eco*_2024.parquet")}


def have_2022() -> set[int]:
    return {int(re.search(r"eco(\d+)", f.name).group(1))
            for f in ECO_DIR_2022.glob("ref_samples_eco*.parquet")}


def load_eco_refs_2024(eco_id: int) -> pd.DataFrame | None:
    p = ECO_DIR_2024 / f"ref_samples_eco{eco_id}_2024.parquet"
    if not p.exists():
        return None
    con = duckdb.connect()
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)
    df = con.execute(
        f'SELECT "natural", {embed_sql} FROM read_parquet(\'{p}\')'
    ).df()
    con.close()
    df = df.dropna(subset=EMBED_COLS).reset_index(drop=True)
    df["ref_state"] = df["natural"].map({1: "good", 0: "bad"})
    return df


def load_target_test_sites() -> pd.DataFrame:
    """One mean 2024 embedding per target PLOTID, from v4+v5+missing parquets.

    The v4/v5 parquets carry PLOTID; the missing parquet is keyed by PLOTID too.
    Target group + eco_id come from target_groups.parquet (+ missing eco_id).
    """
    con = duckdb.connect()
    embed_sql = ", ".join(f'"{c}"' for c in EMBED_COLS)
    frames = []
    for p in (TS_STABLE, TS_CAND):
        frames.append(con.execute(
            f"SELECT CAST(PLOTID AS VARCHAR) PLOTID, {embed_sql} "
            f"FROM read_parquet('{p}')").df())
    if TS_MISSING.exists():
        frames.append(con.execute(
            f"SELECT CAST(PLOTID AS VARCHAR) PLOTID, {embed_sql} "
            f"FROM read_parquet('{TS_MISSING}')").df())
    # target manifest: PLOTID -> group, eco_id (prefer missing-parquet eco_id where
    # the v5 refs never mapped one)
    tgt = con.execute(
        f'SELECT CAST(PLOTID AS VARCHAR) PLOTID, "group", '
        f"CAST(eco_id AS INTEGER) eco_id FROM read_parquet('{TARGETS}')").df()
    miss_eco = None
    if TS_MISSING.exists():
        miss_eco = con.execute(
            f"SELECT CAST(PLOTID AS VARCHAR) PLOTID, CAST(eco_id AS INTEGER) eco_id "
            f"FROM read_parquet('{TS_MISSING}')").df()
    con.close()

    emb = pd.concat(frames, ignore_index=True).dropna(subset=EMBED_COLS)
    emb = emb.groupby("PLOTID", sort=False)[EMBED_COLS].mean().reset_index()

    # fill eco_id from the missing-parquet extraction where target manifest is null
    if miss_eco is not None:
        tgt = tgt.merge(miss_eco.rename(columns={"eco_id": "eco_id_miss"}),
                        on="PLOTID", how="left")
        tgt["eco_id"] = tgt["eco_id"].where(
            tgt["eco_id"].notna() & (tgt["eco_id"] >= 0), tgt["eco_id_miss"])
        tgt = tgt.drop(columns=["eco_id_miss"])

    out = tgt.merge(emb, on="PLOTID", how="inner")
    return out


def eco_baseline(ref_embeds: np.ndarray, rng) -> np.ndarray:
    n = len(ref_embeds)
    idx = rng.choice(n, min(N_BASELINE, n), replace=False)
    return cosine_similarity(ref_embeds[idx], ref_embeds).mean(axis=1)


def main() -> None:
    sites = load_target_test_sites()
    print(f"target sites with a 2024 embedding: {len(sites)}")

    refs2024, refs2022 = have_2024(), have_2022()
    eco_cache: dict[int, pd.DataFrame | None] = {}
    rng = np.random.default_rng(SEED)
    rows = []

    for r in sites.itertuples():
        eid = int(r.eco_id) if pd.notna(r.eco_id) and r.eco_id >= 0 else None
        rec = {"PLOTID": r.PLOTID, "group": getattr(r, "group"), "eco_id": eid,
               "ref_vintage": "2024"}
        x = np.array([getattr(r, c) for c in EMBED_COLS], dtype=float).reshape(1, -1)

        if eid is None:
            rec.update(_nan_rec()); rows.append(rec); continue
        # no silent 2022 fallback: 2022 present but 2024 missing -> hard error
        if eid not in refs2024:
            if eid in refs2022:
                raise RuntimeError(
                    f"eco {eid}: 2022 ref file exists but NO 2024 file — refusing to "
                    f"silently use the 2022 vintage. Extract eco {eid} at 2024 (step 2) "
                    f"or remove its sites from scope.")
            rec.update(_nan_rec()); rows.append(rec); continue   # never sampled at all

        if eid not in eco_cache:
            eco_cache[eid] = load_eco_refs_2024(eid)
        g = eco_cache[eid]
        if g is None:
            rec.update(_nan_rec()); rows.append(rec); continue

        for pool, mask in (("all", slice(None)),
                           ("good", (g["ref_state"] == "good").to_numpy()),
                           ("bad", (g["ref_state"] == "bad").to_numpy())):
            sub = g.iloc[mask] if pool != "all" else g
            emb = sub[EMBED_COLS].to_numpy(dtype=float)
            rec[f"n_refs_{pool}"] = int(len(emb))
            if len(emb) < MIN_REFS:
                rec[f"sim_{pool}"] = np.nan
                rec[f"pct_vs_{pool}"] = np.nan
                continue
            test_sim = float(cosine_similarity(x, emb).mean())
            base = eco_baseline(emb, rng)
            rec[f"sim_{pool}"] = test_sim
            rec[f"pct_vs_{pool}"] = float(percentileofscore(base, test_sim, kind="mean"))

        sg, sb = rec.get("pct_vs_good"), rec.get("pct_vs_bad")
        rec["pct_dor"] = (float((sg + (100.0 - sb)) / 2.0)
                          if sg is not None and sb is not None
                          and not (np.isnan(sg) or np.isnan(sb)) else np.nan)
        rows.append(rec)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    n_scored = int(df["pct_vs_all"].notna().sum())
    print(f"\nWrote {len(df)} sites ({n_scored} scored) -> {OUT}")
    if n_scored < len(df):
        n_files = len(list(ECO_DIR_2024.glob("ref_samples_eco*_2024.parquet")))
        print(f"  ({len(df) - n_scored} sites unscored: their ecoregion has no ref "
              f"file in {ECO_DIR_2024} — only {n_files} ecoregion(s) present. "
              f"This is expected with the shipped sample; see "
              f"data/ecoregion_refs/DOWNLOAD.md for the full reference set.")
    print("\nmedian pct_dor by group (scored only):")
    print(df.dropna(subset=["pct_dor"]).groupby("group")["pct_dor"].median().round(1).to_string())
    print("\nn scored by group:")
    print(df.dropna(subset=["pct_vs_all"]).groupby("group").size().to_string())


def _nan_rec() -> dict:
    d = {"pct_dor": np.nan}
    for pool in ("all", "good", "bad"):
        d[f"n_refs_{pool}"] = 0
        d[f"sim_{pool}"] = np.nan
        d[f"pct_vs_{pool}"] = np.nan
    return d


if __name__ == "__main__":
    main()
