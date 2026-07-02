"""Step 5 — combine the buffer + ecoregion DoR scores and summarise per group.

Joins the two scoring outputs on PLOTID, attaches the target group, and prints a
per-group summary: n scored by each method, median buffer DoR + category breakdown,
median ecoregion pct_dor, and the buffer-vs-ecoregion rank agreement (the cross-check
the methods doc frames the two methods as).

Inputs:
  outputs/data/test_site_dor_buffer_3_8km.csv      (step 3)
  outputs/data/test_site_dor_ecoregion_2024.csv    (step 4)
  outputs/data/target_groups.parquet               (step 0)
Output:
  outputs/data/test_site_scores_combined.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUT_DATA  # noqa: E402

TARGETS = OUT_DATA / "target_groups.parquet"
BUFFER = OUT_DATA / "test_site_dor_buffer_3_8km.csv"
ECO = OUT_DATA / "test_site_dor_ecoregion_2024.csv"
OUT = OUT_DATA / "test_site_scores_combined.csv"

GROUP_ORDER = ["stable_natural", "artificial_reversion", "stable_artificial"]


def main() -> None:
    con = duckdb.connect()
    tgt = con.execute(
        f'SELECT CAST(PLOTID AS VARCHAR) PLOTID, "group", parent_id, eco_id '
        f"FROM read_parquet('{TARGETS}')").df()
    con.close()

    buf = pd.read_csv(BUFFER, dtype={"PLOTID": str})
    eco = pd.read_csv(ECO, dtype={"PLOTID": str})

    buf_cols = ["PLOTID", "dor_knn", "dor_knn_ci_low", "dor_knn_ci_high",
                "category_knn", "n_good", "n_bad", "inner_m", "outer_m"]
    eco_cols = ["PLOTID", "pct_dor", "pct_vs_good", "pct_vs_bad", "pct_vs_all",
                "ref_vintage", "n_refs_all"]

    df = (tgt.merge(buf[buf_cols], on="PLOTID", how="left")
             .merge(eco[[c for c in eco_cols if c in eco.columns]],
                    on="PLOTID", how="left"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Wrote {len(df)} sites -> {OUT}\n")

    summ = (df.groupby("group")
              .agg(n_total=("PLOTID", "size"),
                   n_buffer=("dor_knn", lambda s: s.notna().sum()),
                   n_eco=("pct_dor", lambda s: s.notna().sum()),
                   med_buffer_dor=("dor_knn", "median"),
                   med_eco_pct_dor=("pct_dor", "median"))
              .reindex(GROUP_ORDER).round(3))
    print("Per-group summary:")
    print(summ.to_string())

    print("\nBuffer category_knn by group:")
    print(df.dropna(subset=["category_knn"])
            .groupby("group")["category_knn"].value_counts().to_string())

    # rank agreement between the two methods (where both scored)
    both = df.dropna(subset=["dor_knn", "pct_dor"])
    if len(both) > 10:
        rho, p = spearmanr(both["dor_knn"], both["pct_dor"])
        print(f"\nBuffer vs ecoregion rank agreement (Spearman, n={len(both)}): "
              f"rho={rho:.3f}, p={p:.2g}")
    print("\nExpected rank order (both methods): "
          "stable_natural > artificial_reversion > stable_artificial")


if __name__ == "__main__":
    main()
