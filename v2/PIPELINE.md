# v2 recovery sampling workflow

This v2 folder is for method development and A/B testing of reference sampling.

## What is new in v2

- Exclude crop / built / building loss regions from reference candidates.
- Dynamic local buffering: sample at max radius then keep the **nearest** points,
  so the smallest sufficient buffer is used per parent.
- Configurable balanced or area-proportional class allocation.
- Strategy benchmarking with spatial-autocorrelation-aware effective sample size.
- Comparator now uses the **real test-site embedding** as `x_obs` (operationally
  meaningful DoR), and FSCS uses **all 64 AlphaEarth bands**.
- Empirical n_eff calibration script grounds `n_eff_target` in CI-width plateau
  and cross-seed DoR stability.

## Selected v2 policy (locked)

Backed by 6-scenario exhaustive comparison + n_eff calibration with real test-site
embeddings. See [v2/data/v2real_exhaustive_robust_rank.csv](data/v2real_exhaustive_robust_rank.csv)
and [v2/data/neff_calibration_mask_on/neff_calibration_summary.csv](data/neff_calibration_mask_on/neff_calibration_summary.csv).

| Setting             | Value                       | Rationale                                                                  |
|---------------------|-----------------------------|----------------------------------------------------------------------------|
| Selection method    | `random`                    | Tied with FSCS-64 / feature-diverse on all metrics; simpler, no clustering |
| `target_per_class`  | 100                         | CI-width plateau (slope < 0.005) and cross-seed DoR std 0.012 (< 0.02)     |
| `min_per_class`     | 30                          | Floor before a parent is flagged "insufficient"                            |
| `n_eff_min`         | 30 (corr_range_m = 300 m)   | At target=100, median n_eff_min = 28.9 with 0.49 coverage at threshold 30  |
| Loss mask           | enabled (crop + built)      | Slightly better metrics than mask_off; robustness against trend artifacts  |
| Building-loss layer | optional (`--exclude-...`)  | Marginal effect; off by default                                            |
| Buffer steps        | 1000, 1500, 2000, 3000, 5000, 8000 m | Buffer used per parent recorded as `buffer_m_used` in output     |

## Run the EE sampler (v2 default policy)

```bash
python v2/scripts/sampling/sample_reference_states_v2.py \
  --export --verbose \
  --target-per-class 100 \
  --min-per-class 30 \
  --asset-id projects/ee-gsingh/assets/recover_reference_samples_v2 \
  --description recover_reference_samples_v2
```

Variants:

```bash
# Without loss mask (sensitivity scenario)
python v2/scripts/sampling/sample_reference_states_v2.py --disable-loss-mask --export --verbose

# Area-proportional class allocation
python v2/scripts/sampling/sample_reference_states_v2.py --allocation-mode area_proportional --export --verbose

# Include buildings-loss layer in mask
python v2/scripts/sampling/sample_reference_states_v2.py --exclude-buildings-loss --export --verbose
```

## Strategy comparison (re-evaluate after re-sampling)

```bash
python v2/scripts/analysis/compare_sampling_options.py \
  --input v2/data/recover_reference_samples_v2_mask_on_large_alphaearth.parquet \
  --out-dir v2/data/v2real_mask_on_corr300_exhaustive \
  --plot-dir v2/plots/v2real_mask_on_corr300_exhaustive \
  --corr-range-m 300 \
  --n-boot 800 \
  --exhaustive
```

Repeat for `--corr-range-m` in {150, 300, 500} and the mask-off parquet, then aggregate:

```bash
python v2/scripts/analysis/aggregate_v2real_results.py
```

Produces:
- `v2/data/v2real_exhaustive_summary_all.csv`
- `v2/data/v2real_exhaustive_top5_by_scenario.csv`
- `v2/data/v2real_exhaustive_robust_rank.csv`

## Empirical n_eff calibration

```bash
python v2/scripts/analysis/neff_calibration.py \
  --input v2/data/recover_reference_samples_v2_mask_on_large_alphaearth.parquet \
  --out-dir v2/data/neff_calibration_mask_on \
  --plot-dir v2/plots/neff_calibration_mask_on \
  --sample-sizes 10,20,30,50,75,100,150,200 \
  --n-seeds 8 --n-boot 400 --corr-range-m 300
```

Outputs:
- `neff_calibration_summary.csv` with median CI width, p90 CI width,
  median n_eff_min, median cross-seed DoR std, and CI-width slope per +10 points.
- `neff_calibration.png` showing the plateau curves used to pick `target_per_class`.

## Headline results (with real test-site x_obs)

Top of robust rank (mean rank across 6 scenarios; lower is better):

| Strategy              | Mean rank |
|-----------------------|-----------|
| `random_200`          | 1.0       |
| `fscs_200`            | 2.0       |
| `random_150`          | 3.5       |
| `fscs_150`            | 4.0       |
| `feature_diverse_150` | 4.5       |

Note: `random_200` ≈ `fscs_200` is **degenerate** — the candidate pool is 200
per class so both select the full pool. The decision uses the calibration
plateau (n=100 per class) rather than the cap=200 tie.

At `target_per_class=100`, mask_on, corr=300:
- `random_100`: median CI = 0.0736, median n_eff_min = 28.97, pooled AUC = 0.913

## Methodology notes

- DoR uses cosine distance: `median(d_bad) / (median(d_good) + median(d_bad))`.
- 95% bootstrap CIs use `n_boot=800` over per-set medians.
- `n_eff` ≈ `n² / Σ exp(-(d/L)²)` with `L = corr_range_m`. Sensitivity to L is
  reported across {150, 300, 500} m.
- Comparator's `x_obs` defaults to the test-site embedding (joined by
  `parent_id`) and falls back to the reference mean if the join fails.
  Use `--use-ref-mean` to reproduce the legacy v1 behaviour.
