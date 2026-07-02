# Degree of regeneration

We estimated the degree of regeneration (DoR) for sites classified as
artificial land reversion using a local space-for-time comparison. The
aim was to ask whether a reverted site now looks more like nearby
natural or semi-natural land, or more like the artificial land-cover
state it replaced. Stable natural and stable artificial sites were
scored with the same workflow but were interpreted as controls. 1,248
sites were assessed (747 stable natural, 115 artificial reversion and
386 stable artificial).

For each site, we used the 2024 AlphaEarth annual satellite embedding
(Brown et al., 2025), a 64-band, 10 m representation of the land
surface. The embedding summarises spectral, phenological, structural,
climatic and topographic information, so sites can be compared by their
overall land-surface signature rather than by a single index such as
NDVI. Around each site, we sampled two local reference pools; a
near-natural pool from WorldCover natural and semi-natural pixels,
excluding water and other candidate reversion pixels, and a non-natural
pool from nearby WorldCover artificial land pixels.

Reference pixels were drawn from the site’s RESOLVE 2017 ecoregion
within a 3-8 km buffer. This kept the comparison local while reducing
contamination from the test site itself. The buffer was chosen from a
buffer analysis that balanced contamination removal, class separation,
spatial independence, confidence-interval width and reference retention.
Sites were scored only when both reference pools were available; 884 of
1,248 sites met this condition. Among scored sites, the median reference
pool contained 155 near-natural and 130 non-natural pixels.

DoR was calculated with cosine distance in AlphaEarth embedding space.
Cosine distance compares the shape/pattern of values across the 64
embedding bands while reducing sensitivity to overall vector magnitude.
For each site, we calculated the mean cosine distance to the five
nearest pixels in each reference pool, using the closest local analogues
rather than the average of a mixed reference cloud. Let D<sub>N</sub> be
the mean distance to the near-natural pool and D<sub>A</sub> the mean
distance to the non-natural pool:

DoR = D<sub>A</sub> / (D<sub>N</sub> + D<sub>A</sub>)

Values near 1 indicate greater similarity to local near-natural
conditions, whereas values near 0 indicate greater similarity to local
artificial conditions. The distance to artificial land is the numerator
because a regenerating site is expected to move away from the artificial
state it replaced and toward nearby natural reference conditions.
Uncertainty in sample selection was estimated with 2,000 bootstrap
resamples of the reference pools, and confidence intervals were compared
with empirically calibrated thresholds to classify sites as
regenerating, degraded or indistinguishable.

The local-buffer results followed the expected gradient. Median DoR was
highest for stable natural sites (0.58), intermediate for artificial
reversion sites (0.55), and lowest for stable artificial controls
(0.43). Among locally scored artificial reversion sites, 48% were
classified as regenerating, compared with 14% of stable artificial
controls. This suggests that many reversion sites have shifted toward
local natural reference conditions, while most persistent artificial
sites remain closer to the non-natural reference state.

## Ecoregion-wide reference complement

We also calculated an ecoregion-wide diagnostic to place each site in a
broader biome context. This used the same 2024 AlphaEarth embedding and
the same near-natural and non-natural reference definitions, but sampled
references across the whole RESOLVE 2017 ecoregion using Feature Space
Coverage Sampling on a 10 km grid. The local-buffer score therefore asks
whether a site resembles its nearby surroundings, while the ecoregion
diagnostic asks whether it resembles the wider biome.

For each site, we compared its similarity to the ecoregion’s
near-natural and non-natural reference pixels, then expressed those
values as percentile ranks against the range of similarities observed
among reference pixels from the same ecoregion. We combined the two
percentiles as:

pct_dor = (pct_vs_good + (100 - pct_vs_bad)) / 2

Higher values indicate sites that are more natural-like and less
built-like relative to the ecoregion as a whole. This diagnostic
separated groups less strongly than the local-buffer DoR: median pct_dor
was 54.1 for stable natural sites, 50.0 for stable artificial sites and
46.8 for artificial reversion sites. Across the 749 sites scored by both
methods, the two measures were positively but moderately correlated
(Spearman rho = 0.35, p \< 1e-22), showing that they capture related but
distinct reference frames.

## Data and reproducibility

The local-buffer DoR is written to
`outputs/data/test_site_dor_buffer_3_8km.csv`, the ecoregion-wide
percentile to `outputs/data/test_site_dor_ecoregion_2024.csv`, and the
joined site-level table to
`outputs/data/test_site_scores_combined.csv`. The scoring code is in
`src/degree_of_recovery/core.py`, with workflows in
`scripts/03_score_buffer_dor.py`, `scripts/04_score_ecoregion_dor.py`
and `scripts/05_combine_and_summarise.py`. See the top-level
[README](../README.md) for how to run the pipeline and for the two
run modes (sample data vs. full reproduction).
