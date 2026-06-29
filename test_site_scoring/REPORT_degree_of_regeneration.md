# Degree of Regeneration — methods and results (test-site scoring)

> **Scope.** This report supersedes the *Degree of Regeneration* methods text
> circulated on 26 Jun 2026 (`paper_methods_v26062026.md`). It incorporates the v5
> annulus buffer geometry (inner = 3 km / outer = 8 km), the ecoregion-wide
> percentile complement scored on the matched 2024 AlphaEarth vintage, and the three
> co-author clarifications: an explicit DoR formula, the rationale for cosine
> distance, and the interpretation of the score when distance to artificial land
> increases. The sample comprises 1,248 test sites in three 2018→2024 transition
> groups (crop classes excluded): stable_natural (n = 747), artificial_reversion
> (n = 115), and stable_artificial (n = 386). The metric is the **Degree of
> Regeneration (DoR)**, the positive outcome is **"regenerating"**, and the data
> column `dor_knn` is retained for continuity.

## Degree of regeneration

We estimated the degree of regeneration for the sample units classified as artificial
land reversion using a local space-for-time comparison. The score represents the
extent to which a reverted site more closely resembles nearby (semi-)natural land or
the artificial land-cover state it replaced. Sample units labelled as stable
artificial land were processed with the same workflow but used only as controls and
not interpreted as recovery observations.

For each sample unit we used the 2024 AlphaEarth annual satellite embedding (Brown et
al., 2025), a 64-band, 10 m representation of the land surface capturing spectral,
phenological, structural, climatic and topographic characteristics. In this context, a
foundation model embedding is a compact numerical description of each pixel that
integrates information from multiple Earth-observation data sources, allowing sites to
be compared by their overall environmental similarity rather than by a single index
such as NDVI. Two local reference pools were sampled around each site. The near-natural
pool comprised ESA WorldCover pixels that were not cropland, artificial land or water,
and excluding other candidate artificial land reversion pixels. The non-natural pool
comprised nearby artificial land pixels mapped in WorldCover.

Reference pixels were drawn from a concentric buffer around each site, bounded by an
inner exclusion radius and an outer ceiling that expanded dynamically until reference
targets were met, rather than from a fixed-radius buffer. The sampling region was the
buffer intersected with the RESOLVE 2017 ecoregion polygon in which the site falls, so
that references outside the site's ecoregion were rejected even when within the radius;
this constrains both reference pools to the site's biome-level habitat context. The
inner exclusion removes near-field pixels most likely to be contaminated by the test
site itself, while the outer ceiling keeps references within a comparable landscape
context and admits enough spatially independent pixels to support a stable confidence
interval. The ceiling expanded through 5 km and, where count targets were unmet, to
8 km. The inner/outer geometry was selected by a buffer-desirability analysis over a
grid of inner (0–4 km) × outer (1–8 km) candidates, scored per land-cover class on
contamination removal, separability, spatial independence, confidence-interval width,
and reference retention; the optimum was inner = 3 km / outer = 8 km, on a broad plateau
of near-equivalent settings, and references in the 3–8 km band were used for all scores
reported here. The point estimate is nearly invariant to the outer ceiling (drift
< 0.02 over 3–8 km, because the score uses only the nearest references); the wider
ceiling primarily tightens the confidence interval. Each site obtained ample references
in this band (median ≈ 155 near-natural and ≈ 130 non-natural pixels).

Degree of Regeneration (DoR) was calculated using cosine distance in AlphaEarth
embedding space. We used cosine distance because it compares the pattern of values
across the 64 embedding bands while reducing sensitivity to differences in overall
vector magnitude — capturing the kind of land surface a pixel represents rather than
the strength of its signal — and because it is the similarity measure the embedding
space is built for, which keeps the near-natural and non-natural reference pools
cleanly separable (separability MCC = 0.92, AUC = 0.99). For each site we calculated
the mean cosine distance to the five nearest pixels in each reference pool, using the
closest local analogues rather than the average of a heterogeneous pool. Writing D_N
and D_A for these mean distances to the near-natural and non-natural pools, the score
is

> **DoR = D_A / (D_N + D_A)**   (Eq. 1)

so that values near 1 indicate greater similarity to local near-natural reference
conditions and values near 0 greater similarity to non-natural or built-up reference
conditions. DoR thus represents a site's relative position along a locally defined
artificial-to-natural similarity gradient rather than a direct measure of ecological
recovery. The distance to the non-natural state (D_A) is the numerator because a site
that has genuinely regenerated has moved *away* from the artificial land it replaced
(large D_A) and *toward* near-natural conditions (small D_N), driving the ratio toward
1; a larger distance to artificial land is therefore evidence of more regeneration,
not a paradox. Uncertainty was estimated using 2,000 bootstrap resamples of the local
reference pools, and each site was classified against an empirically calibrated,
per-class threshold via its confidence interval as regenerating, degraded, or
indistinguishable.

Under this design the three groups order as expected — stable_natural (median
DoR = 0.58) > artificial_reversion (0.55) > stable_artificial (0.43). Artificial
reversion sites sit near the natural/artificial boundary, consistent with land that has
lost its artificial cover and is partially regenerating, and 48% are classified as
regenerating. Only 14% of stable artificial controls are classified as regenerating,
confirming that the 3 km / 8 km buffer does not systematically misclassify persistent
artificial land as recovered (884 of 1,248 sites had sufficient buffer references to be
scored this way).

## Ecoregion-wide reference complement

Both the local buffer and the ecoregion layer draw references only from within the
site's RESOLVE 2017 ecoregion; they differ in spatial scope. The local buffer samples
ecoregion-constrained pixels within a few kilometres of the site, whereas the ecoregion
layer samples references across the entire ecoregion, giving a biome-wide rather than a
neighbourhood reference frame. We built it as an ecologically anchored complement
because a near-site buffer, even when clipped to the ecoregion, characterises only the
local landscape and may miss the full range of natural and artificial conditions the
biome contains. Each ecoregion was sampled with Feature Space Coverage Sampling on a
10 km covering grid, spreading the reference sample evenly across the ecoregion's
feature space, using the same two-layer (WorldCover and HABLOSS) natural/non-natural
definition as the buffer pipeline. This is a data-preparation stage only and does not
change the DoR scoring function. References were sampled on the 2024 AlphaEarth vintage
to match the test-site embeddings, resolving the earlier 2022/2024 vintage mismatch.

For each site we computed its mean cosine similarity to its ecoregion's near-natural and
non-natural reference clouds, expressed each as a percentile rank within a null
distribution built from the ecoregion's own references, and combined them into a
directional percentile, pct_dor = (pct_vs_good + (100 − pct_vs_bad)) / 2, so that high
values indicate a site that is natural-like and not built-like relative to its whole
ecoregion. The ecoregion percentile reproduced the broad control contrast but separated
the groups less strongly than the local buffer: median pct_dor was 54.1 for
stable_natural, 50.0 for stable_artificial, and 46.8 for artificial_reversion, with the
medians compressed near the percentile midpoint. Reversion sites carried the highest
pct_vs_bad (43.2), indicating that against their whole biome they still resemble
artificial references more than the controls do — a stricter bar than the local buffer,
which compares only against nearby artificial land. The ecoregion percentile is
therefore reported as a complementary, biome-anchored diagnostic rather than a
replacement for the local-buffer DoR.

Across the 749 sites scored by both methods, the two scores were positively but only
moderately correlated (Spearman ρ = 0.35, p < 10⁻²²), and both recovered the same group
ordering (stable_natural > artificial_reversion > stable_artificial). The moderate
correlation reflects their different reference frames — the local buffer asks whether a
site is natural-like relative to its immediate surroundings, the ecoregion percentile
whether it is natural-like relative to its entire biome — so the two are complementary
rather than redundant.

## Data and reproducibility

The local-buffer DoR is in `test_site_scoring/data/test_site_dor_buffer_3_8km.csv`, the
ecoregion-percentile DoR in `test_site_scoring/data/test_site_dor_ecoregion_2024.csv`,
and the joined per-site table in `test_site_scoring/data/test_site_scores_combined.csv`;
the v5 built-loss point shapefile is `v5/data/test_site_dor_v5_built_loss.shp`. Scoring
primitives (Eq. 1, the five-nearest distances, bootstrap CI and classification) are in
`src/degree_of_recovery/core.py`; the buffer and ecoregion scorers are
`test_site_scoring/scripts/03_score_buffer_dor.py` and `04_score_ecoregion_dor.py`, and
the join and summary are in `05_combine_and_summarise.py`. Citations to add: RESOLVE
2017 Ecoregions (Dinerstein et al., 2017; GEE asset `RESOLVE/ECOREGIONS/2017`),
AlphaEarth annual embeddings (Brown et al., 2025), and the Feature Space Coverage
Sampling lineage (Brus, 2019, or Minasny & McBratney, 2006).
