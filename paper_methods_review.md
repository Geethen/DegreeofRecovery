# Review of `paper_methods.docx`

Comments: 2  ·  Tracked changes: 6


## Comments

### 1. Geethen Singh — 2026-05-20T16:08:00Z
**On:** Ambiguous sites were excluded
**Says:** How many were excluded?
**Paragraph (¶ 15):** 'Stable' sites carried no loss-defined class__ by experts__, so we classified them as stable near-natural, stable cropland, stable built-up, or ambiguous by majority vote across six independent land-cover and building-footprint products (Supplementary Table S1). Ambiguous sites were excluded. Stable cropland and stable built-up controls are expected to score as non-natural; high recovery rates in these classes would indicate false positives.

### 2. Geethen Singh — 2026-05-20T16:13:00Z
**On:** , and the 200-pixel cap kept the two reference states balanced
**Says:** This does not seem valuable. Likely redundant. Consider dropping throughout
**Paragraph (¶ 17):** Reference pixels were drawn from adaptive 1-8 km concentric buffers, expanding through radii of 1, 1.5, 2, 3, 5, and 8 km until at least 30 pixels per pool were obtained, with a target of 100 and a combined cap of 200 (Supplementary Figure S1). These parameters were chosen from sampling experiments: scores and confidence-interval widths stabilised near 100 pixels per pool, 30 pixels was the minimum effective floor, and the 200-pixel cap kept the two reference states balanced.


## Tracked changes

### 1. Inserted — Geethen Singh — 2026-05-20T16:06:00Z
**Text:** e
**Paragraph (¶ 14):** We estimated nature regeneration at 158 candidate abandonment sites (68 cropland-loss, 90 built-loss) by asking whether each site more closely resembles nearby near-natural land or the transformed land it replaced. Th__e__~~is~~ Degree of Recovery (DoR) score is applied after abandonment has been identified; it does not detect abandonment. Stable cropland and stable built-up sites (n = 1,471 scored controls) were processed identically but not interpreted as recovery observations.

### 2. Deleted — Geethen Singh — 2026-05-20T16:06:00Z
**Text:** is
**Paragraph (¶ 14):** We estimated nature regeneration at 158 candidate abandonment sites (68 cropland-loss, 90 built-loss) by asking whether each site more closely resembles nearby near-natural land or the transformed land it replaced. Th__e__~~is~~ Degree of Recovery (DoR) score is applied after abandonment has been identified; it does not detect abandonment. Stable cropland and stable built-up sites (n = 1,471 scored controls) were processed identically but not interpreted as recovery observations.

### 3. Inserted — Geethen Singh — 2026-05-20T16:07:00Z
**Text:** by experts
**Paragraph (¶ 15):** 'Stable' sites carried no loss-defined class__ by experts__, so we classified them as stable near-natural, stable cropland, stable built-up, or ambiguous by majority vote across six independent land-cover and building-footprint products (Supplementary Table S1). Ambiguous sites were excluded. Stable cropland and stable built-up controls are expected to score as non-natural; high recovery rates in these classes would indicate false positives.

### 4. Inserted — Geethen Singh — 2026-05-20T16:09:00Z
**Text:** based on the loss masks from the upstream analy
**Paragraph (¶ 16):** For each test site we used the 2024 AlphaEarth annual satellite embedding (Brown et al., 2025), a 64-band, 10 m representation of the land surface capturing spectral, phenological, structural, climatic, and topographical characteristics. Two local reference pools were sampled in Google Earth Engine around each site. The near-natural pool comprised ESA WorldCover pixels that were not cropland, built-up land, or water, excluding pixels with evidence of recent loss__ based on the loss masks from the upstream analy____sis__. The non-natural pool matched the site's 2018 land cover: cropland pixels for cropland-loss and stable cropland sites, built-up pixels for built-loss and stable built-up sites, and the union of both for stable near-natural sites.

### 5. Inserted — Geethen Singh — 2026-05-20T16:10:00Z
**Text:** sis
**Paragraph (¶ 16):** For each test site we used the 2024 AlphaEarth annual satellite embedding (Brown et al., 2025), a 64-band, 10 m representation of the land surface capturing spectral, phenological, structural, climatic, and topographical characteristics. Two local reference pools were sampled in Google Earth Engine around each site. The near-natural pool comprised ESA WorldCover pixels that were not cropland, built-up land, or water, excluding pixels with evidence of recent loss__ based on the loss masks from the upstream analy____sis__. The non-natural pool matched the site's 2018 land cover: cropland pixels for cropland-loss and stable cropland sites, built-up pixels for built-loss and stable built-up sites, and the union of both for stable near-natural sites.

### 6. Inserted — Geethen Singh — 2026-05-20T15:59:00Z
**Text:** a
**Paragraph (¶ 23):** We quantified nature regeneration at 158 candidate abandonment sites (68 cropland-loss, 90 built-loss) using a Degree of Recovery (DoR) score that asks whether each site more closely resembles nearby near-natural land or the non-natural land it replaced; 1,471 stable-site controls were scored identically. Stable sites had no loss-defined class by experts, so we assigned each to stable near-natural, stable cropland, stable built-up, or ambiguous using __a __majority vote across six independent land-cover and building-footprint datasets (Supplementary Table S1); ambiguous sites were excluded.
