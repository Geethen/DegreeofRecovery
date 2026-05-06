# Presentation Slide Summaries
**Audience:** Ecologists, sociologists, GIS and remote sensing researchers
**Duration:** 30–40 minutes
**Goal:** Communicate research interests and skills to identify collaboration opportunities, proposal alignment, and areas where you can provide technical input or decision-making support.

---

## Slide 1 — Title
**"Satellites, ML & Ecology: From Invasions to Biodiversity Intactness"**
Geethen Singh | [Affiliation]

---

## Slide 2 — Who Am I?
Researcher at the intersection of remote sensing, machine learning, and ecology. Background spans academic research and consulting roles, including experience in medical statistics. Work covers land cover mapping, invasive species, biodiversity, ecosystem condition, burn scar mapping, biomass estimation, and species distribution modelling (SDMs) — primarily in southern Africa.

---

## Slide 3 — The Core Problem
Ecological and environmental decisions need to be spatially explicit and timely. Ground-based surveys are slow, expensive, and sparse. Satellite data + ML bridges that gap — enabling continental-scale monitoring, time series change detection, and evidence-based decision making.

---

## Slide 4 — Why Satellite Data + ML?
- Free, global, repeat coverage enables time series and change detection
- Multi-spectral bands provide proxies for vegetation, water, soil, and fire
- ML handles high-dimensional spatial data where classical statistics struggle
- Tradeoffs exist: interpretability vs. predictive power — choosing the right tool matters
- Spatially explicit outputs support optimal, context-specific decisions

---

## Slide 5 — My Research Threads (Overview)
Visual overview linking to five vignette areas:
1. Land cover mapping & map product evaluation
2. Invasive species — detection, drivers, and range prediction
3. Landscape state — burn scars, ecosystem condition, biodiversity intactness
4. Biomass estimation & SDMs
5. Uncertainty quantification & label-efficient methods

---

## Vignette 1 — Land Cover

### Slide 6 — Which Global Map Should You Trust?
Comparison of Dynamic World, WorldCover, and ESRI Land Cover at 10 m resolution (Singh et al., 2022 — 413 citations). Each product has different accuracy profiles, update frequencies, and use-case fit. Key message: map choice is not neutral — it affects downstream analysis.

**Audience hook (GIS/RS):** If your project uses global land cover data, here is how to choose the right product.

---

## Vignette 2 — Invasive Species

### Slide 7 — Water Hyacinth: A Continental-Scale Problem
Water hyacinth (*Eichhornia crassipes*) is one of the world's worst aquatic invasive species. Ecological impacts: oxygen depletion, biodiversity loss. Socioeconomic impacts: blocked waterways, reduced water access, impacts on livelihoods.

**Audience hook (all three):** Ecologists (invasion biology), sociologists (water security, livelihoods), GIS/RS (monitoring at scale).

### Slide 8 — Detecting It From Space
National-extent mapping of water, aquatic vegetation, and water hyacinth using remote sensing (Singh et al., 2020 — 83 citations). Method: multi-temporal satellite imagery + classification. Key result: accurate detection at national scale, enabling targeted management responses.

### Slide 9 — What's Driving the Invasion?
Explainable ML (XAI/SHAP) used to identify environmental and anthropogenic drivers of water hyacinth spread (Singh et al., 2025). Not just *where* — but *why*. Key finding: [insert top drivers]. Demonstrates how XAI makes ML outputs actionable for managers and policymakers.

### Slide 10 — Predicting Future Range
Species distribution modelling (SDM) applied to invasive tree fern *Sphaeropteris cooperi* — predicted range shrinkage under climate scenarios in two southern hemisphere biodiversity hotspots (2025). Shows how satellite-derived covariates improve SDM predictions beyond climate-only models.

---

## Vignette 3 — Landscape State

### Slide 11 — Burn Scar Mapping
Detecting and monitoring fire extent using satellite time series. Applications: post-fire recovery assessment, fire regime characterisation, input layer for ecosystem condition models. Links fire disturbance to downstream biodiversity and carbon outcomes.

### Slide 12 — Ecosystem Condition & Biodiversity Intactness
A continuum: *land cover → ecosystem condition → biodiversity intactness*. Place-based assessment of biodiversity intactness in sub-Saharan Africa (Singh et al., 2026). Satellite proxies for on-ground biodiversity signals. Relevance for conservation planning, protected area management, and policy reporting.

---

## Vignette 4 — Biomass & SDMs

### Slide 13 — Biomass Estimation & SDMs
Estimating above-ground biomass from satellite data — relevance to carbon accounting, ecosystem services, and climate mitigation. SDMs used both classically and hybridised with remote sensing covariates for improved spatial predictions. Connects to carbon markets, community forestry, and biodiversity offsets.

**Audience hook (proposals):** Carbon and biodiversity are converging funding areas — spatially explicit estimates are increasingly required.

---

## Vignette 5 — Methods

### Slide 14 — My ML Philosophy
- Explainability over black boxes where possible (XAI/SHAP)
- Uncertainty quantification as standard practice, not afterthought
- Label-efficient and semi-supervised methods for data-scarce ecology
- Model choice driven by tradeoffs: interpretability, sample size, spatial autocorrelation

### Slide 15 — When to Trust Your Model
Conformal prediction for uncertainty quantification in earth observation ML (Singh et al., 2024 — 59 citations). Most models give a prediction — this framework tells you *when not to trust it*. Applicable to any ML workflow, not just remote sensing.

### Slide 16 — Reducing the Labelling Burden
Common Ground: semi-automated approach to tracking changes in land cover and species over time using time series (Singh et al., 2026). Less field work, more coverage. Practical value for anyone running monitoring programmes with limited ground-truth data.

### Slide 17 — Toolkit
Google Earth Engine, Python, time series analysis, conformal prediction, explainable ML (SHAP), SDM frameworks, statistical modelling (including medical statistics background for rigorous inference).

---

## Slide 18 — Where I Can Plug Into Your Work

| Audience | How I can contribute |
|---|---|
| **Ecologists** | Habitat/species mapping, change detection, biodiversity proxies, SDMs, fire ecology |
| **Sociologists** | Land use change, human-environment interaction, resource access mapping |
| **GIS / RS** | Map product selection, uncertainty-aware analysis, labelling pipelines, GEE workflows |
| **All** | Methods input on grant proposals, spatial analysis review, decision-support tools |

Consulting background means experience translating research outputs into practical decisions.

---

## Slide 19 — Open Questions I'm Wrestling With
- Scaling conformal prediction to operational monitoring systems
- Improving semi-supervised labelling across ecosystems and regions
- Cross-ecosystem transferability of trained models
- Integrating socioeconomic drivers into ecological models more rigorously

*Invites dialogue — what problems are you stuck on?*

---

## Slide 20 — Let's Find Overlaps
**Prompt to room:** "What spatial or temporal data problem are you stuck on?"

Discussion questions:
- Where do you need better spatial data than you currently have?
- Are you working on anything where field data is the bottleneck?
- Any funding calls where a remote sensing / ML component would strengthen the proposal?

---

## Slide 21 — Contact & Links
- Email: [your email]
- Google Scholar: https://scholar.google.com/citations?user=J4rtU2kAAAAJ&hl=en
- GitHub: [your GitHub]
- Key tools/datasets you maintain: [list if applicable]
