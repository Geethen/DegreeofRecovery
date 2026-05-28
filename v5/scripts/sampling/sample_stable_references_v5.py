"""v5 sampler for stable_stable parents.

Adds to v4 (v4/scripts/sampling/sample_stable_references_v4.py):
  1. Ecoregion intersection: the sampling region is parent.buffer(MAX_R)
     intersected with the RESOLVE 2017 ecoregion polygon the parent sits in.
     Refs outside the ecoregion are rejected even if within the buffer.
  2. GHM attach: every ref carries `ghm_aa` -- the GlobalHumanModification 2022
     90 m all-threats-combined value at the ref point. Sampled native (90 m).
  3. Buffer ceiling kept at 8 km via BUFFER_STEPS_M (1k -> 8k as in v4).
  4. Two-tier selection: the 100 'selected' refs per class are drawn from the
     dist_m >= INNER_EXCLUSION_M (default 4 km) annulus so production scoring
     is structurally insulated from sub-pixel contamination and the inner
     portion of the variogram correlation range. Up to DIAG_PER_CLASS=1000
     additional refs per class are kept tagged 'diagnostic' (the <4 km disk)
     so analyses can still characterise the contamination + autocorrelation
     gradient without re-sampling. (DIAG_PER_CLASS default 200 keeps the v5
     output ~3-4x v4's size; raise to 1000 for richer diagnostics at ~10x v4.)

Motivation: v4 sampling treats refs within ~100 m of the test-site centroid as
independent, which they aren't (sub-pixel pseudoreplication in built-loss).
Extending the buffer alone does not fix this -- the DoR drift past 100 m is
dominated by spatial autocorrelation, not contamination. Constraining the
pool to the parent's ecoregion removes refs that are physically close but
ecologically irrelevant, and adding GHM lets downstream analyses condition on
human-modification gradient (which the AlphaEarth embedding compresses but
does not isolate).

Per-class bad-pool routing (unchanged from v4):
  - stable_class = nature -> bad = WC{40,50}
  - stable_class = crop   -> bad = WC{40}
  - stable_class = built  -> bad = WC{50}

Good pool (unchanged): WC != {40, 50, 80} AND not in the loss-trend mask.

Output schema mirrors v4 plus:
  - ghm_aa: GHM 2022 all-threats (0-1) sampled at the ref point
  - eco_id: RESOLVE 2017 ECO_ID of the ecoregion the parent (and ref) sits in
  - selection: 'selected' (production scoring) or 'diagnostic' (<4 km annulus)
  - inner_exclusion_m: the radius used for the selected/diagnostic split
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ee

PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"

# RESOLVE 2017 Ecoregions
ECOREGIONS_ASSET = "RESOLVE/ECOREGIONS/2017"
# GHM v3 2022 at 90 m -- ImageCollection from sat-io
GHM_ASSET = "projects/sat-io/open-datasets/GHM/HM_2022_90M"
GHM_BAND = "AA"

SCALE = 10
SCALE_GHM = 90
SEED = 234

TARGET_PER_CLASS = 100
MIN_PER_CLASS = 30
TOTAL_POINTS = 200
OVERSAMPLE_FACTOR = 5

# v5: production scoring runs with a minimum exclusion radius around the
# parent centroid. The sampler still draws from the full buffer ∩ ecoregion so
# downstream analyses can study the <INNER_EXCLUSION_M annulus, but the 100
# nearest refs reported as `selection = 'selected'` are drawn only from
# dist_m >= INNER_EXCLUSION_M. Refs in the inner annulus are kept as
# `selection = 'diagnostic'`, up to DIAG_PER_CLASS each, so per-site
# diagnostics (built-loss contamination, m_g/m_b decomposition) remain
# possible without re-sampling.
INNER_EXCLUSION_M = 4000
DIAG_PER_CLASS = 200

BUFFER_STEPS_M = [1000, 1500, 2000, 3000, 5000, 8000]

CROP_LOSS_THR = -7
BUILT_LOSS_THR = -7
BUILDING_LOSS_THR = -7

EXPORT_DESCRIPTION = "recover_reference_samples_v5_stable"
EXPORT_ASSET_ID = "projects/ee-gsingh/assets/recover_reference_samples_v5_stable"


def initialize() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=PROJECT)


def build_exclusion_mask(include_buildings_loss: bool = True) -> ee.Image:
    """Same loss exclusion as v2/v4 -- used to mask GOOD candidates only."""
    worldcereal = (
        ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
        .filter('product == "temporarycrops"')
        .select("classification")
        .max()
    )
    glad = ee.ImageCollection("projects/glad/GLCmap2019").mosaic()
    glad_crop = glad.eq(252)
    glad_urban = glad.gte(240).And(glad.lte(249))

    glc = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual").mosaic()
    glc_urb_2018 = glc.select("b19").eq(190)
    glc_crop_2018 = glc.select("b19").lte(20)

    crop_trend = ee.ImageCollection(
        "projects/gee-zander-nina/assets/HABLOSS/crop_trend_2018_2024_v1"
    ).mosaic()
    cropland_base = (
        ee.Image(0).where(worldcereal, 1).where(glad_crop, 1).where(glc_crop_2018, 1)
    )
    crop_trend = crop_trend.where(cropland_base.eq(0), 0)
    crop_loss = crop_trend.lt(CROP_LOSS_THR)

    built_trend = ee.ImageCollection(
        "projects/gee-zander-nina/assets/HABLOSS/grey_trend_2018_2024_v1"
    ).mosaic()
    built_base = ee.Image(0).where(glad_urban, 1).where(glc_urb_2018, 1)
    built_trend = built_trend.where(built_base.eq(0), 0)
    built_loss = built_trend.lt(BUILT_LOSS_THR)

    if include_buildings_loss:
        building_trend = ee.ImageCollection(
            "projects/gee-zander-nina/assets/HABLOSS/buildings_trend_2018_2023_v1"
        ).mosaic()
        buildings_loss = building_trend.lt(BUILDING_LOSS_THR)
    else:
        buildings_loss = ee.Image(0)

    return crop_loss.Or(built_loss).Or(buildings_loss).unmask(0).toByte()


def _classband_for(stable_class: str, exclusion: ee.Image) -> ee.Image:
    wc = ee.ImageCollection("ESA/WorldCover/v200").mosaic()
    natural = wc.neq(40).And(wc.neq(50)).And(wc.neq(80))

    if stable_class == "nature":
        bad = wc.eq(40).Or(wc.eq(50))
    elif stable_class == "crop":
        bad = wc.eq(40)
    elif stable_class == "built":
        bad = wc.eq(50)
    else:
        raise ValueError(f"Unsupported stable_class: {stable_class!r}")

    cb = (
        natural.multiply(1)
        .add(bad.multiply(2))
        .selfMask()
        .rename("ref")
        .toByte()
    )
    cb = cb.updateMask(cb.eq(2).Or(exclusion.Not()))
    return cb


def _first_radius_ge(dist_m: ee.Number, radii: ee.List) -> ee.Number:
    d = ee.Number(dist_m)
    out = ee.Number(
        radii.iterate(
            lambda cur, acc: ee.Algorithms.If(
                ee.Number(acc).lt(0).And(ee.Number(cur).gte(d)),
                cur,
                acc,
            ),
            -1,
        )
    )
    return ee.Number(ee.Algorithms.If(out.lt(0), radii.get(-1), out))


def get_ecoregions() -> ee.FeatureCollection:
    return ee.FeatureCollection(ECOREGIONS_ASSET)


def get_ghm_image() -> ee.Image:
    """GHM v3 2022 at 90 m, AA threat category (all threats combined).

    The HM_2022_90M collection holds one image per threat with band name
    'constant'; the threat code lives in the system id (..._2022s_AA_90 etc).
    We filter to AA and rename the single band so downstream `reduceRegions`
    returns a column we can identify.
    """
    img = ee.ImageCollection(GHM_ASSET).filter(
        ee.Filter.stringContains("system:index", f"_{GHM_BAND}_90")
    ).first()
    return ee.Image(img).rename("ghm_aa")


def make_sampler(
    classband: ee.Image,
    parent_label: str,
    ecoregions: ee.FeatureCollection,
    ghm_image: ee.Image,
    target_per_class: int = TARGET_PER_CLASS,
    min_per_class: int = MIN_PER_CLASS,
    total_points: int = TOTAL_POINTS,
    oversample_factor: int = OVERSAMPLE_FACTOR,
    buffer_steps_m: list[int] | None = None,
    inner_exclusion_m: int = INNER_EXCLUSION_M,
    diag_per_class: int = DIAG_PER_CLASS,
):
    radii = ee.List(buffer_steps_m if buffer_steps_m is not None else BUFFER_STEPS_M)
    max_r = ee.Number(radii.get(-1))
    inner_r = ee.Number(inner_exclusion_m)
    ref_state_lookup = ee.Dictionary({"1": "good", "2": "bad"})

    def sample_one(ft: ee.Feature) -> ee.FeatureCollection:
        ft = ee.Feature(ft)
        parent_id = ft.id()
        parent_geom = ft.geometry()

        # 1) ecoregion polygon the parent sits in
        eco_match = ecoregions.filterBounds(parent_geom).first()
        # Sentinel feature so the script never breaks if a parent has no eco match
        # (RESOLVE has near-global terrestrial coverage so this is rare).
        eco_id = ee.Algorithms.If(eco_match, ee.Feature(eco_match).get("ECO_ID"), -1)
        eco_geom = ee.Algorithms.If(
            eco_match,
            ee.Feature(eco_match).geometry(),
            parent_geom.buffer(max_r),  # fallback: just use the buffer
        )

        # 2) sampling region = buffer ∩ ecoregion (full disk, not annulus)
        region = parent_geom.buffer(max_r).intersection(ee.Geometry(eco_geom), 1)

        t_good = ee.Number(target_per_class)
        t_bad = ee.Number(target_per_class)
        # Budget the oversample on DIAG_PER_CLASS, not target_per_class. We
        # need enough candidates in the OUTER annulus (dist >= inner_r) for the
        # 100 selected refs to survive, AND enough in the inner disk for
        # diagnostics. Using diag_per_class * oversample as the EE-side request
        # gives both.
        class_points = [
            ee.Number(diag_per_class).multiply(oversample_factor).int(),
            ee.Number(diag_per_class).multiply(oversample_factor).int(),
        ]

        candidates = classband.stratifiedSample(
            numPoints=0,
            classBand="ref",
            classValues=[1, 2],
            classPoints=class_points,
            region=region,
            scale=SCALE,
            seed=SEED,
            geometries=True,
        )

        # 3) attach GHM at the ref point (native 90 m)
        candidates = ghm_image.reduceRegions(
            collection=candidates,
            reducer=ee.Reducer.first(),
            scale=SCALE_GHM,
        )

        # reduceRegions writes its result under property 'first'. Copy it to
        # ghm_aa.
        candidates = candidates.map(
            lambda f: f.set({
                "ref_state": ref_state_lookup.get(
                    ee.Number(f.get("ref")).format("%d")),
                "dist_m": f.geometry().distance(parent_geom, 1),
                "parent_id": parent_id,
                "parent_label": parent_label,
                "eco_id": eco_id,
                "ghm_aa": f.get("first"),
                "target_good": t_good,
                "target_bad": t_bad,
            })
        )

        # 4) Two-tier selection per class:
        #    - selected:   nearest target_per_class refs with dist_m >= inner_r
        #    - diagnostic: nearest diag_per_class refs from the FULL pool that
        #                  are NOT already in selected (i.e. dist_m < inner_r,
        #                  plus any 4-8 km refs beyond the selected 100).
        # Both go into the output, tagged via 'selection'. Production scoring
        # filters to selection='selected'.
        def split_class(class_value: int, target: ee.Number) -> ee.FeatureCollection:
            pool = candidates.filter(ee.Filter.eq("ref", class_value))
            outer = pool.filter(ee.Filter.gte("dist_m", inner_r)).sort("dist_m").limit(target)
            inner_pool = pool.filter(ee.Filter.lt("dist_m", inner_r)).sort("dist_m").limit(diag_per_class)
            outer_tagged = outer.map(lambda f: ee.Feature(f).set("selection", "selected"))
            inner_tagged = inner_pool.map(lambda f: ee.Feature(f).set("selection", "diagnostic"))
            return outer_tagged.merge(inner_tagged)

        good = split_class(1, t_good)
        bad = split_class(2, t_bad)
        selected = good.merge(bad)

        # selected_only used for the buffer_m_used / target_met / min_met flags
        good_sel = good.filter(ee.Filter.eq("selection", "selected"))
        bad_sel = bad.filter(ee.Filter.eq("selection", "selected"))

        max_dist = ee.Number(
            ee.Algorithms.If(
                good_sel.merge(bad_sel).size().gt(0),
                good_sel.merge(bad_sel).aggregate_max("dist_m"),
                max_r,
            )
        )
        used_r = _first_radius_ge(max_dist, radii)

        keep_props = [
            "parent_id", "parent_label", "ref", "ref_state", "dist_m",
            "buffer_m_used", "target_good", "target_bad", "target_met",
            "min_met", "stable_class", "strategy", "eco_id", "ghm_aa",
            "selection", "inner_exclusion_m",
        ]
        return selected.map(
            lambda f: ee.Feature(f).set({
                "buffer_m_used": used_r,
                "target_met": ee.Number(good_sel.size()).gte(t_good)
                .And(ee.Number(bad_sel.size()).gte(t_bad)),
                "min_met": ee.Number(good_sel.size()).gte(min_per_class)
                .And(ee.Number(bad_sel.size()).gte(min_per_class)),
                "stable_class": parent_label.replace("stable_", ""),
                "strategy": f"random_{int(target_per_class)}",
                "inner_exclusion_m": inner_r,
            }).select(propertySelectors=keep_props, retainGeometry=True)
        )

    return sample_one


def load_classification(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cls = row["stable_class"]
            if cls in {"nature", "crop", "built"}:
                out[str(row["parent_id"])] = cls
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--classification",
        default=str(Path(__file__).resolve().parents[2] / "data" /
                    "stable_state_classification.csv"),
        help="Reuses v4's stable_state_classification.csv.",
    )
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--asset-id", default=EXPORT_ASSET_ID)
    parser.add_argument("--description", default=EXPORT_DESCRIPTION)
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS)
    parser.add_argument("--total-points", type=int, default=TOTAL_POINTS)
    parser.add_argument("--oversample-factor", type=int, default=OVERSAMPLE_FACTOR)
    parser.add_argument(
        "--buffer-steps",
        default=",".join(str(v) for v in BUFFER_STEPS_M),
    )
    parser.add_argument(
        "--inner-exclusion-m", type=int, default=INNER_EXCLUSION_M,
        help="Refs with dist_m < this radius are tagged 'diagnostic' instead "
             "of 'selected'. The nearest target_per_class refs with "
             "dist_m >= inner-exclusion-m are tagged 'selected' and used for "
             "production scoring. Default 4000.",
    )
    parser.add_argument(
        "--diag-per-class", type=int, default=DIAG_PER_CLASS,
        help="Up to this many additional refs per class are kept tagged "
             "'diagnostic' (the <inner-exclusion-m annulus). Default 1000.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, sample only the first N matching parents (for dry-runs).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initialize()
    buffer_steps_m = [int(x.strip()) for x in args.buffer_steps.split(",") if x.strip()]

    # v4's classification CSV is reused -- the same parent_id -> stable_class mapping
    # the v4 sampler used. v5 changes the sampling region, not the parent set.
    classification_path = Path(args.classification)
    if not classification_path.exists():
        # fall back to v4's copy
        classification_path = Path(__file__).resolve().parents[3] / "v4" / "data" / "stable_state_classification.csv"
    classification = load_classification(classification_path)
    if args.verbose:
        from collections import Counter
        c = Counter(classification.values())
        print(f"Classifications loaded: {dict(c)} (total {len(classification)})")
        print(f"From: {classification_path}")

    parent_ids_keep = ee.List(list(classification.keys()))
    samples = (
        ee.FeatureCollection(SAMPLES_ASSET)
        .filter(ee.Filter.eq("r", "stable_stable"))
        .filter(ee.Filter.inList("system:index", parent_ids_keep))
        .map(lambda ft: ft.setGeometry(ft.geometry().centroid(5)))
    )
    if args.limit is not None:
        samples = samples.limit(args.limit)

    exclusion = build_exclusion_mask(include_buildings_loss=True)
    ecoregions = get_ecoregions()
    ghm = get_ghm_image()

    fcs: list[ee.FeatureCollection] = []
    for stable_class in ("nature", "crop", "built"):
        ids_for_class = [pid for pid, c in classification.items() if c == stable_class]
        if not ids_for_class:
            continue
        if args.verbose:
            print(f"  {stable_class}: {len(ids_for_class)} parents")

        group = samples.filter(
            ee.Filter.inList("system:index", ee.List(ids_for_class))
        )
        cb = _classband_for(stable_class, exclusion)
        fc = ee.FeatureCollection(
            group.map(make_sampler(
                cb,
                parent_label=f"stable_{stable_class}",
                ecoregions=ecoregions,
                ghm_image=ghm,
                target_per_class=args.target_per_class,
                min_per_class=args.min_per_class,
                total_points=args.total_points,
                oversample_factor=args.oversample_factor,
                buffer_steps_m=buffer_steps_m,
                inner_exclusion_m=args.inner_exclusion_m,
                diag_per_class=args.diag_per_class,
            ))
        ).flatten()
        fcs.append(fc)

    if not fcs:
        raise SystemExit("No non-ambiguous classifications available; nothing to sample.")

    reference_samples = fcs[0]
    for extra in fcs[1:]:
        reference_samples = reference_samples.merge(extra)

    if args.export:
        task = ee.batch.Export.table.toAsset(
            collection=reference_samples,
            description=args.description,
            assetId=args.asset_id,
        )
        task.start()
        print(f"Started export task '{args.description}' -> {args.asset_id}")
        return

    print("Reference sample size (dry run):", reference_samples.size().getInfo())
    if args.verbose:
        first = reference_samples.first().getInfo()
        keys = sorted(first.get("properties", {}).keys())
        print("First feature property keys:", keys)


if __name__ == "__main__":
    main()
