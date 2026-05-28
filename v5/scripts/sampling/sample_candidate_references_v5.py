"""v5 sampler for candidate (loss-site) parents.

Mirrors v2/scripts/sampling/sample_reference_states_v2.py with the same v5
additions made to the stable sampler (sample_stable_references_v5.py):
  1. Sampling region = parent.buffer(MAX_R) intersected with the RESOLVE 2017
     ecoregion polygon containing the parent.
  2. Each ref carries `ghm_aa` (GHM 2022 90 m AA band) and `eco_id`.
  3. Buffer ceiling kept at 8 km (BUFFER_STEPS_M = [1k..8k]).

Parents: built_loss (r=='built_loss') and crop_loss (r=='crop_loss') from
SAMPLES_ASSET; same as v2. Output schema mirrors v2 plus `eco_id`, `ghm_aa`,
`strategy` (which v2 lacked but v4 added).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ee

PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"

ECOREGIONS_ASSET = "RESOLVE/ECOREGIONS/2017"
GHM_ASSET = "projects/sat-io/open-datasets/GHM/HM_2022_90M"
GHM_BAND = "AA"

SCALE = 10
SCALE_GHM = 90
SEED = 234

TARGET_PER_CLASS = 100
MIN_PER_CLASS = 30
TOTAL_POINTS = 200
OVERSAMPLE_FACTOR = 5

# See sample_stable_references_v5.py for the rationale of the two-tier
# selection (selected vs diagnostic). Same constants apply to the candidate
# pool so downstream analyses can compare stable and candidate refs at the
# same minimum exclusion radius.
INNER_EXCLUSION_M = 4000
DIAG_PER_CLASS = 200

BUFFER_STEPS_M = [1000, 1500, 2000, 3000, 5000, 8000]

CROP_LOSS_THR = -7
BUILT_LOSS_THR = -7
BUILDING_LOSS_THR = -7

EXPORT_DESCRIPTION = "recover_reference_samples_v5_candidate"
EXPORT_ASSET_ID = "projects/ee-gsingh/assets/recover_reference_samples_v5_candidate"


def initialize() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=PROJECT)


def build_exclusion_mask(include_buildings_loss: bool = True):
    wc = ee.ImageCollection("ESA/WorldCover/v200").mosaic()

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

    return crop_loss.Or(built_loss).Or(buildings_loss).unmask(0).toByte(), wc


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


def build_classbands(exclusion_mask: ee.Image, wc: ee.Image):
    natural = wc.neq(40).And(wc.neq(50)).And(wc.neq(80))
    cb_built = natural.add(wc.eq(50).multiply(2)).selfMask().rename("ref").toByte()
    cb_crop = natural.add(wc.eq(40).multiply(2)).selfMask().rename("ref").toByte()
    cb_built = cb_built.updateMask(exclusion_mask.Not())
    cb_crop = cb_crop.updateMask(exclusion_mask.Not())
    return cb_built, cb_crop


def get_ecoregions() -> ee.FeatureCollection:
    return ee.FeatureCollection(ECOREGIONS_ASSET)


def get_ghm_image() -> ee.Image:
    img = ee.ImageCollection(GHM_ASSET).filter(
        ee.Filter.stringContains("system:index", f"_{GHM_BAND}_90")
    ).first()
    return ee.Image(img).rename("ghm_aa")


def make_sampler(
    classband: ee.Image,
    ecoregions: ee.FeatureCollection,
    ghm_image: ee.Image,
    buffer_steps_m: list[int] | None = None,
    target_per_class: int = TARGET_PER_CLASS,
    min_per_class: int = MIN_PER_CLASS,
    total_points: int = TOTAL_POINTS,
    oversample_factor: int = OVERSAMPLE_FACTOR,
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
        parent_label = ft.get("r")
        parent_geom = ft.geometry()

        eco_match = ecoregions.filterBounds(parent_geom).first()
        eco_id = ee.Algorithms.If(eco_match, ee.Feature(eco_match).get("ECO_ID"), -1)
        eco_geom = ee.Algorithms.If(
            eco_match,
            ee.Feature(eco_match).geometry(),
            parent_geom.buffer(max_r),
        )
        region = parent_geom.buffer(max_r).intersection(ee.Geometry(eco_geom), 1)

        t_good = ee.Number(target_per_class)
        t_bad = ee.Number(target_per_class)
        # Budget oversample on diag_per_class so both 'selected' (outer
        # annulus) and 'diagnostic' (inner disk) survive the EE-side cap.
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

        candidates = ghm_image.reduceRegions(
            collection=candidates,
            reducer=ee.Reducer.first(),
            scale=SCALE_GHM,
        )

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
            "min_met", "strategy", "eco_id", "ghm_aa",
            "selection", "inner_exclusion_m",
        ]
        return selected.map(
            lambda f: ee.Feature(f).set({
                "buffer_m_used": used_r,
                "target_met": ee.Number(good_sel.size()).gte(t_good)
                .And(ee.Number(bad_sel.size()).gte(t_bad)),
                "min_met": ee.Number(good_sel.size()).gte(min_per_class)
                .And(ee.Number(bad_sel.size()).gte(min_per_class)),
                "strategy": f"random_{int(target_per_class)}",
                "inner_exclusion_m": inner_r,
            }).select(propertySelectors=keep_props, retainGeometry=True)
        )

    return sample_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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
        help="Refs with dist_m < this radius are tagged 'diagnostic'; the "
             "nearest target_per_class refs with dist_m >= it are tagged "
             "'selected' for production scoring. Default 4000.",
    )
    parser.add_argument(
        "--diag-per-class", type=int, default=DIAG_PER_CLASS,
        help="Up to this many additional refs per class are kept tagged "
             "'diagnostic'. Default 1000.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initialize()
    buffer_steps_m = [int(x.strip()) for x in args.buffer_steps.split(",") if x.strip()]

    samples = (
        ee.FeatureCollection(SAMPLES_ASSET)
        .filter(ee.Filter.neq("r", "stable_stable"))
        .map(lambda ft: ft.setGeometry(ft.geometry().centroid(5)))
    )
    if args.limit is not None:
        samples = samples.limit(args.limit)

    exclusion, wc = build_exclusion_mask(include_buildings_loss=True)
    cb_built, cb_crop = build_classbands(exclusion, wc)
    ecoregions = get_ecoregions()
    ghm = get_ghm_image()

    built_loss = samples.filter(ee.Filter.eq("r", "built_loss"))
    crop_loss = samples.filter(ee.Filter.eq("r", "crop_loss"))

    refs_built = ee.FeatureCollection(
        built_loss.map(make_sampler(
            cb_built, ecoregions, ghm,
            buffer_steps_m=buffer_steps_m,
            target_per_class=args.target_per_class,
            min_per_class=args.min_per_class,
            total_points=args.total_points,
            oversample_factor=args.oversample_factor,
            inner_exclusion_m=args.inner_exclusion_m,
            diag_per_class=args.diag_per_class,
        ))
    ).flatten()
    refs_crop = ee.FeatureCollection(
        crop_loss.map(make_sampler(
            cb_crop, ecoregions, ghm,
            buffer_steps_m=buffer_steps_m,
            target_per_class=args.target_per_class,
            min_per_class=args.min_per_class,
            total_points=args.total_points,
            oversample_factor=args.oversample_factor,
            inner_exclusion_m=args.inner_exclusion_m,
            diag_per_class=args.diag_per_class,
        ))
    ).flatten()

    reference_samples = refs_built.merge(refs_crop)

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
