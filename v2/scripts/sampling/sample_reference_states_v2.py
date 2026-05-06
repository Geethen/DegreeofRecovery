"""v2 reference sampler with loss-region exclusion and dynamic local buffering.

Key updates versus v1:
  1. Excludes pixels flagged as crop/built/building loss from candidate refs.
  2. Uses dynamic buffering by sampling in the max radius then selecting
     nearest candidates first, equivalent to radius expansion from 1 km.
  3. Supports balanced or area-proportional class allocation.

Output schema extends v1 with:
  - dist_m: point-to-parent distance (meters)
  - buffer_m_used: smallest configured radius covering selected points
  - target_good, target_bad: requested class counts for that parent
"""

import ee

PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"

SCALE = 10
SEED = 234

TARGET_PER_CLASS = 100
MIN_PER_CLASS = 30
TOTAL_POINTS = 200
OVERSAMPLE_FACTOR = 5

BUFFER_STEPS_M = [1000, 1500, 2000, 3000, 5000, 8000]

CROP_LOSS_THR = -7
BUILT_LOSS_THR = -7
BUILDING_LOSS_THR = -7

EXPORT_DESCRIPTION = "recover_reference_samples_v2"
EXPORT_ASSET_ID = "projects/ee-gsingh/assets/recover_reference_samples_v2"


def initialize():
    try:
        ee.Initialize(project=PROJECT)
    except Exception as e:
        print(e)
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=PROJECT)


def build_exclusion_mask(include_buildings_loss=True):
    """Build exclusion mask for loss regions that should not be sampled."""
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

    exclusion = crop_loss.Or(built_loss).Or(buildings_loss).unmask(0).toByte()
    return exclusion, wc


def _first_radius_ge(dist_m, radii):
    """Return smallest radius in `radii` that is >= dist_m."""
    d = ee.Number(dist_m)
    r = ee.List(radii)
    out = ee.Number(
        r.iterate(
            lambda cur, acc: ee.Algorithms.If(
                ee.Number(acc).lt(0).And(ee.Number(cur).gte(d)),
                cur,
                acc,
            ),
            -1,
        )
    )
    return ee.Number(ee.Algorithms.If(out.lt(0), r.get(-1), out))


def build_classbands(exclusion_mask, wc):
    """Build label-specific classband images after masking out exclusions.

    ref class values:
      1 => good
      2 => bad
    """
    natural = wc.neq(40).And(wc.neq(50)).And(wc.neq(80))

    cb_built = natural.add(wc.eq(50).multiply(2)).selfMask().rename("ref").toByte()
    cb_crop = natural.add(wc.eq(40).multiply(2)).selfMask().rename("ref").toByte()

    # Remove recovering/loss areas from both good and bad candidates.
    cb_built = cb_built.updateMask(exclusion_mask.Not())
    cb_crop = cb_crop.updateMask(exclusion_mask.Not())
    return cb_built, cb_crop


def _class_targets(
    classband, region_geom, mode, target_per_class, min_per_class, total_points
):
    """Compute per-class targets for one parent.

    mode:
      - balanced: target_good = target_bad = TARGET_PER_CLASS
      - area_proportional: split TOTAL_POINTS by eligible class area with class floor
    """
    if mode == "balanced":
        return ee.Dictionary({"good": target_per_class, "bad": target_per_class})

    area = ee.Image.pixelArea().addBands(classband)
    stats = area.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="ref"),
        geometry=region_geom,
        scale=SCALE,
        maxPixels=1e10,
    )

    groups = ee.List(ee.Dictionary(stats).get("groups", ee.List([])))

    def area_for_class(k):
        k_num = ee.Number(k)
        match = groups.filter(ee.Filter.eq("ref", k_num))
        return ee.Number(
            ee.Algorithms.If(
                match.size().gt(0), ee.Dictionary(match.get(0)).get("sum"), 0
            )
        )

    a_good = area_for_class(1)
    a_bad = area_for_class(2)
    total = a_good.add(a_bad)

    good_target = ee.Number(
        ee.Algorithms.If(
            total.gt(0),
            ee.Number(total_points).multiply(a_good.divide(total)).round(),
            target_per_class,
        )
    )

    good_target = good_target.max(min_per_class).min(total_points - min_per_class)
    bad_target = ee.Number(total_points).subtract(good_target)

    return ee.Dictionary({"good": good_target.int(), "bad": bad_target.int()})


def make_sampler(
    classband,
    allocation_mode="balanced",
    buffer_steps_m=None,
    target_per_class=TARGET_PER_CLASS,
    min_per_class=MIN_PER_CLASS,
    total_points=TOTAL_POINTS,
    oversample_factor=OVERSAMPLE_FACTOR,
):
    radii = ee.List(buffer_steps_m if buffer_steps_m is not None else BUFFER_STEPS_M)
    max_r = ee.Number(radii.get(-1))
    ref_state_lookup = ee.Dictionary({"1": "good", "2": "bad"})

    def sample_one(ft):
        ft = ee.Feature(ft)
        parent_id = ft.id()
        parent_label = ft.get("r")
        parent_geom = ft.geometry()
        region = parent_geom.buffer(max_r)

        targets = _class_targets(
            classband,
            region,
            allocation_mode,
            target_per_class=target_per_class,
            min_per_class=min_per_class,
            total_points=total_points,
        )
        t_good = ee.Number(targets.get("good"))
        t_bad = ee.Number(targets.get("bad"))

        class_points = [
            t_good.multiply(oversample_factor).int(),
            t_bad.multiply(oversample_factor).int(),
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

        candidates = candidates.map(
            lambda f: f.set(
                {
                    "ref_state": ref_state_lookup.get(
                        ee.Number(f.get("ref")).format("%d")
                    ),
                    "dist_m": f.geometry().distance(parent_geom, 1),
                    "parent_id": parent_id,
                    "parent_label": parent_label,
                    "target_good": t_good,
                    "target_bad": t_bad,
                }
            )
        )

        good = candidates.filter(ee.Filter.eq("ref", 1)).sort("dist_m").limit(t_good)
        bad = candidates.filter(ee.Filter.eq("ref", 2)).sort("dist_m").limit(t_bad)
        selected = good.merge(bad)

        max_dist = ee.Number(
            ee.Algorithms.If(
                selected.size().gt(0), selected.aggregate_max("dist_m"), max_r
            )
        )
        used_r = _first_radius_ge(max_dist, radii)

        selected = selected.map(
            lambda f: ee.Feature(f).set(
                {
                    "buffer_m_used": used_r,
                    "target_met": ee.Number(good.size())
                    .gte(t_good)
                    .And(ee.Number(bad.size()).gte(t_bad)),
                    "min_met": ee.Number(good.size())
                    .gte(min_per_class)
                    .And(ee.Number(bad.size()).gte(min_per_class)),
                }
            )
        )
        return selected

    return sample_one


def main(
    export=False,
    limit=None,
    asset_id=EXPORT_ASSET_ID,
    description=EXPORT_DESCRIPTION,
    allocation_mode="balanced",
    use_loss_mask=True,
    include_buildings_loss=True,
    buffer_steps_m=None,
    target_per_class=TARGET_PER_CLASS,
    min_per_class=MIN_PER_CLASS,
    total_points=TOTAL_POINTS,
    oversample_factor=OVERSAMPLE_FACTOR,
    verbose=False,
):
    initialize()

    samples = (
        ee.FeatureCollection(SAMPLES_ASSET)
        .filter(ee.Filter.neq("r", "stable_stable"))
        .map(lambda ft: ft.setGeometry(ft.geometry().centroid(5)))
    )

    if limit is not None:
        samples = samples.limit(limit)

    wc = ee.ImageCollection("ESA/WorldCover/v200").mosaic()
    if use_loss_mask:
        exclusion, _ = build_exclusion_mask(
            include_buildings_loss=include_buildings_loss
        )
    else:
        exclusion = ee.Image(0).toByte()

    cb_built, cb_crop = build_classbands(exclusion, wc)

    built_loss = samples.filter(ee.Filter.eq("r", "built_loss"))
    crop_loss = samples.filter(ee.Filter.eq("r", "crop_loss"))

    refs_built = ee.FeatureCollection(
        built_loss.map(
            make_sampler(
                cb_built,
                allocation_mode=allocation_mode,
                buffer_steps_m=buffer_steps_m,
                target_per_class=target_per_class,
                min_per_class=min_per_class,
                total_points=total_points,
                oversample_factor=oversample_factor,
            )
        )
    ).flatten()
    refs_crop = ee.FeatureCollection(
        crop_loss.map(
            make_sampler(
                cb_crop,
                allocation_mode=allocation_mode,
                buffer_steps_m=buffer_steps_m,
                target_per_class=target_per_class,
                min_per_class=min_per_class,
                total_points=total_points,
                oversample_factor=oversample_factor,
            )
        )
    ).flatten()

    reference_samples = refs_built.merge(refs_crop)

    if verbose:
        print("Sampling mode:", allocation_mode)
        print(
            "Use loss mask:",
            use_loss_mask,
            "(include buildings:",
            include_buildings_loss,
            ")",
        )
        print(
            "Buffer steps (m):",
            buffer_steps_m if buffer_steps_m is not None else BUFFER_STEPS_M,
        )
        print(
            "Target/class:",
            target_per_class,
            "Min/class:",
            min_per_class,
            "Total points:",
            total_points,
            "Oversample:",
            oversample_factor,
        )

    if export:
        task = ee.batch.Export.table.toAsset(
            collection=reference_samples,
            description=description,
            assetId=asset_id,
        )
        task.start()
        print(f"Started export task '{description}' -> {asset_id}")
        return reference_samples, task

    return reference_samples, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--asset-id", default=EXPORT_ASSET_ID)
    parser.add_argument("--description", default=EXPORT_DESCRIPTION)
    parser.add_argument(
        "--allocation-mode",
        choices=["balanced", "area_proportional"],
        default="balanced",
    )
    parser.add_argument(
        "--exclude-buildings-loss",
        action="store_true",
        help="When using the loss mask, drop building-trend loss pixels too.",
    )
    parser.add_argument(
        "--disable-loss-mask",
        action="store_true",
        help="Do not exclude crop/built/building loss regions.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=TARGET_PER_CLASS,
        help="Target count per class when allocation-mode=balanced.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=MIN_PER_CLASS,
        help="Minimum class count target used by both allocation modes.",
    )
    parser.add_argument(
        "--total-points",
        type=int,
        default=TOTAL_POINTS,
        help="Total points target when allocation-mode=area_proportional.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=OVERSAMPLE_FACTOR,
        help="Candidate oversampling multiplier before nearest selection.",
    )
    parser.add_argument(
        "--buffer-steps",
        default=",".join(str(v) for v in BUFFER_STEPS_M),
        help="Comma-separated buffer radii in meters, e.g. 1000,1500,2000,3000,5000",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    buffer_steps_m = [int(x.strip()) for x in args.buffer_steps.split(",") if x.strip()]

    main(
        export=args.export,
        limit=args.limit,
        asset_id=args.asset_id,
        description=args.description,
        allocation_mode=args.allocation_mode,
        use_loss_mask=not args.disable_loss_mask,
        include_buildings_loss=args.exclude_buildings_loss,
        buffer_steps_m=buffer_steps_m,
        target_per_class=args.target_per_class,
        min_per_class=args.min_per_class,
        total_points=args.total_points,
        oversample_factor=args.oversample_factor,
        verbose=args.verbose,
    )
