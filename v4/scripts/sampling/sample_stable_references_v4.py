"""v4 sampler for stable_stable parents — routes bad pool by stable_class.

Inputs:
  - SAMPLES_ASSET   : EE FeatureCollection of all RECOVER parents (filtered to
                      stable_stable here).
  - --classification: CSV produced by `classify_stable_state.py` mapping
                      parent_id -> stable_class ∈ {nature, crop, built, ambiguous}.

Per-class bad-pool routing (each class's bad = its own non-natural state):
  - stable_class = nature -> bad = WC{40,50}  (crop ∪ built)
  - stable_class = crop   -> bad = WC{40}     (crop) — sanity: should score 'degraded'
  - stable_class = built  -> bad = WC{50}     (built) — sanity: should score 'degraded'
  - stable_class = ambiguous -> SKIPPED upstream (not in classification CSV's
                                non-ambiguous subset)

Good pool (same as v2): WC ≠ {40, 50, 80} AND not in the loss-trend mask.

Output schema mirrors v2's exhaustive parquet so v3's downstream scripts
(validate, score) can read it with a `--strategy random_100` filter:
  parent_id, parent_label (= 'stable_stable_<class>'), ref_state {good,bad},
  strategy, A00..A63, geo, lon, lat, dist_m, buffer_m_used, ...

Note: v3's `parent_label` is the *disturbance* label. For v4 we encode the
stable_class into parent_label as `stable_<class>` (e.g. `stable_nature`) so
downstream calibration can group by parent_label and the per-class threshold
refit in `validate_steps_within_parent_v4.py` works without schema changes.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

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

EXPORT_DESCRIPTION = "recover_reference_samples_v4_stable"
EXPORT_ASSET_ID = "projects/ee-gsingh/assets/recover_reference_samples_v4_stable"


def initialize() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=PROJECT)


def build_exclusion_mask(include_buildings_loss: bool = True) -> ee.Image:
    """Same loss exclusion as v2 — used to mask GOOD candidates only."""
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


# Class-band builders. Pixel values: 1=good, 2=bad. Other pixels masked.

def _classband_for(stable_class: str, exclusion: ee.Image) -> ee.Image:
    wc = ee.ImageCollection("ESA/WorldCover/v200").mosaic()
    natural = wc.neq(40).And(wc.neq(50)).And(wc.neq(80))

    if stable_class == "nature":
        # bad = crop ∪ built
        bad = wc.eq(40).Or(wc.eq(50))
    elif stable_class == "crop":
        # bad = crop (sanity check; site sits in crop → expected 'degraded')
        bad = wc.eq(40)
    elif stable_class == "built":
        # bad = built (sanity check; site sits in built → expected 'degraded')
        bad = wc.eq(50)
    else:
        raise ValueError(f"Unsupported stable_class: {stable_class!r}")

    cb = (
        natural.multiply(1)            # good=1 where natural
        .add(bad.multiply(2))          # +2 on bad pixels (natural and bad are disjoint by construction)
        .selfMask()                    # drop 0-valued pixels
        .rename("ref")
        .toByte()
    )
    # Mask GOOD candidates with the loss-trend exclusion (same as v2).
    # Bad candidates (already in WC=40 or 50) are NOT excluded here.
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


def make_sampler(
    classband: ee.Image,
    parent_label: str,
    target_per_class: int = TARGET_PER_CLASS,
    min_per_class: int = MIN_PER_CLASS,
    total_points: int = TOTAL_POINTS,
    oversample_factor: int = OVERSAMPLE_FACTOR,
    buffer_steps_m: list[int] | None = None,
):
    radii = ee.List(buffer_steps_m if buffer_steps_m is not None else BUFFER_STEPS_M)
    max_r = ee.Number(radii.get(-1))
    ref_state_lookup = ee.Dictionary({"1": "good", "2": "bad"})

    def sample_one(ft: ee.Feature) -> ee.FeatureCollection:
        ft = ee.Feature(ft)
        parent_id = ft.id()
        parent_geom = ft.geometry()
        region = parent_geom.buffer(max_r)

        t_good = ee.Number(target_per_class)
        t_bad = ee.Number(target_per_class)
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
            lambda f: f.set({
                "ref_state": ref_state_lookup.get(
                    ee.Number(f.get("ref")).format("%d")),
                "dist_m": f.geometry().distance(parent_geom, 1),
                "parent_id": parent_id,
                "parent_label": parent_label,
                "target_good": t_good,
                "target_bad": t_bad,
            })
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

        return selected.map(
            lambda f: ee.Feature(f).set({
                "buffer_m_used": used_r,
                "target_met": ee.Number(good.size()).gte(t_good)
                .And(ee.Number(bad.size()).gte(t_bad)),
                "min_met": ee.Number(good.size()).gte(min_per_class)
                .And(ee.Number(bad.size()).gte(min_per_class)),
                "stable_class": parent_label.replace("stable_", ""),
                "strategy": f"random_{int(target_per_class)}",
            })
        )

    return sample_one


def load_classification(path: Path) -> dict[str, str]:
    """Read the classifier CSV; return {parent_id: stable_class} excluding 'ambiguous'."""
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initialize()
    buffer_steps_m = [int(x.strip()) for x in args.buffer_steps.split(",") if x.strip()]

    classification = load_classification(Path(args.classification))
    if args.verbose:
        from collections import Counter
        c = Counter(classification.values())
        print(f"Classifications loaded: {dict(c)} (total {len(classification)})")

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

    # Group parents by stable_class and run a class-specific sampler on each group.
    fcs: list[ee.FeatureCollection] = []
    for stable_class in ("nature", "crop", "built"):
        ids_for_class = [pid for pid, c in classification.items() if c == stable_class]
        if not ids_for_class:
            if args.verbose:
                print(f"  {stable_class}: 0 parents (skipping)")
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
                target_per_class=args.target_per_class,
                min_per_class=args.min_per_class,
                total_points=args.total_points,
                oversample_factor=args.oversample_factor,
                buffer_steps_m=buffer_steps_m,
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

    print("Reference sample size (collecting):", reference_samples.size().getInfo())


if __name__ == "__main__":
    main()
