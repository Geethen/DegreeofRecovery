"""Sample good/bad reference states around each RECOVER sample point.

For each sample in `projects/nina/RECOVER/samples_recover_w_ref_label`, build a
buffer and stratified-sample reference points:
  - "good" reference state: any terrestrial surface other than cropland (40),
    built (50), or water (80) in WorldCover 2020.
  - "bad" reference state: depends on the sample's `r` label
        built_loss     -> built (50)
        crop_loss      -> cropland (40)

`stable_stable` parents have no degraded counterpart and are filtered out
upfront — downstream DoR scoring marks them not-applicable anyway.
"""

import ee

PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"
BUFFER_M = 1000
N_POINTS = 100
SCALE = 10
SEED = 234

EXPORT_DESCRIPTION = "recover_reference_samples"
EXPORT_ASSET_ID = "projects/ee-gsingh/assets/recover_reference_samples"

def initialize():
    try:
        ee.Initialize(project=PROJECT)
    except Exception as e:
        print(e)
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=PROJECT)


def build_classband_images():
    """Two-class classband images for one-shot stratified sampling.

    Pixel values:
      1 = good  (natural: not cropland / built / water)
      2 = bad   (built for the built-loss image, cropland for the crop-loss
                 image)
    Other pixels are masked.
    """
    wc = ee.ImageCollection("ESA/WorldCover/v200").mosaic()
    natural = wc.neq(40).And(wc.neq(50)).And(wc.neq(80))

    cb_built = (
        natural.add(wc.eq(50).multiply(2)).selfMask().rename("ref").toByte()
    )
    cb_crop = (
        natural.add(wc.eq(40).multiply(2)).selfMask().rename("ref").toByte()
    )
    return cb_built, cb_crop


def make_sampler(classband):
    """Return a function mapping one parent feature to its sampled refs.

    A single `stratifiedSample` returns N_POINTS for each of the two classes;
    `ref_state` is decoded from the class band value via a lookup dict.
    """
    ref_state_lookup = ee.Dictionary({"1": "good", "2": "bad"})

    def sample_one(ft):
        ft = ee.Feature(ft)
        parent_id = ft.id()
        parent_label = ft.get("r")
        buff = ft.geometry().buffer(BUFFER_M)

        sampled = classband.stratifiedSample(
            numPoints=N_POINTS,
            classBand="ref",
            classValues=[1, 2],
            classPoints=[N_POINTS, N_POINTS],
            region=buff,
            scale=SCALE,
            seed=SEED,
            geometries=True,
        )

        return sampled.map(
            lambda f: f.set({
                "ref_state": ref_state_lookup.get(
                    ee.Number(f.get("ref")).format("%d")),
                "parent_id": parent_id,
                "parent_label": parent_label,
            })
        )

    return sample_one


def main(export=False, limit=None, asset_id=EXPORT_ASSET_ID,
         description=EXPORT_DESCRIPTION, verbose=False):
    initialize()

    # Filter stable_stable upfront and centroid once.
    samples = (
        ee.FeatureCollection(SAMPLES_ASSET)
        .filter(ee.Filter.neq("r", "stable_stable"))
        .map(lambda ft: ft.setGeometry(ft.geometry().centroid(5)))
    )

    if verbose:
        print(f"Number of samples (built_loss + crop_loss): "
              f"{samples.size().getInfo()}")

    if limit is not None:
        samples = samples.limit(limit)
        if verbose:
            print(f"Limiting to first {limit} sample(s)")

    cb_built, cb_crop = build_classband_images()

    built_loss = samples.filter(ee.Filter.eq("r", "built_loss"))
    crop_loss = samples.filter(ee.Filter.eq("r", "crop_loss"))

    refs_built = ee.FeatureCollection(
        built_loss.map(make_sampler(cb_built))).flatten()
    refs_crop = ee.FeatureCollection(
        crop_loss.map(make_sampler(cb_crop))).flatten()

    reference_samples = refs_built.merge(refs_crop)

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
    parser.add_argument(
        "--export",
        action="store_true",
        help="Start an Earth Engine export to the configured asset id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input samples (for testing).",
    )
    parser.add_argument(
        "--asset-id",
        default=EXPORT_ASSET_ID,
        help="Override the destination EE asset id for the export.",
    )
    parser.add_argument(
        "--description",
        default=EXPORT_DESCRIPTION,
        help="Override the EE export task description.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic counts (each one is a blocking EE round-trip).",
    )
    args = parser.parse_args()

    fc, _ = main(
        export=args.export,
        limit=args.limit,
        asset_id=args.asset_id,
        description=args.description,
        verbose=args.verbose,
    )
    if not args.export and args.verbose:
        print(f"Reference sample count: {fc.size().getInfo()}")
