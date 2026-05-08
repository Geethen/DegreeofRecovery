"""Extract AlphaEarth embeddings for v4 stable reference points.

Thin wrapper around the v1 extractor that sets v4-specific defaults
(asset ID, output path). Run after the EE export of
`recover_reference_samples_v4_stable` has completed.

Usage:
  python v4/scripts/extraction/extract_stable_refs_alphaearth.py
  python v4/scripts/extraction/extract_stable_refs_alphaearth.py --test_mode
"""

import os
import sys

# Add project root to path so we can import the v1 extractor.
ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, ROOT)

from degreeRecover.scripts.extraction.extract_alphaearth_embeddings import (  # noqa: E402
    init_gee, run,
)

V4_SAMPLES_ASSET = "projects/ee-gsingh/assets/recover_reference_samples_v4_stable"
V4_OUTPUT = os.path.join(ROOT, "v4", "data", "v4_stable_refs_alphaearth.parquet")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--n_shards", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=40)
    parser.add_argument("--output", default=V4_OUTPUT)
    parser.add_argument("--samples_asset", default=V4_SAMPLES_ASSET)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    init_gee()
    run(
        year=args.year,
        n_shards=args.n_shards,
        max_workers=args.max_workers,
        output_path=args.output,
        samples_asset=args.samples_asset,
        test_mode=args.test_mode,
        verbose=args.verbose,
    )
