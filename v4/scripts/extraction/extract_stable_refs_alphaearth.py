"""Extract AlphaEarth embeddings for v4 stable reference points.

v4-specific wrapper around the shared GEE extraction helpers in
`degree_of_recovery.gee`. Sets v4 defaults (asset id, output path) and
calls into the library. Run after the EE export of
`recover_reference_samples_v4_stable` has completed.

For the canonical end-to-end extraction reference (single-file, no
library imports), see v1/scripts/extraction/extract_alphaearth_embeddings.py.

Usage:
  python v4/scripts/extraction/extract_stable_refs_alphaearth.py
  python v4/scripts/extraction/extract_stable_refs_alphaearth.py --test_mode
"""

import argparse
import os

from degree_of_recovery.gee import init_gee, run

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

V4_SAMPLES_ASSET = "projects/ee-gsingh/assets/recover_reference_samples_v4_stable"
V4_OUTPUT = os.path.join(ROOT, "v4", "data", "v4_stable_refs_alphaearth.parquet")


if __name__ == "__main__":
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
