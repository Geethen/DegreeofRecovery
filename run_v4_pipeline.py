"""End-to-end v4 pipeline: extraction -> classification -> scoring -> reporting.

This single script runs every stage of the v4 Degree-of-Recovery workflow in
the correct order. Read top-to-bottom to learn the v4 process.

Pipeline stages (each calls a stage-specific script in v4/scripts/):

  Stage 0 (manual, one-time):
    Sample stable_stable parents in Earth Engine (out of scope here) and
    publish a FeatureCollection asset. v4 expects:
        projects/ee-gsingh/assets/recover_reference_samples_v4_stable
    See `v4/scripts/sampling/sample_stable_references_v4.py --export ...`
    for the EE-side sampling step.

  Stage 1: Classify each parent's stable_class (nature / crop / built / ambiguous)
    Script:  v4/scripts/classification/classify_stable_state.py
    Inputs:  parents asset (EE), WorldCover/DynamicWorld/buildings rasters (EE)
    Output:  v4/data/parent_stable_class.csv

  Stage 2: Sample reference points per stable_class with class-specific bad pools
    Script:  v4/scripts/sampling/sample_stable_references_v4.py
    Inputs:  parents asset + parent_stable_class.csv
    Output:  EE FeatureCollection (--export) at recover_reference_samples_v4_stable
    NOTE: the EE export is async; wait for it to finish before Stage 3.

  Stage 3: Extract AlphaEarth (64 bands) at the reference points
    Script:  v4/scripts/extraction/extract_stable_refs_alphaearth.py
    Inputs:  EE asset from Stage 2
    Output:  v4/data/v4_stable_refs_alphaearth.parquet

  Stage 4: Extract AlphaEarth at the parent (test-site) points
    Script:  v1/scripts/extraction/extract_test_site_embeddings.py
    Inputs:  the same parents asset (we sample it at the parent points)
    Output:  v4/data/test_site_alphaearth_2024_v4.parquet

  Stage 5: Calibrate per-class thresholds via within-parent K-fold validation
    Script:  v4/scripts/analysis/validate_steps_within_parent_v4.py
    Inputs:  v4/data/v4_stable_refs_alphaearth.parquet
    Outputs: v4/data/within_parent_site_scores_v4.csv
             v4/data/within_parent_summary_v4.csv
             v4/data/calibrated_thresholds_v4.json   <- consumed by Stage 6

  Stage 6: Score test-sites against per-class thresholds
    Script:  v4/scripts/analysis/score_test_sites_v4.py
    Inputs:  refs parquet + test-sites parquet + calibrated_thresholds_v4.json
    Output:  v4/data/test_site_dor_v4.csv

  Stage 7: Reporting (charts + shapefile export)
    Scripts: v4/scripts/reporting/make_v4_charts.py
             v4/scripts/reporting/export_v4_shp.py
    Inputs:  v4/data/test_site_dor_v4.csv (+ within-parent summaries)
    Outputs: v4/plots/*.png, v4/data/test_site_dor_v4.shp

Usage:
    # Run every stage end-to-end (assumes Stages 0+2 EE assets already exist)
    python run_v4_pipeline.py

    # Skip the GEE-hitting stages if their parquets are already cached
    python run_v4_pipeline.py --skip 1,2,3,4

    # Run only scoring + reporting (after refs/test-sites parquets cached)
    python run_v4_pipeline.py --only 5,6,7

    # Print the full plan (commands to be run) without executing
    python run_v4_pipeline.py --dry-run

Constraints:
  * Stages 1-4 hit Earth Engine; you must `earthengine authenticate` first.
  * Stage 2 publishes an EE asset asynchronously. The script returns
    immediately; wait for the export task to finish before Stage 3.
  * Stages 5-7 are local-only and reproducible (seed=42).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

V4_DATA = ROOT / "v4" / "data"
V1_DATA = ROOT / "v1" / "data"

V4_TEST_SITES_PARQUET = V4_DATA / "test_site_alphaearth_2024_v4.parquet"
V4_REFS_PARQUET = V4_DATA / "v4_stable_refs_alphaearth.parquet"
V4_THRESHOLDS_JSON = V4_DATA / "calibrated_thresholds_v4.json"
V4_PARENT_CLASS_CSV = V4_DATA / "parent_stable_class.csv"
V4_DOR_CSV = V4_DATA / "test_site_dor_v4.csv"


@dataclass
class Stage:
    n: int
    name: str
    script: Path
    args: list[str]
    needs_ee: bool
    output_hint: str


def stages() -> list[Stage]:
    return [
        Stage(
            n=1,
            name="Classify stable_class per parent",
            script=ROOT / "v4" / "scripts" / "classification" / "classify_stable_state.py",
            args=[],
            needs_ee=True,
            output_hint=str(V4_PARENT_CLASS_CSV),
        ),
        Stage(
            n=2,
            name="Sample stable references (EE export)",
            script=ROOT / "v4" / "scripts" / "sampling" / "sample_stable_references_v4.py",
            args=["--export"],
            needs_ee=True,
            output_hint="EE asset: recover_reference_samples_v4_stable (async)",
        ),
        Stage(
            n=3,
            name="Extract AlphaEarth at reference points",
            script=ROOT / "v4" / "scripts" / "extraction" / "extract_stable_refs_alphaearth.py",
            args=[],
            needs_ee=True,
            output_hint=str(V4_REFS_PARQUET),
        ),
        Stage(
            n=4,
            name="Extract AlphaEarth at test-site (parent) points",
            script=ROOT / "v1" / "scripts" / "extraction" / "extract_test_site_embeddings.py",
            args=["--year", "2024", "--output", str(V4_TEST_SITES_PARQUET)],
            needs_ee=True,
            output_hint=str(V4_TEST_SITES_PARQUET),
        ),
        Stage(
            n=5,
            name="Calibrate per-class thresholds (within-parent K-fold)",
            script=ROOT / "v4" / "scripts" / "analysis" / "validate_steps_within_parent_v4.py",
            args=["--out-dir", str(V4_DATA)],
            needs_ee=False,
            output_hint=f"{V4_THRESHOLDS_JSON} + within_parent_*_v4.csv",
        ),
        Stage(
            n=6,
            name="Score test-sites with per-class thresholds",
            script=ROOT / "v4" / "scripts" / "analysis" / "score_test_sites_v4.py",
            # IMPORTANT: --test-sites override needed because the v4 script's
            # default still points at the v1 parquet (pre-existing bug).
            args=[
                "--test-sites", str(V4_TEST_SITES_PARQUET),
                "--out-dir", str(V4_DATA),
            ],
            needs_ee=False,
            output_hint=str(V4_DOR_CSV),
        ),
        Stage(
            n=7,
            name="Reporting: charts and shapefile",
            script=ROOT / "v4" / "scripts" / "reporting" / "make_v4_charts.py",
            args=[],
            needs_ee=False,
            output_hint="v4/plots/*.png",
        ),
    ]


def parse_stage_set(spec: str | None, all_ns: list[int]) -> set[int]:
    if spec is None:
        return set()
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    unknown = out - set(all_ns)
    if unknown:
        raise SystemExit(f"unknown stage numbers: {sorted(unknown)}")
    return out


def fmt_command(stage: Stage) -> str:
    return " ".join([PYTHON, str(stage.script), *stage.args])


def run_stage(stage: Stage) -> int:
    print(f"\n{'=' * 78}")
    print(f"  Stage {stage.n}: {stage.name}")
    print(f"  -> {stage.output_hint}")
    if stage.needs_ee:
        print("  (requires Earth Engine; run `earthengine authenticate` first)")
    print(f"  $ {fmt_command(stage)}")
    print("=" * 78)
    env = os.environ.copy()
    # Force UTF-8 on Windows so docstrings with arrows don't crash --help.
    env["PYTHONIOENCODING"] = env.get("PYTHONIOENCODING", "utf-8")
    return subprocess.call([PYTHON, str(stage.script), *stage.args], env=env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only", help="Comma/range list of stages to run (e.g. '5,6,7' or '5-7')"
    )
    parser.add_argument(
        "--skip", help="Comma/range list of stages to skip (e.g. '1,2,3,4')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining stages even if one fails (default: stop on first failure).",
    )
    args = parser.parse_args()

    all_stages = stages()
    all_ns = [s.n for s in all_stages]

    only = parse_stage_set(args.only, all_ns)
    skip = parse_stage_set(args.skip, all_ns)
    if only and skip:
        raise SystemExit("--only and --skip are mutually exclusive")

    selected = [
        s for s in all_stages
        if (not only or s.n in only) and s.n not in skip
    ]

    print(f"v4 pipeline: {len(selected)} stage(s) selected\n")
    for s in selected:
        ee_tag = " [EE]" if s.needs_ee else ""
        print(f"  Stage {s.n}{ee_tag}: {s.name}")
        print(f"           $ {fmt_command(s)}")

    if args.dry_run:
        print("\n(dry-run; no stages executed)")
        return 0

    failures: list[int] = []
    for s in selected:
        rc = run_stage(s)
        if rc != 0:
            failures.append(s.n)
            print(f"\n[FAIL] Stage {s.n} exited with rc={rc}")
            if not args.continue_on_error:
                return rc

    if failures:
        print(f"\nDone with failures in stages: {failures}")
        return 1
    print("\nAll selected stages completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
