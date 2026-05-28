"""Build paper_methods.docx from paper_methods.md via pandoc.

Source of truth is the markdown. The docx is a regenerated export, never
hand-edited. Figures are regenerated from their scripts before the conversion
so the embedded PNGs match the current code.

Usage:
    python scripts/build_docx.py
    python scripts/build_docx.py --skip-figures        # md changes only
    python scripts/build_docx.py --open                # open the result in Word

Output: build/paper_methods.docx
Styling: scripts/reference.docx (a Word file whose styles pandoc copies into
the output; edit it in Word and Save when you want to change fonts/margins/
heading styles).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pypandoc

REPO = Path(__file__).resolve().parents[1]
MD = REPO / "paper_methods.md"
OUT_DIR = REPO / "build"
OUT_DOCX = OUT_DIR / "paper_methods.docx"
REFERENCE_DOC = REPO / "scripts" / "reference.docx"

FIGURE_SCRIPTS = [
    REPO / "scripts" / "make_dor_results_figure.py",
    REPO / "v4" / "scripts" / "reporting" / "make_supp_quality_figure.py",
]

# Pin pandoc to the binary bundled inside pypandoc so behaviour is
# reproducible across machines without a system pandoc install.
_PYPANDOC_BIN = (Path(pypandoc.__file__).parent / "files" / "pandoc.exe")
if _PYPANDOC_BIN.exists():
    os.environ.setdefault("PYPANDOC_PANDOC", str(_PYPANDOC_BIN))


def _check_target_writable(path: Path) -> None:
    """Fail fast if the docx target is locked (Word has it open)."""
    if not path.exists():
        return
    try:
        with open(path, "ab"):
            pass
    except PermissionError as exc:
        sys.exit(
            f"error: {path} is locked (likely open in Word). "
            f"Close the document and re-run.\n  {exc}"
        )


def _run_figure_script(script: Path) -> None:
    print(f"[fig] {script.relative_to(REPO)}")
    subprocess.run([sys.executable, str(script)], check=True, cwd=REPO)


def build(skip_figures: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _check_target_writable(OUT_DOCX)

    if not skip_figures:
        for script in FIGURE_SCRIPTS:
            _run_figure_script(script)

    extra_args = [
        "--standalone",
        f"--resource-path={REPO}",
    ]
    if REFERENCE_DOC.exists():
        extra_args.append(f"--reference-doc={REFERENCE_DOC}")
    else:
        print(f"warning: {REFERENCE_DOC.relative_to(REPO)} missing; "
              "using pandoc defaults")

    print(f"[pandoc] {MD.relative_to(REPO)} -> {OUT_DOCX.relative_to(REPO)}")
    pypandoc.convert_file(
        str(MD),
        to="docx",
        format="markdown+tex_math_single_backslash",
        outputfile=str(OUT_DOCX),
        extra_args=extra_args,
    )
    print(f"[ok] wrote {OUT_DOCX}  ({OUT_DOCX.stat().st_size/1024:.0f} KB)")
    return OUT_DOCX


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-figures", action="store_true",
                    help="don't regenerate the PNGs first")
    ap.add_argument("--open", action="store_true",
                    help="open the built docx after a successful build")
    args = ap.parse_args()

    out = build(skip_figures=args.skip_figures)
    if args.open:
        os.startfile(out)  # Windows-only


if __name__ == "__main__":
    main()
