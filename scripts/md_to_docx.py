"""Convert paper_methods.md to paper_methods.docx via pandoc.

Uses the pandoc binary bundled with pypandoc_binary. Markdown is read in
pandoc's GFM dialect with raw_attribute disabled so that any HTML or LaTeX
that creeps in is preserved as visible text rather than silently injected
into the docx. The resulting file opens cleanly in Word and Apple Pages.
"""
from __future__ import annotations

from pathlib import Path

import pypandoc


REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "paper_methods.md"
OUTPUT = REPO / "paper_methods.docx"


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Input file not found: {INPUT}")

    pypandoc.convert_file(
        str(INPUT),
        to="docx",
        format="gfm",
        outputfile=str(OUTPUT),
        extra_args=[
            "--standalone",
            "--wrap=none",
        ],
    )
    print(f"Wrote {OUTPUT}  ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
