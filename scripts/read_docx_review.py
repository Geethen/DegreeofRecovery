"""Extract comments and tracked changes from a reviewed .docx.

Reads the Word document at `paper_methods.docx` (or the path passed on the
command line), pulls comments from `word/comments.xml` and tracked changes
(`w:ins` / `w:del`) from `word/document.xml`, and writes a clean Markdown
review summary to `paper_methods_review.md`.

The .docx is treated as a zip of XML — no python-docx dependency. Output is
designed to be readable by both the human reviewer and Claude, so revisions
can be applied to the source `paper_methods.md` without round-tripping
through Word.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def q(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


# ---------------------------------------------------------------------------
# Tokenisation of paragraph content
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str               # text | ins_open | ins_close | del_open | del_close
                            # | cmt_open | cmt_close | cmt_ref
    text: str = ""          # for kind == text or kind in (del_*)
    cid: str = ""           # comment / change id
    author: str = ""        # ins / del / cmt author
    date: str = ""          # ins / del date


def walk_paragraph(elem: ET.Element, tokens: list[Token]) -> None:
    """Recursively walk a paragraph's children and emit ordered tokens."""
    for child in elem:
        tag = child.tag
        if tag == q("r"):
            for sub in child:
                if sub.tag == q("t"):
                    tokens.append(Token("text", sub.text or ""))
                elif sub.tag == q("delText"):
                    tokens.append(Token("text", sub.text or ""))
                elif sub.tag == q("tab"):
                    tokens.append(Token("text", "\t"))
                elif sub.tag == q("br"):
                    tokens.append(Token("text", "\n"))
        elif tag == q("ins"):
            tokens.append(Token(
                "ins_open",
                cid=child.get(q("id"), ""),
                author=child.get(q("author"), ""),
                date=child.get(q("date"), ""),
            ))
            walk_paragraph(child, tokens)
            tokens.append(Token("ins_close", cid=child.get(q("id"), "")))
        elif tag == q("del"):
            tokens.append(Token(
                "del_open",
                cid=child.get(q("id"), ""),
                author=child.get(q("author"), ""),
                date=child.get(q("date"), ""),
            ))
            walk_paragraph(child, tokens)
            tokens.append(Token("del_close", cid=child.get(q("id"), "")))
        elif tag == q("commentRangeStart"):
            tokens.append(Token("cmt_open", cid=child.get(q("id"), "")))
        elif tag == q("commentRangeEnd"):
            tokens.append(Token("cmt_close", cid=child.get(q("id"), "")))
        elif tag == q("commentReference"):
            tokens.append(Token("cmt_ref", cid=child.get(q("id"), "")))
        else:
            walk_paragraph(child, tokens)


# ---------------------------------------------------------------------------
# Per-paragraph extraction
# ---------------------------------------------------------------------------

@dataclass
class CommentAnchor:
    cid: str
    paragraph_idx: int
    anchored_text: str
    paragraph_plain: str


@dataclass
class TrackedChange:
    kind: str          # "insertion" or "deletion"
    cid: str
    author: str
    date: str
    text: str
    paragraph_idx: int
    paragraph_plain: str


def process_paragraphs(body: ET.Element) -> tuple[
    list[CommentAnchor], list[TrackedChange]
]:
    anchors: list[CommentAnchor] = []
    changes: list[TrackedChange] = []
    paragraphs = body.findall(f".//{q('p')}")
    for p_idx, p in enumerate(paragraphs):
        tokens: list[Token] = []
        walk_paragraph(p, tokens)

        # Pass 1: paragraph plain text (with strikethrough markers for deletions
        # and a visible marker for insertions) so reviewers can see context.
        plain_parts: list[str] = []
        in_ins = 0
        in_del = 0
        for tok in tokens:
            if tok.kind == "text":
                if in_del:
                    plain_parts.append(f"~~{tok.text}~~")
                elif in_ins:
                    plain_parts.append(f"__{tok.text}__")
                else:
                    plain_parts.append(tok.text)
            elif tok.kind == "ins_open":
                in_ins += 1
            elif tok.kind == "ins_close":
                in_ins = max(0, in_ins - 1)
            elif tok.kind == "del_open":
                in_del += 1
            elif tok.kind == "del_close":
                in_del = max(0, in_del - 1)
        paragraph_plain = "".join(plain_parts).strip()

        # Pass 2: build anchored-text spans for each open comment range and
        # collect tracked changes with their text content.
        open_comments: dict[str, list[str]] = {}
        open_inserts: dict[str, dict] = {}  # cid -> {author,date,buf}
        open_deletes: dict[str, dict] = {}
        for tok in tokens:
            if tok.kind == "cmt_open":
                open_comments[tok.cid] = []
            elif tok.kind == "cmt_close":
                if tok.cid in open_comments:
                    anchored = "".join(open_comments.pop(tok.cid)).strip()
                    anchors.append(CommentAnchor(
                        cid=tok.cid,
                        paragraph_idx=p_idx,
                        anchored_text=anchored,
                        paragraph_plain=paragraph_plain,
                    ))
            elif tok.kind == "ins_open":
                open_inserts[tok.cid] = {
                    "author": tok.author, "date": tok.date, "buf": []
                }
            elif tok.kind == "ins_close":
                meta = open_inserts.pop(tok.cid, None)
                if meta is not None:
                    changes.append(TrackedChange(
                        kind="insertion",
                        cid=tok.cid,
                        author=meta["author"],
                        date=meta["date"],
                        text="".join(meta["buf"]).strip(),
                        paragraph_idx=p_idx,
                        paragraph_plain=paragraph_plain,
                    ))
            elif tok.kind == "del_open":
                open_deletes[tok.cid] = {
                    "author": tok.author, "date": tok.date, "buf": []
                }
            elif tok.kind == "del_close":
                meta = open_deletes.pop(tok.cid, None)
                if meta is not None:
                    changes.append(TrackedChange(
                        kind="deletion",
                        cid=tok.cid,
                        author=meta["author"],
                        date=meta["date"],
                        text="".join(meta["buf"]).strip(),
                        paragraph_idx=p_idx,
                        paragraph_plain=paragraph_plain,
                    ))
            elif tok.kind == "text":
                for buf in open_comments.values():
                    buf.append(tok.text)
                for meta in open_inserts.values():
                    meta["buf"].append(tok.text)
                for meta in open_deletes.values():
                    meta["buf"].append(tok.text)

    return anchors, changes


# ---------------------------------------------------------------------------
# Comments.xml extraction
# ---------------------------------------------------------------------------

@dataclass
class Comment:
    cid: str
    author: str
    date: str
    text: str


def load_comments(zf: zipfile.ZipFile) -> dict[str, Comment]:
    if "word/comments.xml" not in zf.namelist():
        return {}
    tree = ET.fromstring(zf.read("word/comments.xml"))
    out: dict[str, Comment] = {}
    for c in tree.findall(q("comment")):
        cid = c.get(q("id"), "")
        text_parts: list[str] = []
        for t in c.findall(f".//{q('t')}"):
            if t.text:
                text_parts.append(t.text)
        out[cid] = Comment(
            cid=cid,
            author=c.get(q("author"), ""),
            date=c.get(q("date"), ""),
            text="".join(text_parts).strip(),
        )
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(
    docx_path: Path,
    comments: dict[str, Comment],
    anchors: list[CommentAnchor],
    changes: list[TrackedChange],
) -> str:
    lines: list[str] = []
    lines.append(f"# Review of `{docx_path.name}`\n")
    lines.append(
        f"Comments: {len(comments)}  ·  "
        f"Tracked changes: {len(changes)}\n"
    )

    # Comments, ordered by appearance in document.
    lines.append("\n## Comments\n")
    if not anchors:
        lines.append("_None._\n")
    else:
        anchors_by_cid = {a.cid: a for a in anchors}
        ordered_cids = [a.cid for a in anchors]
        for i, cid in enumerate(ordered_cids, 1):
            c = comments.get(cid)
            a = anchors_by_cid[cid]
            author = c.author if c else "?"
            date = c.date if c else "?"
            text = c.text if c else "(comment text missing)"
            lines.append(f"### {i}. {author} — {date}")
            lines.append(f"**On:** {a.anchored_text or '_(no anchored text)_'}")
            lines.append(f"**Says:** {text}")
            lines.append(f"**Paragraph (¶ {a.paragraph_idx + 1}):** {a.paragraph_plain}\n")

    # Tracked changes, ordered by appearance.
    lines.append("\n## Tracked changes\n")
    if not changes:
        lines.append("_None._\n")
    else:
        for i, ch in enumerate(changes, 1):
            verb = "Inserted" if ch.kind == "insertion" else "Deleted"
            lines.append(f"### {i}. {verb} — {ch.author} — {ch.date}")
            lines.append(f"**Text:** {ch.text or '_(empty)_'}")
            lines.append(f"**Paragraph (¶ {ch.paragraph_idx + 1}):** {ch.paragraph_plain}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "docx",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "paper_methods.docx"),
        help="Path to the reviewed .docx.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output Markdown path. Defaults to <docx_stem>_review.md next to the docx.",
    )
    args = parser.parse_args()

    docx_path = Path(args.docx)
    if not docx_path.exists():
        sys.exit(f"File not found: {docx_path}")

    out_path = (
        Path(args.output)
        if args.output
        else docx_path.with_name(f"{docx_path.stem}_review.md")
    )

    with zipfile.ZipFile(docx_path) as zf:
        comments = load_comments(zf)
        doc_xml = ET.fromstring(zf.read("word/document.xml"))
        body = doc_xml.find(q("body"))
        if body is None:
            sys.exit("Could not find <w:body> in document.xml.")
        anchors, changes = process_paragraphs(body)

    report = render_report(docx_path, comments, anchors, changes)
    out_path.write_text(report, encoding="utf-8")
    print(
        f"Wrote {out_path}  "
        f"({len(comments)} comments, {len(changes)} tracked changes)"
    )


if __name__ == "__main__":
    main()
