#!/usr/bin/env python3
"""Rewrite notebook markdown headers to portable markdown.

The chapters wrap headings in deprecated presentational HTML
(``<b> <font color='#A20025'> ...</font>``) that GitHub will not render. This
helper strips those ``<b>`` / ``<font>`` tags from markdown cells, leaving clean
portable markdown (``# 📚 Chapter 1: ...``) while preserving heading levels,
emoji, and ``**bold**``.

It is intentionally minimal-diff: the notebook is loaded and re-serialized in
canonical Jupyter format (``json.dumps(..., indent=1, ensure_ascii=False)`` plus
a trailing newline), so only the changed source lines appear in the diff — code
cell outputs are untouched. Review the result with ``nbdime`` before committing.

Usage:
    python qa/modernize_headers.py CH01_FLYNN_ML4DD.ipynb [more.ipynb ...]
    python qa/modernize_headers.py --check CH01_FLYNN_ML4DD.ipynb   # dry run
"""

from __future__ import annotations

import json
import re
import sys

TAG_PATTERNS = [
    re.compile(r"</?b>"),
    re.compile(r"<font[^>]*>"),
    re.compile(r"</font>"),
]
HEADING_SPACE = re.compile(r"^(#{1,6})\s+")


def clean_line(line: str) -> str:
    original = line
    for pat in TAG_PATTERNS:
        line = pat.sub("", line)
    if line is original:
        return line
    # Normalize the space after heading hashes ("##   Foo" -> "## Foo").
    if HEADING_SPACE.match(line):
        line = HEADING_SPACE.sub(r"\1 ", line, count=1)
    # Collapse the double spaces the stripped tags leave behind, but keep any
    # trailing newline the source line carried.
    trailing = "\n" if line.endswith("\n") else ""
    line = re.sub(r"[ \t]{2,}", " ", line.rstrip("\n")).rstrip() + trailing
    return line


def modernize(nb: dict) -> int:
    changes = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        for idx, line in enumerate(source):
            new = clean_line(line)
            if new != line:
                source[idx] = new
                changes += 1
    return changes


def main(argv: list[str]) -> int:
    args = argv[1:]
    check = "--check" in args
    paths = [a for a in args if a != "--check"]
    if not paths:
        print(__doc__)
        return 2

    total = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            nb = json.load(fh)
        changes = modernize(nb)
        total += changes
        if changes and not check:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        verb = "would modernize" if check else "modernized"
        print(f"{path}: {verb} {changes} header line(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
