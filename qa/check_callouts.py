#!/usr/bin/env python3
"""Fail if Manning callout markers leak into notebook *code* cells.

The 3P reviewers' top concern was AsciiDoc callout markers (``[CA]``/``[CB]``
and trailing ``#A``/``#B`` annotations) ending up inside runnable code, where
they raise ``SyntaxError``. An exhaustive sweep found zero such leaks in the
current notebooks — this guard exists to keep it that way as chapters are
edited. It is deliberately stdlib-only (parses the ``.ipynb`` JSON directly) so
it runs anywhere: CI, pre-commit, or ``make lint``.

Usage:
    python qa/check_callouts.py [notebook.ipynb ...]

With no arguments it scans every chapter + appendix + CH12 notebook. Exits 1 on
any leak (printing file / cell / line), 0 otherwise.
"""

from __future__ import annotations

import glob
import json
import re
import sys

# [CA], [CB], ... — Manning bracketed inline callout references.
BRACKET_CALLOUT = re.compile(r"\[C[A-Z]\]")

# Trailing #A / #B / #A1 annotation markers. Anchored so a normal "# comment"
# (hash + space) never matches, and hex colours like color="#A20025" never
# match (the '#' there is not preceded by whitespace/start-of-line and the
# marker is not at end-of-line as a bare token).
HASH_CALLOUT = re.compile(r"(?:^|\s)#[A-Z]{1,2}\d*\s*$")


def find_default_notebooks() -> list[str]:
    patterns = ["CH*.ipynb", "APPENDIX*.ipynb", "CH12_FLYNN_ML4DD/notebooks/*.ipynb"]
    found: list[str] = []
    for pat in patterns:
        found.extend(sorted(glob.glob(pat)))
    return found


def scan_notebook(path: str) -> list[str]:
    """Return a list of human-readable leak descriptions for one notebook."""
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)

    leaks: list[str] = []
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        lines = source if isinstance(source, list) else source.splitlines(keepends=True)
        for line_idx, line in enumerate(lines, start=1):
            text = line.rstrip("\n")
            if BRACKET_CALLOUT.search(text) or HASH_CALLOUT.search(text):
                leaks.append(f"{path}: code cell {cell_idx}, line {line_idx}: {text.strip()!r}")
    return leaks


def main(argv: list[str]) -> int:
    notebooks = argv[1:] or find_default_notebooks()
    all_leaks: list[str] = []
    for nb_path in notebooks:
        all_leaks.extend(scan_notebook(nb_path))

    if all_leaks:
        print("Manning callout markers leaked into notebook code cells:\n")
        for leak in all_leaks:
            print(f"  {leak}")
        print(f"\n{len(all_leaks)} leak(s) found. See qa/check_callouts.py.")
        return 1

    print(f"No callout leaks in {len(notebooks)} notebook(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
