#!/usr/bin/env python3
"""Static validation harness for the ML4DD companion notebooks.

This is the *standing guard* for the QA pass: it enforces the machine floor of
"static-all" — every code cell in every notebook must parse as valid Python —
and flags the recurring manuscript artifacts that turn a listing into a
SyntaxError (stray callout markers, line-continuation-then-comment, REPL prompts
pasted into a code cell, Unicode ellipses).

Design goals:
  * **stdlib only** (``json`` + ``ast``). No nbformat / jupytext / third-party
    deps, so it runs in a bare interpreter, in CI, and inside any chapter env.
  * **fast + deterministic**: no execution, no imports of notebook code.
  * **CI-friendly exit codes**: non-zero if any hard error is found.

What it catches
---------------
ERRORS (fail the run):
  * A code cell that does not ``ast.parse`` (this is exactly the failure mode of
    ``\\`` + ``#A``, a stray ``[CA]`` token, ``PolynomialFeatures(.. , , ..)``,
    a bad indent, a Unicode ``…``, or a ``>>>`` prompt at statement level).

WARNINGS (reported, do not fail by default; use --strict to fail):
  * A trailing manuscript callout marker (``  #A`` / ``#B`` / ``#C`` at end of a
    line) that happens to be valid Python but is a book-build artifact.

Note on false positives: ``>>>`` inside a docstring (e.g. CH10's Vocab examples)
is valid Python and parses fine — it is intentionally *not* flagged. Only a
``>>>`` that reaches the parser as code (statement level) trips an ERROR.

Usage
-----
    python tools/validate_notebooks.py                  # all repo notebooks
    python tools/validate_notebooks.py CH08_FLYNN_ML4DD.ipynb ...
    python tools/validate_notebooks.py --strict         # warnings also fail
    python tools/validate_notebooks.py --json           # machine-readable

Exit codes: 0 = clean, 1 = errors found (or warnings under --strict).
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import warnings
from pathlib import Path

# Trailing manuscript callout marker, e.g. ``    return x    #A`` or ``... #B12``.
# Requires >=2 spaces before the marker so we don't flag ``x = 1  # a note``.
_CALLOUT_RE = re.compile(r"\s{2,}#[A-Z]\d?\s*$")
# Stray bracketed markers such as [CA], [CB], [CA1] that leak from the manuscript.
_BRACKET_MARKER_RE = re.compile(r"\[C[A-Z]\d?\]")


def _iter_code_cells(nb: dict):
    """Yield (index, source_str) for every code cell in a parsed notebook."""
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        # Strip IPython line/cell magics and shell escapes, which are valid in a
        # notebook kernel but not valid Python for ast.parse. Preserve the
        # original indentation so an ``!cmd`` inside an ``if:`` block still
        # satisfies the expected-indented-block rule.
        lines = []
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                indent = line[: len(line) - len(stripped)]
                lines.append(f"{indent}pass  # (magic/shell line elided for static check)")
            else:
                lines.append(line)
        yield i, "\n".join(lines), src


def validate_notebook(path: Path) -> dict:
    """Return a result dict: {path, n_code_cells, errors[], warnings[]}."""
    result = {"path": str(path), "n_code_cells": 0, "errors": [], "warnings": []}
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        result["errors"].append({"cell": None, "kind": "unreadable", "detail": str(e)})
        return result

    for idx, code, raw in _iter_code_cells(nb):
        result["n_code_cells"] += 1

        # Hard check: does the cell parse as Python? (Suppress SyntaxWarnings
        # such as invalid escape sequences — those are lint, not parse errors.)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(code)
        except SyntaxError as e:
            bad_line = ""
            if e.lineno:
                src_lines = code.splitlines()
                if 0 < e.lineno <= len(src_lines):
                    bad_line = src_lines[e.lineno - 1].strip()
            result["errors"].append(
                {
                    "cell": idx,
                    "kind": "syntax",
                    "detail": f"{e.msg} (line {e.lineno})",
                    "line": bad_line,
                }
            )

        # Soft checks: manuscript-marker artifacts that still parse.
        for lineno, line in enumerate(raw.splitlines(), start=1):
            if _CALLOUT_RE.search(line):
                result["warnings"].append(
                    {"cell": idx, "kind": "callout-marker", "line": line.strip()}
                )
            if _BRACKET_MARKER_RE.search(line):
                result["warnings"].append(
                    {"cell": idx, "kind": "bracket-marker", "line": line.strip()}
                )
    return result


def default_targets() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    targets = sorted(root.glob("*.ipynb"))
    targets += sorted((root / "CH12_FLYNN_ML4DD" / "notebooks").glob("*.ipynb"))
    return targets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="*", help="Notebook paths (default: all repo notebooks)")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args(argv)

    if args.notebooks:
        targets: list[Path] = []
        for pat in args.notebooks:
            targets.extend(Path(p) for p in glob.glob(pat)) if glob.has_magic(pat) else targets.append(Path(pat))
    else:
        targets = default_targets()

    results = [validate_notebook(p) for p in targets]
    n_err = sum(len(r["errors"]) for r in results)
    n_warn = sum(len(r["warnings"]) for r in results)
    n_cells = sum(r["n_code_cells"] for r in results)

    if args.json:
        print(json.dumps({"results": results, "totals": {"errors": n_err, "warnings": n_warn, "code_cells": n_cells, "notebooks": len(results)}}, indent=2))
    else:
        for r in results:
            name = Path(r["path"]).name
            if not r["errors"] and not r["warnings"]:
                print(f"  ✓ {name:32s} {r['n_code_cells']:>3} code cells OK")
                continue
            status = "✗" if r["errors"] else "⚠"
            print(f"  {status} {name:32s} {r['n_code_cells']:>3} code cells")
            for e in r["errors"]:
                loc = f"cell {e['cell']}" if e["cell"] is not None else "file"
                print(f"      ERROR [{loc}] {e['kind']}: {e['detail']}")
                if e.get("line"):
                    print(f"            > {e['line']}")
            for w in r["warnings"]:
                print(f"      WARN  [cell {w['cell']}] {w['kind']}: {w['line']}")
        print(
            f"\nSummary: {len(results)} notebooks, {n_cells} code cells, "
            f"{n_err} errors, {n_warn} warnings"
        )

    if n_err:
        return 1
    if args.strict and n_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
