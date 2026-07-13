#!/usr/bin/env python3
"""Execute a notebook in place, honoring ``skip-execution`` cell tags.

Why this exists (from the QA pilot): local "output refresh" execution of the
book notebooks must NOT run two kinds of cells:

  1. **Colab-only setup cells** (`!pip` / `!wget` / `condacolab` / and in CH02 a
     literal `os.kill(os.getpid(), 9)` that would kill the kernel), and
  2. **Intentionally long-running cells** that the notebook already pairs with a
     committed-artifact *reload* cell (e.g. CH02's ~20-min PAINS/BRENK filter,
     served from `artifacts/ch02/*` on the fast path).

Both are marked with the Jupyter cell tag ``skip-execution``, which nbclient
honors natively. This wrapper is the one uniform way every chapter agent runs
the execution step: it skips tagged cells, uses a sane per-cell timeout, fails
loudly on a *real* error (no ``--allow-errors``), and prints what ran vs. skipped.

So "full execution" for a CPU chapter means: run every cell EXCEPT those tagged
``skip-execution``. If a cell is too slow to run in CI/locally, tag it
``skip-execution`` (and make sure the notebook reloads its result from a
committed artifact) rather than raising the timeout.

Usage (must run inside the chapter env, e.g. via uv):
    uv run python tools/execute_notebook.py CH02_FLYNN_ML4DD.ipynb
    uv run python tools/execute_notebook.py CH08_FLYNN_ML4DD.ipynb --timeout 1800

Exit codes: 0 = executed clean, 1 = an executed cell raised, 2 = usage/env error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SKIP_TAG = "skip-execution"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebook", help="path to the .ipynb to execute in place")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds (default 900)")
    ap.add_argument("--kernel", default="python3", help="kernel name (default python3)")
    ap.add_argument("--dry-run", action="store_true", help="report skip/run counts without executing")
    args = ap.parse_args(argv)

    nb_path = Path(args.notebook)
    if not nb_path.exists():
        print(f"not found: {nb_path}", file=sys.stderr)
        return 2

    try:
        import nbformat
        from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
    except ImportError as e:  # pragma: no cover - depends on env
        print(f"missing execution deps ({e}); run inside the chapter env via `uv run`", file=sys.stderr)
        return 2

    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    skipped = [c for c in code_cells if SKIP_TAG in c.get("metadata", {}).get("tags", [])]
    to_run = len(code_cells) - len(skipped)
    print(f"{nb_path.name}: {len(code_cells)} code cells; {len(skipped)} tagged '{SKIP_TAG}'; {to_run} to run")

    if args.dry_run:
        return 0

    # ExecutePreprocessor (via nbclient) skips cells tagged `skip-execution`.
    ep = ExecutePreprocessor(timeout=args.timeout, kernel_name=args.kernel, allow_errors=False)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent) or "."}})
    except CellExecutionError as e:
        nbformat.write(nb, nb_path)  # persist partial outputs to aid debugging
        print(f"EXECUTION FAILED (partial outputs written): {e}", file=sys.stderr)
        return 1

    nbformat.write(nb, nb_path)
    print(f"OK: executed {to_run} cells, skipped {len(skipped)} ('{SKIP_TAG}'), 0 errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
