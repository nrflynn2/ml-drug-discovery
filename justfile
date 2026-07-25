# Justfile — QA + notebook workflow for the ML4DD companion notebooks.
# Install `just` (https://github.com/casey/just), or read these as the canonical
# commands and run them by hand. Python steps require the env from INSTALL.md.

# Show available recipes
default:
    @just --list

# --- Static validation (the standing marker guard) -------------------------
# Every code cell must parse. Pass notebook paths, or omit for all.
validate *nbs:
    python tools/validate_notebooks.py {{nbs}}

# --- Jupytext pairing / sync ----------------------------------------------
# Pair a notebook <-> percent .py (run ONCE per notebook to add metadata).
pair nb:
    jupytext --set-formats ipynb,py:percent {{nb}}

# Pair every top-level chapter notebook + Appendix C in one go.
pair-all:
    for nb in CH*_FLYNN_ML4DD.ipynb APPENDIX_C_FLYNN_ML4DD.ipynb; do \
        jupytext --set-formats ipynb,py:percent "$nb"; \
    done

# Sync edits from a paired .py back into its .ipynb (inputs only, keeps outputs).
sync file:
    jupytext --sync {{file}}

# --- Execution (refresh outputs) ------------------------------------------
# Execute a notebook in place, honoring `skip-execution` tags (Colab + heavy cells).
# `timeout` is per-cell seconds (default 15 min). Long cells should be skip-tagged, not timed out.
execute nb timeout="900":
    python tools/execute_notebook.py {{nb}} --timeout {{timeout}}

# Full gate for a CPU chapter: sync .py -> validate -> execute .ipynb.
check-cpu file nb:
    just sync {{file}}
    just validate {{nb}}
    just execute {{nb}}

# --- Environment / housekeeping -------------------------------------------
# Regenerate the pip requirements exports from pyproject extras (needs uv).
export-reqs:
    uv export --no-hashes -o requirements-core.txt
    uv export --no-hashes --extra advanced -o requirements-advanced.txt
    uv export --no-hashes --extra full -o requirements-full.txt

lint:
    ruff check .

fmt:
    black .
