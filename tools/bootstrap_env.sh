#!/usr/bin/env bash
# Bootstrap the ML4DD environment with uv + Python 3.12 (WSL2 / Linux / macOS).
#
# Usage:  bash tools/bootstrap_env.sh [core|advanced|full]   (default: advanced)
#
# Chapter 9 needs the conda island instead — see the note printed at the end.
set -euo pipefail

TIER="${1:-advanced}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Install uv if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv (Astral) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo ">> uv: $(command -v uv)"

# 2. Create a Python 3.12 virtual environment (uv fetches 3.12 if needed).
echo ">> Creating .venv (Python 3.12) ..."
uv venv --python 3.12

# 3. Install the requested tier from pyproject.toml.
echo ">> Installing tier: $TIER"
case "$TIER" in
    core)     uv sync ;;
    advanced) uv sync --extra advanced ;;
    full)     uv sync --extra full ;;
    *) echo "Unknown tier '$TIER' (expected core|advanced|full)"; exit 1 ;;
esac

# 4. QA/authoring tooling used by the notebook workflow.
uv pip install "jupytext>=1.16.0" "nbconvert>=7.16.0"

# 5. Sanity check: every notebook code cell must still parse.
echo ">> Validating notebooks ..."
uv run python tools/validate_notebooks.py

cat <<'EOF'

============================================================
Environment ready.  Activate it with:
    source .venv/bin/activate

Chapter 9 (openmm / pdbfixer / vina) uses the conda island:
    conda env create -f ml4dd2025.yml && conda activate ml4dd2025

Chapter 12 is a self-contained package:
    cd CH12_FLYNN_ML4DD && uv pip install -e ".[notebooks]"
============================================================
EOF
