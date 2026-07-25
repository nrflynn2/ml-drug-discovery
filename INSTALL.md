# Installation Guide

This repository supports several installation paths. **uv is the recommended path** — it is fast,
reproducible (via the committed `uv.lock`), and manages the Python 3.12 interpreter for you.

## Prerequisites

- Git
- Python 3.12 (uv can install this for you — see Method 1)

## Tiered installs

Install only what a chapter needs:

| Tier | Chapters | How |
|------|----------|-----|
| **Core** — basic ML & QSAR | 1–4 | base install |
| **Advanced** — boosting, deep learning, GNNs | 5–8, 10, 11, Appendix C | `advanced` extra |
| **Full** — + pip-installable docking/MD helpers | adds Chapter 9 helpers | `full` extra |
| **Chapter 9 conda island** | 9 (openmm/pdbfixer/vina) | conda `ml4dd2025.yml` |
| **Chapter 12** | 12 | its own `CH12_FLYNN_ML4DD/` package |

## Method 1: uv (recommended)

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

# Create a Python 3.12 environment and install a tier
uv venv --python 3.12
source .venv/bin/activate            # Windows: .venv\Scripts\activate

uv sync                     # core     (Chapters 1–4)
uv sync --extra advanced    # + Chapters 5–8, 10, 11, Appendix C
uv sync --extra full        # + Chapter 9 pip-installable docking/MD helpers

# Or run without activating:  uv run --extra advanced jupyter lab
```

> Chapter 9 also needs conda-only tools (openmm, pdbfixer, vina). See Method 2.

## Method 2: Conda island for Chapter 9

Molecular-dynamics/docking packages (openmm, pdbfixer, vina) install reliably only via conda:

```bash
conda env create -f ml4dd2025.yml
conda activate ml4dd2025
```

## Method 3: pip + pyproject extras

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[advanced]"         # or ".[full]", or "." for core only
```

## Chapter 12

Chapter 12 is a self-contained package under `CH12_FLYNN_ML4DD/`:

```bash
cd CH12_FLYNN_ML4DD
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[notebooks]"     # ESM-2 (transformers), BioPython, viz/widgets
```

## Google Colab

Open any notebook in Colab and run its install cell at the top. Non-Chapter-9 notebooks use a fast
pip install for that chapter's tier; Chapter 9 uses `condacolab` for the openmm/vina/pdbfixer stack.

## Reproducibility

- Every notebook calls `set_seed(42)` (from `utils.py`) to seed Python / NumPy / PyTorch (+CUDA).
- The committed `uv.lock` pins exact versions for the uv path.
- `numpy` is no longer capped below 2.0 — the previous cap came only from `mordred`, which the
  notebooks do not use and which has been removed. If a package ever breaks under numpy 2.x, pin
  `numpy>=1.26.4,<2.0.0` in `pyproject.toml` and note the reason here.
