# Installation Guide

This repository provides multiple installation methods. Choose the one that best fits your needs.

## Prerequisites

- Git
- Python 3.12

> **Which chapters need what?** Chapters 1–8, 10–11, and Appendix C install entirely under uv/pip.
> **Only Chapter 9** (structure-based design: docking + molecular dynamics) needs conda, for
> `openmm`, `pdbfixer`, and `vina`, which have no reliable PyPI wheels.

## Method 1: Using uv (Recommended — fast + reproducible)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. A committed `uv.lock`
pins the full pip-installable environment, so everyone resolves the same versions.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

# Create the environment and install the tier you need
uv venv --python 3.12
uv sync                       # Chapters 1-4 (core)
uv sync --extra advanced      # Chapters 5-8 + Appendix C
uv sync --extra full          # Chapters 10-11 (adds GNN + docking-analysis, pip-only)
uv sync --extra dev           # Adds lint/test tooling (ruff, black, pytest, nbmake, nbqa, nbdime)

# Run anything inside the environment without activating it:
uv run python -c "import rdkit, pandas, sklearn; import bookutils"
```

To activate the environment directly instead of using `uv run`:

```bash
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

## Method 2: Using Conda (Chapter 9 only)

Chapter 9 needs the conda-forge docking/molecular-dynamics stack. Create its dedicated environment:

```bash
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

conda env create -f ml4dd2025.yml
conda activate ml4dd2025
```

This environment is self-contained (it also installs the ML/cheminformatics packages via pip), so
you can run every chapter from it if you prefer a single conda environment.

## Method 3: Using pip + requirements files

If you would rather use plain pip, the tiered requirements files mirror the uv extras (and are what
the Colab setup cells install from):

```bash
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements-core.txt        # Chapters 1-4
pip install -r requirements-advanced.txt    # Chapters 5-8 + Appendix C
pip install -r requirements-full.txt        # Chapters 10-11
# Chapter 9: use the conda method above.
```

## Method 4: Editable install of the package

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[advanced]"   # or ".[full]", ".[dev]"
```
