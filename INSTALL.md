# Installation Guide

This repository provides multiple installation methods. Choose the one that best fits your needs.

## Prerequisites

- Git
- Python 3.11 or later

## Installation Methods

### Method 1: Using uv (Recommended - Fastest)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer that's significantly faster than pip.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Note: For molecular dynamics tools (openmm, pdbfixer, vina), use conda method below. These tools are only necessary to complete chapter 9.
```

### Method 2: Using Conda (Complete Installation)

This method installs all dependencies including molecular dynamics tools that require conda.

```bash
# Clone repository
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

# Create and activate conda environment
conda env create -f ml4dd2025.yml
conda activate ml4dd2025
```

### Method 3: Using pip + pyproject.toml

```bash
# Clone repository
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Note: For molecular dynamics tools (openmm, pdbfixer, vina), install via conda:
# conda install -c conda-forge openmm pdbfixer vina
```

### Method 4: Using pip + requirements.txt (Traditional)

```bash
# Clone repository
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Note: For molecular dynamics tools (openmm, pdbfixer, vina), install via conda:
# conda install -c conda-forge openmm pdbfixer vina
```
