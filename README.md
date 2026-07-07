# "Build AI Drug Discovery Pipelines" - Code and Data Repository

### 👋 Welcome to the Build AI Drug Discovery Pipelines Repository

This repository contains code and data for [Build AI Drug Discovery Pipelines (Manning Publications)](http://mng.bz/DdVn). The companion material within this repository covers introductory topics at the intersection of machine learning, deep learning, and drug discovery applied to real world scenarios in each chapter. The code and notebooks are released under the Apache 2.0 license. 

For readability, the chapter notebooks only contain runnable code blocks and section titles. They omit the rest of the material in the book, i.e., text paragraphs, figures (unless generated as part of one of the code blocks), equations, and pseudocode. **If you want to be able to follow what's going on, I recommend reading the notebooks side-by-side with your copy of the book!**

Encounter any issues? Please let me know -- I can't fix a problem if I am not aware of its existence!

### 📚 Table of Contents

#### 💊 Part 1: Fundamentals of Cheminformatics & Machine Learning
* [Chapter 1: The Drug Discovery Process](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH01_FLYNN_ML4DD.ipynb)
* [Chapter 2: Ligand-based Screening: Filtering & Similarity Searching](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH02_FLYNN_ML4DD.ipynb)
* [Chapter 3: Ligand-based Screening: Machine Learning](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH03_FLYNN_ML4DD.ipynb)
* [Chapter 4: Solubility Deep Dive with Linear Models](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH04_FLYNN_ML4DD.ipynb)
* [Chapter 5: Classification: Cytochrome P450 Inhibition](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH05_FLYNN_ML4DD.ipynb)
* [Chapter 6: Case Study: Small Molecule Binding to an RNA Target](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH06_FLYNN_ML4DD.ipynb)
* [Chapter 7: Unsupervised Learning: Repurposing Drugs, Curating Compounds, & Screening Fragments](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH07_FLYNN_ML4DD.ipynb)

#### 🧬 Part 2: Deep Learning for Molecules & Structural Biology
* [Chapter 8: Introduction to Deep Learning](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH08_FLYNN_ML4DD.ipynb)
* [Chapter 9: Structure-based Drug Design with Active Learning](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH09_FLYNN_ML4DD.ipynb)
* [Chapter 10: Generative Models for De Novo Design](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH10_FLYNN_ML4DD.ipynb)
* [Chapter 11: Graph Neural Networks for Drug Target Affinity Prediction](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH11_FLYNN_ML4DD.ipynb)
* [Chapter 12: Transformer Architectures for Protein Structure Prediction](https://github.com/nrflynn2/ml-drug-discovery/tree/main/CH12_FLYNN_ML4DD)
* [Chapter 13: Multimodal AI Systems for End-to-End Drug Discovery Pipelines](https://www.manning.com/books/machine-learning-for-drug-discovery)

#### Appendices
* [Appendix A: Glossary](https://livebook.manning.com/book/machine-learning-for-drug-discovery/appendix-a)
* [Appendix B: Chemical Data Repositories](https://livebook.manning.com/book/machine-learning-for-drug-discovery/appendix-b/v-8)
* [Appendix C: Knowledge Distillation: Shrinking Models for Efficient, Hierarchical Molecular Generation](https://github.com/nrflynn2/ml-drug-discovery/blob/main/APPENDIX_C_FLYNN_ML4DD.ipynb)
* Appendix D: Technical Deep Dive into Protein Structure Prediction

### 💻 Getting Started

#### Option 1: Google Colab (No Installation Required)

Open any notebook in Colab and run the installation cells at the top! <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Each notebook includes two Colab installation options:
- **Quick Install**: Fast pip-based setup (3-10 minutes) with only the packages needed for that chapter
- **Full Install**: Complete conda environment (15-20 minutes) with all packages for all chapters

#### Option 2: Local Installation

**Prerequisites**: Python 3.12 and git. We recommend [uv](https://github.com/astral-sh/uv) for a fast, reproducible install (a committed `uv.lock` pins the full environment).

**Recommended (uv):**
```bash
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery
uv venv --python 3.12
uv sync                          # base = Chapters 1-4 (core)
uv sync --extra advanced         # Chapters 5-8 + Appendix C
uv sync --extra full             # Chapters 10-11 (adds GNN/docking-analysis, pip-only)
```

We also provide **tiered `requirements-*.txt`** files (used by the Colab cells) if you prefer pip:

**Core (Chapters 1-4)** — Basic ML & QSAR
```bash
pip install -r requirements-core.txt
```
*Includes: numpy, pandas, matplotlib, seaborn, rdkit, scikit-learn*

**Advanced (Chapters 5-8, Appendix C)** — Gradient Boosting & Deep Learning
```bash
pip install -r requirements-advanced.txt
```
*Adds: torch, xgboost, lightgbm, catboost, shap, umap, statsmodels*

**Full (Chapters 10-11)** — Generative Models & GNNs
```bash
pip install -r requirements-full.txt
```
*Adds: torch-geometric, mdtraj, prolif, gemmi, meeko, openbabel — all pip-installable*

**Chapter 9 only** — Structure-based Design (Docking & Molecular Dynamics)
```bash
conda env create -f ml4dd2025.yml
conda activate ml4dd2025
```
*Adds the conda-only stack: openmm, vina, pdbfixer.*

**Note**: **Only Chapter 9** requires conda, for the docking/molecular-dynamics packages (openmm, vina, pdbfixer) that have no reliable pip wheels. Every other chapter — including 10 and 11 — installs under uv/pip.

**Quick Reference**:
- **Chapters 1-4**: `uv sync` (or `requirements-core.txt`)
- **Chapters 5-8, Appendix C**: `uv sync --extra advanced` (or `requirements-advanced.txt`)
- **Chapters 10-11**: `uv sync --extra full` (or `requirements-full.txt`)
- **Chapter 9**: `ml4dd2025.yml` (conda required)
- **Chapter 12**: follow the notebooks within `CH12_FLYNN_ML4DD`

For detailed installation instructions and troubleshooting, see [INSTALL.md](INSTALL.md).

### 👥 Contribution & Support

Feel free to contribute, raise issues, or propose enhancements to make this repository a comprehensive resource for everyone venturing into machine learning, drug discovery, and related applications.

### 🔎 Citations

If you wish to cite the book, you may use the following:

```
@book{flynn2025mldd,
title={Machine Learning for Drug Discovery},
author={Flynn, N.},
isbn={9781633437661},
url={https://www.manning.com/books/machine-learning-for-drug-discovery},
year={2025},
publisher={Manning Publications}
}
```
