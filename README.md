# "Build AI Drug Discovery Pipelines" - Code and Data Repository

### Welcome to the Build AI Drug Discovery Pipelines Repository

This repository contains code and data for [Build AI Drug Discovery Pipelines (Manning Publications)](https://www.manning.com/books/build-ai-drug-discovery-pipelines). The companion material within this repository covers introductory topics at the intersection of machine learning, deep learning, and drug discovery applied to real world scenarios in each chapter. The code and notebooks are released under the Apache 2.0 license. 

For readability, the chapter notebooks only contain runnable code blocks and section titles. They omit the rest of the material in the book, i.e., text paragraphs, figures (unless generated as part of one of the code blocks), equations, and pseudocode. **If you want to be able to follow what's going on, I recommend reading the notebooks side-by-side with your copy of the book!**

Encounter any issues? Please let me know -- I can't fix a problem if I am not aware of its existence!

### 📚 Table of Contents

#### 💊 Part 1: Fundamentals of Cheminformatics & Machine Learning
* [Chapter 1: The Drug Discovery Process](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH01_FLYNN_ML4DD.ipynb)
* [Chapter 2: Ligand-based Screening: Filtering & Similarity Searching](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH02_FLYNN_ML4DD.ipynb)
* [Chapter 3: Solubility Deep Dive with Linear Models](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH03_FLYNN_ML4DD.ipynb)
* [Chapter 4: Case Study: Small Molecule Binding to an RNA Target](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH04_FLYNN_ML4DD.ipynb)

#### 🧬 Part 2: Deep Learning for Molecules & Structural Biology
* [Chapter 5: Structure-based Drug Design with Active Learning](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH05_FLYNN_ML4DD.ipynb)
* [Chapter 6: Generative Models for De Novo Design](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH06_FLYNN_ML4DD.ipynb)
* [Chapter 7: Transformer Architectures for Protein Structure Prediction](https://github.com/nrflynn2/ml-drug-discovery/tree/main/CH07_FLYNN_ML4DD)
* [Chapter 8: Multimodal AI Systems for End-to-End Drug Discovery Pipelines](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH08_FLYNN_ML4DD.ipynb)

#### 🌐 Supplemental Chapters (online edition)
* [Supplemental Chapter 1: Ligand-based Screening: Machine Learning](https://github.com/nrflynn2/ml-drug-discovery/blob/main/SUPP01_FLYNN_ML4DD.ipynb)
* [Supplemental Chapter 2: Classification: Cytochrome P450 Inhibition](https://github.com/nrflynn2/ml-drug-discovery/blob/main/SUPP02_FLYNN_ML4DD.ipynb)
* [Supplemental Chapter 3: Unsupervised Learning: Repurposing Drugs, Curating Compounds, & Screening Fragments](https://github.com/nrflynn2/ml-drug-discovery/blob/main/SUPP03_FLYNN_ML4DD.ipynb)
* [Supplemental Chapter 4: Introduction to Deep Learning](https://github.com/nrflynn2/ml-drug-discovery/blob/main/SUPP04_FLYNN_ML4DD.ipynb)
* [Supplemental Chapter 5: Graph Neural Networks for Drug Target Affinity Prediction](https://github.com/nrflynn2/ml-drug-discovery/blob/main/SUPP05_FLYNN_ML4DD.ipynb)

#### Appendices
* [Appendix A: Glossary](https://livebook.manning.com/book/build-ai-drug-discovery-pipelines/appendix-a)
* [Appendix B: Chemical Data Repositories](https://livebook.manning.com/book/build-ai-drug-discovery-pipelines/appendix-b/v-8)
* [Appendix C: Knowledge Distillation: Shrinking Models for Efficient, Hierarchical Molecular Generation](https://github.com/nrflynn2/ml-drug-discovery/blob/main/APPENDIX_C_FLYNN_ML4DD.ipynb)
* Appendix D: Technical Deep Dive into Protein Structure Prediction
* Appendix E: Extended Technical Material
* Appendix F: Chapter References
* Appendix G: Chapter Exercises
* Appendix H: Computational Drug Target Discovery
* [Appendix I: Diffusion &amp; Flow Matching with FlowMol3, DiffSBDD, RFdiffusion, &amp; Boltz-2](https://github.com/nrflynn2/ml-drug-discovery/tree/main/APPENDIX_I_FLYNN_ML4DD)

#### Reading the early-access (MEAP) edition?

The early-access edition had 13 chapters. For the final release, 8 chapters are in the printed
book and 5 moved online as supplemental chapters, so the notebooks and their `data/`,
`artifacts/`, and `figures/` folders were renumbered to match. The appendices did not change.

| Early-access chapter | Final edition | Notebook | Data/artifact/figure folder |
|---|---|---|---|
| 1 | Chapter 1 | `CH01_FLYNN_ML4DD.ipynb` | `ch01` |
| 2 | Chapter 2 | `CH02_FLYNN_ML4DD.ipynb` | `ch02` |
| 3 | Supplemental Chapter 1 | `SUPP01_FLYNN_ML4DD.ipynb` | `supp01` |
| 4 | Chapter 3 | `CH03_FLYNN_ML4DD.ipynb` | `ch03` |
| 5 | Supplemental Chapter 2 | `SUPP02_FLYNN_ML4DD.ipynb` | `supp02` |
| 6 | Chapter 4 | `CH04_FLYNN_ML4DD.ipynb` | `ch04` |
| 7 | Supplemental Chapter 3 | `SUPP03_FLYNN_ML4DD.ipynb` | `supp03` |
| 8 | Supplemental Chapter 4 | `SUPP04_FLYNN_ML4DD.ipynb` | `supp04` |
| 9 | Chapter 5 | `CH05_FLYNN_ML4DD.ipynb` | `ch05` |
| 10 | Chapter 6 | `CH06_FLYNN_ML4DD.ipynb` | `ch06` |
| 11 | Supplemental Chapter 5 | `SUPP05_FLYNN_ML4DD.ipynb` | `supp05` |
| 12 | Chapter 7 | `CH07_FLYNN_ML4DD/` | `CH07_FLYNN_ML4DD/data` |
| 13 | Chapter 8 | `CH08_FLYNN_ML4DD.ipynb` | `ch08` |

If your copy of the book still uses the old chapter numbers in its paths, use the frozen
13-chapter layout at the [`meap-13-chapters`](https://github.com/nrflynn2/ml-drug-discovery/tree/meap-13-chapters) tag.

### Getting Started

#### Option 1: Google Colab (No Installation Required)

Open any notebook in Colab and run the installation cells at the top! <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Each notebook includes a Colab install cell that pip-installs only the packages that chapter needs
(3-10 minutes). **Chapter 5** is the exception: its docking stack needs conda-only tools, so it uses a
conda install cell on Colab.

#### Option 2: Local Installation with uv (recommended)

**Prerequisites**: git and Python 3.12 — [uv](https://github.com/astral-sh/uv) installs and manages the interpreter for you.

```bash
# Install uv if needed:  curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery
uv venv --python 3.12 && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

We provide **tiered installs** so you install only what a chapter needs:

| Tier | Chapters | Command |
|------|----------|---------|
| **Core** — basic ML & QSAR | 1–3, 8; Supplemental 1 | `uv sync` |
| **Advanced** — boosting, deep learning, GNNs | 4, 6; Supplemental 2–5; Appendix C | `uv sync --extra advanced` |
| **Full** — + pip-installable docking/MD helpers | adds Chapter 5 helpers | `uv sync --extra full` |

**Chapter 5** additionally needs conda-only tools (openmm, vina, pdbfixer):
```bash
conda env create -f ml4dd2025.yml && conda activate ml4dd2025
```

**Chapter 7** is a self-contained package — see [`CH07_FLYNN_ML4DD/`](CH07_FLYNN_ML4DD).

Prefer classic pip? `pip install -e ".[advanced]"` works too. For detailed instructions and troubleshooting, see [INSTALL.md](INSTALL.md).

### Contribution & Support

Feel free to contribute, raise issues, or propose enhancements to make this repository a comprehensive resource for everyone venturing into machine learning, drug discovery, and related applications.

### Citations

If you wish to cite the book, you may use the following:

```
@book{flynn2025mldd,
title={Build AI Drug Discovery Pipelines},
author={Flynn, N.},
isbn={9781633437661},
url={https://www.manning.com/books/build-ai-drug-discovery-pipelines},
year={2025},
publisher={Manning Publications}
}
```
