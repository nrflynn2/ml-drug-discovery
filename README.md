# "Machine Learning for Drug Discovery" - Code and Data Repository

### 👋 Welcome to the Machine Learning for Drug Discovery Repository

This repository contains code and data for the first edition of [Machine Learning for Drug Discovery (Manning Publications)](http://mng.bz/DdVn). The companion material within this repository covers introductory topics at the intersection of machine learning, deep learning, and drug discovery applied to real world scenarios in each chapter. The code and notebooks are released under the Apache 2.0 license. 

For readability, the chapter notebooks only contain runnable code blocks and section titles. They omit the rest of the material in the book, i.e., text paragraphs, figures (unless generated as part of one of the code blocks), equations, and pseudocode. **If you want to be able to follow what's going on, I recommend reading the notebooks side-by-side with your copy of the book!**

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
* Chapter 12: Transformer Architectures for Protein Structure Prediction
* Chapter 13: Multimodal AI Systems for End-to-End Drug Discovery Pipelines

#### Appendices
* [Appendix A: Glossary](https://livebook.manning.com/book/machine-learning-for-drug-discovery/appendix-a)
* [Appendix B: Chemical Data Repositories](https://livebook.manning.com/book/machine-learning-for-drug-discovery/appendix-b/v-8)
* [Appendix C: Knowledge Distillation: Shrinking Models for Efficient, Hierarchical Molecular Generation](https://github.com/nrflynn2/ml-drug-discovery/blob/main/APPENDIX_C_FLYNN_ML4DD.ipynb)
* Appendix D: Reinforcement Learning for Targeted Optimizations
* Appendix E: Agentic Systems in Drug Discovery

### 🚧 Under Construction

Note that this project is a work in progress and notebooks will be released as they are drafted. We anticipate a full release of the book in Winter 2025. We recommend interacting with notebooks through Colab.

Purchase of the book through Manning's Early Access Program (MEAP) guarantees access to current and future chapters. I appreciate your patience and support!

Encounter any issues? Please let me know -- I can't fix a problem if I am not aware of its existence!

### 💻 Getting Started

#### Option 1: Google Colab (No Installation Required)

Open the repository in Colab to walk through the notebooks without needing to install anything! <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#### Option 2: Local Installation

**Prerequisites**: Python 3.11+ and git

Choose one of the following installation methods:

**Quick Start with uv (Fastest)**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Complete Installation with Conda (Recommended for All Features)**

```bash
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery
conda env create -f ml4dd2025.yml
conda activate ml4dd2025
```

**Traditional pip Installation**

```bash
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# For molecular dynamics tools (optional except for chapter 9):
# conda install -c conda-forge openmm pdbfixer vina
```

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
