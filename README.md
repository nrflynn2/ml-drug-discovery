# "Machine Learning for Drug Discovery" - Code and Data Repository [Work In Progress]

### 👋 Welcome to the Machine Learning for Drug Discovery Repository!!!

This repository contains code and data for the first edition of [Machine Learning for Drug Discovery (Manning Publications)](http://mng.bz/DdVn). The companion material within this repository covers introductory topics at the intersection of machine learning, deep learning, and drug discovery applied to real world scenarios in each chapter. The code and notebooks are released under the Apache 2.0 license. 

For readability, the chapter notebooks only contain runnable code blocks and section titles. They omit the rest of the material in the book, i.e., text paragraphs, figures (unless generated as part of one of the code blocks), equations, and pseudocode. **If you want to be able to follow what's going on, I recommend reading the notebooks side-by-side with your copy of the book!**

### 🚧 Under Construction

Note that this project is a work in progress and notebooks will be released as they are drafted. We anticipate a full release of the book in Summer 2025. We recommend interacting with notebooks through Colab.

Purchase of the book through Manning's Early Access Program (MEAP) guarantees access to current and future chapters. I appreciate your patience and support!

### 💻 Getting Started

Open the repository in Colab to walk through the notebooks without needing to install anything! <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you want to run and modify the code locally, install [Anaconda](https://www.anaconda.com/products/distribution) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) and [git](https://git-scm.com/downloads) if you don't already have access to them. Clone this repository by typing the following within a terminal (ignoring the first `$` character):

    $ git clone https://github.com/nrflynn2/ml-drug-discovery.git
    $ cd ml-drug-discovery

Set up a conda environment with prerequisite installs, we recommend:

    $ conda env create -f ml4dd2025.yml

An alternative method is:

    $ conda create --name ml-drug-discovery python=3.10 pip
    $ conda activate ml-drug-discovery
    $ pip install -r requirements.txt
    $ conda install -c conda-forge vina openmm pdbfixer

Finally, start Jupyter in the terminal via `jupyter notebook` or through your favorite IDE to embark on an exciting journey. Happy learning!

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
* Chapter 10: Generative Models for De Novo Design
* Chapter 11: Graph Neural Networks for Molecular Representation & Interaction Prediction
* Chapter 12: Transformer Architectures for Protein Structure Prediction
* Chapter 13: Multimodal AI Systems for End-to-End Drug Discovery Pipelines

#### Appendices
* Appendix C: Agentic Systems in Drug Discovery

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
