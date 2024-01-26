# "Machine Learning for Drug Discovery" - Code and Data Repository [Work In Progress]

### 👋 Welcome to the Machine Learning for Drug Discovery Repository!!!

This repository contains code and data for the first edition of Machine Learning for Drug Discovery (Manning Publications). The companion material within this repository covers introductory topics at the intersection of machine learning, deep learning, and drug discovery applied to real world scenarios in each chapter. The code and notebooks are released under the Apache 2.0 license. 

The first edition is a work in progress and notebooks will be released as they are drafted. We anticipate a full release of the book in Fall 2024. In the meantime, chapters will be released and made available as part of Manning's Early Access Program in early 2024. We recommend interacting with notebooks through Colab.

### 💻 Getting Started

Open the repository in Colab to walk through the notebooks without needing to install anything! <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you want to run and modify the code locally, install [Anaconda](https://www.anaconda.com/products/distribution) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) and [git](https://git-scm.com/downloads) if you don't already have access to them. Clone this repository by typing the following within a terminal (ignoring the first `$` character):

    $ git clone https://github.com/nrflynn2/ml-drug-discovery.git
    $ cd ml-drug-discovery

Set up a conda environment with prerequisite installs:

    $ conda create --name ml-drug-discovery python=3.10 pip
    $ conda activate ml-drug-discovery
    $ pip install -r requirements.txt

Finally, start Jupyter in the terminal via `jupyter notebook` or through your favorite IDE to embark on an exciting journey. Happy learning!

### 📚 Table of Contents

#### Part 1: Fundamentals of Cheminformatics & Machine Learning
* [Chapter 1: The Drug Discovery Process](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH01_FLYNN_ML4DD.ipynb)
* [Chapter 2: Ligand-based Screening: Filtering & Similarity Searching](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH02_FLYNN_ML4DD.ipynb)
* [Chapter 3: Ligand-based Screening: Machine Learning](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH03_FLYNN_ML4DD.ipynb)
* [Chapter 4: Solubility Deep Dives with Linear Models](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH04_FLYNN_ML4DD.ipynb)
* [Chapter 5: Classification: Cytochrome P450 Inhibition](https://github.com/nrflynn2/ml-drug-discovery/blob/main/CH05_FLYNN_ML4DD.ipynb)
* Chapter 6: Searching in Chemical Space
* Chapter 7: Curating Diverse Compounds

#### Part 2: Deep Learning for Molecules & Structural Biology
* Chapter 8: Introduction to Deep Learning
* Chapter 9: Generative Models for Library Design
* Chapter 10: Molecules as a Language
* Chapter 11: Drug-Target Binding Affinity with Transformers
* Chapter 12: Graph Neural Networks for Molecules
* Chapter 13: GNN Applications in Drug Discovery
* Chapter 14: Diffusion Models
* Chapter 15: Closing Remarks & Next Steps

### 👥 Contribution & Support

Feel free to contribute, raise issues, or propose enhancements to make this repository a comprehensive resource for everyone venturing into machine learning, drug discovery, and related applications.
