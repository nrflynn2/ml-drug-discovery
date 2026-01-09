# Chapter 12: Protein Language Models with Transformers

This repository contains production-quality, instructional implementations of Transformer-based protein models for the textbook chapter on "Transformers for Protein Structure Prediction."

## Overview

This codebase demonstrates three key applications of Transformers to protein sequences:

1. **Protein Language Model (PLM)**: Self-supervised learning with masked language modeling
2. **Antibody Classifier**: Supervised classification using Transformer encoders
3. **Mutation Scoring**: Use ESM-2 to perturb proteins and computationally assess whether the mutated sequence is beneficial or deleterious.

## Repository Structure

```
ch12-protein-transformer/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── model.py                  # Transformer architecture components
│   ├── data.py                   # Tokenization and data loading
│   └── utils.py                  # Utility functions
├── scripts/                      # Training and evaluation scripts
│   ├── train_plm.py              # Train protein language model
│   ├── train.py                  # Train antibody classifier
│   ├── evaluate.py               # Model evaluation
│   └── tune.py                   # Hyperparameter tuning
├── notebooks/                    # Interactive tutorials corresponding to material from Chapter 12 and Appendix D
├── data/                         # Dataset files
│   ├── bcr_train.parquet
│   ├── bcr_test.parquet
│   ├── bcr.parquet
│   └── bcr_full_v3.csv
├── runs/                         # Training outputs (created automatically)
├── pyproject.toml                # Modern Python package config
├── requirements.txt              # Legacy requirements file
└── README.md
```

## Installation

This project supports multiple installation methods to accommodate different workflows.
We recommend following the analagous installation instructions for the book repository as a whole and then walking through the Python notebooks while reading Chapter 12 and Appendix D in the book.

## Quick Start

### 1. Train a Protein Language Model

Train a Transformer to predict masked amino acids:

```bash
python scripts/train_plm.py \
    --run-id my_first_plm \
    --dataset-loc data/bcr_train.parquet \
    --embedding-dim 128 \
    --num-layers 4 \
    --num-heads 4 \
    --ffn-dim 512 \
    --batch-size 32 \
    --lr 5e-4 \
    --num-epochs 50 \
    --warmup-steps 500
```

**What this does:**
- Trains a 4-layer Transformer encoder on protein sequences
- Uses masked language modeling (MLM) objective
- Implements learning rate warmup and cosine decay
- Saves model and metrics to `runs/my_first_plm/`

**Expected results** (on ~1000 sequences):
- Training time: ~30 minutes (GPU) / ~2 hours (CPU)
- Final validation perplexity: ~3-4
- Final validation accuracy: ~60-70%

### 2. Train an Antibody Classifier

Train a Transformer-based binary classifier:

```bash
python scripts/train.py \
    --run-id my_classifier \
    --dataset-loc data/bcr_train.parquet \
    --embedding-dim 64 \
    --num-layers 8 \
    --num-heads 2 \
    --ffn-dim 128 \
    --batch-size 32 \
    --lr 2e-5 \
    --num-epochs 20
```

### 3. Evaluate a Trained Model

```bash
python scripts/evaluate.py \
    --run-dir runs/my_classifier \
    --dataset-loc data/bcr_test.parquet \
    --batch-size 64
```

This computes and saves test metrics: accuracy, AUC, precision, recall, F1.

### 4. Hyperparameter Tuning

Use Ray Tune to find optimal hyperparameters:

```bash
# Make sure you installed with tuning support
pip install -e ".[dev]"

python scripts/tune.py \
    --run-id tuning_run \
    --dataset-loc data/bcr_train.parquet \
    --num-samples 50 \
    --num-epochs 30
```

## Architecture Details

### Protein Language Model

```python
ProteinLanguageModel(
    vocab_size=25,          # 20 amino acids + 5 special tokens
    padding_idx=1,          # Index of <pad> token
    embedding_dim=128,      # Embedding dimensions
    num_layers=4,           # Number of Transformer layers
    num_heads=4,            # Attention heads per layer
    ffn_dim=512,            # Feed-forward hidden dimension
    dropout=0.1,            # Dropout rate
)
```

**Training Objective:** Masked Language Modeling (MLM)
- Randomly mask 15% of amino acids
- 80% replaced with [MASK], 10% random, 10% unchanged
- Model predicts original amino acid at masked positions

### Antibody Classifier

```python
AntibodyClassifier(
    vocab_size=25,
    padding_idx=1,
    embedding_dim=64,
    num_layers=8,
    num_heads=2,
    ffn_dim=128,
    dropout=0.05,
    num_classes=2,
)
```

**Components:**
1. Embedding + Positional Encoding
2. Transformer Encoder
3. Mean Pooling: Averages across sequence positions
4. Classification Head: Projects to class logits

## Troubleshooting

### Out of Memory (OOM)
1. Reduce batch size: `--batch-size 16` or `--batch-size 8`
2. Reduce model size: Smaller `--embedding-dim` or `--num-layers`
3. Use CPU if needed (automatic fallback)

### Slow Training
1. Ensure PyTorch detects your GPU: `torch.cuda.is_available()`
2. Increase batch size for better GPU utilization
3. Enable mixed precision training

### Poor Performance
1. Increase model size: More layers, larger embedding dimension
2. More training epochs
3. Tune learning rate: Try 1e-4, 5e-4, 1e-3
4. More data (small datasets lead to overfitting)