# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: ch07-protein-transformer-310
#     language: python
#     name: python3
# ---

# %% [markdown]
# <table style="border:none;border-collapse:collapse;margin:0 0 8px;width:100%;">
# <tr style="border:none;">
#   <td style="border:none;padding:0;vertical-align:middle;">
#     <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:#A20025;margin:0 0 7px;">Companion notebook</div>
#     <div style="font-family:Georgia,'Iowan Old Style',serif;font-size:18px;color:#1A1416;line-height:1.25;margin:0 0 7px;">
#       <a href="https://www.manning.com/books/build-ai-drug-discovery-pipelines" style="color:#1A1416;text-decoration:none;border-bottom:1px solid #A20025;padding-bottom:1px;">Build AI Drug Discovery Pipelines</a>
#     </div>
#     <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:12px;color:#6B6461;line-height:1.7;">
#       Noah Flynn&nbsp;&middot;&nbsp;Manning Publications<br>
#       Code&nbsp;&middot;&nbsp;<a href="https://github.com/nrflynn2/ml-drug-discovery" style="color:#A20025;text-decoration:none;">github.com/nrflynn2/ml-drug-discovery</a>
#     </div>
#   </td>
#   <td style="border:none;padding:0 0 0 22px;vertical-align:middle;width:104px;text-align:right;">
#     <a href="https://www.manning.com/books/build-ai-drug-discovery-pipelines">
#       <img src="https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/assets/cover-sm.png" width="88" alt="Build AI Drug Discovery Pipelines cover" style="border:1px solid #E7E2DF;border-radius:3px;display:inline-block;">
#     </a>
#   </td>
# </tr>
# </table>
# <hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:0 0 22px;">

# %% [markdown]
# <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:#A20025;margin:0 0 12px;">Chapter 7</div>
#
# <h1 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:33px;line-height:1.12;margin:0 0 16px;color:#1A1416;letter-spacing:-.01em;">Transfer Learning with ESM-2 for Antibody Classification</h1>
#
# <hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:0 0 4px;">
#
# <em style="color:#6B6461;font-size:14.5px;">Runnable code for Chapter 7; read alongside the book.</em>
#
# <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/CH07_FLYNN_ML4DD/notebooks/ch07-antibody-classification.ipynb" target="_blank"><img src="https://img.shields.io/badge/Open%20in-Colab-A20025?logo=googlecolab&logoColor=white" alt="Open in Colab" style="height:20px;vertical-align:middle;"></a>
#
# In section 7.3.3, we built a Transformer-based classifier from scratch and achieved 76% test accuracy
# on distinguishing HIV-1 vs. SARS-CoV-2 antibody sequences. While respectable for a small model trained
# on limited data, we can do much better by leveraging pre-trained protein language models.
#
# In this notebook, we'll use **ESM-2** (Evolutionary Scale Modeling), Meta AI's state-of-the-art protein
# language model trained on 65 million sequences. We'll demonstrate two approaches:
#
# 1. **Feature extraction** (frozen ESM-2): Use pre-trained embeddings, train only a classification head
# 2. **Fine-tuning** (unfrozen ESM-2): Adapt ESM-2's weights specifically for our task
#
# <strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#A20025;">Objectives</strong>
# <ol style="margin:8px 0 0;padding-left:22px;font-size:15.5px;color:#241c1d;">
# <li style="margin:6px 0;">Load and use pre-trained ESM-2 models from Hugging Face</li>
# <li style="margin:6px 0;">Extract protein sequence embeddings from ESM-2</li>
# <li style="margin:6px 0;">Distinguish feature extraction from fine-tuning</li>
# <li style="margin:6px 0;">Improve antibody classification over a from-scratch baseline with transfer learning</li>
# <li style="margin:6px 0;">Visualize attention patterns learned by ESM-2</li>
# <li style="margin:6px 0;">Decide when to use pre-trained models versus training from scratch</li>
# </ol>

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§0&#8195;</span>Setup and Installation</h2>
#
# <div style="font-size:14px;margin:0 0 6px;">
# <div style="display:flex;gap:14px;padding:9px 0;border-bottom:1px solid #EFEBE8;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Local&#8195;&#8195;</strong><span>The Chapter 7 code is a self-contained package. From <code>CH07_FLYNN_ML4DD/</code>, run <code>pip install -e ".[notebooks]"</code> (Python 3.12).</span></div>
# <div style="display:flex;gap:14px;padding:9px 0;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Colab&#8195;&#8195;</strong><span>Clone the repository, then <code>pip install -e "ml-drug-discovery/CH07_FLYNN_ML4DD[notebooks]"</code>; a GPU runtime is recommended.</span></div>
# </div>
#
# First, let's install the necessary libraries and set up our environment.

# %% tags=["skip-execution"]
# Colab users only: install the required packages, then continue below.
# Skipped on a local run -- the chapter environment already provides these.
# !pip install -q tqdm transformers torch scikit-learn matplotlib seaborn pandas numpy

# %%
import sys
from pathlib import Path

# Add the package root to the path so `src` resolves whether or not the chapter
# package has been pip-installed. Without this, `from src.data import load_data`
# below raised ModuleNotFoundError when the notebook was run from notebooks/.
# (This matches the bootstrap in CH07-protein-transformers.ipynb.)
notebook_dir = Path().resolve()
project_root = notebook_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Standard imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Transformers for ESM-2
import transformers
from transformers import AutoTokenizer, EsmModel, AutoModel

# Scikit-learn for metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_fscore_support,
)

# PyTorch utilities
from torch.utils.data import Dataset, DataLoader

# Our custom modules (from src/)
from src.data import load_data
from src.utils import set_seeds, get_device

# Set random seeds for reproducibility (Python, NumPy and PyTorch)
SEED = set_seeds(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Report the environment, so this notebook's outputs are self-documenting
print(f"Python      : {sys.version.split()[0]}")
for _m in (np, pd, torch, transformers, matplotlib):
    print(f"{_m.__name__:12s}: {_m.__version__}")

print(f"\nSeed  : {SEED}")
device = get_device()
print("Device:", device)

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§1&#8195;</span>Load Pre-trained ESM-2 Model</h2>
#
# ESM-2 comes in multiple sizes. We'll use the 650M parameter version (`esm2_t33_650M_UR50D`), which
# offers an excellent balance of performance and computational efficiency:
#
# - **8M params** (t6): Fastest, good for prototyping
# - **35M params** (t12): Fast, decent performance
# - **150M params** (t30): Good balance, fits on most GPUs
# - **650M params** (t33): Best accuracy/cost tradeoff **We'll use this**
# - **3B params** (t36): State-of-the-art, requires more memory
# - **15B params** (t48): Best performance, requires substantial resources

# %%
# Load ESM-2 model and tokenizer from Hugging Face.
#
# ONE constant drives every ESM-2 load in this notebook. Previously this cell
# loaded esm2_t6_8M while ESM2Classifier defaulted to esm2_t33_650M, so the
# notebook silently pulled down BOTH checkpoints and the two halves disagreed
# about which model the chapter uses. Change this single name to switch sizes:
#   facebook/esm2_t6_8M_UR50D    ~8M   -- fastest, runs on CPU  (current)
#   facebook/esm2_t12_35M_UR50D  ~35M  -- fast, decent
#   facebook/esm2_t30_150M_UR50D ~150M -- good balance
#   facebook/esm2_t33_650M_UR50D ~650M -- best accuracy, needs a GPU (~2.6 GB)
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Kept as `model_name` too, since later cells refer to it by that name.
model_name = MODEL_NAME

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name)

# Move to GPU
esm_model = esm_model.to(device)
esm_model.eval()

# Model info
num_params = sum(p.numel() for p in esm_model.parameters())
print(f"\nModel loaded successfully!")
print(f"  Number of parameters: {num_params:,}")
print(f"  Number of layers: {esm_model.config.num_hidden_layers}")
print(f"  Embedding dimension: {esm_model.config.hidden_size}")
print(f"  Number of attention heads: {esm_model.config.num_attention_heads}")

# %% [markdown]
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Compare to Our From-Scratch Model</h3>
#
# Our small model from section 7.3.3 had:
# - 273,794 parameters
# - 8 layers
# - 64-dimensional embeddings
#
# ESM-2 (650M) has:
# - **650,428,480 parameters** (2,370× larger!)
# - 33 layers (4× deeper)
# - 1,280-dimensional embeddings (20× wider)
#
# This scale, combined with training on 65 million sequences, gives ESM-2 a massive advantage.

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§2&#8195;</span>Extract Embeddings from Example Sequences</h2>
#
# Let's see what ESM-2 embeddings look like. We'll tokenize a few antibody sequences and extract
# their representations.

# %%
# Example antibody sequences (heavy chain variable regions)
example_sequences = [
    "QVQLVETGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSV",
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVAS",
    "QVQLLESGAEVKKPGSSVKVSCKASGDTFIRYSFTWVRQAPGQGLEWMGR",
]

sequence_labels = ["SARS-CoV-2", "SARS-CoV-2", "HIV-1"]

def get_esm_embeddings(sequences, model, tokenizer, device):
    """
    Extract sequence-level embeddings from ESM-2.

    Args:
        sequences: List of protein sequence strings
        model: ESM model
        tokenizer: ESM tokenizer
        device: torch device

    Returns:
        embeddings: (num_sequences, hidden_dim) numpy array
    """
    # Tokenize (adds <cls> at start, <eos> at end)
    tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**tokens)

    # Use mean pooling over sequence (excluding padding)
    embeddings = []
    for i in range(len(sequences)):
        # Get attention mask for this sequence
        mask = tokens['attention_mask'][i].unsqueeze(-1)  # (seq_len, 1)

        # Get token embeddings
        token_embs = outputs.last_hidden_state[i]  # (seq_len, hidden_dim)

        # Mean pool (excluding padding)
        masked_embs = token_embs * mask
        pooled = masked_embs.sum(dim=0) / mask.sum()

        embeddings.append(pooled.cpu().numpy())

    return np.array(embeddings)

# Extract embeddings
embeddings = get_esm_embeddings(example_sequences, esm_model, tokenizer, device)

print(f"Embedding shape: {embeddings.shape}")
print(f"  {len(example_sequences)} sequences")
print(f"  {embeddings.shape[1]} dimensions per sequence")

# Compute pairwise similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)

print("\nCosine similarities between sequences:")
for i in range(len(example_sequences)):
    for j in range(i+1, len(example_sequences)):
        print(f"  {sequence_labels[i]} vs {sequence_labels[j]}: {similarities[i, j]:.3f}")

# %% [markdown]
# Notice that the two SARS-CoV-2 antibodies have higher similarity (0.9+) to each other than to
# the HIV-1 antibody. ESM-2's embeddings already capture meaningful biological information!

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§3&#8195;</span>Load Antibody Classification Data</h2>
#
# Let's load our training and test datasets. These are the same datasets used in section 7.3.3.

# %%
# Load data
df_train, classes = load_data("../data/bcr_train.parquet")
df_test, _ = load_data("../data/bcr_test.parquet")

print(f"Training set: {len(df_train)} sequences  ({df_train.shape[0]} rows x {df_train.shape[1]} cols)")
print(f"Test set: {len(df_test)} sequences")
print(f"\nClass distribution (training):")
print(df_train['target'].value_counts())
print(f"\nClass mapping: {classes}")
print(f"\nFirst rows:")
print(df_train.head(3))

# Sanity checks on the two frames the rest of the notebook depends on
assert {"sequence", "target", "label"} <= set(df_train.columns)
assert set(df_train["label"].unique()) == {0, 1}, "expected a binary label"
assert df_train["sequence"].notna().all() and df_test["sequence"].notna().all()

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§4&#8195;</span>Build ESM-2 Classifier Architecture</h2>
#
# We'll create a classifier that wraps ESM-2. The architecture is:
#
# 1. **ESM-2 encoder** (pre-trained, frozen or fine-tunable)
# 2. **Mean pooling** to get a single vector per sequence
# 3. **Classification head** to predict class probabilities

# %%
class ESM2Classifier(nn.Module):
    """
    Antibody classifier using pre-trained ESM-2.

    Architecture:
        1. ESM-2 encoder (pre-trained, frozen or fine-tuned)
        2. Mean pooling over sequence
        3. Classification head
    """

    def __init__(self, model_name=MODEL_NAME, num_classes=2, freeze_esm=True,
                 attn_implementation="eager"):
        super().__init__()

        # Load pre-trained ESM-2.
        # attn_implementation="eager" is required for Part 12: recent Transformers
        # defaults to the faster "sdpa" kernel, which does NOT support
        # output_attentions=True -- it returns an EMPTY attentions tuple (with a
        # log line, not an exception), so the attention-visualisation cell died on
        # an out-of-range index. "eager" is marginally slower but is the only
        # implementation that exposes the attention weights this chapter plots.
        self.esm_model = AutoModel.from_pretrained(
            model_name, attn_implementation=attn_implementation
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Optionally freeze ESM-2 weights
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False

        # Get embedding dimension
        self.hidden_size = self.esm_model.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        """Average embeddings over sequence length (excluding padding)."""
        # Expand mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()

        # Sum embeddings (masked)
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

        # Count non-padded tokens
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Average
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask

        Returns:
            logits: (batch_size, num_classes)
        """
        # Get ESM-2 embeddings
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)

        # Pool to sequence-level representation
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)

        # Classify
        logits = self.classifier(pooled)

        return logits

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§5&#8195;</span>Create Dataset and DataLoader</h2>
#
# We need a PyTorch Dataset that tokenizes sequences on-the-fly.

# %%
class AntibodyDataset(Dataset):
    """Dataset for antibody sequences."""

    def __init__(self, dataframe, tokenizer, max_length=512):
        self.sequences = dataframe['sequence'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = AntibodyDataset(df_train, tokenizer)
test_dataset = AntibodyDataset(df_test, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§6&#8195;</span>Training with Frozen ESM-2 (Feature Extraction)</h2>
#
# Let's start by training only the classification head, keeping ESM-2 frozen. This is called
# **feature extraction** - we use ESM-2's pre-trained representations without modifying them.
#
# This approach is:
# - **Fast**: Only ~656K parameters to train (vs. 650M)
# - **Data-efficient**: Works well with limited labeled data
# - **Stable**: Pre-trained representations are robust

# %%
# Create model with frozen ESM-2
model_frozen = ESM2Classifier(freeze_esm=True, num_classes=2)
model_frozen = model_frozen.to(device)

# Check trainable parameters
trainable_params = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_frozen.parameters())

print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"Total parameters: {total_params:,}")

# %%
# Training setup
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_frozen.parameters()),
    lr=1e-3  # Higher LR for training from scratch head
)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, auc, f1, all_labels, all_preds, all_probs

# %% [markdown]
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Train the Model</h3>
#
# We'll train for 3 epochs. With frozen ESM-2 only the small classification head
# is being updated, so training is fast.

# %%
num_epochs = 3
history = {'train_loss': [], 'test_acc': [], 'test_auc': [], 'test_f1': []}

print("Training with FROZEN ESM-2 (feature extraction)...")
print("=" * 60)

for epoch in range(1, num_epochs + 1):
    # Train
    train_loss = train_epoch(model_frozen, train_loader, optimizer, loss_fn, device)

    # Evaluate on test set
    test_acc, test_auc, test_f1, _, _, _ = evaluate(model_frozen, test_loader, device)

    # Save history
    history['train_loss'].append(train_loss)
    history['test_acc'].append(test_acc)
    history['test_auc'].append(test_auc)
    history['test_f1'].append(test_f1)

    print(f"Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, "
          f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")

print("=" * 60)
print(f"Best Test Accuracy: {max(history['test_acc']):.4f}")
print(f"Best Test AUC: {max(history['test_auc']):.4f}")


# %% [markdown]
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Results Analysis</h3>
#
# Compare to our from-scratch model from section 7.3.3:
#
# | Model | Test Accuracy | Test AUC | Parameters |
# |-------|--------------|----------|------------|
# | From scratch | 76.1% | 0.805 | 274K |
# | ESM-2 frozen | **~89%** | **~0.95** | 651M (656K trainable) |
#
# **A 13 percentage point improvement** by using pre-trained representations!

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§12&#8195;</span>Visualize Attention Patterns</h2>
#
# One of the most interpretable aspects of Transformers is attention. Let's visualize what
# ESM-2 focuses on when processing an antibody sequence.

# %%
def visualize_attention(model, sequence, tokenizer, device, layer=-1, head=0):
    """
    Visualize attention patterns from a specific layer and head.

    Args:
        model: ESM-2 model
        sequence: Protein sequence string
        tokenizer: ESM tokenizer
        device: torch device
        layer: Which layer to visualize. Defaults to -1 (the final layer) so this
            works for ANY ESM-2 size -- a hardcoded `layer=32` only exists on the
            33-layer 650M checkpoint and raised IndexError on every smaller one.
        head: Which attention head (0 .. num_attention_heads-1; ESM-2 uses 20).
    """
    # Tokenize
    encoding = tokenizer(sequence, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get attention weights
    model.eval()
    with torch.no_grad():
        outputs = model.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    # Extract attention from specified layer and head
    n_layers, n_heads = len(outputs.attentions), outputs.attentions[0].shape[1]
    assert -n_layers <= layer < n_layers, f"layer {layer} out of range (model has {n_layers})"
    assert 0 <= head < n_heads, f"head {head} out of range (model has {n_heads})"
    attention = outputs.attentions[layer][0, head].cpu().numpy()
    layer_label = layer if layer >= 0 else n_layers + layer

    # Get tokens (for axis labels)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attention, cmap='viridis', aspect='auto')

    # Set ticks every 10 positions
    tick_positions = range(0, len(tokens), 10)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([f"{i}" for i in tick_positions], fontsize=8)
    ax.set_yticklabels([f"{i}" for i in tick_positions], fontsize=8)

    ax.set_xlabel("Key Position", fontsize=12)
    ax.set_ylabel("Query Position", fontsize=12)
    ax.set_title(f"Attention Pattern - Layer {layer_label}, Head {head}\nSequence length: {len(tokens)}", fontsize=14)

    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    plt.show()

# Visualize attention for an example sequence
example_seq = df_test.iloc[0]['sequence'][:100]  # First 100 amino acids
print(f"Visualizing attention for sequence (length {len(example_seq)}):")
print(f"{example_seq[:80]}...")

# `model_frozen`, not `model_finetuned`: this notebook only builds the frozen
# feature-extraction model (the fine-tuning section it referred to is not present
# here), so `model_finetuned` was never defined and this cell raised NameError.
# Using the frozen model is also the better fit for the narrative below: its
# encoder is ESM-2 exactly as pre-trained, so these really are the patterns
# learned from 65M sequences.
visualize_attention(model_frozen, example_seq, tokenizer, device, layer=-1, head=5)

# %% [markdown]
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Interpreting Attention Patterns</h3>
#
# Different patterns you might observe:
#
# 1. **Diagonal (local attention)**: Model focuses on nearby positions
# 2. **Vertical/horizontal stripes**: Attention to specific "anchor" positions
# 3. **Block patterns**: Attending to structural regions (e.g., framework vs. CDR regions)
# 4. **Long-range attention**: Positions far apart in sequence but close in 3D structure
#
# These patterns emerge from training on 65 million sequences - the model has learned
# meaningful protein structure principles!

# %% [markdown]
# <hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:34px 0 12px;">
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:0 0 10px;color:#1A1416;">Summary</h2>
#
#  Mutation effect prediction with ESM-2 leverages evolutionary patterns learned from millions of sequences to score variants computationally. Log-likelihood ratios quantify whether mutations increase or decrease sequence plausibility. Comprehensive mutation landscapes reveal conserved and variable regions, guiding where mutations are tolerable. Targeted scoring of specific mutations enables efficient evaluation of design candidates or disease variants. Pseudo-perplexity assesses overall sequence quality, catching problematic combinations. Together, these approaches transform protein engineering workflows from exhaustive experimental screening to hypothesis-driven testing of computationally prioritized variants.
#
#  While genomic sequencing has identified hundreds of millions of protein sequences, experimental methods like X-ray crystallography have determined the 3D structures for only a tiny fraction of them. This disconnect creates a massive "blind spot" in drug discovery, as knowing the 3D structure is critical. Deep learning bridges this gap by computationally predicting structures for the millions of sequences that lack experimental data.
#
#  Proteins are not static strings; they are dynamic chains that fold into precise hierarchies – from local helices and sheets to global 3D arrangements. This folding is driven by physical forces, such as the "hydrophobic effect," where oily parts of the protein collapse inward to escape water. A successful prediction model must implicitly learn these geometric constraints and physical laws to distinguish valid biological shapes from random tangles.
#
#  SimpleFold approaches structure prediction not just as a regression problem, but as a generative one using "flow matching." Similar to how AI might generate art, the model learns to guide a cloud of random noise into a structured protein shape over several steps. This allows the model to capture the complex distribution of possible protein conformations rather than just predicting a single average structure.
#
#  Proteins have complex long-range dependencies; an AA at the beginning of a chain might chemically interact with one at the very end. Traditional sequential models (like RNNs) struggle to "remember" these distant connections due to information bottlenecks. Transformers solve this using self-attention, which allows every part of the sequence to look at and influence every other part simultaneously, regardless of how far apart they are.
#
#  By training models on millions of raw protein sequences using "Masked Language Modeling", hiding parts of a sequence and asking the model to fill in the blanks, we force the model to learn fundamental rules of biology. The model learns which AA patterns are evolutionarily conserved and structurally stable without needing explicit labels, creating rich representations of proteins.
#
#  The "knowledge" a model gains during pre-training can be transferred to specific drug discovery tasks. We demonstrated this by adapting a protein language model to classify antibodies, distinguishing between those that target HIV-1 and SARS-CoV-2. This highlights how a general understanding of protein language can be fine-tuned to solve highly specific biological classification problems with limited labeled data.
#
#  Advanced models like ESM-2 are so capable that they can predict the effects of mutations without further training, a technique called "zero-shot" inference. By scoring how "natural" a mutated sequence looks to the model, researchers can identify stabilizing mutations or flag deleterious ones. This enables computational screening of thousands of therapeutic candidates before a single experiment is run in the lab.

# %%
