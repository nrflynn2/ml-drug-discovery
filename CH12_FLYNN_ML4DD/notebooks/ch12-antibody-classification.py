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
#     display_name: ch12-protein-transformer-310
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transfer Learning with ESM-2 for Antibody Classification
#
# In section 12.3.3, we built a Transformer-based classifier from scratch and achieved 76% test accuracy
# on distinguishing HIV-1 vs. SARS-CoV-2 antibody sequences. While respectable for a small model trained
# on limited data, we can do much better by leveraging pre-trained protein language models.
#
# In this notebook, we'll use **ESM-2** (Evolutionary Scale Modeling), Meta AI's state-of-the-art protein
# language model trained on 65 million sequences. We'll demonstrate two approaches:
#
# 1. **Feature extraction** (frozen ESM-2): Use pre-trained embeddings, train only a classification head
# 2. **Fine-tuning** (unfrozen ESM-2): Adapt ESM-2's weights specifically for our task
#
# ## Learning Objectives
#
# By the end of this notebook, you will understand:
# 1. How to load and use pre-trained ESM-2 models from Hugging Face
# 2. How to extract rich protein sequence embeddings from ESM-2
# 3. The difference between feature extraction and fine-tuning
# 4. How transfer learning dramatically improves performance (76% → 89%)
# 5. How to visualize attention patterns learned by ESM-2
# 6. When to use pre-trained models vs. training from scratch

# %% [markdown]
# ## Setup and Installation
#
# First, let's install the necessary libraries and set up our environment.

# %%
# !pip install tqdm

# %%
# !pip install transformers

# %%
# Install required packages (uncomment if running on Colab)
# # !pip install -q transformers torch scikit-learn matplotlib seaborn pandas numpy

# Standard imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm

# Transformers for ESM-2
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

# Set random seeds for reproducibility
set_seeds()

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("PyTorch version:", torch.__version__)
device = get_device()
print("Device:", device)

# %% [markdown]
# ## Part 1: Load Pre-trained ESM-2 Model
#
# ESM-2 comes in multiple sizes. We'll use the 650M parameter version (`esm2_t33_650M_UR50D`), which
# offers an excellent balance of performance and computational efficiency:
#
# - **8M params** (t6): Fastest, good for prototyping
# - **35M params** (t12): Fast, decent performance
# - **150M params** (t30): Good balance, fits on most GPUs
# - **650M params** (t33): Best accuracy/cost tradeoff ← **We'll use this**
# - **3B params** (t36): State-of-the-art, requires more memory
# - **15B params** (t48): Best performance, requires substantial resources

# %%
# Load ESM-2 model and tokenizer from Hugging Face
model_name = "facebook/esm2_t6_8M_UR50D"

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
# ### Compare to Our From-Scratch Model
#
# Our small model from section 12.3.3 had:
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
# ## Part 2: Extract Embeddings from Example Sequences
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
# ## Part 3: Load Antibody Classification Data
#
# Let's load our training and test datasets. These are the same datasets used in section 12.3.3.

# %%
# Load data
df_train, classes = load_data("../data/bcr_train.parquet")
df_test, _ = load_data("../data/bcr_test.parquet")

print(f"Training set: {len(df_train)} sequences")
print(f"Test set: {len(df_test)} sequences")
print(f"\nClass distribution (training):")
print(df_train['target'].value_counts())
print(f"\nClass mapping: {classes}")

# %% [markdown]
# ## Part 4: Build ESM-2 Classifier Architecture
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

    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", num_classes=2, freeze_esm=True):
        super().__init__()

        # Load pre-trained ESM-2
        self.esm_model = AutoModel.from_pretrained(model_name)
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
# ## Part 5: Create Dataset and DataLoader
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
# ## Part 6: Training with Frozen ESM-2 (Feature Extraction)
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
# ### Train the Model
#
# We'll train for 10 epochs. With frozen ESM-2, training is fast!

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
# ### Results Analysis
#
# Compare to our from-scratch model from section 12.3.3:
#
# | Model | Test Accuracy | Test AUC | Parameters |
# |-------|--------------|----------|------------|
# | From scratch | 76.1% | 0.805 | 274K |
# | ESM-2 frozen | **~89%** | **~0.95** | 651M (656K trainable) |
#
# **A 13 percentage point improvement** by using pre-trained representations!

# %% [markdown]
# ## Part 12: Visualize Attention Patterns
#
# One of the most interpretable aspects of Transformers is attention. Let's visualize what
# ESM-2 focuses on when processing an antibody sequence.

# %%
def visualize_attention(model, sequence, tokenizer, device, layer=32, head=0):
    """
    Visualize attention patterns from a specific layer and head.

    Args:
        model: ESM-2 model
        sequence: Protein sequence string
        tokenizer: ESM tokenizer
        device: torch device
        layer: Which layer to visualize (0-32 for 650M model)
        head: Which attention head (0-19 for 650M model)
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
    attention = outputs.attentions[layer][0, head].cpu().numpy()

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
    ax.set_title(f"Attention Pattern - Layer {layer}, Head {head}\nSequence length: {len(tokens)}", fontsize=14)

    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    plt.show()

# Visualize attention for an example sequence
example_seq = df_test.iloc[0]['sequence'][:100]  # First 100 amino acids
print(f"Visualizing attention for sequence (length {len(example_seq)}):")
print(f"{example_seq[:80]}...")

visualize_attention(model_finetuned, example_seq, tokenizer, device, layer=32, head=5)

# %% [markdown]
# ### Interpreting Attention Patterns
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
# ## Summary and Key Takeaways
#
# ### What We Learned
#
# 1. **Pre-training matters**: ESM-2 dramatically outperforms training from scratch
#    - From scratch: 76% accuracy
#    - ESM-2 frozen: 89% accuracy (+13 points!)
#    - ESM-2 fine-tuned: 89-91% accuracy
#
# 2. **Scale matters**: 650M parameters trained on 65M sequences capture rich protein knowledge
#    - Secondary structure propensities
#    - Structural contacts
#    - Evolutionary constraints
#    - Functional motifs
#
# 3. **Feature extraction often sufficient**: Frozen ESM-2 performed nearly as well as fine-tuned
#    - Much faster training
#    - Lower risk of overfitting
#    - Good choice for small datasets
#
# 4. **Transfer learning is data-efficient**: Achieved SOTA with only 364 labeled examples
#
# ### When to Use Each Approach
#
# **Feature Extraction (Frozen ESM-2)**:
# - Limited labeled data (< 1000 examples)
# - Fast iteration needed
# - Limited computational resources
#
# **Fine-tuning (Unfrozen ESM-2)**:
# - More labeled data available (> 1000 examples)
# - Task-specific adaptation needed
# - Computational resources available
#
# **From Scratch**:
# - Massive task-specific data (millions of examples)
# - Highly specialized task unrelated to general protein properties
# - Strict computational constraints
# - Educational purposes
#
# ### Computational Requirements
#
# - **GPU Memory**: 16GB+ recommended for 650M model
# - **Training Time**: 10-30 minutes for 10 epochs (depends on frozen vs. fine-tuned)
# - **Inference**: ~0.1-1 second per sequence
#
# ### Next Steps
#
# - Try different ESM-2 model sizes (35M, 150M, 3B)
# - Apply to other protein classification tasks
# - Explore ESMFold for structure prediction
# - Use ESM-1v for mutation effect prediction (see next notebook!)

# %%
