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
# <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:#A20025;margin:0 0 12px;">Chapter 12</div>
#
# <h1 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:33px;line-height:1.12;margin:0 0 16px;color:#1A1416;letter-spacing:-.01em;">Training a Protein Language Model with Transformers</h1>
#
# <hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:0 0 4px;">
#
# <em style="color:#6B6461;font-size:14.5px;">Runnable code for Chapter 12; read alongside the book.</em>
#
# <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/CH12_FLYNN_ML4DD/notebooks/CH12-protein-transformers.ipynb" target="_blank"><img src="https://img.shields.io/badge/Open%20in-Colab-A20025?logo=googlecolab&logoColor=white" alt="Open in Colab" style="height:20px;vertical-align:middle;"></a>
#
# In this notebook, we'll walk through the process of training a small protein language
# model using the Transformer architecture. This model learns representations of protein
# sequences by predicting masked amino acids, similar to how BERT learns language representations.
#
# <strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#A20025;">Objectives</strong>
# <ol style="margin:8px 0 0;padding-left:22px;font-size:15.5px;color:#241c1d;">
# <li style="margin:6px 0;">Tokenize protein sequences for Transformer models</li>
# <li style="margin:6px 0;">Build the architecture of a protein language model</li>
# <li style="margin:6px 0;">Apply masked language modeling to protein sequences</li>
# <li style="margin:6px 0;">Train and evaluate a Transformer on protein sequences</li>
# <li style="margin:6px 0;">Use modern training practices: warmup, gradient clipping, cosine scheduling</li>
# </ol>

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§0&#8195;</span>Setup and Imports</h2>
#
# <div style="font-size:14px;margin:0 0 6px;">
# <div style="display:flex;gap:14px;padding:9px 0;border-bottom:1px solid #EFEBE8;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Local&#8195;&#8195;</strong><span>The Chapter 12 code is a self-contained package. From <code>CH12_FLYNN_ML4DD/</code>, run <code>pip install -e ".[notebooks]"</code> (Python 3.12).</span></div>
# <div style="display:flex;gap:14px;padding:9px 0;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Colab&#8195;&#8195;</strong><span>Clone the repository, then <code>pip install -e "ml-drug-discovery/CH12_FLYNN_ML4DD[notebooks]"</code>; a GPU runtime is recommended.</span></div>
# </div>
#
# First, let's import the necessary libraries and modules:

# %%
import sys
from pathlib import Path

# Add parent directory to path so we can import src and scripts modules
notebook_dir = Path().resolve()
project_root = notebook_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import math

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Our custom modules
from src.data import Tokenizer, ProteinSequenceDataset, mlm_collate_fn, mask_sequences
from src.model import ProteinLanguageModel, MultiheadAttention, EncoderLayer
from src.utils import set_seeds, get_device, scale_dot_product_attention
from scripts.train_plm import get_cosine_schedule_with_warmup, train_epoch, val_epoch, get_dataloaders

# Set random seeds for reproducibility (Python, NumPy and PyTorch)
SEED = set_seeds(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 4)

# Report the environment, so this notebook's outputs are self-documenting
print(f"Python      : {sys.version.split()[0]}")
for _m in (np, pd, torch, matplotlib, sns):
    print(f"{_m.__name__:12s}: {_m.__version__}")
print(f"\nSeed  : {SEED}")
print(f"Device: {get_device()}   (CPU is fine here -- the model below is ~0.8M parameters)")

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§1&#8195;</span>Understanding Protein Tokenization</h2>
#
# Unlike natural language, proteins have a well-defined vocabulary: the 20 canonical amino acids.
# We'll add special tokens for sequence boundaries and masking.

# %%
# Initialize the tokenizer
tokenizer = Tokenizer()

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens:")
print(f"  - CLS token: '<cls>' (ID: {tokenizer.cls_token_id})")
print(f"  - PAD token: '<pad>' (ID: {tokenizer.pad_token_id})")
print(f"  - EOS token: '<eos>' (ID: {tokenizer.eos_token_id})")
print(f"  - MASK token: '<mask>' (ID: {tokenizer.mask_token_id})")
print(f"\nAmino acids in vocabulary:")
print(tokenizer.token_to_index)

# %% [markdown]
# Let's tokenize some example protein sequences:

# %%
# Example sequences (small peptides for demonstration)
example_sequences = [
    "ACDEFGHIKLMNPQRSTVWY",  # All 20 amino acids
    "MKTAYIAKQRQISFVKSHFS",  # Example sequence 1
    "KALTARQQEVFDLIRD",      # Example sequence 2 (shorter)
]

# Tokenize the sequences
tokenized = tokenizer(example_sequences, padding=True)

print("Tokenized sequences:")
for i, seq in enumerate(example_sequences):
    print(f"\nSequence {i+1}: {seq}")
    print(f"  Length: {len(seq)}")
    print(f"  Tokens: {tokenized['input_ids'][i][:len(seq)+2]}")  # Show first few tokens
    print(f"  Attention mask: {tokenized['attention_mask'][i][:len(seq)+2]}")

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§2&#8195;</span>Understanding Scaled Dot-Product Attention</h2>
#
# The core of the Transformer is the attention mechanism. Let's visualize how it works:

# %%
def visualize_attention(attention_weights, seq_length, title="Attention Pattern"):
    """Visualize attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Take first head, first batch for visualization
    attn = attention_weights[0, 0, :seq_length, :seq_length].detach().cpu().numpy()

    sns.heatmap(attn, annot=True, fmt='.2f', cmap='viridis',
                square=True, cbar_kws={'label': 'Attention Weight'})
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    plt.tight_layout()
    return fig

# Create a small example
seq_len = 6
d_k = 8
batch_size = 1
num_heads = 1

# Random query, key, value vectors
torch.manual_seed(42)
q = torch.randn(batch_size, num_heads, seq_len, d_k)
k = torch.randn(batch_size, num_heads, seq_len, d_k)
v = torch.randn(batch_size, num_heads, seq_len, d_k)

# Compute attention
values, attention = scale_dot_product_attention(q, k, v)

print("Input shapes:")
print(f"  Query (Q): {q.shape}")
print(f"  Key (K): {k.shape}")
print(f"  Value (V): {v.shape}")
print(f"\nOutput shapes:")
print(f"  Attention weights: {attention.shape}")
print(f"  Output values: {values.shape}")

# Visualize
visualize_attention(attention, seq_len, "Scaled Dot-Product Attention Weights")
plt.show()

# %% [markdown]
# Notice how each row sums to 1.0 - these are the attention weights that determine
# how much each position attends to every other position.

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§3&#8195;</span>Multi-Head Attention</h2>
#
# Multi-head attention allows the model to attend to different types of patterns simultaneously.
# Let's see it in action:

# %%
# Create a multi-head attention layer
embedding_dim = 64
num_heads = 4
mha = MultiheadAttention(embedding_dim=embedding_dim, num_heads=num_heads)

# Create example input (batch_size=2, seq_len=10, embedding_dim=64)
x = torch.randn(2, 10, embedding_dim)

# Forward pass
output, attention_weights = mha(x, return_attention=True)

print("Multi-Head Attention:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Attention weights shape: {attention_weights.shape}")
print(f"    (batch_size, num_heads, seq_len, seq_len)")

# Visualize different attention heads
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
seq_len = 10

for head_idx in range(num_heads):
    ax = axes[head_idx]
    attn = attention_weights[0, head_idx, :, :].detach().cpu().numpy()

    sns.heatmap(attn, annot=False, cmap='viridis', square=True,
                cbar=True, ax=ax, vmin=0, vmax=0.3)
    ax.set_title(f'Head {head_idx + 1}')
    ax.set_xlabel('Key Position')
    if head_idx == 0:
        ax.set_ylabel('Query Position')

plt.suptitle('Different Attention Patterns in Each Head', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# Each head learns to attend to different patterns. Some heads might focus on local context,
# while others capture long-range dependencies.

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§4&#8195;</span>The Complete Transformer Encoder Layer</h2>
#
# An encoder layer combines multi-head attention with a feed-forward network:

# %%
# Create an encoder layer
encoder_layer = EncoderLayer(
    embedding_dim=64,
    num_heads=4,
    ffn_dim=256,
    dropout=0.1
)

# Forward pass
x = torch.randn(2, 10, 64)
output = encoder_layer(x)

print("Transformer Encoder Layer:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"\nLayer components:")
print(f"  1. Multi-head attention")
print(f"  2. Add & Norm (residual connection + layer normalization)")
print(f"  3. Feed-forward network (Linear -> ReLU -> Dropout -> Linear)")
print(f"  4. Add & Norm (residual connection + layer normalization)")

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§5&#8195;</span>Masked Language Modeling</h2>
#
# The key to training a protein language model is masked language modeling (MLM).
# We randomly mask some amino acids and train the model to predict them.

# %%
# Load some example sequences
df = pd.read_parquet("../data/bcr_train.parquet")
print(f"Loaded {len(df)} protein sequences  ({df.shape[0]} rows x {df.shape[1]} cols)")
print(f"Sequence length: min {df['sequence'].str.len().min()}, "
      f"max {df['sequence'].str.len().max()}, "
      f"mean {df['sequence'].str.len().mean():.0f} residues")
print(f"\nFirst few sequences:")
print(df["sequence"].head())

# Sanity checks before we build a model around this frame
assert {"sequence", "target"} <= set(df.columns), f"unexpected columns: {list(df.columns)}"
assert df["sequence"].notna().all(), "found empty sequences"
_alphabet = set("ACDEFGHIKLMNPQRSTVWY")
_non_canonical = {c for s in df["sequence"] for c in s.upper()} - _alphabet
print(f"\nNon-canonical residues present: {_non_canonical or 'none'} "
      f"(any would tokenize to <unk>)")

# Take a few sequences for demonstration
demo_sequences = df["sequence"].iloc[:5].tolist()

# Tokenize them
tokenized = tokenizer(demo_sequences, padding=True)
input_ids = torch.tensor(tokenized["input_ids"])

print(f"\nTokenized shape: {input_ids.shape}")

# Apply masking
special_token_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id]
masked_input_ids, labels, mask_positions = mask_sequences(
    input_ids=input_ids,
    mask_token_id=tokenizer.mask_token_id,
    vocab_size=tokenizer.vocab_size,
    special_token_ids=special_token_ids,
    mask_prob=0.15,
)

print(f"\nMasking statistics:")
print(f"  Total tokens: {input_ids.numel()}")
print(f"  Masked tokens: {(labels != -100).sum().item()}")
print(f"  Masking percentage: {(labels != -100).sum().item() / input_ids.numel() * 100:.1f}%")

# %% [markdown]
# Let's visualize what masking looks like for one sequence:

# %%
def decode_tokens(token_ids, tokenizer):
    """Convert token IDs back to amino acids."""
    return ''.join([tokenizer.index_to_token.get(int(tid), '?') for tid in token_ids])

# Take first sequence
seq_idx = 0
original_ids = input_ids[seq_idx]
masked_ids = masked_input_ids[seq_idx]
target_labels = labels[seq_idx]

# Find non-padding positions
non_pad = (original_ids != tokenizer.pad_token_id)
seq_length = non_pad.sum().item()

print("Original sequence:")
print(decode_tokens(original_ids[:seq_length], tokenizer))
print("\nMasked sequence:")
print(decode_tokens(masked_ids[:seq_length], tokenizer))
print("\nPositions to predict (green = masked):")

# Show which positions are masked
for i in range(seq_length):
    original_token = tokenizer.index_to_token[int(original_ids[i])]
    masked_token = tokenizer.index_to_token[int(masked_ids[i])]
    is_masked = target_labels[i] != -100

    if is_masked:
        print(f"  Position {i:2d}: {masked_token:6s} -> {original_token:6s} ✓")

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§6&#8195;</span>Building the Protein Language Model</h2>
#
# Now let's create a complete protein language model:

# %%
# Model hyperparameters
model_config = {
    "vocab_size": tokenizer.vocab_size,
    "padding_idx": tokenizer.pad_token_id,
    "embedding_dim": 128,
    "num_layers": 4,
    "num_heads": 4,
    "ffn_dim": 512,
    "dropout": 0.1,
}

# Create model
model = ProteinLanguageModel(**model_config)
device = get_device()
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Protein Language Model Architecture:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"\nModel configuration:")
for key, value in model_config.items():
    print(f"  {key}: {value}")

# %% [markdown]
# Let's do a forward pass to see what the model outputs:

# %%
# Create a small batch
batch_sequences = demo_sequences[:2]
tokenized = tokenizer(batch_sequences, padding=True)

batch = {
    "input_ids": torch.tensor(tokenized["input_ids"], device=device),
    "attention_mask": torch.tensor(tokenized["attention_mask"], device=device),
}

# Forward pass
with torch.no_grad():
    logits = model(batch)

print("Model output:")
print(f"  Input shape: {batch['input_ids'].shape} (batch_size, seq_len)")
print(f"  Output shape: {logits.shape} (batch_size, seq_len, vocab_size)")
print(f"\nFor each position, the model outputs a probability distribution over {tokenizer.vocab_size} tokens")

# Show predictions for first sequence
seq_len = (batch['attention_mask'][0] == 1).sum()
predicted_ids = torch.argmax(logits[0, :seq_len], dim=-1)

print(f"\nExample predictions (before training, random):")
print(f"  Input:     {decode_tokens(batch['input_ids'][0, :seq_len], tokenizer)}")
print(f"  Predicted: {decode_tokens(predicted_ids, tokenizer)}")

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§7&#8195;</span>Training the Model</h2>
#
# Now let's train the model with modern best practices:
# - AdamW optimizer with weight decay
# - Learning rate warmup
# - Cosine annealing schedule
# - Gradient clipping

# %%
# Training configuration
train_config = {
    "batch_size": 1,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "num_epochs": 10,  # Small number for demo
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "mask_prob": 0.15,
    "val_size": 0.15,
}

print("Training configuration:")
for key, value in train_config.items():
    print(f"  {key}: {value}")

# Create dataloaders
train_dl, val_dl = get_dataloaders(
    df=df,
    val_size=train_config["val_size"],
    tokenizer=tokenizer,
    device=device,
    batch_size=train_config["batch_size"],
    mask_prob=train_config["mask_prob"],
)

print(f"\nDataset splits:")
print(f"  Training batches: {len(train_dl)}")
print(f"  Validation batches: {len(val_dl)}")
print(f"  Training samples: {len(train_dl.dataset)}")
print(f"  Validation samples: {len(val_dl.dataset)}")

# %% [markdown]
# Set up the optimizer and learning rate scheduler:

# %%
# Loss function (ignores -100 labels for non-masked positions)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config["lr"],
    weight_decay=train_config["weight_decay"],
)

# Learning rate scheduler with warmup
num_training_steps = len(train_dl) * train_config["num_epochs"]
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=train_config["warmup_steps"],
    num_training_steps=num_training_steps,
)

print(f"Total training steps: {num_training_steps}")
print(f"Warmup steps: {train_config['warmup_steps']}")

# Visualize learning rate schedule
lrs = []
for _ in range(num_training_steps):
    lrs.append(scheduler.get_last_lr()[0])
    optimizer.step()
    scheduler.step()

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule (Warmup + Cosine Decay)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Reset optimizer and scheduler for actual training
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config["lr"],
    weight_decay=train_config["weight_decay"],
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=train_config["warmup_steps"],
    num_training_steps=num_training_steps,
)

# %% [markdown]
# Train the model:

# %%
# Training loop
history = {
    "train_loss": [],
    "train_perplexity": [],
    "val_loss": [],
    "val_perplexity": [],
    "val_accuracy": [],
}

print("Starting training...")
print("-" * 80)

best_val_loss = float('inf')

for epoch in range(1, train_config["num_epochs"] + 1):
    # Train
    train_loss, train_ppl = train_epoch(
        model=model,
        train_dl=train_dl,
        loss_fn=loss_fn,
        opt=optimizer,
        scheduler=scheduler,
        max_grad_norm=train_config["max_grad_norm"],
    )

    # Validate
    val_loss, val_ppl, val_acc = val_epoch(
        model=model,
        val_dl=val_dl,
        loss_fn=loss_fn,
    )

    # Track history
    history["train_loss"].append(train_loss)
    history["train_perplexity"].append(train_ppl)
    history["val_loss"].append(val_loss)
    history["val_perplexity"].append(val_ppl)
    history["val_accuracy"].append(val_acc)

    # Update best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

    # Print progress
    print(
        f"Epoch {epoch:2d}/{train_config['num_epochs']}: "
        f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:6.2f} | "
        f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:6.2f}, Val Acc: {val_acc:.4f}"
    )

print("-" * 80)
print(f"Training complete! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

# %% [markdown]
# Visualize training progress:

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot loss
ax = axes[0]
ax.plot(history["train_loss"], label="Train", marker='o', markersize=4)
ax.plot(history["val_loss"], label="Validation", marker='s', markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot perplexity
ax = axes[1]
ax.plot(history["train_perplexity"], label="Train", marker='o', markersize=4)
ax.plot(history["val_perplexity"], label="Validation", marker='s', markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Perplexity')
ax.set_title('Perplexity (lower is better)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot accuracy
ax = axes[2]
ax.plot(history["val_accuracy"], label="Validation", marker='s', markersize=4, color='green')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Accuracy on Masked Tokens')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§8&#8195;</span>Evaluating the Trained Model</h2>
#
# Let's see how well our model can predict masked amino acids:

# %%
model.eval()

# Take some test sequences
test_sequences = df["sequence"].iloc[100:103].tolist()

for seq_idx, sequence in enumerate(test_sequences):
    print(f"\nSequence {seq_idx + 1}: {sequence}")
    print("-" * 60)

    # Tokenize
    tokenized = tokenizer([sequence], padding=True)
    input_ids = torch.tensor(tokenized["input_ids"], device=device)
    attention_mask = torch.tensor(tokenized["attention_mask"], device=device)

    # Manually mask a few positions for demonstration
    original_ids = input_ids.clone()
    seq_len = (attention_mask[0] == 1).sum()

    # Mask positions 3, 7, 11 (if they exist)
    mask_positions = [3, 7, 11]
    mask_positions = [p for p in mask_positions if p < seq_len]

    for pos in mask_positions:
        input_ids[0, pos] = tokenizer.mask_token_id

    # Get predictions
    with torch.no_grad():
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits = model(batch)
        predictions = torch.argmax(logits, dim=-1)

    # Show results
    print("Predictions:")
    for pos in mask_positions:
        original_aa = tokenizer.index_to_token[int(original_ids[0, pos])]
        predicted_aa = tokenizer.index_to_token[int(predictions[0, pos])]

        # Get top 3 predictions with probabilities
        probs = torch.softmax(logits[0, pos], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=3)

        top_aas = [tokenizer.index_to_token[int(idx)] for idx in top_indices]
        top_prob_values = [float(p) for p in top_probs]

        match = "✓" if predicted_aa == original_aa else "✗"
        print(f"  Position {pos}: {original_aa} -> {predicted_aa} {match}")
        print(f"    Top 3: {top_aas[0]}({top_prob_values[0]:.3f}), "
              f"{top_aas[1]}({top_prob_values[1]:.3f}), "
              f"{top_aas[2]}({top_prob_values[2]:.3f})")

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§9&#8195;</span>Visualizing Learned Attention Patterns</h2>
#
# One of the most interesting aspects of Transformers is that we can visualize
# what the model has learned to attend to:

# %%
# Get attention weights for a sequence
test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEDESGK"
tokenized = tokenizer([test_seq], padding=False)

batch = {
    "input_ids": torch.tensor(tokenized["input_ids"], device=device),
    "attention_mask": torch.tensor(tokenized["attention_mask"], device=device),
}

# Get attention weights from the encoder
model.eval()
with torch.no_grad():
    x = model.embedding(batch["input_ids"])
    x = model.pe(x)
    attention_maps = model.encoder.get_attentions(x, batch["attention_mask"])

# Visualize attention from the last layer
last_layer_attention = attention_maps[-1]  # Shape: (1, num_heads, seq_len, seq_len)
seq_len = (batch["attention_mask"][0] == 1).sum().item()

# Plot all heads from the last layer
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for head_idx in range(4):
    ax = axes[head_idx]
    attn = last_layer_attention[0, head_idx, 1:seq_len-1, 1:seq_len-1].cpu().numpy()

    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_title(f'Layer 4, Head {head_idx + 1}')
    ax.set_xlabel('Key Position')
    if head_idx == 0:
        ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f'Attention Patterns in Final Layer\nSequence: {test_seq[:30]}...',
             fontsize=12, y=1.05)
plt.tight_layout()
plt.show()

# %%
#import matplotlib.pyplot as plt
#import cmocean  # You may need to: pip install cmocean

# Get attention weights for a sequence
test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEDESGK"
tokenized = tokenizer([test_seq], padding=False)

batch = {
    "input_ids": torch.tensor(tokenized["input_ids"], device=device),
    "attention_mask": torch.tensor(tokenized["attention_mask"], device=device),
}

# Get attention weights from the encoder
model.eval()
with torch.no_grad():
    x = model.embedding(batch["input_ids"])
    x = model.pe(x)
    attention_maps = model.encoder.get_attentions(x, batch["attention_mask"])

# Visualize attention from the last layer
last_layer_attention = attention_maps[-1]  # Shape: (1, num_heads, seq_len, seq_len)
seq_len = (batch["attention_mask"][0] == 1).sum().item()

# Set up figure with larger size for publication quality
fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=600)

# Configure font sizes for publication
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 14
SUPTITLE_SIZE = 20

for head_idx in range(4):
    ax = axes[head_idx]
    attn = last_layer_attention[0, head_idx, 1:seq_len-1, 1:seq_len-1].cpu().numpy()

    # Use cmocean diff_r colormap
    im = ax.imshow(attn, cmap="viridis", aspect='auto')
    
    # Set larger font sizes
    ax.set_title(f'Layer 4, Head {head_idx + 1}', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=LABEL_SIZE)
    if head_idx == 0:
        ax.set_ylabel('Query Position', fontsize=LABEL_SIZE)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    # Add colorbar with larger font
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

plt.suptitle(f'Attention Patterns in Final Layer\nSequence: {test_seq[:30]}...',
             fontsize=SUPTITLE_SIZE, fontweight='bold', y=1.02)
plt.tight_layout()

# Save as high-quality PNG (uncomment the SVG line for a vector copy too)
output_path = Path('attention_patterns_final_layer')
plt.savefig(f'{output_path}.png', dpi=600, bbox_inches='tight', facecolor='white')
#plt.savefig(f'{output_path}.svg', bbox_inches='tight', facecolor='white')

# Only announce what was actually written: the SVG savefig above is commented
# out, so the previous version of this print named a file that never existed.
print(f"Saved figures:")
print(f"  - {output_path}.png (high-res raster)")

plt.show()

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
