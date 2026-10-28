"""
Training script for Protein Language Model with Masked Language Modeling.

This module provides a complete training pipeline for a Transformer-based protein
language model using masked language modeling (MLM), similar to BERT but for protein sequences.
"""

import torch
import typer
import json
import functools
import pandas as pd
import numpy as np
import torch.nn as nn
import math

from pathlib import Path
from collections import defaultdict
from src.data import Tokenizer, ProteinSequenceDataset, mlm_collate_fn
from src.model import ProteinLanguageModel
from src.utils import set_seeds, get_device
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR


# Create a typer app
app = typer.Typer()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with linear warmup and cosine decay.

    This scheduler is commonly used in transformer training. The learning rate:
    1. Increases linearly from 0 to the initial LR during warmup
    2. Decreases following a cosine curve after warmup

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of steps for the warmup phase.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine cycles (default 0.5 means cosine goes to 0).
        last_epoch: The index of the last epoch.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_dataloaders(
    df: pd.DataFrame,
    val_size: float,
    tokenizer: Tokenizer,
    device: torch.device,
    batch_size: int,
    mask_prob: float = 0.15,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and validation dataloaders for MLM.

    Args:
        df: DataFrame containing protein sequences.
        val_size: Proportion of data to use for validation.
        tokenizer: Tokenizer for sequences.
        device: Device to place tensors on.
        batch_size: Batch size for dataloaders.
        mask_prob: Probability of masking each token.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    # Split sequences (no stratification needed for unsupervised learning)
    sequences = df["sequence"].tolist()
    train_seqs, val_seqs = train_test_split(
        sequences, random_state=0, test_size=val_size
    )

    # Create datasets
    train_ds = ProteinSequenceDataset(train_seqs)
    val_ds = ProteinSequenceDataset(val_seqs)

    # Create collate function
    collate_fn_partial = functools.partial(
        mlm_collate_fn, tokenizer=tokenizer, device=device, mask_prob=mask_prob
    )

    # Create dataloaders
    train_dl = DataLoader(
        train_ds, collate_fn=collate_fn_partial, batch_size=batch_size, shuffle=True
    )
    val_dl = DataLoader(val_ds, collate_fn=collate_fn_partial, batch_size=batch_size)

    return train_dl, val_dl


def train_epoch(
    model: nn.Module,
    train_dl: DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: LambdaLR = None,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """
    Performs one training epoch for masked language modeling.

    Args:
        model: The protein language model.
        train_dl: Training dataloader.
        loss_fn: Loss function (typically CrossEntropyLoss).
        opt: Optimizer.
        scheduler: Learning rate scheduler (optional).
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Tuple of (average_loss, average_perplexity).
    """
    model.train()

    total_loss = 0
    total_tokens = 0

    for batch in train_dl:
        # Zero gradients
        opt.zero_grad()

        # Forward pass
        logits = model(batch)  # (batch_size, seq_len, vocab_size)

        # Reshape for loss computation
        # loss_fn expects (N, C) for predictions and (N,) for targets
        logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        labels_flat = batch["labels"].view(-1)  # (batch_size * seq_len)

        # Compute loss (CrossEntropyLoss ignores -100 labels)
        loss = loss_fn(logits_flat, labels_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update weights
        opt.step()

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Track loss
        # Count only the tokens we actually predict (not -100)
        num_tokens = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def val_epoch(
    model: nn.Module,
    val_dl: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, float, float]:
    """
    Performs one validation epoch for masked language modeling.

    Args:
        model: The protein language model.
        val_dl: Validation dataloader.
        loss_fn: Loss function.

    Returns:
        Tuple of (average_loss, perplexity, accuracy).
    """
    model.eval()

    total_loss = 0
    total_tokens = 0
    correct_predictions = 0

    with torch.inference_mode():
        for batch in val_dl:
            # Forward pass
            logits = model(batch)

            # Reshape for loss computation
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = batch["labels"].view(-1)

            # Compute loss
            loss = loss_fn(logits_flat, labels_flat)

            # Count tokens and track loss
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Compute accuracy on masked tokens
            predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
            mask = batch["labels"] != -100
            correct_predictions += (predictions[mask] == batch["labels"][mask]).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = correct_predictions / total_tokens

    return avg_loss, perplexity, accuracy


@app.command()
def train_model(
    run_id: Annotated[str, typer.Option(help="Name for the training run ID")],
    dataset_loc: Annotated[
        str, typer.Option(help="Path to the dataset in parquet or CSV format")
    ],
    val_size: Annotated[
        float, typer.Option(help="Proportion of the dataset to use for validation")
    ] = 0.15,
    embedding_dim: Annotated[
        int, typer.Option(help="Dimensionality of token embeddings")
    ] = 128,
    num_layers: Annotated[
        int, typer.Option(help="Number of Transformer encoder layers")
    ] = 4,
    num_heads: Annotated[
        int, typer.Option(help="Number of attention heads in the encoder")
    ] = 4,
    ffn_dim: Annotated[
        int,
        typer.Option(help="Dimensionality of the feed-forward layer in the encoder"),
    ] = 512,
    dropout: Annotated[
        float, typer.Option(help="Dropout probability for regularization")
    ] = 0.1,
    mask_prob: Annotated[
        float, typer.Option(help="Probability of masking each token")
    ] = 0.15,
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 32,
    lr: Annotated[
        float, typer.Option(help="The learning rate for the optimizer")
    ] = 5e-4,
    weight_decay: Annotated[
        float, typer.Option(help="Weight decay (L2 regularization)")
    ] = 0.01,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs for training")] = 50,
    warmup_steps: Annotated[
        int, typer.Option(help="Number of warmup steps for learning rate scheduler")
    ] = 500,
    max_grad_norm: Annotated[
        float, typer.Option(help="Maximum gradient norm for clipping")
    ] = 1.0,
    verbose: Annotated[
        bool, typer.Option(help="Whether to print verbose training messages")
    ] = True,
    output_dir: Annotated[
        str, typer.Option(help="Path to save the best model and training results")
    ] = "runs",
) -> None:
    """
    Trains a protein language model using masked language modeling.

    This function trains a Transformer-based model to predict masked amino acids
    in protein sequences, learning representations of protein structure and function.

    Example:
        python train_plm.py --run-id my_plm --dataset-loc bcr_train.parquet --num-epochs 50
    """

    # Create output directory
    save_path = Path(f"{output_dir}/{run_id}")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Load dataset
    if dataset_loc.endswith(".parquet"):
        df = pd.read_parquet(dataset_loc)
    else:
        df = pd.read_csv(dataset_loc)

    # Initialize tokenizer and device
    tokenizer = Tokenizer()
    device = get_device()

    # Create dataloaders
    train_dl, val_dl = get_dataloaders(
        df, val_size, tokenizer, device, batch_size, mask_prob
    )

    # Calculate total training steps
    num_training_steps = len(train_dl) * num_epochs

    # Initialize model
    model = ProteinLanguageModel(
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.pad_token_id,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )
    model.to(device)

    # Loss function (ignores -100 labels)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer (AdamW with weight decay)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    # Track training results
    results = defaultdict(list)

    # Save best model based on validation loss
    best_val_loss = float("inf")

    # Save model parameters
    params = {
        "vocab_size": tokenizer.vocab_size,
        "padding_idx": tokenizer.pad_token_id,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "dropout": dropout,
    }
    with open(save_path / "args.json", "w") as f:
        json.dump(params, f, indent=4, sort_keys=False)

    if verbose:
        print(f"Training Protein Language Model: {run_id}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training samples: {len(train_dl.dataset)}")
        print(f"Validation samples: {len(val_dl.dataset)}")
        print(f"Total training steps: {num_training_steps}")
        print("-" * 80)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        results["epoch"].append(epoch)

        # Train
        train_loss, train_ppl = train_epoch(
            model, train_dl, loss_fn, opt, scheduler, max_grad_norm
        )

        # Validation
        val_loss, val_ppl, val_acc = val_epoch(model, val_dl, loss_fn)

        # Track metrics
        results["train_loss"].append(train_loss)
        results["train_perplexity"].append(train_ppl)
        results["val_loss"].append(val_loss)
        results["val_perplexity"].append(val_ppl)
        results["val_accuracy"].append(val_acc)
        results["learning_rate"].append(scheduler.get_last_lr()[0])

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path / "best_model.pt")

        if verbose:
            print(
                f"Epoch {epoch:3d}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f} | "
                f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}, Val Acc: {val_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # Save training results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path / "results.csv", index=False)

    if verbose:
        print("-" * 80)
        print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
        print(f"Model and results saved to: {save_path}")


if __name__ == "__main__":
    set_seeds()
    app()
