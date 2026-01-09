import torch
import random
import numpy as np


def set_seeds() -> None:
    """
    Sets random seeds for NumPy, Python's random module, and PyTorch
    for reproducibility purposes.
    """

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def scale_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements scaled dot-product attention, a core component in transformer architectures.

    This function calculates attention weights based on the query (q), key (k), and value (v) vectors,
    optionally applying a mask to exclude irrelevant positions.

    Args:
        q (torch.Tensor): Query vectors of shape (*, seq_length, dk).
        k (torch.Tensor): Key vectors of shape (*, seq_length, dk).
        v (torch.Tensor): Value vectors of shape (*, seq_length, dv).
        mask (torch.Tensor, optional): Attention mask of shape (*, seq_length, seq_length)
            used to mask out irrelevant positions during attention computation. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - values (torch.Tensor): Weighted sum of value vectors of shape (*, seq_length, dv).
            - attention (torch.Tensor): Attention weights of shape (*, seq_length, seq_length).
    """

    dk = q.shape[-1]
    attn_logits = q @ k.transpose(-2, -1) / np.sqrt(dk)  # (*, seq_length, seq_length)

    if mask is not None:
        # Mask out with a very large negative value
        attn_logits.masked_fill_(mask == 0, -1e9)

    attention = torch.softmax(attn_logits, dim=-1)  # (*, seq_length, seq_length)
    values = attention @ v  # (*, seq_length, dv)

    return values, attention


def get_device() -> torch.device:
    """
    Returns the appropriate device for PyTorch computations, prioritizing CUDA
    if available, otherwise defaulting to CPU.

    Returns:
        torch.device: A `torch.device` object representing the chosen device for
                     PyTorch computations (either CUDA or CPU).
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
