import torch
import random
import numpy as np


def set_seeds(seed: int = 42) -> int:
    """
    Sets random seeds for NumPy, Python's random module, and PyTorch
    for reproducibility purposes.

    `torch.manual_seed` already covers the accelerator: it fans out to
    `torch.cuda.manual_seed_all` / `torch.mps.manual_seed` / `torch.xpu` / `torch.mtia`
    internally, so no extra per-backend seeding is needed here.

    Args:
        seed (int, optional): The seed applied to every RNG. Defaults to 42, the
            seed used across the rest of the book (this chapter previously
            hardcoded 0).

    Returns:
        int: The seed that was applied, so callers can write ``SEED = set_seeds()``.
    """

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    return seed


def scale_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements scaled dot-product attention, a core component in transformer architectures.

    This function calculates attention weights based on the query (q), key (k), and value (v) vectors,
    optionally applying a mask to exclude irrelevant positions.

    Mask convention used throughout this chapter: masks are **1/0 (or True/False)
    keep-or-drop** masks, NOT additive `0/-inf` masks. ``1``/``True`` means "this
    position may be attended to"; ``0``/``False`` means "ignore it". The additive
    form is produced internally, here, by filling the masked logits with the
    minimum value of the logits' dtype.

    Args:
        q (torch.Tensor): Query vectors of shape (*, seq_length, dk).
        k (torch.Tensor): Key vectors of shape (*, seq_length, dk).
        v (torch.Tensor): Value vectors of shape (*, seq_length, dv).
        mask (torch.Tensor, optional): Attention mask broadcastable to
            (*, query_length, seq_length) -- callers coming from
            `MultiheadAttention.expand_mask` pass
            (batch_size, num_heads_or_1, query_len_or_1, seq_length). Any dtype that
            compares against 0 works (bool/int/float). Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - values (torch.Tensor): Weighted sum of value vectors of shape (*, seq_length, dv).
            - attention (torch.Tensor): Attention weights of shape (*, seq_length, seq_length).
    """

    dk = q.shape[-1]
    attn_logits = q @ k.transpose(-2, -1) / np.sqrt(dk)  # (*, seq_length, seq_length)

    if mask is not None:
        # Mask out with the most negative value the logits' dtype can hold, so the
        # softmax sends these positions to ~0. A hardcoded -1e9 OVERFLOWS float16
        # (whose max is 65504) straight to -inf, so under autocast/fp16 the masked
        # logits become -inf instead of "very negative". torch.finfo(...).min is
        # dtype-correct for fp16, bf16 and fp32 alike, and is numerically identical
        # to -1e9 on fp32.
        attn_logits = attn_logits.masked_fill(mask == 0, torch.finfo(attn_logits.dtype).min)

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

    return torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")
