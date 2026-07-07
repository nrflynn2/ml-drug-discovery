"""Unit tests for src/model.py — pins the expand_mask regression (fix #4)."""

import torch

from src.model import MultiheadAttention


def test_expand_mask_2d_to_4d():
    attn = MultiheadAttention(embedding_dim=8, num_heads=2)
    mask = torch.ones(3, 5)  # (batch, seq_len)
    out = attn.expand_mask(mask)
    assert out is not None
    assert out.shape == (3, 1, 1, 5)


def test_expand_mask_3d_to_4d():
    attn = MultiheadAttention(embedding_dim=8, num_heads=2)
    mask = torch.ones(3, 5, 5)  # (batch, seq_len, seq_len)
    out = attn.expand_mask(mask)
    assert out.shape == (3, 1, 5, 5)


def test_expand_mask_4d_passthrough_is_not_none():
    """Regression: a 4-D mask must be returned as-is, never dropped to None."""
    attn = MultiheadAttention(embedding_dim=8, num_heads=2)
    mask = torch.ones(3, 2, 5, 5)  # (batch, num_heads, seq_len, seq_len)
    out = attn.expand_mask(mask)
    assert out is not None, "4-D mask was silently dropped (attention left unmasked)"
    assert out.shape == (3, 2, 5, 5)
    assert torch.equal(out, mask)


def test_forward_with_4d_mask_runs():
    """A 4-D mask must flow through forward without a None-mask crash."""
    attn = MultiheadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(3, 5, 8)
    mask = torch.ones(3, 2, 5, 5)
    out = attn(x, mask=mask)
    assert out.shape == (3, 5, 8)
