"""Unit tests for src/data.py — pins the .iloc (fix #10) and device (fix #2) fixes."""

import pandas as pd
import torch

from src.data import BCRDataset, Tokenizer, mask_sequences


def test_bcrdataset_getitem_with_non_reset_index():
    """Regression: __getitem__ must index positionally, so a non-default
    (non-zero-based) DataFrame index still works instead of raising KeyError."""
    df = pd.DataFrame(
        {"sequence": ["ACDE", "FGHI", "KLMN"], "label": [0, 1, 0]},
        index=[10, 11, 12],
    )
    ds = BCRDataset(df)
    seq, label = ds[0]
    assert seq == "ACDE"
    assert label == 0
    seq, label = ds[2]
    assert seq == "KLMN"


def test_mask_sequences_shapes_and_device():
    """mask_sequences must return tensors on the same device as input_ids and
    preserve shape; helper tensors must not force a device mismatch on CPU/GPU."""
    tok = Tokenizer()
    input_ids = torch.tensor([[tok.token_to_index[c] for c in "ACDEFG"]], dtype=torch.long)
    special_ids = [tok.token_to_index[t] for t in ("<cls>", "<pad>", "<eos>", "<unk>")]

    masked, labels, positions = mask_sequences(
        input_ids,
        mask_token_id=tok.token_to_index["<mask>"],
        vocab_size=tok.vocab_size,
        special_token_ids=special_ids,
        mask_prob=0.5,
    )

    assert masked.shape == input_ids.shape
    assert labels.shape == input_ids.shape
    assert positions.shape == input_ids.shape
    assert masked.device == input_ids.device
    assert labels.device == input_ids.device
    # Positions that were not selected for prediction are ignored (-100).
    assert torch.all(labels[~positions] == -100)
