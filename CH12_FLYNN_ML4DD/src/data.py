import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class Tokenizer:
    """
    Tokenizer class for converting protein sequences into numerical representations.

    This tokenizer handles a vocabulary of special tokens for classification (`<cls>`, `<eos>`),
    padding (`<pad>`), and unknown amino acids (`<unk>`), along with the 20 standard amino acids.

    Attributes:
        token_to_index (Dict[str, int]): Mapping from token to its index in the vocabulary.
        index_to_token (Dict[int, str]): Mapping from index to its corresponding token.
        vocab_size (int): Size of the vocabulary (number of tokens).
        pad_token_id (int): Index of the padding token (`<pad>`).
    """

    def __init__(self):
        """
        Initializes the tokenizer with a vocabulary of special tokens and amino acids.
        """

        # special tokens
        vocab = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
        # 20 canonical amino acids
        vocab += list("ACDEFGHIKLMNPQRSTVWY")
        # mapping
        self.token_to_index = {tok: i for i, tok in enumerate(vocab)}
        self.index_to_token = {i: tok for i, tok in enumerate(vocab)}

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary (number of tokens).
        """
        return len(self.token_to_index)

    @property
    def pad_token_id(self):
        """
        Returns the index of the padding token (`<pad>`).
        """
        return self.token_to_index["<pad>"]

    @property
    def mask_token_id(self):
        """
        Returns the index of the mask token (`<mask>`).
        """
        return self.token_to_index["<mask>"]

    @property
    def cls_token_id(self):
        """
        Returns the index of the CLS token (`<cls>`).
        """
        return self.token_to_index["<cls>"]

    @property
    def eos_token_id(self):
        """
        Returns the index of the EOS token (`<eos>`).
        """
        return self.token_to_index["<eos>"]

    def __call__(
        self, seqs: list[str], padding: bool = True
    ) -> dict[str, list[list[int]]]:
        """
        Tokenizes a list of protein sequences and creates input representations with attention masks.

        Args:
            seqs (List[str]): List of protein sequences to tokenize.
            padding (bool, optional): Whether to pad sequences to a maximum length. Defaults to True.

        Returns:
            Dict[str, List[List[int]]]: A dictionary containing:
                - input_ids (List[List[int]]): List of token IDs for each sequence.
                - attention_mask (List[List[int]]): List of attention masks for each sequence.
        """

        input_ids = []
        attention_mask = []

        if padding:
            max_len = max(len(seq) for seq in seqs)

        for seq in seqs:
            # Preprocessing: strip whitespace, convert to uppercase
            seq = seq.strip().upper()

            # Add special tokens
            toks = ["<cls>"] + list(seq) + ["<eos>"]

            if padding:
                # Pad with '<pad>' tokens to reach max_len
                toks += ["<pad>"] * (max_len - len(seq))

            # Convert tokens to IDs (handling unknown amino acids)
            unk_id = self.token_to_index["<unk>"]
            input_ids.append([self.token_to_index.get(tok, unk_id) for tok in toks])

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask.append([1 if tok != "<pad>" else 0 for tok in toks])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_data(data_loc: str) -> tuple[pd.DataFrame, dict]:
    """
    Loads and preprocesses data from a Parquet file.

    This function reads a Pandas DataFrame from the specified Parquet file location (`data_loc`),
    encodes categorical target labels using LabelEncoder, and returns the preprocessed DataFrame.

    Args:
        data_loc (str): Path to the Parquet file containing BCR data.

    Returns:
        tuple: A tuple containing:
            * pd.DataFrame: Preprocessed Pandas DataFrame containing:
                - Existing features from the original data.
                - "label" (int): Encoded representation of the target variable.
            * dict: A dictionary mapping encoded labels (integers) to their original class values.
    """
    df = pd.read_parquet(data_loc)

    # Create LabelEncoder for target variable
    le = LabelEncoder()

    # Encode target labels
    df["label"] = le.fit_transform(df["target"])

    # class mapping
    classes = {i: c for i, c in enumerate(le.classes_)}

    return df, classes


class BCRDataset(Dataset):
    """
    BCRDataset class for loading and preparing B-cell receptor (BCR) dataset.

    This class inherits from `torch.utils.data.Dataset` and is used to load and prepare
    BCR data from a Pandas DataFrame for training or evaluation with a model.

    Attributes:
        df (pd.DataFrame): Pandas DataFrame containing BCR data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the BCRDataset object.

        Args:
            df (pd.DataFrame): Pandas DataFrame containing BCR data.
                - sequence (str): Amino acid sequence of the heavy chain.
                - label (int): Label associated with the BCR sample.
        """
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        """
        Returns the length of the dataset (number of samples).

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i) -> tuple[str, int]:
        """
        Retrieves a data point (sequence and label) at a specific index.

        Args:
            i (int): Index of the data point to retrieve.

        Returns:
            Tuple[str, int]: A tuple containing:
                - x (str): Amino acid sequence of the heavy chain.
                - y (int): Label associated with the BCR sample.
        """

        x = self.df.loc[i, "sequence"]
        y = self.df.loc[i, "label"]

        return x, y


def collate_fn(
    batch: list[tuple[str, int]],  # Tuples of (sequence, label)
    tokenizer: Tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Collate function to prepare a batch of data for training or evaluation.

    This function takes a batch of data points, each containing a sequence (str) and its
    corresponding label (int), and processes them into a dictionary suitable for
    training or evaluation with a model. It performs the following steps:

    Args:
        batch (List[Tuple[str, int]]): Batch of data points, where each data point
            is a tuple containing a sequence (str) and its corresponding label (int ).
        tokenizer (Tokenizer): Tokenizer object used to convert sequences into numerical representations.
        device (torch.device): Device (CPU or GPU) where the tensors should be placed.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the processed data:
            - input_ids (torch.Tensor): Tokenized sequence IDs (shape: batch_size, max_len).
            - attention_mask (torch.Tensor): Attention masks (shape: batch_size, max_len).
            - label (torch.Tensor): Labels (shape: batch_size).
    """

    # Unpack sequences and labels from the batch
    seqs, labels = zip(*batch)

    # Tokenize sequences and create attention masks with padding
    batch = tokenizer(seqs, padding=True)

    # convert to tensor
    for k in batch.keys():
        batch[k] = torch.tensor(batch[k], dtype=torch.long, device=device)

    # labels
    batch["label"] = torch.tensor(labels, dtype=torch.long, device=device)

    return batch


def mask_sequences(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: list[int],
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates masked sequences for masked language modeling (MLM).

    This function implements the masking strategy from BERT:
    - Select 15% of tokens for masking (by default)
    - Of those selected:
        - 80% are replaced with [MASK] token
        - 10% are replaced with a random amino acid
        - 10% are kept unchanged (but still predicted)

    Args:
        input_ids (torch.Tensor): Input sequence token IDs (batch_size, seq_len).
        mask_token_id (int): ID of the mask token.
        vocab_size (int): Size of the vocabulary.
        special_token_ids (list[int]): List of special token IDs that should not be masked
            (e.g., [CLS], [PAD], [EOS]).
        mask_prob (float, optional): Probability of selecting a token for masking. Defaults to 0.15.
        mask_token_prob (float, optional): Probability of replacing selected token with [MASK].
            Defaults to 0.8.
        random_token_prob (float, optional): Probability of replacing selected token with random token.
            Defaults to 0.1.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - masked_input_ids: Input IDs with masking applied (batch_size, seq_len)
            - labels: Target labels for MLM, -100 for non-masked positions (batch_size, seq_len)
            - mask_positions: Boolean tensor indicating which positions were masked (batch_size, seq_len)
    """

    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()

    # Create probability matrix
    probability_matrix = torch.full(input_ids.shape, mask_prob)

    # Don't mask special tokens
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in special_token_ids:
        special_tokens_mask |= input_ids == special_id

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Select positions to mask
    mask_positions = torch.bernoulli(probability_matrix).bool()

    # Create labels (-100 for positions we don't predict)
    labels[~mask_positions] = -100

    # 80% of the time: replace with [MASK] token
    indices_replaced = (
        torch.bernoulli(torch.full(input_ids.shape, mask_token_prob)).bool()
        & mask_positions
    )
    masked_input_ids[indices_replaced] = mask_token_id

    # 10% of the time: replace with random amino acid
    indices_random = (
        torch.bernoulli(torch.full(input_ids.shape, random_token_prob)).bool()
        & mask_positions
        & ~indices_replaced
    )

    # Random tokens from vocabulary (excluding special tokens)
    # Create a list of valid token IDs (amino acids only)
    num_special = len(special_token_ids)
    amino_acid_ids = torch.arange(num_special, vocab_size)

    random_tokens = amino_acid_ids[
        torch.randint(len(amino_acid_ids), input_ids.shape)
    ]
    masked_input_ids[indices_random] = random_tokens[indices_random]

    # 10% of the time: keep the token unchanged (but still predict it)

    return masked_input_ids, labels, mask_positions


class ProteinSequenceDataset(Dataset):
    """
    Dataset class for protein language modeling with masked language modeling.

    This dataset loads protein sequences and prepares them for masked language modeling,
    where the model learns to predict masked amino acids.

    Attributes:
        sequences (list[str]): List of protein sequences.
    """

    def __init__(self, sequences: list[str]):
        """
        Initializes the ProteinSequenceDataset.

        Args:
            sequences (list[str]): List of protein sequences as strings.
        """
        super().__init__()
        self.sequences = sequences

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, sequence_col: str = "sequence"):
        """
        Creates a ProteinSequenceDataset from a pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing protein sequences.
            sequence_col (str, optional): Name of the column containing sequences.
                Defaults to "sequence".

        Returns:
            ProteinSequenceDataset: Dataset instance.
        """
        sequences = df[sequence_col].tolist()
        return cls(sequences)

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        """
        Retrieves a sequence at the specified index.

        Args:
            idx (int): Index of the sequence.

        Returns:
            str: Protein sequence.
        """
        return self.sequences[idx]


def mlm_collate_fn(
    batch: list[str],
    tokenizer: Tokenizer,
    device: torch.device,
    mask_prob: float = 0.15,
) -> dict[str, torch.Tensor]:
    """
    Collate function for masked language modeling.

    This function tokenizes sequences, applies masking, and prepares batches for MLM training.

    Args:
        batch (list[str]): List of protein sequences.
        tokenizer (Tokenizer): Tokenizer for converting sequences to token IDs.
        device (torch.device): Device to place tensors on.
        mask_prob (float, optional): Probability of masking each token. Defaults to 0.15.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing:
            - input_ids: Masked input token IDs (batch_size, seq_len)
            - attention_mask: Attention mask (batch_size, seq_len)
            - labels: Target labels for MLM (batch_size, seq_len)
    """

    # Tokenize sequences
    tokenized = tokenizer(batch, padding=True)

    # Convert to tensors
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        tokenized["attention_mask"], dtype=torch.long, device=device
    )

    # Apply masking
    special_token_ids = [
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
    ]

    masked_input_ids, labels, _ = mask_sequences(
        input_ids=input_ids,
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        special_token_ids=special_token_ids,
        mask_prob=mask_prob,
    )

    return {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
