import torch
import torch.nn as nn
import numpy as np

from src.utils import scale_dot_product_attention


class MultiheadAttention(nn.Module):
    """
    Multi-head attention layer for Transformer models.

    This module performs multi-head attention on the input, splitting it into multiple
    heads, performing scaled dot-product attention, and concatenating the results.

    Args:
        embedding_dim (int): The dimension of the input and output embeddings.
        num_heads (int): The number of attention heads.
    """

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "Embedding dimensionality should be a multiple of num_heads"

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear layer for transforming X into Q, K, V
        self.input = nn.Linear(embedding_dim, 3 * embedding_dim)

        # output layer
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def expand_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Expands the attention mask to the required 4D shape for multi-head attention.

        Args:
            mask (torch.Tensor): Attention mask with various possible shapes:
                - (batch_size, seq_len)
                - (batch_size, seq_len, seq_len)
                - (batch, num_heads, seq_len, seq_len)
                None: No mask is applied.

        Returns:
            torch.Tensor: Expanded mask with 4D shape
        """

        assert 2 <= mask.ndim <= 4, "Wrong mask dimensionality"
        if mask.ndim == 2:
            return mask[:, None, None, :]
        elif mask.ndim == 3:
            return mask[:, None, :, :]

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Performs multi-head attention on the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
            mask (torch.Tensor, optional): Attention mask to prevent attention to padded tokens.
                Can have various shapes based on documentation. Defaults to None.
            return_attention (bool, optional): Whether to return the attention weights.
                Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
            torch.Tensor, optional: Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """

        batch_size, seq_len = x.shape[0], x.shape[1]

        x = self.input(x)  # (batch_size, seq_len, 3 * embedding_dim)

        # split heads
        x = x.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)

        # swap dims
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, 3 * head_dim)

        # q, k, v
        q, k, v = x.chunk(3, dim=-1)  # (batch_size, num_heads, seq_len, head_dim)

        # expand mask to 4D if needed
        if mask is not None:
            mask = self.expand_mask(mask)

        # Perform scaled dot-product attention
        # values: (batch_size, num_heads, seq_len, head_dim)
        # attention: (batch_size, num_heads, seq_len, seq_len)
        values, attention = scale_dot_product_attention(q, k, v, mask=mask)

        # change dims
        values = values.permute(
            0, 2, 1, 3
        )  # (batch_size, seq_len, num_heads, head_dim)

        # concat heads
        values = values.reshape(
            batch_size, seq_len, -1
        )  # (batch_size, seq_len, embedding_dim)

        # output linear layer
        out = self.output(values)  # (batch_size, seq_len, embedding_dim)

        if return_attention:
            return out, attention

        return out


class EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer.

    The encoder layer consists of a multi-head attention layer followed by a feed-forward network.
    Both layers incorporate residual connections, layer normalization, and dropout for regularization.

    Args:
        embedding_dim (int): The dimension of the word embeddings.
        num_heads (int): The number of attention heads in the multi-head attention layer.
        ffn_dim (int): The dimension of the feed-forward network.
        dropout (float, optional): The dropout probability for regularization. Defaults to 0.0.
    """

    def __init__(
        self, embedding_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        # d_embed is the same as d_model
        self.attention = MultiheadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
        )

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes the input sequence through the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            mask (torch.Tensor, optional): Attention mask to prevent attention to padded tokens.
                Shape: (batch_size, seq_length). Defaults to None.

        Returns:
            torch.Tensor: The encoded output tensor of shape (batch_size, seq_length, embedding_dim).
        """

        #  Multi-head attention layer
        attn_out = self.attention(x, mask)  # (batch, seq_len, embedding_dim)

        # residual connections
        x = x + self.dropout(attn_out)
        # layer norm
        x = self.norm1(x)

        # feed forward
        ffn_out = self.ffn(x)  # (batch, seq_len, d_model)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    """
    A stack of Transformer encoder layers.

    The encoder processes the input sequence and generates a contextual representation
    of each token, taking into account its relationships with other tokens in the sequence.

    Args:
        num_layers (int): The number of encoder layers to stack.
        embedding_dim (int): The dimension of the token embeddings.
        num_heads (int): The number of attention heads in the multi-head attention layer.
        ffn_dim (int): The dimension of the feed-forward network in the encoder Layer.
        dropout (float, optional): The dropout probability for regularization. Defaults to 0.0.
    """

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Encoder consists of a list of EncoderLayers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes the input sequence through all encoder layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            mask (torch.Tensor, optional): Attention mask to prevent attention to padded tokens.
                Shape: (batch_size, seq_length). Defaults to None.

        Returns:
            torch.Tensor: The encoded output tensor of shape (batch_size, seq_length, embedding_dim).
        """

        for layer in self.layers:
            x = layer(x, mask)

        return x

    def get_attentions(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> list[torch.Tensor]:
        """
        Retrieves the attention weights from each encoder layer.

        This can be useful for visualizing attention patterns or analyzing how the model attends
        to different parts of the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            mask (torch.Tensor, optional): Attention mask. Shape: (batch_size, seq_length).
                Defaults to None.

        Returns:
            list[torch.Tensor]: A list of attention weights tensors, one for each encoder layer.
                Each tensor has shape (batch_size, num_heads, seq_length, seq_length).
        """

        attention_maps = []

        for layer in self.layers:
            _, attn = layer.attention(x, mask=mask, return_attention=True)
            attention_maps.append(attn)

            x = layer(x)

        return attention_maps


class PositionalEncoding(nn.Module):
    """
    Injects positional information into token embeddings using sine and cosine functions.

    The positional encoding is added to the token embeddings to provide the model with
    information about the position of each token in a sequence.

    Args:
        embedding_dim (int): The dimension of the token embeddings.
        max_length (int, optional): The maximum sequence length for which positional
            encodings will be pre-computed. Defaults to 5000.

    Attributes:
        pe (torch.Tensor): A pre-computed positional encoding tensor of shape
            (1, max_length, embedding_dim).
    """

    def __init__(self, embedding_dim: int, max_length: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_length, embedding_dim)

        # add an addtitional dimention for broadcasting
        position = torch.arange(max_length).float().unsqueeze(1)

        # div_term is of length embedding_dim//2:
        div_term = torch.exp(
            -torch.arange(0, embedding_dim, 2) / embedding_dim * np.log(1e4)
        )

        # populate even and odd indices
        # position*div_term: (max_length, embedding_dim//2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshape for broadcasting: (max_length, embedding_dim) => (1, max_length, embedding_dim)
        pe = pe.unsqueeze(0)

        # pe is not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Adds positional encodings to input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            torch.Tensor: Input tensor with positional encodings added, of the same shape.
        """

        return x + self.pe[:, : x.shape[1], :]  # (batch, seq_len, embedding_dim)


class ProteinLanguageModel(nn.Module):
    """
    Protein language model using a Transformer-based encoder with masked language modeling.

    This model learns representations of protein sequences by predicting masked amino acids.
    It uses an embedding layer, positional encoding, and a Transformer encoder to process
    sequences, followed by a language modeling head that predicts tokens at each position.

    This architecture is similar to BERT but adapted for protein sequences. It learns
    which amino acid patterns are biologically plausible by training on large protein datasets.

    Attributes:
        embedding (nn.Embedding): Embedding layer for mapping amino acid tokens to vectors.
        pe (PositionalEncoding): Positional encoding layer to inject position information.
        encoder (Encoder): Transformer encoder for sequence processing with self-attention.
        lm_head (nn.Linear): Language modeling head that projects to vocabulary size.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        padding_idx (int): Index of the padding token in the vocabulary.
        embedding_dim (int): Dimensionality of token embeddings.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads in each encoder layer.
        ffn_dim (int): Dimensionality of the feed-forward network in encoder layers.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.pe = PositionalEncoding(embedding_dim)
        self.encoder = Encoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        # Language modeling head: project back to vocabulary size
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the protein language model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data containing:
                - input_ids (torch.Tensor): Tokenized sequence IDs (batch_size, seq_len).
                - attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for each position and token (batch_size, seq_len, vocab_size).
        """

        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

        # Map tokens to vectors
        x = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # Add positional encoding
        x = self.pe(x)  # (batch_size, seq_len, embedding_dim)

        # Pass through Transformer encoder
        x = self.encoder(x, attention_mask)  # (batch_size, seq_len, embedding_dim)

        # Project to vocabulary size for language modeling
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits


class AntibodyClassifier(nn.Module):
    """
    Antibody classifier model using a Transformer-based encoder.

    This class defines a neural network architecture for antibody classification.
    It utilizes an embedding layer to map sequence tokens to vectors,
    followed by a positional encoding layer to inject positional information,
    and a multi-head self-attention Transformer encoder for sequence processing.
    Finally, a feed-forward network with dropout is used for classification.

    Attributes:
        embedding (nn.Embedding): Embedding layer for mapping tokens to vectors.
        pe (PositionalEncoding): Positional encoding layer.
        encoder (Encoder): Transformer encoder for sequence processing.
        classifier (nn.Sequential): Classification network with linear layers and activation functions.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        padding_idx (int): Index of the padding token in the vocabulary.
        embedding_dim (int): Dimensionality of token embeddings.
        num_layers (int): Number of encoder layers.
        num_heads (int): Number of attention heads in the encoder.
        ffn_dim (int): Dimensionality of the feed-forward layer in the encoder.
        dropout (float): Dropout rate for regularization.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.pe = PositionalEncoding(embedding_dim)
        self.encoder = Encoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, num_classes),
        )

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs mean pooling of token embeddings based on attention mask.

        This function averages the embedding vectors of all amino acids within a sequence,
        considering only the unmasked tokens based on the attention mask.

        Args:
            token_embeddings (torch.Tensor): Token embeddings from the encoder (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor): Attention mask indicating valid tokens (batch_size, seq_len).

        Returns:
            torch.Tensor: Mean-pooled embedding vectors of each sequence (batch_size, embedding_dim).
        """

        # Expand the mask for broadcasting
        expanded_mask = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        )

        # sum unmasked token embeddings
        sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)

        # number of unmasked tokens (clamp to avoid division by zero)
        num_tokens = torch.clamp(expanded_mask.sum(1), min=1e-9)

        # mean pooling
        mean_embeddings = sum_embeddings / num_tokens
        return mean_embeddings

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of training/evaluation data containing:
                - input_ids (torch.Tensor): Tokenized sequence IDs (batch_size, seq_len).
                - attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: Model output, representing class logits (batch_size, num_classes).
        """

        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

        # map tokens to vectors
        x = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # add positional encoding
        x = self.pe(x)  # (batch, seq_len, embedding_dim)

        # pass through the Transformer encoder for sequence processing
        x = self.encoder(x, attention_mask)  # (batch_size, seq_len, embedding_dim)

        # Mean pooling
        x = self.mean_pooling(x, attention_mask)  # (batch_size, embedding_dim)

        # Classification head
        logits = self.classifier(x)  # (batch_size, num_classes)

        return logits
