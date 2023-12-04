import torch
import torch.nn as nn

from .const import DEFAULT_EMBEDDING_DIM


class Word2Vec(nn.Module):
    """
    Word2Vec model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.projection = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


class Region2Vec(Word2Vec):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        super().__init__(vocab_size, embedding_dim)
