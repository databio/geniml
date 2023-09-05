import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenization.main import InMemTokenizer
from .const import DEFAULT_EMBEDDING_SIZE, DEFAULT_HIDDEN_DIM


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_SIZE,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.projection = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = F.relu(self.hidden(x))
        x = self.output(x)


class Region2VecExModel(Word2Vec):
    def __init__(self, model_path: str = None, tokenizer: InMemTokenizer = None, **kwargs):
        """
        Initialize Region2VecExModel.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.tokenizer: InMemTokenizer = tokenizer

        if model_path is not None:
            self._init_from_huggingface(model_path)
        else:
            self.tokenizer = tokenizer
