from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenization.main import InMemTokenizer
from ..models.main import ExModel
from ..io.io import RegionSet
from .const import (
    DEFAULT_EMBEDDING_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_COUNT,
    DEFAULT_N_SHUFFLES,
)


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


class Region2VecExModel(ExModel):
    def __init__(self, model_path: str = None, tokenizer: InMemTokenizer = None, **kwargs):
        """
        Initialize Region2VecExModel.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        super().__init__()
        self.model_path = model_path
        self.tokenizer: InMemTokenizer = tokenizer

        if model_path is not None:
            self._init_from_huggingface(model_path)
        else:
            self.tokenizer = tokenizer
            self._model = Word2Vec(
                len(self.tokenizer.universe),
                embedding_dim=kwargs.get("embedding_dim", DEFAULT_EMBEDDING_SIZE),
                hidden_dim=kwargs.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            )

    def _validate_data_for_training(
        self, data: Union[List[RegionSet], List[str]]
    ) -> List[RegionSet]:
        """
        Validate the data for training. This will return a list of RegionSets if the data is valid.

        :param Union[List[RegionSet], List[str]] data: List of data to train on. This is either
                                                       a list of RegionSets or a list of paths to bed files.
        :return: List of RegionSets.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list or RegionSets or a list of paths to bed files.")
        if len(data) == 0:
            raise ValueError("data must not be empty.")

        # check if the data is a list of RegionSets
        if isinstance(data[0], RegionSet):
            return data
        elif isinstance(data[0], str):
            return [RegionSet(f) for f in data]

    def train(
        self,
        data: Union[List[RegionSet], List[str]],
        window_size: int = DEFAULT_WINDOW_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        min_count: int = DEFAULT_MIN_COUNT,
        n_shuffles: int = DEFAULT_N_SHUFFLES,
    ):
        """
        Train the model.

        :param Union[List[RegionSet], List[str]] data: List of data to train on. This is either
                                                       a list of RegionSets or a list of paths to bed files.
        :param int window_size: Window size for the model.
        :param int epochs: Number of epochs to train for.
        :param int min_count: Minimum count for a region to be included in the vocabulary.
        :param int n_shuffles: Number of shuffles to perform on the data.
        """
        # validate the data
        data = self._validate_data_for_training(data)

        # train the model
        raise NotImplementedError("Training is not yet implemented.")
