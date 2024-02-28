from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn

from ..io import RegionSet

if TYPE_CHECKING:
    from ..tokenization import Tokenizer


class Model(nn.Module):
    """Class representing an *actual* model, that is, weights, etc"""

    pass


class ExModel(ABC):
    """
    An Extended Model is a 3-part object consisting of a tokenizer (T), a
    universe/vocabulary (U), and a model (M). The tokenizer is used to tokenize
    region sets into the universe. The model is defined on the universe.
    """

    model: Model
    universe: RegionSet
    tokenizer: "Tokenizer"

    @abstractmethod
    def __init__(self, model_path: str = None, tokenizer: "Tokenizer" = None, device: str = None):
        """
        Initialize the model. If model_path is not None, load the model from
        huggingface.

        :param str model_path: Path to the model on huggingface to load
        :param Tokenizer tokenizer: Tokenizer to use
        :param str device: Device to use (e.g. "cpu", "cuda:0", etc)
        """
        pass
