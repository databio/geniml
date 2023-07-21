from abc import ABC, abstractmethod
from ..io import RegionSet
from ..tokenization import Tokenizer

class Model(ABC):
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
    tokenizer: Tokenizer

    @abstractmethod
    def __init__(
        self, model: Model = None, universe: RegionSet = None, tokenizer: Tokenizer = None
    ) -> None:
        raise NotImplementedError
