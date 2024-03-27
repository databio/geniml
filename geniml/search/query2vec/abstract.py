from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from ...io import RegionSet
from ...text2bednn.embedder.abstract import TextEmbedder


class Query2Vec(ABC):
    """
    An abstract class representing query embedder. In retrieval of region sets,
    this allows embedding a query region set/ bed file, and embedding a query
    natural language string.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, query: Union[str, RegionSet]) -> np.ndarray:
        raise NotImplementedError
