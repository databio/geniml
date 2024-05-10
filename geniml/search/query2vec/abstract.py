from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from ...io import RegionSet


class Query2Vec(ABC):
    """
    An abstract class representing query embedder. In retrieval of region sets,
    it embeds the query into a vector, which is used for KNN search in backend
    """

    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, query: Union[str, RegionSet]) -> np.ndarray:
        """

        Parameters
        ----------
        query : a natural language string (query term or path to a BED file in disk),
        or a RegionSet object

        Returns
        -------
        the embedding vector
        """
        raise NotImplementedError
