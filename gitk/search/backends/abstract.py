from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class EmSearchBackend(ABC):
    """
    An abstract class representing Embedding Search Backends. This allows
    backends to be either a qdrant server or a local in-memory NMS index, or
    anything, really. This allows us to use the same interface for both.
    """

    def __init__(self, embeddings: np.ndarray = None, labels: list = None) -> None:
        if embeddings:
            self.load(embeddings, labels)

    @abstractmethod
    def load(self, embeddings: np.ndarray, labels: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Dict]:
        """
        Search for the nearest neighbors of the given embedding

        :param query: the embedding to search for
        :param k: the number of results to return
        :return: a list of (id, score) pairs
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of embeddings in the backend
        """
        raise NotImplementedError()

    @abstractmethod
    def retrieve_info(self, key) -> List[Dict]:
        """
        with a list of storage ids, return matching vectors and their information
        """
        raise NotImplementedError()
