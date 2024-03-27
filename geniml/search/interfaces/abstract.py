from abc import ABC, abstractmethod
from typing import Union, List, Dict

import numpy as np

from ...io import RegionSet
from ..backends.abstract import EmSearchBackend
from ..query2vec.abstract import Query2Vec


class BEDSearchInterface(ABC):
    """
    An abstract class representing BED files search interface.
    Consist of a backend that stores BED embedding vectors,
    and a module that embeds the query (natural langauge string or region set)
    """

    def __init__(self, backend: EmSearchBackend, query2vec: Query2Vec) -> None:
        self.backend = backend
        self.query2vec = query2vec

    @abstractmethod
    def query_search(
        self,
        query: Union[str, RegionSet, np.ndarray],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
    ) -> List[Dict]:
        raise NotImplementedError
