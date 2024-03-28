from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

from ...io import RegionSet
from ..backends.abstract import EmSearchBackend
from ..query2vec.abstract import Query2Vec


class BEDSearchInterface(ABC):
    """
    An abstract class representing BED files search interface.
    The query will be embedded by one of the subclass of Query2Vec,
    and the embedding is used to do KNN search in the backend
    where BED embeddings are stored.
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
