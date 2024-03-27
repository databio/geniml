from typing import Union, List, Dict

import numpy as np

from ..query2vec import Text2Vec
from ..backends import QdrantBackend, HNSWBackend
from .abstract import BEDSearchInterface
from ...io import RegionSet


class Text2BEDSearchInterface(BEDSearchInterface):
    def __init__(
        self, backend: Union[QdrantBackend, HNSWBackend], query2vec: Union[Text2Vec],
    ):
        self.query2vec = query2vec
        self.backend = backend

    def query_search(
        self,
        query: Union[str, RegionSet, np.ndarray],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
    ) -> List[Dict]:
        if isinstance(query, np.ndarray):
            search_vec = query
        else:
            search_vec = self.query2vec.forward(query)

        return self.backend.search(search_vec, limit, with_payload, with_vectors, offset)
