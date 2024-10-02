from typing import Dict, List, Union

import numpy as np

from ..backends import BiVectorBackend
from ..query2vec import Text2Vec
from .abstract import BEDSearchInterface


class BiVectorSearchInterface(BEDSearchInterface):
    """Search interface for ML free bi-vectors searching backend"""

    def __init__(self, backend: BiVectorBackend, query2vec: Union[str, Text2Vec]) -> None:
        """
        Initiate the search interface

        :param backend: the backend where vectors are stored
        :param query2vec: a Text2Vec, for details, see docstrings in geniml.search.query2vec.text2vec
        """
        if isinstance(query2vec, str):
            self.query2vec = Text2Vec(query2vec, v2v=None)
        else:
            self.query2vec = query2vec
        self.backend = backend

    def query_search(
        self,
        query: Union[str, np.ndarray],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
        p: float = 1.0,
        q: float = 1.0,
        distance: bool = False,
        rank: bool = False,
    ) -> List[Dict]:
        """

        :param query: the natural language query string, or a vector in the embedding space of region sets

        for rest of the parameters, check the docstring of QdrantBackend.search() or HNSWBackend.search()
        """
        if isinstance(query, np.ndarray):
            search_vec = query
        else:
            search_vec = self.query2vec.forward(query)

        return self.backend.search(
            query=search_vec,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            offset=offset,
            p=p,
            q=q,
            distance=distance,
            rank=rank,
        )
