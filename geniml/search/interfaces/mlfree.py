from typing import Dict, List, Union

import numpy as np

from ..backends import BiVectorBackend
from ..query2vec import Text2Vec
from .abstract import BEDSearchInterface


class BiVectorSearchInterface(BEDSearchInterface):
    """Search interface for ML free bi-vectors searching backend"""

    def __init__(self, backend: BiVectorBackend, query2vec: Union[str, Text2Vec]) -> None:
        """Initialize the BiVectorSearchInterface.

        Args:
            backend: the backend where vectors are stored
            query2vec: a Text2Vec instance (see geniml.search.query2vec.text2vec for details)
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
        """Search BED files using natural language query or embedding vector.

        Args:
            query: the natural language query string or a vector in the embedding space of region sets
            limit: number of results to return
            with_payload: whether payload is included in the result
            with_vectors: whether the stored vector is included in the result
            offset: the offset of the search results
            p: weight for metadata search score
            q: weight for BED file search score
            distance: whether to return distance or similarity scores
            rank: whether to rank by maximum rank or weighted score

        Returns:
            A list of dictionaries containing search results.
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
