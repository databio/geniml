from typing import Dict, List, Union

import numpy as np

from ...io import RegionSet
from ..backends import HNSWBackend, QdrantBackend
from ..query2vec import BED2Vec
from .abstract import BEDSearchInterface


class BED2BEDSearchInterface(BEDSearchInterface):
    """Search interface for the query that is a region set"""

    def __init__(
        self,
        backend: Union[QdrantBackend, HNSWBackend],
        query2vec: Union[str, BED2Vec],
    ):
        """

        :param backend: the backend where vectors are stored

        :param query2vec: a BED2Vec model, or a hugging face model repository of geniml.region2vec.Region2VecExModel
        """
        if isinstance(query2vec, str):
            self.query2vec = BED2Vec(query2vec)
        else:
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
        """
        :param query: a region set, s path to a BED file in disk, or a region set embedding vector

        for rest of the parameters, check the docstring of QdrantBackend.search() or HNSWBackend.search()
        """
        if isinstance(query, np.ndarray):
            search_vec = query
        else:
            search_vec = self.query2vec.forward(query)

        return self.backend.search(search_vec, limit, with_payload, with_vectors, offset)
