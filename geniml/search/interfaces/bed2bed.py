from typing import Dict, List, Union

import numpy as np

from ...io import RegionSet
from ..backends import HNSWBackend, QdrantBackend
from ..query2vec import Bed2Vec
from .abstract import BEDSearchInterface


class BED2BEDSearchInterface(BEDSearchInterface):
    """Search interface for the query that is a region set"""

    def __init__(
        self,
        backend: Union[QdrantBackend, HNSWBackend],
        query2vec: Union[str, Bed2Vec],
    ):
        """
        Initiate the search interface

        Parameters
        ----------
        backend : vector backend where the BED embeddings are stored
        query2vec : used to embed query region set
        """
        if isinstance(query2vec, str):
            self.query2vec = Bed2Vec(query2vec)
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

        Parameters
        ----------
        query : the region set, path to a BED file in disk, or region set embedding vector
        limit : see docstrings of def search() in QdrantBackend or HNSWBackend
        with_payload :
        with_vectors :
        offset :

        Returns
        -------
        see docstrings of def search() in QdrantBackend or HNSWBackend
        """
        if isinstance(query, np.ndarray):
            search_vec = query
        else:
            search_vec = self.query2vec.forward(query)

        return self.backend.search(search_vec, limit, with_payload, with_vectors, offset)
