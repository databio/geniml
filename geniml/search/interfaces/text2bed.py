import logging
from typing import Dict, List, Union

import numpy as np
from geniml.const import PKG_NAME

from ..backends import HNSWBackend, QdrantBackend
from ..query2vec import Text2Vec
from ..utils import single_query_eval
from .abstract import BEDSearchInterface

_LOGGER = logging.getLogger(PKG_NAME)


class Text2BEDSearchInterface(BEDSearchInterface):
    """Search interface for the query that is a natural language string"""

    def __init__(
        self,
        backend: Union[QdrantBackend, HNSWBackend],
        query2vec: Text2Vec,
    ):
        """
        Initiate the search interface

        :param backend: the backend where vectors are stored
        :param query2vec: a Text2Vec, for details, see docstrings in geniml.search.query2vec.text2vec
        """
        self.query2vec = query2vec
        self.backend = backend

    def query_search(
        self,
        query: Union[str, np.ndarray],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
    ) -> List[Dict]:
        """

        :param query: the natural language query string, or a vector in the embedding space of region sets

        for rest of the parameters, check the docstring of QdrantBackend.search() or HNSWBackend.search()
        """
        if isinstance(query, np.ndarray):
            search_vec = query
        else:
            search_vec = self.query2vec.forward(query)

        return self.backend.search(search_vec, limit, with_payload, with_vectors, offset)

    def eval(self, query_dict: Dict[str, List[Union[int, np.int64]]]) -> Dict[str, float]:
        """
        With a query dictionary, return the Mean Average Precision, AUC-ROC and R-precision of query retrieval

        :param query_dict:a dictionary that contains query and relevant results in this format:
            {
                <query string>:[
                    <store id in backend>,
                    ...
                ],
                ...
            }

        :return: a Tuple of (Mean Average Precision, Average AUC-ROC, Average R-precision)
        """

        # number
        n = len(self.backend)

        # set ef for search
        # ef cannot be set lower than the number of queried nearest neighbors k

        if isinstance(self.backend, HNSWBackend):
            self.backend.idx.set_ef(n)

        sum_ap = 0  # sum of all average precision
        sum_auc = 0  # sum of all AUC-ROC
        sum_rp = 0  # sum of all R-Precision

        # total number of queries
        query_count = 0

        k = n  # to rank all results
        # evaluate each retrieval
        for query_str in query_dict.keys():
            relevant_results = query_dict[query_str]  # set of relevant ids

            try:
                search_results = self.query_search(
                    query=query_str, limit=k, with_vectors=False, with_payload=False
                )
                query_count += 1
            except Exception as e:
                _LOGGER.error(f"{query_str} caused {e}")
                continue

            ap, auc, rp = single_query_eval(search_results, relevant_results)
            sum_ap += ap
            sum_auc += auc
            sum_rp += rp

        if query_count > 0:
            return {
                "Mean Average Precision": sum_ap / query_count,
                "Mean AUC-ROC": sum_auc / query_count,
                "Average R-Precision": sum_rp / query_count,
            }

        else:
            return {"Mean Average Precision": 0.0, "Mean AUC-ROC": 0.0, "Average R-Precision": 0.0}
