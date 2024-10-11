import logging
import os
from typing import Dict, List, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest
from qdrant_client.models import Distance, PointStruct, VectorParams

from geniml.const import PKG_NAME
from geniml.search.const import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DIM,
    DEFAULT_QDRANT_DIST,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
    DEFAULT_QUANTIZATION_CONFIG,
)

from ..utils import verify_load_inputs
from .abstract import EmSearchBackend

_LOGGER = logging.getLogger(PKG_NAME)


def queries_to_requests(
    queries: np.ndarray,
    limit: int,
    with_payload: bool = True,
    with_vectors: bool = True,
    offset: int = 0,
) -> List[SearchRequest]:
    """
    Prepare all search requests for each query vector in a batch

    :param queries: see docstring of QdrantBackend.batch_search
    :param limit:
    :param with_payload:
    :param with_vectors:
    :param offset:
    """
    requests = []
    for query in queries:
        if query.ndim > 1:
            # that each request is from one single query vector
            requests.extend(queries_to_requests(query, limit, with_payload, with_vectors, offset))
        else:
            requests.append(
                SearchRequest(
                    vector=query,
                    limit=limit,
                    with_vector=with_vectors,
                    with_payload=with_payload,
                    offset=offset,
                )
            )
    return requests


def results_processing(search_results, with_payload: bool, with_vectors: bool) -> List[Dict]:
    """
    Process the search result into unified format: list of dictionaries

    :param search_results: result of qdrant client similarity search
    :type search_results: search result of qdrant client
    :param with_payload: see docstring of QdrantBackend.search
    :param with_vectors:
    """
    output_list = []
    for result in search_results:
        # build each dictionary
        result_dict = {"id": result.id, "score": result.score}
        if with_payload:
            result_dict["payload"] = result.payload
        if with_vectors:
            result_dict["vector"] = result.vector
        output_list.append(result_dict)
    return output_list


class QdrantBackend(EmSearchBackend):
    """A search backend that uses a qdrant server to store and search embeddings"""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        dist: Distance = DEFAULT_QDRANT_DIST,
        collection: str = DEFAULT_COLLECTION_NAME,
        qdrant_host: str = DEFAULT_QDRANT_HOST,
        qdrant_port: int = DEFAULT_QDRANT_PORT,
        qdrant_api_key: str = None,
    ):
        """
        Connect to Qdrant on commandline first:
        (Ubuntu Linux terminal)
        sudo docker run -p 6333:6333     -v $(pwd)/qdrant_storage:/qdrant/storage     qdrant/qdrant

        :param config: the vector parameter
        :param collection: name of collection
        :param qdrant_host: host of qdrant server
        :param qdrant_port: port of qdrant server
        :param qdrant_api_key: api key
        """
        super().__init__()
        self.collection = collection
        self.config = VectorParams(size=dim, distance=dist)
        self.url = os.environ.get("QDRANT_HOST", qdrant_host)
        self.port = os.environ.get("QDRANT_PORT", qdrant_port)
        self.qd_client = QdrantClient(
            url=self.url,
            port=self.port,
            api_key=os.environ.get("QDRANT_API_KEY", qdrant_api_key),
        )

        # Create collection only if it does not exist
        try:
            collection_info = self.qd_client.get_collection(collection_name=self.collection)
            _LOGGER.info(
                f"Using collection {self.collection} with {collection_info.points_count} points."
            )
        except Exception:  # qdrant_client.http.exceptions.UnexpectedResponse
            _LOGGER.info(f"Collection {self.collection} does not exist, creating it.")
            self.qd_client.recreate_collection(
                collection_name=self.collection,
                vectors_config=self.config,
                quantization_config=DEFAULT_QUANTIZATION_CONFIG,
            )

    def load(
        self,
        vectors: np.ndarray,
        ids: Union[List[str], None] = None,
        payloads: Union[List[Dict[str, str]], None] = None,
    ):
        """
        Upload vectors and their labels into qdrant storage.

        :param vectors: embedding vectors, a np.ndarray with shape of (n, <vector size>)
        :param ids: list of n point ids, or None to generate ids automatically
        :param payloads: optional list of n dictionaries that contain vector metadata
        :return:
        """

        if not ids:
            start = len(self)
            ids = list(range(start, start + len(payloads)))

        verify_load_inputs(vectors, ids, payloads)

        points = [
            PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=payloads[i])
            for i in range(len(payloads))
        ]
        self.qd_client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(
        self,
        query: np.ndarray,
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
    ) -> Union[
        List[Dict[str, Union[int, float, Dict[str, str], List[float]]]],
        List[List[Dict[str, Union[int, float, Dict[str, str], List[float]]]]],
    ]:
        """
         with a given query vector, get k nearest neighbors from vectors in the collection

        :param query: a vector to search
        :param limit: number of returned results
        :param with_payload: whether payload is included in the result
        :param with_vectors: whether the stored vector is included in the result
        :param offset: the offset of the search results
        :return: a list of dictionary that contains the search results in this format:
        {
            "id": <id>
            "score": <score>
            "payload": {
                <information of the vector>
            }
            "vector": [<the vector>]
        }
        """
        if query.ndim > 1:
            return self.batch_search(query, limit, with_payload, with_vectors, offset)
        # KNN search in qdrant client
        search_results = self.qd_client.search(
            collection_name=self.collection,
            query_vector=query,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            offset=offset,
        )

        # add the results in to the output list
        return results_processing(search_results, with_payload, with_vectors)

    def batch_search(
        self,
        queries: np.ndarray,
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
    ) -> List[List[Dict[str, Union[int, float, Dict[str, str], List[float]]]]]:
        """

        :param queries: multiple search vectors, np.ndarray with shape of (n, dim)
        :param limit: see docstring of def search
        :type limit:
        :param with_payload:
        :param with_vectors:
        :param offset:
        :return: results of all search requests with each vector in queries
        """
        output_list = []
        # build all search requests
        requests = queries_to_requests(queries, limit, with_payload, with_vectors, offset)

        search_results = self.qd_client.search_batch(
            collection_name=self.collection, requests=requests
        )

        # add the results in to the output list
        for batch in search_results:
            batch_list = results_processing(batch, with_payload, with_vectors)
            output_list.append(batch_list)
        return output_list

    def __len__(self) -> int:
        """
        Return the number of embeddings in the backend
        """
        return self.qd_client.get_collection(collection_name=self.collection).vectors_count

    def retrieve_info(
        self, ids: Union[List[int], int, List[str], str], with_vectors: bool = False
    ) -> Union[
        Dict[str, Union[int, str, List[float], Dict[str, str]]],
        List[Dict[str, Union[int, str, List[float], Dict[str, str]]]],
    ]:
        """
        With a given list of storage ids, return the information of these vectors

        :param ids: list of ids, or a single id
        :param with_vectors:  whether the vectors themselves will also be returned in the output
        :return: if ids is one id, a dictionary similar to the output of search() will be returned, without "score";
        if ids is a list, a list of dictionaries will be returned
        """
        if not isinstance(ids, list):
            # retrieve() only takes iterable input
            ids = [ids]

        # add hyphen to uuid if missing
        for i in range(len(ids)):
            id_ = ids[i]
            if isinstance(id_, str):
                if not "-" in id_:
                    ids[i] = f"{id_[:8]}-{id_[8:12]}-{id_[12:16]}-{id_[16:20]}-{id_[20:]}"

        output_list = []
        retrievals = self.qd_client.retrieve(
            collection_name=self.collection,
            ids=ids,
            with_payload=True,
            with_vectors=with_vectors,  # no need vectors
        )

        retrieval_dict = {result.id: result for result in retrievals}

        # retrieve() of qd client does not return result in the order of ids in the list
        # get the retrieval result in output by id order
        for id_ in ids:
            try:
                result = retrieval_dict[id_]
            except:
                _LOGGER.warning(f"Warning: no id stored in backend matches {id_}.")
                continue
            result_dict = {"id": result.id, "payload": result.payload}
            if with_vectors:
                result_dict["vector"] = result.vector
            output_list.append(result_dict)

        # with just one id, only the dictionary instead of the list will be returned
        if len(output_list) == 1:
            return output_list[0]
        else:
            return output_list

    def __str__(self):
        n_items = len(self)
        msg = f"""QdrantBackend
            n items: {n_items}
            url: {self.url}:{self.port},
            collection: {self.collection}
            """
        return msg

    def __repr__(self):
        n_items = len(self)
        msg = f"""QdrantBackend
            n items: {n_items}
            url: {self.url}:{self.port},
            collection: {self.collection}
            """
        return msg
