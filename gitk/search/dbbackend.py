from .abstract import EmSearchBackend
from .const import *
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
import numpy as np
from typing import List, Dict, Union


class QdrantBackend(EmSearchBackend):
    """A search backend that uses a qdrant server to store and search embeddings"""

    def __init__(self, config: VectorParams = DEFAULT_CONFIG,
                 collection: str = DEFAULT_COLLECTION):
        """
        Connect to Qdrant on commandline first:
        (Ubuntu Linux terminal)
        sudo docker run -p 6333:6333     -v $(pwd)/qdrant_storage:/qdrant/storage     qdrant/qdrant

        :param config: the vector parameter
        :param collection: name of collection
        """
        self.collection = collection
        self.config = config
        # self.qd_client = get_qdrant(True)
        self.qd_client = QdrantClient('http://localhost:6333')
        self.qd_client.recreate_collection(collection_name=self.collection,
                                           vectors_config=self.config)
        # TODO: initialize connection to qdrant server

    def load(self,
             embeddings: np.ndarray,
             labels: List[Dict[str, str]]):
        """
        upload vectors and their labels onto qdrant storage

        :param embeddings: embedding vectors of bed files, an np.ndarray with shape of (n, <vector size>)
        :param labels: list of labels, can be name of bed files
        :return:
        """

        if embeddings.shape[0] != len(labels):
            raise KeyError("The number of embeddings does not match the number of labels")

        start = len(self)
        self.qd_client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=i + start,
                    vector=embeddings[i].tolist(),
                    payload=labels[i]
                )
                for i in range(len(labels))
            ]
        )

    def search(self, query: np.ndarray, k: int) -> List:
        """
        with a given query vector, get k nearest neighbors from vectors in the collection

        :param query: a vector to search
        :param k: number of returned results
        :return: a list of search results
        """
        search_results = self.qd_client.search(
            collection_name=self.collection,
            query_vector=query,
            limit=k,
            with_payload=True
        )
        # raise NotImplementedError
        return search_results

    def __len__(self) -> int:
        """
        Return the number of embeddings in the backend
        """
        return self.qd_client.get_collection(collection_name=self.collection).vectors_count

    def retrieve_info(self, key: Union[List[int], int],
                      with_vecs: bool = False) -> Dict[Union[str, int], Union[str, List[int]]]:
        """
        With a given list of storage ids, return the information of these vectors

        :param key: list of ids
        :param with_vecs: whether the vectors themselves will also be returned in the output
        :return: a dictionary in this format:
        {
            <id>: {
                ...(information from payloads)
                "vector": <vector>
            }
        }
        """

        if not isinstance(key, list):
            # retrieve() only takes iterable input
            key = [key]

        output_dict = {}

        # get the information
        retrievals = self.qd_client.retrieve(
            collection_name=self.collection,
            ids=key,
            with_payload=True,
            with_vectors=with_vecs  # no need vectors
        )

        for result in retrievals:
            output_dict[result.id] = result.payload

            if with_vecs:
                output_dict[result.id]["embedding"] = result.vector

        return output_dict
