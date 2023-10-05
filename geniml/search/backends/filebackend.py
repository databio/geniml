from typing import Dict, List, Tuple, Union

import hnswlib
import numpy as np
from ..const import *
from ..utils import verify_load_inputs

from .abstract import EmSearchBackend


class HNSWBackend(EmSearchBackend):
    """A search backend that uses a local HNSW index to store and search embeddings"""

    # the index
    idx: hnswlib.Index
    payloads: dict  # in the format of {<id>: <info dict>}, equivalent to payloads in Qdrant
    idx_path: str  # local path where the index is saved to

    def __init__(
        self,
        local_index_path: str = DEFAULT_INDEX_PATH,
        space: str = DEFAULT_HNSW_SPACE,
        dim: int = DEFAULT_DIM,
        ef: int = DEFAULT_EF,
        m: int = DEFAULT_M,
    ):
        """
        Initiate the backend

        :param local_index_path: local path where the index is saved to
        :param space: possible options are l2, cosine or ip
        :param dim: dimension of vectors that will be stored
        :param ef: defines a construction time/accuracy trade-off, higher ef -> more accurate but slower
        :param m: connected with internal dimensionality of the data, higher M -> higher accuracy/run_time
        when ef is fixed
        """

        # initiate the index
        self.idx = hnswlib.Index(space=space, dim=dim)  # possible options are l2, cosine or ip
        self.idx.init_index(max_elements=0, ef_construction=ef, M=m)

        # save the index to local file path
        self.idx.save_index(local_index_path)
        self.payloads = {}
        self.idx_path = local_index_path

    def load(self,
        vectors: np.ndarray,
        ids: Union[np.ndarray, None]=None,
        payloads: Union[List[Dict[str, str]], None]=None):
        """
        Upload embedding vectors into the hnsw index, and store their hnsw index id and payloads into metadata

        :param vectors: embedding vectors, a np.ndarray with shape of (n, <vector size>)
        :param ids: list of n point ids, or None to generate ids automatically
        :param payloads: optional list of n dictionaries that contain vector metadata
        :return:
        """
            
        # increase max_elements to contain new loadings
        current_max = self.idx.get_max_elements()
        
        if not ids:
            new_max = current_max + embeddings.shape[0]
            ids = np.arange(start=current_max, stop=new_max)
        else:
            new_max = ids.amax()
            
        # check if the number of embedding vectors and labels are same
        verify_load_inputs(vectors, ids, payloads)

        if payloads:
            for i in range(len(payloads)):
                self.payloads[ids[i]] = payloads[i]

        # update hnsw index and load embedding vectors
        self.idx.load_index(self.idx_path, max_elements=new_max)
        self.idx.add_items(vectors, ids)

        # save hnsw index to local file
        self.idx.save_index(self.idx_path)

    def search(
        self, query: np.ndarray, k: int, with_payload: bool = True, with_vec: bool = True
    ) -> Union[
        List[Dict[str, Union[int, float, Dict[str, str], List[float]]]],
        List[List[Dict[str, Union[int, float, Dict[str, str], List[float]]]]],
    ]:
        """
        With query vector(s), get the k nearest neighbors.

        :param query: the query vector, np.ndarray with shape of (1, dim) or (dim, )
        :param k: number of nearest neighbors to search for query vector
        :param with_payload: whether payload is included in the result
        :param with_vec: whether the stored vector is included in the result
        :return: if the shape of query vector is (<dim>, ), a list of k dictionaries will be returned,
        the format of dictionary will be:
        {
            "id": <id>
            "distance": <distance>
            "payload": {
                <information of the vector>
            }
            "vector": [<the vector>]
        }
        if the shape of query vector is (n, <dim>), a 2d list will be returned,
        which is a list of n * list of k dictionaries
        """
        ids, distances = self.idx.knn_query(query, k)
        # ids and distances are 2d array
        ids = ids.tolist()
        distances = distances.tolist()

        output_list = []
        for i in range(len(ids)):
            search_list = []
            result_id = ids[i]
            result_distances = distances[i]
            if with_vec:
                result_vectors = self.idx.get_items(result_id)
            for j in range(k):
                output_dict = {"id": result_id[j], "distance": result_distances[j]}
                if with_payload:
                    output_dict["payload"] = self.payloads[result_id[j]]
                if with_vec:
                    output_dict["vector"] = result_vectors[j]
                search_list.append(output_dict)
            output_list.append(search_list)

        if len(output_list) == 1:
            return output_list[0]
        else:
            return output_list

    def __len__(self) -> int:
        return self.idx.element_count

    def retrieve_info(
        self, ids: Union[List[int], int], with_vec: bool = False
    ) -> Union[
        Dict[str, Union[int, List[float], Dict[str, str]]],
        List[Dict[str, Union[int, List[float], Dict[str, str]]]],
    ]:
        """
        With an id or a list of storage ids, return the information of these vectors
        :param ids: storage id, or a list of ids
        :param with_vec: whether the stored vector is included in the result
        :return:
        """
        if not isinstance(ids, list):
            # retrieve() only takes iterable input
            ids = [ids]
        output_list = []
        for id_ in ids:
            output_dict = {"id": id_, "payload": self.payloads[id_]}
            output_list.append(output_dict)

        if with_vec:
            vecs = self.idx.get_items(ids)
            for i in range(len(vecs)):
                output_list[i]["vector"] = vecs[i]

        # with just one id, only the dictionary instead of the list will be returned
        if len(output_list) == 1:
            return output_list[0]
        else:
            return output_list

    def __str__(self):
        return "HNSWBackend with {} items".format(len(self))

    def __repr__(self):
        return "HNSWBackend with {} items".format(len(self))
