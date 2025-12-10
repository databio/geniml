import json
import os.path
import pickle
from typing import Dict, List, Union

import hnswlib
import yaml

from ... import _LOGGER

DEP_HNSWLIB = True
# try:
#
#
#
# except ImportError:
#     DEP_HNSWLIB = False
#     _LOGGER.error(
#         "HNSWBackend requires hnswlib. Install hnswlib, or ignore this if you don't need HNSWBackend"
#     )

import numpy as np

from geniml.search.const import (
    DEFAULT_DIM,
    DEFAULT_EF,
    DEFAULT_HNSW_SPACE,
    DEFAULT_INDEX_PATH,
    DEFAULT_M,
)

from ..utils import verify_load_inputs
from .abstract import EmSearchBackend

# if not DEP_HNSWLIB:
#
#     class HNSWBackend(EmSearchBackend):
#         pass
#
# else:


class HNSWBackend(EmSearchBackend):
    """A search backend that uses a local HNSW index to store and search embeddings"""

    # instance variables, should not be class variables

    def __init__(
        self,
        local_index_path: str = DEFAULT_INDEX_PATH,
        payloads: Union[dict, str] = dict(),
        space: str = DEFAULT_HNSW_SPACE,
        dim: int = DEFAULT_DIM,
        ef: int = DEFAULT_EF,
        m: int = DEFAULT_M,
    ):
        """Initialize the HNSWBackend.

        Args:
            local_index_path: local path where the index is saved to
            space: possible options are l2, cosine or ip
            dim: dimension of vectors that will be stored
            ef: defines a construction time/accuracy trade-off, higher ef -> more accurate but slower
            m: connected with internal dimensionality of the data, higher M -> higher accuracy/run_time when ef is fixed
            payloads: optional dictionary or path to payloads file
        """
        # super(HNSWBackend, self).__init__()
        # initiate the index
        self.idx = hnswlib.Index(space=space, dim=dim)  # possible options are l2, cosine or ip
        self.idx.init_index(max_elements=0, ef_construction=ef, M=m)

        # load from local index that already store vectors
        if os.path.exists(local_index_path):
            self.idx.load_index(local_index_path)
            _LOGGER.info(f"Using index {local_index_path} with {self.idx.element_count} points.")

            # load payloads:
            if isinstance(payloads, str):
                if payloads.endswith(".json"):
                    with open(payloads, "r") as f:
                        self.payloads = json.load(f)
                elif payloads.endswith(".pkl"):
                    self.payloads = pickle.load(open(payloads, "rb"))
                elif payloads.endswith(".yaml"):
                    with open(payloads, "r") as f:
                        self.payloads = yaml.load(f, Loader=yaml.SafeLoader)

                else:
                    raise ValueError(
                        f"payload should be either a json, pickle, or yaml file. you supplied: {payloads.split('.')[-1]}"
                    )
            else:
                self.payloads = payloads
            # self.payloads = {}
        # save the index to local file path
        else:
            _LOGGER.info(f"Index {local_index_path} does not exist, creating it.")
            self.idx.save_index(local_index_path)
            self.payloads = {}
        # self.payloads = payloads
        self.idx_path = local_index_path

    def load(
        self,
        vectors: np.ndarray,
        ids: Union[np.ndarray, None] = None,
        payloads: Union[List[Dict[str, str]], None] = None,
    ):
        """Upload embedding vectors into the HNSW index and store metadata.

        Args:
            vectors: embedding vectors, a np.ndarray with shape of (n, <vector size>)
            ids: list of n point ids, or None to generate ids automatically
            payloads: optional list of n dictionaries that contain vector metadata
        """

        # increase max_elements to contain new loadings
        current_max = self.idx.get_max_elements()

        if ids is None:
            new_max = current_max + vectors.shape[0]
            ids = np.arange(start=current_max, stop=new_max)
        else:
            new_max = ids.max() + 1

        # check if the number of embedding vectors and labels are same
        verify_load_inputs(vectors, ids, payloads)

        if payloads:
            for i in range(len(payloads)):
                self.payloads[ids[i]] = payloads[i]

        # update hnsw index and load embedding vectors
        self.idx.load_index(self.idx_path, max_elements=new_max)
        self.idx.add_items(vectors)

        # save hnsw index to local file
        self.idx.save_index(self.idx_path)

    def search(
        self,
        query: np.ndarray,
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
    ) -> Union[
        List[Dict[str, Union[int, float, Dict[str, str], np.ndarray]]],
        List[List[Dict[str, Union[int, float, Dict[str, str], np.ndarray]]]],
    ]:
        """Get the limit nearest neighbors for query vector(s).

        Args:
            query: the query vector, np.ndarray with shape of (1, dim) or (dim, )
            limit: number of nearest neighbors to search for query vector
            with_payload: whether payload is included in the result
            with_vectors: whether the stored vector is included in the result
            offset: the offset of the search results

        Returns:
            If query shape is (<dim>, ), returns a list of limit dictionaries in format:
            {"id": <id>, "distance": <distance>, "payload": {...}, "vector": [...]}
            If query shape is (n, <dim>), returns a 2D list where each sublist has limit dictionaries.
        """
        ids, distances = self.idx.knn_query(query, k=limit + offset)
        # ids and distances are 2d array
        ids = ids.tolist()
        distances = distances.tolist()

        output_list = []
        for i in range(len(ids)):
            search_list = []
            result_id = ids[i]
            result_distances = distances[i]
            if with_vectors:
                result_vectors = self.idx.get_items(result_id, return_type="numpy")
            for j in range(limit):
                output_dict = {"id": result_id[j], "distance": result_distances[j]}
                if with_payload:
                    output_dict["payload"] = self.payloads[result_id[j]]
                if with_vectors:
                    output_dict["vector"] = result_vectors[j]
                search_list.append(output_dict)
            output_list.append(search_list)

        if len(output_list) == 1:
            return output_list[0]
        else:
            return output_list

    def __len__(self) -> int:
        return self.idx.element_count

    def retrieve_info(self, ids: Union[List[int], int], with_vectors: bool = False) -> Union[
        Dict[str, Union[int, List[float], Dict[str, str]]],
        List[Dict[str, Union[int, List[float], Dict[str, str]]]],
    ]:
        """Return the information of vectors for given storage ids.

        Args:
            ids: storage id, or a list of ids
            with_vectors: whether the stored vector is included in the result

        Returns:
            If single id, returns a dictionary. If list of ids, returns a list of dictionaries.
        """
        if not isinstance(ids, list):
            # retrieve() only takes iterable input
            ids = [ids]
        output_list = []
        for id_ in ids:
            output_dict = {"id": id_, "payload": self.payloads[id_]}
            output_list.append(output_dict)

        if with_vectors:
            vecs = self.idx.get_items(ids, return_type="numpy")
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
