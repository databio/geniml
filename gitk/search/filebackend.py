from .abstract import EmSearchBackend
from .const import *
import hnswlib
import numpy as np
from typing import List, Tuple, Union


class HNSWBackend(EmSearchBackend):
    """A search backend that uses a local HNSW index to store and search embeddings"""

    # the index
    idx: hnswlib.Index
    metadata: dict # in the format of {<id>: <label>}
    idx_path: str # local path where the index is saved to

    def __init__(self,
                 local_index_path: str = DEFAULT_INDEX_PATH,
                 space: str = DEFAULT_HNSW_SPACE,
                 dim: int = DEFAULT_DIM,
                 ef: int = DEFAULT_EF,
                 m: int = DEFAULT_M):
        """
        initiate the backend

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
        self.metadata = {}
        self.idx_path = local_index_path

    def load(self,
             embeddings: np.ndarray,
             labels: List[str]):
        """
        Load embedding vectors into the hnsw index, and store their hnsw index id and label into metddata

        :param embeddings: embedding vectors
        :param labels: labels
        :return:
        """
        # check if the number of embedding vectors and labels are same
        if embeddings.shape[0] != len(labels):
            raise KeyError("The number of embeddings does not match the number of labels")

        # increase max_elements to contain new loadings
        current_max = self.idx.get_max_elements()
        new_max = current_max + embeddings.shape[0]

        # set ids and store id: label into metadata
        ids = np.arange(start=current_max, stop=new_max)
        for i in range(len(labels)):
            self.metadata[ids[i]] = labels[i]

        # update hnsw index and load embedding vectors
        self.idx.load_index(self.idx_path, max_elements=new_max)
        self.idx.add_items(embeddings, ids)

        # save hnsw index to local file
        self.idx.save_index(self.idx_path)

    def search(self, query: np.ndarray, k: int) -> Tuple[Union[List[str], List[List[str]]], np.ndarray, np.ndarray]:
        """
        with query vector(s), get the k nearest neighbors

        :param query: the query vector, np.ndarray with shape of (1, dim) or (dim, )
        :param k: number of nearest neighbors to search for query vector
        :return: a tuple of search results that consist of labels, ids, and distances
        """
        ids, distances = self.idx.knn_query(query, k)
        if len(query.shape) == 1:
            ids = ids.reshape(ids.shape[1], )
            distances = distances.reshape(distances.shape[1], )
            labels = self._get_labels(ids)
        else:
            labels = []
            for search_result in ids:
                labels.append(self._get_labels(search_result))
        return labels, ids, distances

    def __len__(self) -> int:
        return self.idx.element_count

    def _get_labels(self, ids: Union[np.uint64, int, np.ndarray]) -> Union[str, List[str]]:
        """
        With given one or a list of storage ids, return the matching labels

        :param ids:
        :return:
        """
        if isinstance(ids, np.ndarray):
            return[self.metadata[num] for num in ids]
        else:
            return self.metadata[ids]

    # def __getitem__(self, key) -> np.ndarray:
    #     if metadata:
    #         ret = metadata[key]
    #     else:
    #         ret = {}
    #     ret["embedding"] = self.idx.get_items([key])[0]
    #     return ret

    def __str__(self):
        return "HNSWBackend with {} items".format(len(self))