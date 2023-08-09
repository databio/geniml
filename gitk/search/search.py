from abc import ABC, abstractmethod
import numpy as np
from qdrant_client import QdrantClient
from .const import *
from typing import List, Tuple, Union
from ..models import ExModel
from ..region2vec import Region2VecExModel
from .utils import get_qdrant
from ..tokenization import InMemTokenizer
from qdrant_client.models import VectorParams, Distance, PointStruct


class EmSearchBackend(ABC):
    """
    An abstract class representing Embedding Search Backends. This allows
    backends to be either a qdrant server or a local in-memory NMS index, or
    anything, really. This allows us to use the same interface for both.
    """

    def __init__(self, embeddings: np.ndarray = None, labels: list = None) -> None:
        if embeddings:
            self.load(embeddings, labels)

    @abstractmethod
    def load(self, embeddings: np.ndarray, labels: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Search for the nearest neighbors of the given embedding

        :param query: the embedding to search for
        :param k: the number of results to return
        :return: a list of (id, score) pairs
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of embeddings in the backend
        """
        raise NotImplementedError()


class QdrantBackend:
    """A search backend that uses a qdrant server to store and search embeddings"""

    def __init__(self, config: VectorParams,
                 collection: str = "embeddings"):
        self.collection = collection
        self.config = config
        # self.qd_client = get_qdrant(True)
        self.qd_client = QdrantClient('http://localhost:6333')
        self.qd_client.recreate_collection(collection_name=self.collection,
                                           vectors_config=self.config)
        # TODO: initialize connection to qdrant server

    def load(self,
             embeddings: Union[np.ndarray, List[Union[List, np.ndarray]]],
             labels: list):
        if embeddings.shape[0] != len(labels):
            raise KeyError("The number of embeddings does not match the number of labels")
        self.qd_client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload={"label": labels[i]}
                )
                for i in range(len(labels))
            ]
        )

    def search(self, query: np.ndarray, k: int) -> List:
        search_results = self.qd_client.search(
            collection_name=self.collection,
            query_vector=query,
            limit=k,
            with_payload=True
        )
        # raise NotImplementedError
        return search_results

#
# class HNSWBackend(EmSearchBackend):
#     """A search backend that uses a local HNSW index to store and search embeddings"""
#
#     idx: hnswlib.Index
#
#     def __init__(self, embeddings: np.ndarray, labels: list, metadata: dict = None) -> None:
#         self.labels = labels
#         self.metadata = metadata
#
#         # create an HNSW index for the embeddings
#         dim = embeddings.shape[1]
#         self.idx = hnswlib.Index(space="l2", dim=dim)  # possible options are l2, cosine or ip
#         self.idx.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
#         self.idx.add_items(embeddings, labels)
#
#     def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
#         return self.idx.knn_query(query, k)
#
#     def __len__(self) -> int:
#         return idx.element_count()
#
#     def __getitem__(self, key) -> np.ndarray:
#         if metadata:
#             ret = metadata[key]
#         else:
#             ret = {}
#         ret["embedding"] = self.idx.get_items([key])[0]
#         return ret
#
#     def __str__(self):
#         return "HNSWBackend with {} items".format(len(self))
#
#     def save(self, path):
#         with open(path, "wb") as f:
#             pickle.dump(self, f)
#
#     def load(self, path):
#         self = pickle.load(path)
#
#
# class BEDSpaceSearchInterface(object):
#     def __init__(self, region_embeddings, label_embeddings, universe, tokenizer):
#         self.universe = universe
#         self.tokenizer = tokenizer
#         # TODO: Allow for different backends. It would just require connecting to the backend here
#         self.label_backend = HNSWBackend(label_embeddings, labels, metadata)
#         self.region_set_backend = HNSWBackend(region_embeddings, labels, metadata)
#
#     def search_labels_by_embedding(self, embedding, k: int = 10) -> embeddings:
#         """Given an input region set embedding, suggest labels for that region set"""
#         return self.label_backend.search(region_set_embedding, k)
#
#     def search_region_sets_by_embedding(self, embedding, k: int = 10) -> embeddings:
#         return self.region_set_backend.search(embedding, k)
#
#     def search_region_sets(self, label, k: int = 10) -> embeddings:
#         """Given an input label, suggest region sets for that label"""
#
#         # get the embedding for the label
#         # TODO: handle case where label is not in the universe
#         label_embedding = self.label_backend[label]
#         return self.region_set_backend.search(label_embedding, k)
#
#     def search_labels(self, region_set, k: int = 10) -> embeddings:
#         """Given an input region set, suggest labels for that region set"""
#
#         region_set_embedding = self.get_region_set_embedding(tokenized_region_set)
#         return self.label_backend.search(region_set_embedding, k)
#
#     def search_region_sets_by_region_set(self, region_set, k: int = 10) -> embeddings:
#         """Given an input region set, suggest region sets similar to that region set"""
#
#         region_set_embedding = self.get_region_set_embedding(region_set)
#         return self.region_set_backend.search(region_set_embedding, k)
#
#     def get_region_set_embedding(self, region_set):
#         """Given a region set, return the embedding for that region set"""
#
#         # first, tokenize the region set using the universe
#         tokenized_region_set = self.tokenizer.tokenize(region_set, self.universe)
#         # average the embeddings of each region in the region set
#         region_set_embedding = self.get_region_set_embedding(tokenized_region_set)
#         return region_set_embedding
#
#
# # Demo of how to use the BEDspace search interface
#
# BBSI = BEDspaceSearchInterface(...)
#
# # Find region sets for a label (Scenario 1, l2r)
# BBSI.search_region_sets_by_label("K562")  # returns nearest embeddings
#
# # Annotate region sets with labels (Scenario 2, r2l)
# path_to_bed_file = "path/to/bed/file.bed"
# region_set = RegionSet(path_to_bed_file)
# suggested_labels = BBSI.search_labels_by_region_set(region_set, k=15)
#
# # Find region sets similar to a query region set (Scenario 3, r2r)
# path_to_bed_file = "path/to/bed/file.bed"
# region_set = RegionSet(path_to_bed_file)
# BBSI.search_region_sets_by_region_set(region_set, k=10)
